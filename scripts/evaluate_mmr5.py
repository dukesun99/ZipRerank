#!/usr/bin/env python3
"""
Evaluate MM-R5 reranker on MMDocIR benchmark.

MM-R5 is a Multimodal Reasoning-Enhanced ReRanker that uses Qwen2.5-VL with
chain-of-thought reasoning for document retrieval.

Reference: https://github.com/i2vec/MM-R5
Paper: https://arxiv.org/abs/2506.12364
"""

import argparse
import io
import json
import os
import pickle
import random
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class InferenceTimer:
    """Timer for measuring inference component timings using forward hooks."""
    
    def __init__(self, model):
        self.model = model
        self.vision_time_ms = 0.0
        self._vision_start = None
        self._hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to measure component timings."""
        visual_module = None
        if hasattr(self.model, 'visual'):
            visual_module = self.model.visual
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'visual'):
            visual_module = self.model.model.visual
        
        if visual_module is not None:
            def vision_pre_hook(module, input):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                self._vision_start = time.perf_counter()
            
            def vision_post_hook(module, input, output):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                if self._vision_start is not None:
                    self.vision_time_ms = (time.perf_counter() - self._vision_start) * 1000
                    self._vision_start = None
            
            self._hooks.append(visual_module.register_forward_pre_hook(vision_pre_hook))
            self._hooks.append(visual_module.register_forward_hook(vision_post_hook))
    
    def reset(self):
        """Reset timings for a new inference."""
        self.vision_time_ms = 0.0
        self._vision_start = None
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


# Global inference timer (set after model loading)
_inference_timer: Optional[InferenceTimer] = None


@dataclass
class EvalStats:
    """Statistics accumulator for MM-R5 evaluation metrics."""
    
    total_queries: int = 0
    total_windows: int = 0
    total_vision_time_ms: float = 0.0
    total_llm_time_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_image_tokens: int = 0
    
    def add_window(
        self,
        llm_time_ms: float,
        input_tokens: int,
        output_tokens: int,
        image_tokens: int,
        vision_time_ms: float = 0.0,
    ):
        """Record stats for a single reranking window."""
        self.total_windows += 1
        self.total_llm_time_ms += llm_time_ms
        self.total_vision_time_ms += vision_time_ms
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_image_tokens += image_tokens
    
    @property
    def total_inference_time_ms(self) -> float:
        return self.total_vision_time_ms + self.total_llm_time_ms
    
    @property
    def avg_vision_time_ms(self) -> float:
        if self.total_windows == 0:
            return 0.0
        return self.total_vision_time_ms / self.total_windows
    
    @property
    def avg_llm_time_ms(self) -> float:
        if self.total_windows == 0:
            return 0.0
        return self.total_llm_time_ms / self.total_windows
    
    @property
    def avg_input_tokens_per_window(self) -> float:
        if self.total_windows == 0:
            return 0.0
        return self.total_input_tokens / self.total_windows
    
    @property
    def avg_output_tokens_per_window(self) -> float:
        if self.total_windows == 0:
            return 0.0
        return self.total_output_tokens / self.total_windows
    
    @property
    def avg_image_tokens_per_window(self) -> float:
        if self.total_windows == 0:
            return 0.0
        return self.total_image_tokens / self.total_windows
    
    @property
    def image_token_ratio(self) -> float:
        if self.total_input_tokens == 0:
            return 0.0
        return self.total_image_tokens / self.total_input_tokens
    
    def report(self) -> str:
        """Generate a formatted statistics report."""
        lines = []
        lines.append("=" * 80)
        lines.append("EVALUATION STATISTICS (MM-R5)")
        lines.append("=" * 80)
        
        lines.append(f"Queries processed: {self.total_queries}")
        lines.append(f"Reranking windows: {self.total_windows}")
        
        lines.append("")
        lines.append("Token Statistics (avg per window):")
        lines.append(f"  Input tokens: {self.avg_input_tokens_per_window:,.0f}")
        lines.append(f"  Image tokens: {self.avg_image_tokens_per_window:,.0f} ({self.image_token_ratio:.1%} of input)")
        text_tokens = self.avg_input_tokens_per_window - self.avg_image_tokens_per_window
        text_ratio = 1.0 - self.image_token_ratio if self.total_input_tokens > 0 else 0.0
        lines.append(f"  Text tokens: {text_tokens:,.0f} ({text_ratio:.1%} of input)")
        lines.append(f"  Output tokens: {self.avg_output_tokens_per_window:,.0f}")
        
        lines.append("")
        lines.append("Inference Timing:")
        lines.append(f"  Mode: generation (autoregressive with CoT)")
        
        if self.total_vision_time_ms > 0:
            lines.append("")
            lines.append("  Component Breakdown (avg per window):")
            lines.append(f"    Vision Tower:     {self.avg_vision_time_ms:>8.1f} ms")
            lines.append(f"    LLM Decoder:      {self.avg_llm_time_ms:>8.1f} ms")
            avg_total = self.avg_vision_time_ms + self.avg_llm_time_ms
            lines.append(f"    ─────────────────────────────")
            lines.append(f"    Total:            {avg_total:>8.1f} ms/window")
            lines.append("")
            lines.append("  Component Breakdown (totals):")
            lines.append(f"    Vision Tower:     {self.total_vision_time_ms / 1000:>8.2f} s")
            lines.append(f"    LLM Decoder:      {self.total_llm_time_ms / 1000:>8.2f} s")
            lines.append(f"    ─────────────────────────────")
            lines.append(f"    Total:            {self.total_inference_time_ms / 1000:>8.2f} s")
        else:
            lines.append(f"  Avg time per window: {self.avg_llm_time_ms:.1f} ms")
            lines.append(f"  Total LLM time: {self.total_llm_time_ms / 1000:.2f} s")
        
        if self.total_llm_time_ms > 0:
            output_tokens_per_sec = (self.total_output_tokens / self.total_llm_time_ms) * 1000
            lines.append(f"  Output tokens/sec: {output_tokens_per_sec:.1f}")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)


# Global stats accumulator (set in main)
_eval_stats: Optional[EvalStats] = None

# MM-R5 prompts (from their reranker.py)
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

QUESTION_TEMPLATE = (
    "Please rank the following images according to their relevance to the question. "
    "Provide your response in the format: <think>your reasoning process here</think><answer>[image_id_1, image_id_2, ...]</answer> "
    "where the numbers in the list represent the ranking order of images'id from most to least relevant. "
    "Before outputting the answer, you need to analyze each image and provide your analysis process."
    "For example: <think>Image 1 shows the most relevant content because...</think><answer>[id_most_relevant, id_second_relevant, ...]</answer>"
    "\nThe question is: {Question}"
    "\n\nThere are {image_num} images, id from 1 to {image_num_end}, Image ID to image mapping:\n"
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate MM-R5 reranker on MMDocIR benchmark"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="i2vec/MM-R5",
        help="Path to MM-R5 model (default: i2vec/MM-R5 from HuggingFace)",
    )
    parser.add_argument(
        "--first_stage_file",
        type=str,
        required=True,
        help="Path to first-stage retrieval results pickle file",
    )
    parser.add_argument(
        "--pages_parquet",
        type=str,
        default="MMDocIR/dataset/MMDocIR_pages.parquet",
        help="Path to pages parquet file",
    )
    parser.add_argument(
        "--layouts_parquet",
        type=str,
        default="MMDocIR/dataset/MMDocIR_layouts.parquet",
        help="Path to layouts parquet file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["page", "layout"],
        required=True,
        help="Evaluation mode: page or layout",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=20,
        help="Number of candidates to rank at once",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Stride for sliding window (how much to shift window each round)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature for generation (default: 0.3 as in MM-R5)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8192,
        help="Maximum new tokens for generation",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save reranking results",
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=None,
        help="Number of queries to process (for testing). Overrides --sample_size.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Number of queries to sample for evaluation (default: 100). Use 0 for all queries.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling (default: 42)",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to log file for model outputs",
    )
    
    return parser.parse_args()


def get_image_from_binary(binary_data: bytes) -> Image.Image:
    """Convert binary image data to PIL Image at native resolution."""
    return Image.open(io.BytesIO(binary_data)).convert("RGB")


def save_temp_images(images: List[Image.Image], temp_dir: str) -> List[str]:
    """Save PIL images to temporary files and return paths."""
    paths = []
    for i, img in enumerate(images):
        path = os.path.join(temp_dir, f"img_{i}.jpg")
        img.save(path, format="JPEG", quality=95)
        paths.append(path)
    return paths


def parse_mmr5_output(output_text: str, num_candidates: int) -> List[int]:
    """
    Parse MM-R5's output format: <answer>[1, 2, 3, ...]</answer>
    
    Returns 0-indexed ranking with exactly num_candidates elements.
    """
    match = re.search(r'<answer>\[(.*?)\]</answer>', output_text)
    
    if match:
        try:
            predicted_order = []
            for x in match.group(1).split(','):
                x = x.strip()
                if not x:
                    continue
                
                if '-' in x and not x.startswith('-'):
                    first_part = x.split('-')[0].strip()
                    if first_part:
                        try:
                            predicted_order.append(int(float(first_part)) - 1)
                        except ValueError:
                            continue
                else:
                    try:
                        predicted_order.append(int(float(x)) - 1)
                    except ValueError:
                        continue
            
            seen = set()
            valid_order = []
            for idx in predicted_order:
                if 0 <= idx < num_candidates and idx not in seen:
                    valid_order.append(idx)
                    seen.add(idx)
            
            for i in range(num_candidates):
                if i not in seen:
                    valid_order.append(i)
                    seen.add(i)
            
            return valid_order[:num_candidates]
        except Exception as e:
            print(f"Parsing error: {str(e)}, output: {output_text[:200]}...")
    
    return list(range(num_candidates))


def rerank_with_mmr5(
    model,
    processor,
    query: str,
    image_paths: List[str],
    temperature: float = 0.3,
    max_new_tokens: int = 8192,
    log_file=None,
) -> Tuple[List[int], str]:
    """Rerank candidates using MM-R5."""
    global _eval_stats
    
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "qwen-vl-utils", "-q"])
        from qwen_vl_utils import process_vision_info
    
    num_candidates = len(image_paths)
    device = model.device
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(
                        Question=query,
                        image_num=num_candidates,
                        image_num_end=num_candidates
                    ),
                },
            ],
        },
    ]
    
    for i, image_path in enumerate(image_paths):
        messages[-1]["content"].extend([
            {"type": "text", "text": f"\nImage {i+1}: "},
            {"type": "image", "image": image_path},
        ])
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    
    input_tokens = inputs.input_ids.shape[1]
    
    image_tokens = 0
    if hasattr(inputs, 'image_grid_thw') and inputs.image_grid_thw is not None:
        spatial_merge_size = getattr(model.config, 'vision_config', {})
        if hasattr(spatial_merge_size, 'spatial_merge_size'):
            spatial_merge_size = spatial_merge_size.spatial_merge_size
        else:
            spatial_merge_size = 2
        image_tokens = (inputs.image_grid_thw.prod(dim=-1) // (spatial_merge_size ** 2)).sum().item()
    
    if _inference_timer is not None:
        _inference_timer.reset()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total_forward_time_ms = (time.perf_counter() - start_time) * 1000
    
    vision_time_ms = 0.0
    if _inference_timer is not None:
        vision_time_ms = _inference_timer.vision_time_ms
    
    llm_time_ms = total_forward_time_ms - vision_time_ms
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    output_tokens = generated_ids_trimmed[0].shape[0]
    
    if _eval_stats is not None:
        _eval_stats.add_window(
            llm_time_ms=llm_time_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            image_tokens=image_tokens,
            vision_time_ms=vision_time_ms,
        )
    
    ranked_indices = parse_mmr5_output(output_text, num_candidates)
    
    if log_file:
        log_file.write("="*80 + "\n")
        log_file.write(f"Query: {query}\n")
        log_file.write(f"Input tokens: {input_tokens} (image: {image_tokens})\n")
        log_file.write(f"Output tokens: {output_tokens}\n")
        log_file.write(f"Generation time: {llm_time_ms:.1f} ms\n")
        log_file.write("-"*80 + "\n")
        log_file.write(f"Generated output:\n{output_text}\n")
        log_file.write("-"*80 + "\n")
        log_file.write(f"Parsed ranking: {ranked_indices}\n")
        log_file.write("="*80 + "\n\n")
        log_file.flush()
    
    return ranked_indices, output_text


def sliding_window_rerank_mmr5(
    model,
    processor,
    query: str,
    qid: str,
    candidate_images: List[Image.Image],
    temp_dir: str,
    window_size: int,
    stride: int,
    temperature: float = 0.3,
    max_new_tokens: int = 8192,
    log_file=None,
) -> List[int]:
    """Apply sliding window reranking using MM-R5."""
    n_candidates = len(candidate_images)
    
    indices = list(range(n_candidates))
    current_images = list(candidate_images)
    
    end_pos = n_candidates
    start_pos = end_pos - window_size
    
    while end_pos > 0 and start_pos + stride != 0:
        start_pos = max(start_pos, 0)
        
        window_images = current_images[start_pos:end_pos]
        image_paths = save_temp_images(window_images, temp_dir)
        
        ranked_positions, _ = rerank_with_mmr5(
            model,
            processor,
            query,
            image_paths,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            log_file=log_file,
        )
        
        window_size_actual = end_pos - start_pos
        
        if len(ranked_positions) != window_size_actual:
            print(f"WARNING: ranked_positions length {len(ranked_positions)} != window size {window_size_actual}, using fallback")
            ranked_positions = list(range(window_size_actual))
        
        ranked_positions = [pos for pos in ranked_positions if 0 <= pos < window_size_actual]
        seen_pos = set(ranked_positions)
        for pos in range(window_size_actual):
            if pos not in seen_pos:
                ranked_positions.append(pos)
        ranked_positions = ranked_positions[:window_size_actual]
        
        reordered_indices = [indices[start_pos + pos] for pos in ranked_positions]
        reordered_images = [current_images[start_pos + pos] for pos in ranked_positions]
        
        for i, (idx, img) in enumerate(zip(reordered_indices, reordered_images)):
            if start_pos + i < len(indices):
                indices[start_pos + i] = idx
                current_images[start_pos + i] = img
        
        end_pos = end_pos - stride
        start_pos = start_pos - stride
    
    return indices


def load_first_stage_results(first_stage_file: str) -> Dict:
    """Load first-stage retrieval results."""
    with open(first_stage_file, 'rb') as f:
        return pickle.load(f)


def convert_results_to_official_format(results: List[Dict], mode: str) -> List[Dict]:
    """Convert reranker results to official MMDocIR format."""
    official_results = []
    
    for result in results:
        official_item = {
            "domain": result["domain"],
            "page_id": result["page_id"],
            "layout_mapping": result["layout_mapping"],
        }
        
        if mode == "page":
            start_idx = result["start_idx"]
            end_idx = result["end_idx"]
            num_pages_in_doc = end_idx - start_idx + 1
            
            scores_page = [-float('inf')] * num_pages_in_doc
            
            top_k_global_indices = result["top_k_global_indices"]
            rerank_scores = result["rerank_scores"]
            
            for i, global_idx in enumerate(top_k_global_indices):
                local_idx = global_idx - start_idx
                if 0 <= local_idx < num_pages_in_doc and i < len(rerank_scores):
                    scores_page[local_idx] = rerank_scores[i]
            
            official_item["scores_page"] = scores_page
            
        else:
            layout_indices = result.get("layout_indices", [])
            rerank_scores = result["rerank_scores"]
            
            official_item["layout_indices"] = layout_indices
            official_item["scores_layout"] = rerank_scores
        
        official_results.append(official_item)
    
    return official_results


def convert_first_stage_to_official_format(results: List[Dict], mode: str) -> List[Dict]:
    """Convert first-stage results to official MMDocIR format."""
    official_results = []
    
    for result in results:
        official_item = {
            "domain": result["domain"],
            "page_id": result["page_id"],
            "layout_mapping": result["layout_mapping"],
        }
        
        if mode == "page":
            start_idx = result["start_idx"]
            end_idx = result["end_idx"]
            num_pages_in_doc = end_idx - start_idx + 1
            
            scores_page = [-float('inf')] * num_pages_in_doc
            top_k_global_indices = result["top_k_global_indices"]
            n_candidates = len(top_k_global_indices)
            
            for i, global_idx in enumerate(top_k_global_indices):
                local_idx = global_idx - start_idx
                if 0 <= local_idx < num_pages_in_doc:
                    scores_page[local_idx] = n_candidates - i
            
            official_item["scores_page"] = scores_page
            
        else:
            layout_indices = result.get("layout_indices", [])
            top_k_global_indices = result["top_k_global_indices"]
            n_candidates = len(top_k_global_indices)
            
            scores_layout = [n_candidates - i for i in range(n_candidates)]
            official_item["layout_indices"] = layout_indices
            official_item["scores_layout"] = scores_layout
        
        official_results.append(official_item)
    
    return official_results


def print_domain_header():
    """Print the header row for domain-wise results table."""
    domain_list = [
        "Research", "Admin", "Tutorial", "Academic", "Brochure",
        "Financial", "Guidebook", "Government", "Laws", "News", "Avg", "Overall"
    ]
    header = " | ".join(f"{d:>8}" for d in domain_list)
    print(f"Domains: {header}")
    print("-"*120)


def compute_mmdocir_metrics(
    rerank_results: List[Dict],
    first_stage_results: List[Dict],
    mode: str,
    model_name: str = "MM-R5",
) -> Dict[int, float]:
    """Compute MMDocIR metrics using official evaluation functions."""
    from utils.metric_eval import evaluate_page, evaluate_layout
    
    print("\n" + "="*120)
    print(f"MMDocIR {mode.upper()}-LEVEL EVALUATION RESULTS")
    print("="*120)
    print("Using OFFICIAL MMDocIR evaluation functions")
    print("-"*120)
    
    print_domain_header()
    
    rerank_recalls = {}
    
    if mode == "page":
        topk_values = [1, 3, 5]
    else:
        topk_values = [1, 5, 10]
    
    print("\n[First-Stage]")
    for topk in topk_values:
        if mode == "page":
            evaluate_page(first_stage_results, model_name="First-Stage", topk=topk, metric="recall")
        else:
            evaluate_layout(first_stage_results, model_name="First-Stage", topk=topk, metric="recall")
    
    print(f"\n[Reranker ({model_name})]")
    for topk in topk_values:
        if mode == "page":
            evaluate_page(rerank_results, model_name=model_name, topk=topk, metric="recall")
        else:
            evaluate_layout(rerank_results, model_name=model_name, topk=topk, metric="recall")
    
    print("="*120)
    
    return rerank_recalls


def evaluate_mmdocir_mmr5(
    model,
    processor,
    first_stage_results: Dict,
    parquet_df: pd.DataFrame,
    mode: str,
    window_size: int,
    stride: int,
    temperature: float,
    max_new_tokens: int,
    num_queries: Optional[int] = None,
    sample_size: int = 100,
    seed: int = 42,
    log_file=None,
) -> List[Dict]:
    """Evaluate reranking on MMDocIR using MM-R5."""
    results = []
    
    query_keys = list(first_stage_results.keys())
    total_queries = len(query_keys)
    
    if num_queries:
        query_keys = query_keys[:num_queries]
        sampling_method = f"first {num_queries}"
    elif sample_size > 0 and sample_size < total_queries:
        random.seed(seed)
        query_keys = random.sample(query_keys, sample_size)
        sampling_method = f"random {sample_size} (seed={seed})"
    else:
        sampling_method = "all"
    
    print(f"Evaluating {len(query_keys)}/{total_queries} queries ({sampling_method})...")
    print(f"  Window size: {window_size}")
    print(f"  Stride: {stride}")
    print(f"  Temperature: {temperature}")
    print(f"  Max new tokens: {max_new_tokens}")
    print()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for key in tqdm(query_keys, desc="Reranking"):
            item = first_stage_results[key]
            query = item["query"]
            qid = f"{item['doc_name']}_{item['q_idx']}"
            top_k_global_indices = item["top_k_global_indices"]
            
            candidate_images = []
            for global_idx in top_k_global_indices:
                row = parquet_df.iloc[global_idx]
                img = get_image_from_binary(row['image_binary'])
                candidate_images.append(img)
            
            n_candidates = len(candidate_images)
            
            if n_candidates <= window_size:
                image_paths = save_temp_images(candidate_images, temp_dir)
                ranked_indices, output_text = rerank_with_mmr5(
                    model=model,
                    processor=processor,
                    query=query,
                    image_paths=image_paths,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    log_file=log_file,
                )
            else:
                ranked_indices = sliding_window_rerank_mmr5(
                    model=model,
                    processor=processor,
                    query=query,
                    qid=qid,
                    candidate_images=candidate_images,
                    temp_dir=temp_dir,
                    window_size=window_size,
                    stride=stride,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    log_file=log_file,
                )
            
            rerank_scores = [0.0] * n_candidates
            for rank, idx in enumerate(ranked_indices):
                if idx < n_candidates:
                    rerank_scores[idx] = n_candidates - rank
            
            result = {
                "doc_name": item["doc_name"],
                "domain": item["domain"],
                "q_idx": item["q_idx"],
                "query": query,
                "page_id": item["page_id"],
                "type": item["type"],
                "layout_mapping": item["layout_mapping"],
                "start_idx": item["start_idx"],
                "end_idx": item["end_idx"],
                "top_k_global_indices": top_k_global_indices,
                "ranked_indices": ranked_indices,
                "rerank_scores": rerank_scores,
            }
            
            if mode == "layout":
                result["layout_indices"] = item.get("top_k_layout_indices", [])
            
            results.append(result)
    
    return results


def main():
    args = parse_args()
    
    print("="*80)
    print("MMDocIR Reranking Evaluation (MM-R5)")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Mode: {args.mode}")
    print(f"First-stage file: {args.first_stage_file}")
    print(f"Window size: {args.window_size}")
    print(f"Stride: {args.stride}")
    print(f"Temperature: {args.temperature}")
    print(f"Sample size: {args.sample_size if args.sample_size > 0 else 'all'} (seed={args.seed})")
    print("="*80)
    print()
    
    # Load parquet data
    print("Loading parquet data...")
    if args.mode == "page":
        parquet_df = pd.read_parquet(args.pages_parquet)
    else:
        parquet_df = pd.read_parquet(args.layouts_parquet)
    print(f"Loaded {len(parquet_df)} rows")
    
    # Load first-stage results
    print("Loading first-stage retrieval results...")
    first_stage_results = load_first_stage_results(args.first_stage_file)
    print(f"Loaded {len(first_stage_results)} queries")
    
    # Load model
    print(f"Loading MM-R5 model from {args.model_path}...")
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.eval()
    print("Model loaded successfully!")
    
    # Initialize inference timer
    global _inference_timer
    _inference_timer = InferenceTimer(model)
    print(f"  Inference Timer: Initialized")
    
    # Initialize global stats accumulator
    global _eval_stats
    _eval_stats = EvalStats()
    
    # Open log file if specified
    log_file = None
    if args.log_file:
        print(f"Opening log file: {args.log_file}")
        log_file = open(args.log_file, 'w', encoding='utf-8')
        log_file.write("="*80 + "\n")
        log_file.write("MM-R5 LOG - MMDocIR Reranking Evaluation\n")
        log_file.write("="*80 + "\n")
        log_file.write(f"Model: {args.model_path}\n")
        log_file.write(f"Mode: {args.mode}\n")
        log_file.write(f"Window size: {args.window_size}\n")
        log_file.write("="*80 + "\n\n")
        log_file.flush()
    
    try:
        print("\nStarting reranking evaluation...")
        
        results = evaluate_mmdocir_mmr5(
            model=model,
            processor=processor,
            first_stage_results=first_stage_results,
            parquet_df=parquet_df,
            mode=args.mode,
            window_size=args.window_size,
            stride=args.stride,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            num_queries=args.num_queries,
            sample_size=args.sample_size,
            seed=args.seed,
            log_file=log_file,
        )
        
        official_rerank_results = convert_results_to_official_format(results, args.mode)
        official_first_stage_results = convert_first_stage_to_official_format(results, args.mode)
        
        compute_mmdocir_metrics(
            rerank_results=official_rerank_results,
            first_stage_results=official_first_stage_results,
            mode=args.mode,
            model_name="MM-R5",
        )
        
        _eval_stats.total_queries = len(results)
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + "\n")
            print(f"\nResults saved to {args.output_file}")
        
        print()
        print(_eval_stats.report())
    
    finally:
        if log_file:
            log_file.write("\n" + "="*80 + "\n")
            log_file.write("END OF LOG\n")
            log_file.write("="*80 + "\n")
            log_file.close()
            print(f"Log saved to: {args.log_file}")
    
    print()
    print("="*80)
    print("Evaluation completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()

