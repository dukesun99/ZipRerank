#!/usr/bin/env python3
"""
Evaluate ZipRerank reranking on MMDocIR benchmark.

Uses first-stage retrieval results (top-20) from DSE or ColQwen and applies
the Qwen3-VL reranker. Supports Query-Image Early Interaction (QI-EI) for
efficient visual token pruning.
"""

import argparse
import io
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoConfig
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "MMDocIR"))

# Reuse shared functions from utils
from utils.data_utils import create_ranking_prompt_for_training, prepare_ranking_inputs
from utils.train_utils import resize_image_if_needed
from models.qwen3vl_with_qi_early import Qwen3VLWithQIEarly

import re


def parse_generated_ranking(generated_text: str, window_size: int) -> List[int]:
    """
    Parse a generated ranking string like "A] > [B] > [C]" or "[A] > [B] > [C]"
    and return the ranking as indices.
    
    Args:
        generated_text: The generated text from the model
        window_size: Number of passages in the window
        
    Returns:
        List of indices [0, 1, 2, ...] representing the ranking
    """
    generated_text = generated_text.strip()
    
    # Try letters first: [A], A], or just A
    pattern_letters = r'\[?\s?([A-T])\s?\]?'
    letter_matches = re.findall(pattern_letters, generated_text)
    
    if letter_matches:
        ranking = []
        seen = set()
        for letter in letter_matches:
            idx = ord(letter) - ord('A')
            if idx < window_size and idx not in seen:
                ranking.append(idx)
                seen.add(idx)
        
        # Add any missing indices at the end
        for i in range(window_size):
            if i not in seen:
                ranking.append(i)
        
        return ranking[:window_size]
    
    # Try numbers: [1], 1], or just 1
    pattern_numbers = r'\[?\s?(\d+)\s?\]?'
    number_matches = re.findall(pattern_numbers, generated_text)
    
    if number_matches:
        ranking = []
        seen = set()
        for num_str in number_matches:
            idx = int(num_str) - 1  # Convert to 0-indexed
            if 0 <= idx < window_size and idx not in seen:
                ranking.append(idx)
                seen.add(idx)
        
        for i in range(window_size):
            if i not in seen:
                ranking.append(i)
        
        return ranking[:window_size]
    
    # Default order if no matches
    print(f"WARNING: Could not parse ranking from: {generated_text[:100]}")
    return list(range(window_size))


class InferenceTimer:
    """
    Timer for measuring inference component timings using forward hooks.
    
    Tracks wall-clock time for:
    - Vision tower (ViT encoder + projector)
    - QI-EI filtering (if enabled)
    - LLM decoder (remaining forward pass time)
    """
    
    def __init__(self, model):
        self.model = model
        self.vision_time_ms = 0.0
        self.filtering_time_ms = 0.0
        self._vision_start = None
        self._filtering_start = None
        self._hooks = []
        
        # Register hooks for vision encoder
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to measure component timings."""
        # Find the visual encoder module
        visual_module = None
        if hasattr(self.model, 'visual'):
            visual_module = self.model.visual
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'visual'):
            visual_module = self.model.model.visual
        
        if visual_module is not None:
            # Pre-hook to start timing
            def vision_pre_hook(module, input):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                self._vision_start = time.perf_counter()
            
            # Post-hook to record timing
            def vision_post_hook(module, input, output):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                if self._vision_start is not None:
                    self.vision_time_ms = (time.perf_counter() - self._vision_start) * 1000
                    self._vision_start = None
            
            self._hooks.append(visual_module.register_forward_pre_hook(vision_pre_hook))
            self._hooks.append(visual_module.register_forward_hook(vision_post_hook))
        
        # Find the QI-EI pruner module (if exists)
        pruner_module = None
        if hasattr(self.model, 'qi_early_pruner'):
            pruner_module = self.model.qi_early_pruner
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'qi_early_pruner'):
            pruner_module = self.model.model.qi_early_pruner
        
        if pruner_module is not None:
            def filtering_pre_hook(module, input):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                self._filtering_start = time.perf_counter()
            
            def filtering_post_hook(module, input, output):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                if self._filtering_start is not None:
                    self.filtering_time_ms = (time.perf_counter() - self._filtering_start) * 1000
                    self._filtering_start = None
            
            self._hooks.append(pruner_module.register_forward_pre_hook(filtering_pre_hook))
            self._hooks.append(pruner_module.register_forward_hook(filtering_post_hook))
    
    def reset(self):
        """Reset timings for a new inference."""
        self.vision_time_ms = 0.0
        self.filtering_time_ms = 0.0
        self._vision_start = None
        self._filtering_start = None
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


# Global inference timer (set after model loading)
_inference_timer: Optional[InferenceTimer] = None


@dataclass
class EvalStats:
    """Statistics accumulator for evaluation metrics."""
    
    # Query counts
    total_queries: int = 0
    total_windows: int = 0
    
    # Timing stats (in milliseconds) - broken down by component
    total_vision_time_ms: float = 0.0      # ViT encoding + projection
    total_filtering_time_ms: float = 0.0   # QI-EI compression filtering
    total_llm_time_ms: float = 0.0         # LLM prefill + decode
    
    # Token counts
    total_input_tokens: int = 0
    total_image_tokens: int = 0
    total_output_tokens: int = 0
    
    # Mode info
    use_logits: bool = True
    
    def add_window(
        self,
        llm_time_ms: float,
        input_tokens: int,
        image_tokens: int,
        output_tokens: int = 1,
        vision_time_ms: float = 0.0,
        filtering_time_ms: float = 0.0,
    ):
        """Record stats for a single reranking window."""
        self.total_windows += 1
        self.total_llm_time_ms += llm_time_ms
        self.total_vision_time_ms += vision_time_ms
        self.total_filtering_time_ms += filtering_time_ms
        self.total_input_tokens += input_tokens
        self.total_image_tokens += image_tokens
        self.total_output_tokens += output_tokens
    
    @property
    def total_inference_time_ms(self) -> float:
        """Get total inference time (vision + filtering + LLM)."""
        return self.total_vision_time_ms + self.total_filtering_time_ms + self.total_llm_time_ms
    
    @property
    def avg_vision_time_ms(self) -> float:
        """Get average vision encoding time per window."""
        if self.total_windows == 0:
            return 0.0
        return self.total_vision_time_ms / self.total_windows
    
    @property
    def avg_filtering_time_ms(self) -> float:
        """Get average filtering time per window."""
        if self.total_windows == 0:
            return 0.0
        return self.total_filtering_time_ms / self.total_windows
    
    @property
    def avg_llm_time_ms(self) -> float:
        """Get average LLM time per window."""
        if self.total_windows == 0:
            return 0.0
        return self.total_llm_time_ms / self.total_windows
    
    @property
    def avg_input_tokens_per_window(self) -> float:
        """Get average input tokens per window."""
        if self.total_windows == 0:
            return 0.0
        return self.total_input_tokens / self.total_windows
    
    @property
    def avg_image_tokens_per_window(self) -> float:
        """Get average image tokens per window."""
        if self.total_windows == 0:
            return 0.0
        return self.total_image_tokens / self.total_windows
    
    @property
    def avg_output_tokens_per_window(self) -> float:
        """Get average output tokens per window."""
        if self.total_windows == 0:
            return 0.0
        return self.total_output_tokens / self.total_windows
    
    @property
    def image_token_ratio(self) -> float:
        """Get ratio of image tokens to input tokens."""
        if self.total_input_tokens == 0:
            return 0.0
        return self.total_image_tokens / self.total_input_tokens
    
    def report(self) -> str:
        """Generate a formatted statistics report."""
        lines = []
        lines.append("=" * 80)
        lines.append("EVALUATION STATISTICS")
        lines.append("=" * 80)
        
        # Query/Window counts
        lines.append(f"Queries processed: {self.total_queries}")
        lines.append(f"Reranking windows: {self.total_windows}")
        
        # Token stats
        lines.append("")
        lines.append("Token Statistics (avg per window):")
        lines.append(f"  Input tokens: {self.avg_input_tokens_per_window:,.0f}")
        lines.append(f"  Image tokens: {self.avg_image_tokens_per_window:,.0f} ({self.image_token_ratio:.1%} of input)")
        text_tokens = self.avg_input_tokens_per_window - self.avg_image_tokens_per_window
        text_ratio = 1.0 - self.image_token_ratio if self.total_input_tokens > 0 else 0.0
        lines.append(f"  Text tokens: {text_tokens:,.0f} ({text_ratio:.1%} of input)")
        lines.append(f"  Output tokens: {self.avg_output_tokens_per_window:,.0f}")
        
        # Timing stats with component breakdown
        lines.append("")
        mode_str = "logits (single forward pass)" if self.use_logits else "generation (autoregressive)"
        lines.append("Inference Timing:")
        lines.append(f"  Mode: {mode_str}")
        
        # Show component breakdown if available
        if self.total_vision_time_ms > 0 or self.total_filtering_time_ms > 0:
            lines.append("")
            lines.append("  Component Breakdown (avg per window):")
            lines.append(f"    Vision Tower:     {self.avg_vision_time_ms:>8.1f} ms")
            if self.total_filtering_time_ms > 0:
                lines.append(f"    QI-EI Filtering:  {self.avg_filtering_time_ms:>8.1f} ms")
            lines.append(f"    LLM Decoder:      {self.avg_llm_time_ms:>8.1f} ms")
            avg_total = self.avg_vision_time_ms + self.avg_filtering_time_ms + self.avg_llm_time_ms
            lines.append(f"    ─────────────────────────────")
            lines.append(f"    Total:            {avg_total:>8.1f} ms/window")
            lines.append("")
            lines.append("  Component Breakdown (totals):")
            lines.append(f"    Vision Tower:     {self.total_vision_time_ms / 1000:>8.2f} s")
            if self.total_filtering_time_ms > 0:
                lines.append(f"    QI-EI Filtering:  {self.total_filtering_time_ms / 1000:>8.2f} s")
            lines.append(f"    LLM Decoder:      {self.total_llm_time_ms / 1000:>8.2f} s")
            lines.append(f"    ─────────────────────────────")
            lines.append(f"    Total:            {self.total_inference_time_ms / 1000:>8.2f} s")
        else:
            # Fallback to simple format if no component breakdown
            lines.append(f"  Avg time per window: {self.avg_llm_time_ms:.1f} ms")
            lines.append(f"  Total LLM time: {self.total_llm_time_ms / 1000:.2f} s")
        
        # Tokens per second (more relevant for generation mode)
        if self.total_llm_time_ms > 0:
            output_tokens_per_sec = (self.total_output_tokens / self.total_llm_time_ms) * 1000
            lines.append(f"  Output tokens/sec: {output_tokens_per_sec:.1f}")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)


# Global stats accumulator (set in main)
_eval_stats: Optional[EvalStats] = None


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate ZipRerank reranker on MMDocIR benchmark"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained Qwen3-VL model",
    )
    
    # Data arguments
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
    
    # Reranking arguments
    parser.add_argument(
        "--window_size",
        type=int,
        default=20,
        help="Number of candidates to rank at once (sliding window size)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Stride for sliding window (how much to shift window each round)",
    )
    parser.add_argument(
        "--use_logits",
        action="store_true",
        dest="use_logits",
        help="Use logits from first token for ranking (faster).",
    )
    parser.add_argument(
        "--no_logits",
        action="store_true",
        dest="no_logits",
        help="Use full text generation (autoregressive, default). More interpretable.",
    )
    
    # Output arguments
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
        "--llm_log_file",
        type=str,
        default=None,
        help="Path to log file for LLM calls (prompts and outputs)",
    )
    
    # Query-Image Early Interaction arguments (visual token pruning)
    parser.add_argument(
        "--use_qi_early",
        action="store_true",
        help="Enable Query-Image Early Interaction visual token pruning. Uses query text to select most relevant visual tokens.",
    )
    parser.add_argument(
        "--qi_early_keep_ratio",
        type=float,
        default=0.25,
        help="Fraction of visual tokens to keep per image when using QI-EI (default: 0.25 = 75%% reduction).",
    )
    parser.add_argument(
        "--qi_early_temperature",
        type=float,
        default=0.1,
        help="Temperature for T2I similarity softmax in QI-EI (default: 0.1).",
    )
    
    return parser.parse_args()


def get_image_from_binary(binary_data: bytes, max_size: int = 1024) -> Image.Image:
    """Convert binary image data to PIL Image, resized to max_size if needed.
    
    Args:
        binary_data: Binary image data
        max_size: Maximum size for largest dimension (default: 1024)
        
    Returns:
        PIL Image resized so largest dim <= max_size
    """
    img = Image.open(io.BytesIO(binary_data)).convert("RGB")
    return resize_image_if_needed(img, max_size=max_size)


def rerank_window_with_images(
    model,
    processor,
    query: str,
    qid: str,
    candidate_images: List[Image.Image],
    start_pos: int,
    end_pos: int,
    use_logits: bool = True,
    log_file=None,
) -> List[int]:
    """
    Rerank a single window of candidates using pre-loaded images.
    
    Args:
        model: Qwen3-VL model
        processor: Qwen3-VL processor
        query: Query text
        qid: Query ID for logging
        candidate_images: List of PIL images (candidates in this window)
        start_pos: Start position in original candidate list (for logging)
        end_pos: End position in original candidate list (for logging)
        use_logits: If True, use logits from first token. If False, use full text generation.
        log_file: Optional log file handle
        
    Returns:
        List of indices [0, 1, 2, ...] representing the ranking WITHIN the window
        (indices relative to window, NOT absolute indices)
    """
    global _eval_stats
    
    tokenizer = processor.tokenizer
    window_size = len(candidate_images)
    
    # Create prompt (matches training format EXACTLY)
    prompt = create_ranking_prompt_for_training(query, window_size, use_query_adapter=False)
    
    # Log prompt if requested
    if log_file:
        log_file.write("="*80 + "\n")
        log_file.write(f"Query ID: {qid}\n")
        log_file.write(f"Window: [{start_pos}:{end_pos}] (size={window_size})\n")
        log_file.write("="*80 + "\n")
        log_file.write("PROMPT (text only, images excluded):\n")
        log_file.write("-"*80 + "\n")
        log_file.write(prompt)
        log_file.write("\n")
        log_file.write("-"*80 + "\n")
        log_file.write("Candidates: [Native resolution images from MMDocIR]\n")
        for i in range(window_size):
            letter = chr(ord('A') + i)
            img = candidate_images[i]
            log_file.write(f"[{letter}] Image size: {img.size}\n")
        log_file.write("-"*80 + "\n")
        log_file.flush()
    
    # Use shared preparation function to ensure consistency with training
    ranking_inputs = prepare_ranking_inputs(prompt, candidate_images, processor)
    
    # Convert to tensors and move to device
    input_ids = torch.tensor([ranking_inputs['input_ids']], dtype=torch.long).to(model.device)
    pixel_values = ranking_inputs['pixel_values'].to(model.device)
    image_grid_thw = ranking_inputs['image_grid_thw'].to(model.device)
    
    # Create inputs dict
    inputs = {
        'input_ids': input_ids,
        'attention_mask': torch.ones_like(input_ids),
        'pixel_values': pixel_values,
        'image_grid_thw': image_grid_thw,
    }
    
    # Compute token statistics for this window
    total_tokens = input_ids.shape[1]
    # Image tokens = product of grid dimensions / spatial_merge_size^2 for each image
    spatial_merge_size = getattr(model.config, 'vision_config', {})
    if hasattr(spatial_merge_size, 'spatial_merge_size'):
        spatial_merge_size = spatial_merge_size.spatial_merge_size
    else:
        spatial_merge_size = 2  # Default
    image_tokens = (image_grid_thw.prod(dim=-1) // (spatial_merge_size ** 2)).sum().item()
    
    # Track original image tokens before compression
    original_image_tokens = image_tokens
    pruned_per_image = None
    
    # Track output tokens
    num_output_tokens = 1  # Default for logits mode
    
    # Reset inference timer for this window
    if _inference_timer is not None:
        _inference_timer.reset()
    
    # Generate ranking
    with torch.no_grad():
        if use_logits:
            # MODE 1: Use logits from forward pass (matches training exactly)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()
            
            outputs = model(**inputs)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            total_forward_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Extract component timings from inference timer
            vision_time_ms = 0.0
            filtering_time_ms = 0.0
            if _inference_timer is not None:
                vision_time_ms = _inference_timer.vision_time_ms
                filtering_time_ms = _inference_timer.filtering_time_ms
            
            # LLM decoder time = total - vision - filtering
            llm_time_ms = total_forward_time_ms - vision_time_ms - filtering_time_ms
            
            # Update image_tokens count if QI-EI compression was applied
            if hasattr(model, '_last_pruned_per_image'):
                pruned_per_image = model._last_pruned_per_image
            elif hasattr(model, 'model') and hasattr(model.model, '_last_pruned_per_image'):
                pruned_per_image = model.model._last_pruned_per_image
            
            if pruned_per_image is not None:
                image_tokens = sum(pruned_per_image)
                text_tokens = total_tokens - original_image_tokens
                total_tokens = text_tokens + image_tokens
            
            # Get logits at the last position
            first_token_logits = outputs.logits[0, -1, :]  # Shape: [vocab_size]
            
            # Extract logits for individual letter tokens 'A', 'B', 'C', etc.
            letter_ids = []
            for i in range(window_size):
                letter = chr(ord('A') + i)
                token_ids = tokenizer.encode(letter, add_special_tokens=False)
                if not token_ids:
                    raise RuntimeError(f"Could not tokenize letter '{letter}' for query {qid}")
                letter_ids.append(token_ids[0])
            
            # Get scores for each letter
            scores = [first_token_logits[token_id].item() for token_id in letter_ids]
            
            # Rank by scores (descending)
            ranked_indices = sorted(range(window_size), key=lambda x: scores[x], reverse=True)
            
            # Log generation output if requested
            if log_file:
                log_file.write("\nGENERATION OUTPUT (logits mode):\n")
                log_file.write("-"*80 + "\n")
                log_file.write("Letter token logits:\n")
                for i, (letter_id, score) in enumerate(zip(letter_ids, scores)):
                    letter = chr(ord('A') + i)
                    log_file.write(f"  [{letter}] (token {letter_id}): {score:.6f}\n")
                log_file.write("\n")
                log_file.write("Ranking (by logit score):\n")
                ranking_str = " > ".join([chr(ord('A') + idx) for idx in ranked_indices])
                log_file.write(f"  {ranking_str}\n")
                log_file.write("="*80 + "\n\n")
                log_file.flush()
        
        else:
            # MODE 2: Use full text generation (slower, more interpretable)
            max_tokens = window_size * 5  # Roughly "[X] > " = 5 tokens per passage
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=1.0,
            )
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            total_forward_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Extract component timings from inference timer
            vision_time_ms = 0.0
            filtering_time_ms = 0.0
            if _inference_timer is not None:
                vision_time_ms = _inference_timer.vision_time_ms
                filtering_time_ms = _inference_timer.filtering_time_ms
            
            llm_time_ms = total_forward_time_ms - vision_time_ms - filtering_time_ms
            
            # Update image_tokens count if QI-EI compression was applied
            if hasattr(model, '_last_pruned_per_image'):
                pruned_per_image = model._last_pruned_per_image
            elif hasattr(model, 'model') and hasattr(model.model, '_last_pruned_per_image'):
                pruned_per_image = model.model._last_pruned_per_image
            
            if pruned_per_image is not None:
                image_tokens = sum(pruned_per_image)
                text_tokens = total_tokens - original_image_tokens
                total_tokens = text_tokens + image_tokens
            
            # Decode the generated text
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            num_output_tokens = len(generated_ids)
            
            # Parse the generated ranking
            ranked_indices = parse_generated_ranking(generated_text, window_size)
            
            # Log generation output if requested
            if log_file:
                log_file.write("\nGENERATION OUTPUT (full text mode):\n")
                log_file.write("-"*80 + "\n")
                log_file.write("Generated text:\n")
                log_file.write(f"  {generated_text}\n")
                log_file.write("\n")
                log_file.write("Parsed ranking:\n")
                ranking_str = " > ".join([chr(ord('A') + idx) for idx in ranked_indices])
                log_file.write(f"  {ranking_str}\n")
                log_file.write("="*80 + "\n\n")
                log_file.flush()
    
    # Record stats to global accumulator if available
    if _eval_stats is not None:
        _eval_stats.add_window(
            llm_time_ms=llm_time_ms,
            input_tokens=total_tokens,
            image_tokens=image_tokens,
            output_tokens=num_output_tokens,
            vision_time_ms=vision_time_ms,
            filtering_time_ms=filtering_time_ms,
        )
    
    return ranked_indices


def sliding_window_rerank_with_images(
    model,
    processor,
    query: str,
    qid: str,
    candidate_images: List[Image.Image],
    window_size: int,
    stride: int,
    use_logits: bool = True,
    log_file=None,
) -> List[int]:
    """
    Apply sliding window reranking to a list of candidate images.
    
    Args:
        model: Qwen3-VL model
        processor: Qwen3-VL processor
        query: Query text
        qid: Query ID for logging
        candidate_images: List of candidate images
        window_size: Size of ranking window (e.g., 20)
        stride: How much to shift window (e.g., 10)
        use_logits: Use logits for ranking
        log_file: Optional log file handle
        
    Returns:
        List of indices representing the final ranking
    """
    n_candidates = len(candidate_images)
    
    # Create a list of indices that we'll reorder
    indices = list(range(n_candidates))
    
    # Track the images in current order
    current_images = list(candidate_images)
    
    # Sliding window algorithm
    end_pos = n_candidates
    start_pos = end_pos - window_size
    
    # Slide window from end to start
    while end_pos > 0 and start_pos + stride != 0:
        start_pos = max(start_pos, 0)
        
        # Get window images in current order
        window_images = current_images[start_pos:end_pos]
        
        # Rerank this window
        ranked_positions = rerank_window_with_images(
            model,
            processor,
            query,
            qid,
            window_images,
            start_pos,
            end_pos,
            use_logits=use_logits,
            log_file=log_file,
        )
        
        # Apply the ranking to both indices and images lists
        reordered_indices = [indices[start_pos + pos] for pos in ranked_positions]
        reordered_images = [current_images[start_pos + pos] for pos in ranked_positions]
        
        # Replace the window with reranked items
        for i, (idx, img) in enumerate(zip(reordered_indices, reordered_images)):
            indices[start_pos + i] = idx
            current_images[start_pos + i] = img
        
        # Move window backwards
        end_pos = end_pos - stride
        start_pos = start_pos - stride
    
    return indices


def load_first_stage_results(first_stage_file: str) -> Dict:
    """Load first-stage retrieval results."""
    with open(first_stage_file, 'rb') as f:
        return pickle.load(f)


def evaluate_mmdocir(
    model,
    processor,
    first_stage_results: Dict,
    parquet_df: pd.DataFrame,
    mode: str,
    window_size: int,
    stride: int,
    use_logits: bool,
    num_queries: Optional[int] = None,
    sample_size: int = 100,
    seed: int = 42,
    log_file=None,
) -> List[Dict]:
    """
    Evaluate reranking on MMDocIR.
    
    Args:
        model: Qwen3-VL model
        processor: Processor
        first_stage_results: First-stage retrieval results
        parquet_df: DataFrame with page/layout data
        mode: "page" or "layout"
        window_size: Sliding window size
        stride: Sliding window stride
        use_logits: Use logits for ranking
        num_queries: Limit number of queries (for testing). Overrides sample_size.
        sample_size: Number of queries to sample (0 for all).
        seed: Random seed for sampling.
        log_file: Optional log file handle
        
    Returns:
        List of result dicts with reranking scores
    """
    results = []
    
    query_keys = list(first_stage_results.keys())
    total_queries = len(query_keys)
    
    # Apply subsampling
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
    print(f"  Ranking mode: {'Logits (single-token)' if use_logits else 'Full text generation'}")
    
    for key in tqdm(query_keys, desc="Reranking"):
        item = first_stage_results[key]
        query = item["query"]
        qid = f"{item['doc_name']}_{item['q_idx']}"
        top_k_global_indices = item["top_k_global_indices"]
        
        # Load candidate images at native resolution
        candidate_images = []
        for global_idx in top_k_global_indices:
            row = parquet_df.iloc[global_idx]
            img = get_image_from_binary(row['image_binary'])
            candidate_images.append(img)
        
        # Apply sliding window reranking
        if len(candidate_images) <= window_size:
            # Single window, no need for sliding
            ranked_indices = rerank_window_with_images(
                model,
                processor,
                query,
                qid,
                candidate_images,
                start_pos=0,
                end_pos=len(candidate_images),
                use_logits=use_logits,
                log_file=log_file,
            )
        else:
            # Sliding window reranking
            ranked_indices = sliding_window_rerank_with_images(
                model,
                processor,
                query,
                qid,
                candidate_images,
                window_size=window_size,
                stride=stride,
                use_logits=use_logits,
                log_file=log_file,
            )
        
        # Create scores based on ranking position (higher score = better rank)
        n_candidates = len(candidate_images)
        rerank_scores = [0.0] * n_candidates
        for rank, idx in enumerate(ranked_indices):
            rerank_scores[idx] = n_candidates - rank
        
        # Build result in MMDocIR format
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


def convert_results_to_official_format(
    results: List[Dict],
    mode: str,
) -> List[Dict]:
    """
    Convert reranker results to the official MMDocIR format.
    """
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


def convert_first_stage_to_official_format(
    results: List[Dict],
    mode: str,
) -> List[Dict]:
    """
    Convert first-stage results to official MMDocIR format for evaluation.
    """
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
    model_name: str = "ZipRerank",
) -> Dict[int, float]:
    """
    Compute MMDocIR metrics using the OFFICIAL evaluate_page/evaluate_layout functions.
    """
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
    
    # Evaluate FIRST-STAGE results
    print("\n[First-Stage]")
    for topk in topk_values:
        if mode == "page":
            evaluate_page(first_stage_results, model_name="First-Stage", topk=topk, metric="recall")
        else:
            evaluate_layout(first_stage_results, model_name="First-Stage", topk=topk, metric="recall")
    
    # Evaluate RERANKER results
    print(f"\n[Reranker ({model_name})]")
    for topk in topk_values:
        if mode == "page":
            evaluate_page(rerank_results, model_name=model_name, topk=topk, metric="recall")
        else:
            evaluate_layout(rerank_results, model_name=model_name, topk=topk, metric="recall")
    
    print("="*120)
    
    return rerank_recalls


def main():
    args = parse_args()
    
    print("="*80)
    print("ZipRerank Evaluation (MMDocIR)")
    print("="*80)
    print(f"Model: {args.model_path}")
    
    # Determine use_logits: default False, unless --use_logits is passed
    use_logits = args.use_logits and not args.no_logits
    
    print(f"Mode: {args.mode}")
    print(f"First-stage file: {args.first_stage_file}")
    print(f"Window size: {args.window_size}")
    print(f"Stride: {args.stride}")
    print(f"Ranking mode: {'Logits (single-token)' if use_logits else 'Full text generation (autoregressive)'}")
    print(f"Sample size: {args.sample_size if args.sample_size > 0 else 'all'} (seed={args.seed})")
    
    if args.use_qi_early:
        print(f"QI-EI Visual Token Pruning: Enabled (keep {args.qi_early_keep_ratio*100:.0f}%)")
    else:
        print(f"QI-EI Visual Token Pruning: Disabled")
    
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
    print(f"Loading model from {args.model_path}...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Load model - choose class based on QI-EI mode
    if args.use_qi_early:
        print(f"QI-EI mode: Loading Qwen3VLWithQIEarly with {args.qi_early_keep_ratio*100:.0f}% token retention...")
        model = Qwen3VLWithQIEarly.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        # Configure QI-EI settings
        model.set_qi_early_keep_ratio(args.qi_early_keep_ratio)
        if hasattr(model.qi_early_pruner, 'temperature'):
            model.qi_early_pruner.temperature = args.qi_early_temperature
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    model.eval()
    print("Model loaded successfully!")
    print(f"  QI-EI Pruning: {'Enabled (keep ' + str(int(args.qi_early_keep_ratio*100)) + '%)' if args.use_qi_early else 'Disabled'}")
    
    # Initialize inference timer for component-level timing
    global _inference_timer
    _inference_timer = InferenceTimer(model)
    print(f"  Inference Timer: Initialized")
    
    # Initialize global stats accumulator
    global _eval_stats
    _eval_stats = EvalStats(use_logits=use_logits)
    
    # Open log file if specified
    log_file = None
    if args.llm_log_file:
        print(f"Opening LLM log file: {args.llm_log_file}")
        log_file = open(args.llm_log_file, 'w', encoding='utf-8')
        log_file.write("="*80 + "\n")
        log_file.write("LLM CALL LOG - ZipRerank Evaluation\n")
        log_file.write("="*80 + "\n")
        log_file.write(f"Model: {args.model_path}\n")
        log_file.write(f"Mode: {args.mode}\n")
        log_file.write(f"Window size: {args.window_size}\n")
        log_file.write(f"Stride: {args.stride}\n")
        if args.use_qi_early:
            log_file.write(f"QI-EI: Enabled (keep {args.qi_early_keep_ratio*100:.0f}%, temp={args.qi_early_temperature})\n")
        log_file.write("="*80 + "\n\n")
        log_file.flush()
    
    try:
        # Run evaluation
        print("\nStarting reranking evaluation...")
        
        results = evaluate_mmdocir(
            model=model,
            processor=processor,
            first_stage_results=first_stage_results,
            parquet_df=parquet_df,
            mode=args.mode,
            window_size=args.window_size,
            stride=args.stride,
            use_logits=use_logits,
            num_queries=args.num_queries,
            sample_size=args.sample_size,
            seed=args.seed,
            log_file=log_file,
        )
        
        # Convert results to official MMDocIR format
        official_rerank_results = convert_results_to_official_format(results, args.mode)
        official_first_stage_results = convert_first_stage_to_official_format(results, args.mode)
        
        compute_mmdocir_metrics(
            rerank_results=official_rerank_results,
            first_stage_results=official_first_stage_results,
            mode=args.mode,
            model_name="ZipRerank",
        )
        
        # Update query count in stats
        _eval_stats.total_queries = len(results)
        
        # Save results if requested
        if args.output_file:
            with open(args.output_file, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + "\n")
            print(f"\nResults saved to {args.output_file}")
        
        # Print evaluation statistics
        print()
        print(_eval_stats.report())
    
    finally:
        if log_file:
            log_file.write("\n" + "="*80 + "\n")
            log_file.write("END OF LOG\n")
            log_file.write("="*80 + "\n")
            log_file.close()
            print(f"LLM log saved to: {args.llm_log_file}")
    
    print()
    print("="*80)
    print("Evaluation completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()

