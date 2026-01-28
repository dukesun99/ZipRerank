#!/usr/bin/env python3
"""
Evaluate reranking on MMDocIR benchmark using OpenAI's Vision API.

Uses first-stage retrieval results (top-20) and applies OpenAI vision model for reranking.
Supports multiple images in a single request using base64 encoding.
"""

import argparse
import base64
import io
import json
import os
import pickle
import random
import re
import sys
import time
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# Pricing per million tokens (USD)
MODEL_PRICING = {
    "gpt-5.1": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


class CostTracker:
    """Thread-safe tracker for API usage and costs."""
    
    def __init__(self, model: str):
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0
        self.num_requests = 0
        self.total_api_time_ms = 0.0
        self._lock = threading.Lock()
    
    def get_pricing(self) -> Tuple[float, float]:
        """Get input/output price per million tokens for the model."""
        if self.model in MODEL_PRICING:
            pricing = MODEL_PRICING[self.model]
        else:
            for model_name, prices in MODEL_PRICING.items():
                if self.model.startswith(model_name.split("-")[0]):
                    return prices["input"], prices["output"]
            pricing = {"input": 0.05, "output": 0.40}
        return pricing["input"], pricing["output"]
    
    def add_usage(self, usage_dict: dict, api_time_ms: float = 0.0):
        """Thread-safe: Add usage from an API response."""
        with self._lock:
            self.input_tokens += usage_dict.get("input_tokens", 0)
            self.output_tokens += usage_dict.get("output_tokens", 0)
            if "input_tokens_details" in usage_dict:
                self.cached_tokens += usage_dict["input_tokens_details"].get("cached_tokens", 0)
            self.num_requests += 1
            self.total_api_time_ms += api_time_ms
    
    def get_cost(self) -> Tuple[float, float, float]:
        """Thread-safe: Calculate costs. Returns (input_cost, output_cost, total_cost)."""
        with self._lock:
            input_price, output_price = self.get_pricing()
            input_cost = (self.input_tokens / 1_000_000) * input_price
            output_cost = (self.output_tokens / 1_000_000) * output_price
            return input_cost, output_cost, input_cost + output_cost
    
    def get_stats(self) -> Dict:
        """Thread-safe: Get current stats as dict."""
        with self._lock:
            return {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cached_tokens": self.cached_tokens,
                "num_requests": self.num_requests,
                "total_api_time_ms": self.total_api_time_ms,
            }


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate OpenAI vision model reranker on MMDocIR benchmark"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini-2025-08-07",
        help="OpenAI model to use (default: gpt-5-mini-2025-08-07)",
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
        "--max_image_size",
        type=int,
        default=1024,
        help="Maximum dimension for images (to reduce API costs). Set to 0 to disable resizing.",
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
        help="Path to log file for API calls",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (defaults to OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="Custom base URL for OpenAI API (for proxy services). Defaults to OPENAI_BASE_URL env var.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum number of retries for API calls",
    )
    parser.add_argument(
        "--retry_delay",
        type=float,
        default=1.0,
        help="Initial delay between retries (exponential backoff)",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="Number of parallel threads for API calls (default: 4)",
    )
    
    return parser.parse_args()


def get_image_from_binary(binary_data: bytes) -> Image.Image:
    """Convert binary image data to PIL Image at native resolution."""
    return Image.open(io.BytesIO(binary_data)).convert("RGB")


def resize_image_if_needed(img: Image.Image, max_size: int) -> Image.Image:
    """Resize image if any dimension exceeds max_size, maintaining aspect ratio."""
    if max_size <= 0:
        return img
    
    w, h = img.size
    if w <= max_size and h <= max_size:
        return img
    
    if w > h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)
    
    return img.resize((new_w, new_h), Image.LANCZOS)


def encode_image_to_base64(img: Image.Image) -> str:
    """Encode PIL Image to base64 string for OpenAI API."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def create_ranking_prompt(query: str, num_candidates: int) -> str:
    """Create ranking prompt for OpenAI."""
    letters = [chr(ord('A') + i) for i in range(num_candidates)]
    letter_list = ", ".join(f"[{l}]" for l in letters)
    
    prompt = f"""You are a document retrieval expert. Given a query and {num_candidates} document page images, rank them from most relevant to least relevant to the query.

Query: {query}

The {num_candidates} document pages are labeled {letter_list}.

Instructions:
1. Analyze each document page image and determine its relevance to the query.
2. Rank all documents from most to least relevant.
3. Output ONLY the ranking in the format: [X] > [Y] > [Z] > ...
4. Include ALL {num_candidates} documents in your ranking.

Your ranking:"""
    
    return prompt


def parse_openai_ranking(response_text: str, num_candidates: int) -> List[int]:
    """Parse OpenAI's ranking response into indices."""
    letters_found = re.findall(r'\[?([A-T])\]?', response_text.upper())
    
    if not letters_found:
        print(f"WARNING: Could not parse ranking from: {response_text[:100]}...")
        return list(range(num_candidates))
    
    ranked_indices = []
    seen = set()
    for letter in letters_found:
        idx = ord(letter) - ord('A')
        if idx < num_candidates and idx not in seen:
            ranked_indices.append(idx)
            seen.add(idx)
    
    for i in range(num_candidates):
        if i not in seen:
            ranked_indices.append(i)
    
    return ranked_indices


def rerank_with_openai(
    client,
    model: str,
    query: str,
    qid: str,
    candidate_images: List[Image.Image],
    max_image_size: int,
    cost_tracker: Optional[CostTracker] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    log_file=None,
) -> Tuple[List[int], str]:
    """Rerank candidates using OpenAI's vision API."""
    num_candidates = len(candidate_images)
    
    prompt = create_ranking_prompt(query, num_candidates)
    
    content = [{"type": "input_text", "text": prompt}]
    
    for i, img in enumerate(candidate_images):
        letter = chr(ord('A') + i)
        
        img_resized = resize_image_if_needed(img, max_image_size)
        base64_img = encode_image_to_base64(img_resized)
        
        content.append({
            "type": "input_text",
            "text": f"\n[{letter}]:"
        })
        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{base64_img}",
        })
    
    if log_file:
        log_file.write("="*80 + "\n")
        log_file.write(f"Query ID: {qid}\n")
        log_file.write(f"Query: {query}\n")
        log_file.write(f"Num candidates: {num_candidates}\n")
        log_file.write("-"*80 + "\n")
        log_file.flush()
    
    response_text = ""
    api_time_ms = 0.0
    for attempt in range(max_retries):
        try:
            start_time = time.perf_counter()
            response = client.responses.create(
                model=model,
                input=[{
                    "role": "user",
                    "content": content,
                }],
            )
            api_time_ms = (time.perf_counter() - start_time) * 1000
            response_text = response.output_text
            
            if cost_tracker and hasattr(response, 'usage') and response.usage:
                usage_dict = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
                if hasattr(response.usage, 'input_tokens_details') and response.usage.input_tokens_details:
                    usage_dict["input_tokens_details"] = {
                        "cached_tokens": getattr(response.usage.input_tokens_details, 'cached_tokens', 0)
                    }
                cost_tracker.add_usage(usage_dict, api_time_ms=api_time_ms)
            elif cost_tracker:
                cost_tracker.add_usage({}, api_time_ms=api_time_ms)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                delay = retry_delay * (2 ** attempt)
                print(f"  API error (attempt {attempt+1}/{max_retries}): {e}")
                print(f"  Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                print(f"  API error (final attempt): {e}")
                response_text = ""
    
    ranked_indices = parse_openai_ranking(response_text, num_candidates)
    
    if log_file:
        log_file.write(f"Response:\n{response_text}\n")
        log_file.write(f"Parsed ranking: {ranked_indices}\n")
        log_file.write("="*80 + "\n\n")
        log_file.flush()
    
    return ranked_indices, response_text


def _process_single_query(
    key: str,
    item: Dict,
    client,
    model: str,
    parquet_df: pd.DataFrame,
    mode: str,
    window_size: int,
    max_image_size: int,
    max_retries: int,
    retry_delay: float,
    cost_tracker: CostTracker,
) -> Optional[Dict]:
    """Process a single query for reranking."""
    try:
        query = item["query"]
        qid = f"{item['doc_name']}_{item['q_idx']}"
        top_k_global_indices = item["top_k_global_indices"][:window_size]
        
        candidate_images = []
        for global_idx in top_k_global_indices:
            row = parquet_df.iloc[global_idx]
            img = get_image_from_binary(row['image_binary'])
            candidate_images.append(img)
        
        ranked_indices, response_text = rerank_with_openai(
            client=client,
            model=model,
            query=query,
            qid=qid,
            candidate_images=candidate_images,
            max_image_size=max_image_size,
            cost_tracker=cost_tracker,
            max_retries=max_retries,
            retry_delay=retry_delay,
            log_file=None,
        )
        
        n_candidates = len(candidate_images)
        rerank_scores = [0.0] * n_candidates
        for rank, idx in enumerate(ranked_indices):
            rerank_scores[idx] = n_candidates - rank
        
        result = {
            "key": key,
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
            "openai_response": response_text,
        }
        
        if mode == "layout":
            result["layout_indices"] = item.get("top_k_layout_indices", [])
        
        return result
    except Exception as e:
        print(f"Error processing query {key}: {e}")
        return None


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
    model_name: str = "OpenAI",
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


def evaluate_mmdocir_openai(
    client,
    model: str,
    first_stage_results: Dict,
    parquet_df: pd.DataFrame,
    mode: str,
    window_size: int,
    max_image_size: int,
    max_retries: int,
    retry_delay: float,
    num_queries: Optional[int] = None,
    sample_size: int = 100,
    seed: int = 42,
    log_file=None,
    num_threads: int = 4,
) -> Tuple[List[Dict], CostTracker]:
    """Evaluate reranking on MMDocIR using OpenAI with multithreading."""
    cost_tracker = CostTracker(model=model)
    
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
    print(f"  Model: {model}")
    print(f"  Window size: {window_size}")
    print(f"  Max image size: {max_image_size if max_image_size > 0 else 'Native'}")
    print(f"  Threads: {num_threads}")
    input_price, output_price = cost_tracker.get_pricing()
    print(f"  Pricing: ${input_price}/M input, ${output_price}/M output")
    print()
    
    results = []
    
    if num_threads <= 1:
        pbar = tqdm(query_keys, desc="Reranking")
        for key in pbar:
            item = first_stage_results[key]
            result = _process_single_query(
                key=key,
                item=item,
                client=client,
                model=model,
                parquet_df=parquet_df,
                mode=mode,
                window_size=window_size,
                max_image_size=max_image_size,
                max_retries=max_retries,
                retry_delay=retry_delay,
                cost_tracker=cost_tracker,
            )
            if result:
                results.append(result)
            
            _, _, total_cost = cost_tracker.get_cost()
            stats = cost_tracker.get_stats()
            avg_time = stats['total_api_time_ms'] / stats['num_requests'] if stats['num_requests'] > 0 else 0
            pbar.set_postfix_str(f"${total_cost:.4f} | {stats['input_tokens'] + stats['output_tokens']:,} tok | {avg_time:.0f}ms/req")
        pbar.close()
    else:
        completed = 0
        pbar = tqdm(total=len(query_keys), desc=f"Reranking ({num_threads} threads)")
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(
                    _process_single_query,
                    key=key,
                    item=first_stage_results[key],
                    client=client,
                    model=model,
                    parquet_df=parquet_df,
                    mode=mode,
                    window_size=window_size,
                    max_image_size=max_image_size,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    cost_tracker=cost_tracker,
                ): key for key in query_keys
            }
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                
                completed += 1
                pbar.update(1)
                
                _, _, total_cost = cost_tracker.get_cost()
                stats = cost_tracker.get_stats()
                avg_time = stats['total_api_time_ms'] / stats['num_requests'] if stats['num_requests'] > 0 else 0
                pbar.set_postfix_str(f"${total_cost:.4f} | {stats['input_tokens'] + stats['output_tokens']:,} tok | {avg_time:.0f}ms/req")
        
        pbar.close()
    
    # Print final cost and timing summary
    print()
    print("="*60)
    print("API USAGE SUMMARY")
    print("="*60)
    stats = cost_tracker.get_stats()
    print(f"  Model: {model}")
    print(f"  Threads: {num_threads}")
    print(f"  Requests: {stats['num_requests']:,}")
    print(f"  Input tokens: {stats['input_tokens']:,}")
    print(f"  Output tokens: {stats['output_tokens']:,}")
    print(f"  Cached tokens: {stats['cached_tokens']:,}")
    input_cost, output_cost, total_cost = cost_tracker.get_cost()
    print(f"  Input cost: ${input_cost:.4f}")
    print(f"  Output cost: ${output_cost:.4f}")
    print(f"  TOTAL COST: ${total_cost:.4f}")
    print("-"*60)
    print("  TIMING STATISTICS")
    print("-"*60)
    total_time_s = stats['total_api_time_ms'] / 1000
    avg_time_ms = stats['total_api_time_ms'] / stats['num_requests'] if stats['num_requests'] > 0 else 0
    print(f"  Total API time: {total_time_s:.2f}s")
    print(f"  Avg time/request: {avg_time_ms:.1f}ms")
    if stats['input_tokens'] + stats['output_tokens'] > 0:
        tokens_per_sec = (stats['input_tokens'] + stats['output_tokens']) / total_time_s if total_time_s > 0 else 0
        print(f"  Throughput: {tokens_per_sec:.1f} tokens/sec")
    print("="*60)
    print()
    
    return results, cost_tracker


def main():
    args = parse_args()
    
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: No OpenAI API key provided.")
        print("Either set OPENAI_API_KEY environment variable or use --api_key argument.")
        sys.exit(1)
    
    print("="*80)
    print("MMDocIR Reranking Evaluation (OpenAI Vision API)")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"First-stage file: {args.first_stage_file}")
    print(f"Window size: {args.window_size}")
    print(f"Max image size: {args.max_image_size if args.max_image_size > 0 else 'Native'}")
    print(f"Sample size: {args.sample_size if args.sample_size > 0 else 'all'} (seed={args.seed})")
    print("="*80)
    print()
    
    from openai import OpenAI
    
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    
    client_kwargs = {"api_key": api_key}
    
    if base_url:
        print(f"Using custom base URL: {base_url}")
        client_kwargs["base_url"] = base_url
    
    client = OpenAI(**client_kwargs)
    
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
    
    # Open log file if specified
    log_file = None
    if args.log_file:
        print(f"Opening log file: {args.log_file}")
        log_file = open(args.log_file, 'w', encoding='utf-8')
        log_file.write("="*80 + "\n")
        log_file.write("OpenAI API CALL LOG - MMDocIR Reranking Evaluation\n")
        log_file.write("="*80 + "\n")
        log_file.write(f"Model: {args.model}\n")
        log_file.write(f"Mode: {args.mode}\n")
        log_file.write(f"Window size: {args.window_size}\n")
        log_file.write("="*80 + "\n\n")
        log_file.flush()
    
    try:
        print("\nStarting reranking evaluation...")
        
        results, cost_tracker = evaluate_mmdocir_openai(
            client=client,
            model=args.model,
            first_stage_results=first_stage_results,
            parquet_df=parquet_df,
            mode=args.mode,
            window_size=args.window_size,
            max_image_size=args.max_image_size,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            num_queries=args.num_queries,
            sample_size=args.sample_size,
            seed=args.seed,
            log_file=log_file,
            num_threads=args.num_threads,
        )
        
        official_rerank_results = convert_results_to_official_format(results, args.mode)
        official_first_stage_results = convert_first_stage_to_official_format(results, args.mode)
        
        compute_mmdocir_metrics(
            rerank_results=official_rerank_results,
            first_stage_results=official_first_stage_results,
            mode=args.mode,
            model_name=f"OpenAI-{args.model}",
        )
        
        if args.output_file:
            stats = cost_tracker.get_stats()
            output_data = {
                "cost_summary": {
                    "model": args.model,
                    "num_requests": cost_tracker.num_requests,
                    "input_tokens": cost_tracker.input_tokens,
                    "output_tokens": cost_tracker.output_tokens,
                    "cached_tokens": cost_tracker.cached_tokens,
                    "total_cost_usd": cost_tracker.get_cost()[2],
                },
                "timing_summary": {
                    "total_api_time_ms": stats['total_api_time_ms'],
                    "avg_time_per_request_ms": stats['total_api_time_ms'] / stats['num_requests'] if stats['num_requests'] > 0 else 0,
                },
                "results": results,
            }
            with open(args.output_file, 'w') as f:
                f.write(json.dumps(output_data, indent=2))
            print(f"\nResults saved to {args.output_file}")
    
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

