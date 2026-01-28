#!/usr/bin/env python3
"""
Generate listwise reranking training data for MMDocIR using OpenAI Vision API.

Takes first-stage retrieval results and uses OpenAI to generate reranking labels.
Reuses functions from evaluate_mmdocir_openai.py for consistency.
Supports multithreading with thread-safe cost tracking.

Output format:
training_data.jsonl - Training examples with rankings
{
    "query_id": ...,
    "query": ...,
    "doc_name": ...,
    "domain": ...,
    "candidate_page_ids": [...],
    "gt_page_ids": [...],
    "openai_ranking": [...],
    "ranked_page_ids": [...],
}
"""

import argparse
import io
import json
import os
import pickle
import random
import sys
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Add parent to path and import from evaluate script
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import shared functions from evaluate script
from evaluate_openai import (
    MODEL_PRICING,
    CostTracker,
    get_image_from_binary,
    resize_image_if_needed,
    encode_image_to_base64,
    create_ranking_prompt,
    parse_openai_ranking,
    rerank_with_openai,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate reranking training data using OpenAI")
    
    parser.add_argument("--model", type=str, default="gpt-5-mini",
                        help="OpenAI model (gpt-5-nano, gpt-5-mini, gpt-5.1)")
    parser.add_argument("--first_stage_file", type=str, required=True,
                        help="First-stage results pickle file")
    parser.add_argument("--parquet_dir", type=str, default="MMDocIR/MMDocIR_Train_Dataset/parquet",
                        help="Directory containing MMDocIR parquet files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for training data")
    parser.add_argument("--window_size", type=int, default=20,
                        help="Number of candidates to rank")
    parser.add_argument("--max_image_size", type=int, default=0,
                        help="Max image dimension (0 = native resolution)")
    parser.add_argument("--num_queries", type=int, default=None,
                        help="Limit number of queries (for testing)")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Random sample size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_delay", type=float, default=1.0)
    parser.add_argument("--num_threads", type=int, default=4,
                        help="Number of parallel threads (default: 4)")
    parser.add_argument("--domains", type=str, nargs='+', default=None,
                        help="Filter to specific domains")
    
    return parser.parse_args()


def load_first_stage_results(path: str) -> List[Dict]:
    """Load first-stage retrieval results."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_train_parquets(parquet_dir: str, domains: List[str] = None) -> pd.DataFrame:
    """Load training parquet files."""
    if domains is None:
        # All 7 domains in the training set
        domains = ["ArxivQA", "DUDE", "MP-DocVQA", "SciQAG", "SlideVQA", "TAT-DQA", "Wiki-ss"]
    
    dfs = []
    for domain in domains:
        if domain == "DUDE":
            parquet_name = "DUDE_filter.parquet"
        elif domain == "MP-DocVQA":
            parquet_name = "MP-DocVQA_filter.parquet"
        else:
            parquet_name = f"{domain}_filter.parquet"
        
        parquet_file = os.path.join(parquet_dir, parquet_name)
        if os.path.exists(parquet_file):
            print(f"Loading {parquet_name}...")
            df = pd.read_parquet(parquet_file)
            df['domain'] = domain
            dfs.append(df)
    
    if not dfs:
        raise ValueError(f"No parquet files found in {parquet_dir}")
    
    return pd.concat(dfs, ignore_index=True)


def build_page_lookup(df: pd.DataFrame) -> Dict[Tuple[str, int], int]:
    """Build lookup: (doc_name, page_id) -> row index."""
    lookup = {}
    for idx, row in df.iterrows():
        key = (row['file_name'], row['page'])
        lookup[key] = idx
    return lookup


def _process_single_train_query(
    item: Dict,
    client,
    model: str,
    pages_df: pd.DataFrame,
    page_lookup: Dict,
    window_size: int,
    max_image_size: int,
    max_retries: int,
    retry_delay: float,
    cost_tracker: CostTracker,
) -> Optional[Dict]:
    """Process a single training query. Returns result dict or None on failure."""
    try:
        query_id = item['query_id']
        query = item['query']
        doc_name = item['doc_name']
        top_k_page_ids = item['top_k_page_ids'][:window_size]
        
        # Load candidate images
        candidate_images = []
        valid_page_ids = []
        
        for page_id in top_k_page_ids:
            key = (doc_name, page_id)
            if key in page_lookup:
                row_idx = page_lookup[key]
                row = pages_df.iloc[row_idx]
                try:
                    img = get_image_from_binary(row['image'])
                    candidate_images.append(img)
                    valid_page_ids.append(page_id)
                except Exception:
                    continue
        
        if len(candidate_images) < 2:
            return None
        
        # Rerank with OpenAI (reusing shared function)
        ranked_indices, response_text = rerank_with_openai(
            client=client,
            model=model,
            query=query,
            qid=str(query_id),
            candidate_images=candidate_images,
            max_image_size=max_image_size,
            cost_tracker=cost_tracker,
            max_retries=max_retries,
            retry_delay=retry_delay,
            log_file=None,
        )
        
        # Convert indices to page IDs
        ranked_page_ids = [valid_page_ids[i] for i in ranked_indices if i < len(valid_page_ids)]
        
        # Build result
        result = {
            'query_id': query_id,
            'query': query,
            'doc_name': doc_name,
            'domain': item.get('domain', 'unknown'),
            'candidate_page_ids': valid_page_ids,
            'gt_page_ids': item['gt_page_ids'],
            'neg_page_ids': item.get('neg_page_ids', []),
            'openai_ranking': ranked_indices,
            'ranked_page_ids': ranked_page_ids,
            'openai_response': response_text,
            'first_stage_scores': item['top_k_scores'][:len(valid_page_ids)],
        }
        
        return result
    except Exception as e:
        print(f"Error processing query {item.get('query_id', 'unknown')}: {e}")
        return None


def main():
    args = parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: No OpenAI API key. Set OPENAI_API_KEY or use --api_key")
        sys.exit(1)
    
    # Setup output paths
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    training_data_path = os.path.join(output_dir, "training_data.jsonl")
    
    print("="*60)
    print("MMDocIR Training Data Generation (OpenAI Reranking)")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"First-stage file: {args.first_stage_file}")
    print(f"Output directory: {output_dir}")
    print(f"  - Training data: {training_data_path}")
    print(f"Window size: {args.window_size}")
    print(f"Max image size: {args.max_image_size if args.max_image_size > 0 else 'Native'}")
    print(f"Threads: {args.num_threads}")
    print("="*60)
    
    # Initialize OpenAI client
    from openai import OpenAI
    import httpx
    
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    http_proxy = os.getenv("APIFY_HTTP_PROXY_URL") or os.getenv("HTTP_PROXY")
    
    client_kwargs = {"api_key": api_key}
    if base_url:
        print(f"Using base URL: {base_url}")
        client_kwargs["base_url"] = base_url
    if http_proxy:
        print(f"Using HTTP proxy: {http_proxy}")
        client_kwargs["http_client"] = httpx.Client(proxy=http_proxy, timeout=httpx.Timeout(120.0))
    
    client = OpenAI(**client_kwargs)
    cost_tracker = CostTracker(model=args.model)
    
    # Load data
    print("\nLoading first-stage results...")
    first_stage_results = load_first_stage_results(args.first_stage_file)
    print(f"Loaded {len(first_stage_results)} queries")
    
    print("\nLoading parquet files...")
    pages_df = load_train_parquets(args.parquet_dir, args.domains)
    print(f"Loaded {len(pages_df)} pages")
    
    print("\nBuilding page lookup...")
    page_lookup = build_page_lookup(pages_df)
    print(f"Indexed {len(page_lookup)} (doc, page) pairs")
    
    # Handle resume - use composite key (domain, query_id) since query_ids are NOT unique across domains
    processed_keys = set()
    if args.resume and os.path.exists(training_data_path):
        print(f"\nResuming from {training_data_path}...")
        with open(training_data_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    # Use composite key: (domain, query_id)
                    key = (item.get('domain', ''), item['query_id'])
                    processed_keys.add(key)
                except:
                    continue
        print(f"Already processed: {len(processed_keys)} queries")
    
    # Filter/sample queries - use composite key for matching
    def get_query_key(r):
        domain = r.get('domain', r.get('source_domain', ''))
        return (domain, r['query_id'])
    
    queries_to_process = [r for r in first_stage_results if get_query_key(r) not in processed_keys]
    
    if args.num_queries:
        queries_to_process = queries_to_process[:args.num_queries]
    elif args.sample_size and args.sample_size < len(queries_to_process):
        random.seed(args.seed)
        queries_to_process = random.sample(queries_to_process, args.sample_size)
    
    print(f"\nProcessing {len(queries_to_process)} queries with {args.num_threads} threads...")
    input_price, output_price = cost_tracker.get_pricing()
    print(f"Pricing: ${input_price}/M input, ${output_price}/M output\n")
    
    # Thread-safe file writing
    write_lock = threading.Lock()
    
    def write_result(result: Dict, output_file: str):
        """Thread-safe result writing."""
        with write_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result) + '\n')
    
    # Process queries
    mode = 'a' if args.resume else 'w'
    if mode == 'w':
        # Clear file if not resuming
        open(training_data_path, 'w').close()
    
    completed = 0
    pbar = tqdm(total=len(queries_to_process), desc=f"Reranking ({args.num_threads} threads)")
    
    if args.num_threads <= 1:
        # Single-threaded mode
        for item in queries_to_process:
            result = _process_single_train_query(
                item=item,
                client=client,
                model=args.model,
                pages_df=pages_df,
                page_lookup=page_lookup,
                window_size=args.window_size,
                max_image_size=args.max_image_size,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
                cost_tracker=cost_tracker,
            )
            
            if result:
                write_result(result, training_data_path)
            
            completed += 1
            pbar.update(1)
            _, _, total_cost = cost_tracker.get_cost()
            stats = cost_tracker.get_stats()
            pbar.set_postfix_str(f"${total_cost:.4f} | {stats['input_tokens'] + stats['output_tokens']:,} tok")
    else:
        # Multi-threaded mode
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = {
                executor.submit(
                    _process_single_train_query,
                    item=item,
                    client=client,
                    model=args.model,
                    pages_df=pages_df,
                    page_lookup=page_lookup,
                    window_size=args.window_size,
                    max_image_size=args.max_image_size,
                    max_retries=args.max_retries,
                    retry_delay=args.retry_delay,
                    cost_tracker=cost_tracker,
                ): item['query_id'] for item in queries_to_process
            }
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    write_result(result, training_data_path)
                
                completed += 1
                pbar.update(1)
                _, _, total_cost = cost_tracker.get_cost()
                stats = cost_tracker.get_stats()
                pbar.set_postfix_str(f"${total_cost:.4f} | {stats['input_tokens'] + stats['output_tokens']:,} tok")
    
    pbar.close()
    
    # Print API cost summary
    print("\n" + "="*60)
    print("API USAGE SUMMARY")
    print("="*60)
    print(cost_tracker.get_summary())
    
    # Load all results for statistics and image extraction
    results = []
    with open(training_data_path, 'r') as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except:
                continue
    
    # Calculate statistics
    print("\n" + "-"*60)
    print("Reranking Quality (vs Ground Truth)")
    print("-"*60)
    
    if results:
        for k in [1, 3, 5]:
            hits = 0
            for r in results:
                gt_pages = set(r['gt_page_ids'])
                ranked_pages = set(r['ranked_page_ids'][:k])
                if gt_pages & ranked_pages:
                    hits += 1
            recall = hits / len(results)
            print(f"OpenAI Recall@{k}: {recall*100:.1f}% ({hits}/{len(results)})")
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"Training data: {training_data_path} ({len(results)} queries)")


if __name__ == "__main__":
    main()
