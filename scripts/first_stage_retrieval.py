#!/usr/bin/env python3
"""
First-stage retrieval for MMDocIR benchmark using DSE retriever.

Generates top-k retrieval results for each query and caches them for reranking.
Supports multi-GPU parallel processing.
"""

import argparse
import json
import os
import pickle
import sys
import math
import tempfile
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Tuple
import io

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# MMDocIR path for DSE model (vision_wrapper)
# Set MMDOCIR_PATH environment variable or clone MMDocIR repo to ../MMDocIR
MMDOCIR_PATH = os.environ.get("MMDOCIR_PATH", str(Path(__file__).parent.parent.parent / "MMDocIR"))
if Path(MMDOCIR_PATH).exists():
    sys.path.insert(0, MMDOCIR_PATH)
else:
    print(f"WARNING: MMDocIR not found at {MMDOCIR_PATH}")
    print("DSE first-stage retrieval requires MMDocIR's vision_wrapper.py")
    print("Clone MMDocIR or set MMDOCIR_PATH environment variable")


def load_annotations(annotation_file: str) -> List[Dict]:
    """Load MMDocIR annotations."""
    annotations = []
    with open(annotation_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            annotations.append(item)
    return annotations


def load_pages_parquet(parquet_file: str) -> pd.DataFrame:
    """Load pages parquet file."""
    print(f"Loading pages from {parquet_file}...")
    df = pd.read_parquet(parquet_file)
    print(f"Loaded {len(df)} pages")
    return df


def load_layouts_parquet(parquet_file: str) -> pd.DataFrame:
    """Load layouts parquet file."""
    print(f"Loading layouts from {parquet_file}...")
    df = pd.read_parquet(parquet_file)
    print(f"Loaded {len(df)} layouts")
    return df


def get_image_from_binary(binary_data: bytes) -> Image.Image:
    """Convert binary image data to PIL Image."""
    return Image.open(io.BytesIO(binary_data)).convert("RGB")


def flatten_queries(annotations: List[Dict], mode: str = "page") -> List[Dict]:
    """Flatten all queries with their document context."""
    all_queries = []
    for doc_idx, doc in enumerate(annotations):
        doc_name = doc["doc_name"]
        domain = doc["domain"]
        
        if mode == "page":
            start_idx, end_idx = doc["page_indices"]
        else:  # layout
            start_idx, end_idx = doc["layout_indices"]
        
        for q_idx, qa in enumerate(doc["questions"]):
            all_queries.append({
                "doc_idx": doc_idx,
                "doc_name": doc_name,
                "domain": domain,
                "q_idx": q_idx,
                "query": qa["Q"],
                "answer": qa["A"],
                "page_id": qa["page_id"],
                "type": qa["type"],
                "layout_mapping": qa.get("layout_mapping", []),
                "start_idx": start_idx,
                "end_idx": end_idx,
            })
    return all_queries


def _encode_documents_chunk(args):
    """
    Worker to encode a chunk of documents on a dedicated GPU.
    
    Args tuple contains:
        rank: worker rank
        device_id: CUDA device to use
        doc_indices: list of (doc_idx, start_idx, end_idx) tuples to process
        parquet_file: path to parquet file
        model_name: path to DSE model
        batch_size: batch size
        mode: "page" or "layout"
        tmp_dir: where to write results
        flash_attn: whether to use flash attention
    """
    (
        rank,
        device_id,
        doc_indices,
        parquet_file,
        model_name,
        batch_size,
        mode,
        tmp_dir,
        flash_attn,
    ) = args
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    torch.cuda.set_device(0)  # After setting CUDA_VISIBLE_DEVICES, device 0 is the target GPU
    
    # Import DSE here to avoid CUDA initialization issues
    from vision_wrapper import DSE
    
    # Load parquet data
    if mode == "page":
        df = pd.read_parquet(parquet_file)
    else:
        df = pd.read_parquet(parquet_file)
    
    # Initialize retriever
    retriever = DSE(model_name=model_name, bs=batch_size, flash_attn=flash_attn)
    
    results = {}
    
    for doc_idx, start_idx, end_idx in tqdm(doc_indices, desc=f"GPU {device_id}", position=rank):
        # Get candidate images for this document
        candidate_df = df.iloc[start_idx:end_idx + 1]
        candidate_images = [
            get_image_from_binary(row['image_binary']) 
            for _, row in candidate_df.iterrows()
        ]
        
        if len(candidate_images) == 0:
            continue
        
        # Get candidate indices (layout indices include metadata)
        if mode == "layout":
            candidate_indices = [
                (row_idx, row['page_id'], *row['bbox'])
                for row_idx, (_, row) in enumerate(candidate_df.iterrows())
            ]
        else:
            candidate_indices = list(range(len(candidate_images)))
        
        # Encode candidate images
        candidate_embeds = retriever.embed_quotes(candidate_images)
        candidate_embeds = np.array(candidate_embeds)
        
        # Store results
        results[(doc_idx, start_idx, end_idx)] = {
            'embeddings': candidate_embeds,
            'indices': candidate_indices,
        }
    
    # Save results to temp file
    part_path = os.path.join(tmp_dir, f"doc_embeddings.part{rank}.pkl")
    with open(part_path, 'wb') as f:
        pickle.dump(results, f)
    
    return {'part_path': part_path, 'rank': rank}


def run_first_stage_retrieval_multigpu(
    annotations: List[Dict],
    parquet_file: str,
    model_name: str,
    top_k: int = 20,
    batch_size: int = 2,
    mode: str = "page",
    num_gpus: int = 4,
    flash_attn: bool = True,
) -> Dict:
    """
    Run first-stage retrieval using multiple GPUs.
    
    Args:
        annotations: List of annotation dicts
        parquet_file: Path to parquet file
        model_name: Path to DSE model
        top_k: Number of top results to retrieve
        batch_size: Batch size for encoding
        mode: "page" or "layout"
        num_gpus: Number of GPUs to use
        flash_attn: Whether to use flash attention
    
    Returns:
        Dict mapping query key -> retrieval results
    """
    # Flatten queries
    all_queries = flatten_queries(annotations, mode)
    print(f"Total queries: {len(all_queries)}")
    
    # Group queries by document
    queries_by_doc = {}
    for q in all_queries:
        doc_key = (q["doc_idx"], q["start_idx"], q["end_idx"])
        if doc_key not in queries_by_doc:
            queries_by_doc[doc_key] = []
        queries_by_doc[doc_key].append(q)
    
    print(f"Total documents to process: {len(queries_by_doc)}")
    
    # Distribute documents across GPUs
    doc_keys = list(queries_by_doc.keys())
    chunk_size = math.ceil(len(doc_keys) / num_gpus)
    
    # Create temp directory in /dev/shm (RAM-based, much faster)
    shm_base = "/dev/shm"
    if os.path.exists(shm_base) and os.access(shm_base, os.W_OK):
        tmp_dir = tempfile.mkdtemp(prefix="mmdocir_retrieval_", dir=shm_base)
    else:
        tmp_dir = tempfile.mkdtemp(prefix="mmdocir_retrieval_")
    print(f"Using temp directory: {tmp_dir}")
    
    # Prepare tasks for each GPU
    tasks = []
    for rank in range(num_gpus):
        start = rank * chunk_size
        end = min((rank + 1) * chunk_size, len(doc_keys))
        if start >= end:
            continue
        
        chunk_doc_keys = doc_keys[start:end]
        
        tasks.append((
            rank,
            rank,  # device_id
            chunk_doc_keys,
            parquet_file,
            model_name,
            batch_size,
            mode,
            tmp_dir,
            flash_attn,
        ))
    
    print(f"Launching {len(tasks)} GPU workers...")
    
    # Run encoding in parallel
    # Use spawn to avoid CUDA initialization issues
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(tasks)) as pool:
        results = pool.map(_encode_documents_chunk, tasks)
    
    print("All GPU workers completed. Merging results...")
    
    # Merge results from all workers
    all_doc_embeddings = {}
    for res in results:
        part_path = res['part_path']
        with open(part_path, 'rb') as f:
            part_results = pickle.load(f)
        all_doc_embeddings.update(part_results)
        os.remove(part_path)
    
    # Clean up temp directory
    os.rmdir(tmp_dir)
    
    print("Encoding query embeddings...")
    
    # Now encode queries and compute retrieval scores
    # Use single GPU for queries (they're just text, fast to encode)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    from vision_wrapper import DSE
    retriever = DSE(model_name=model_name, bs=batch_size, flash_attn=flash_attn)
    
    # Process each document's queries
    final_results = {}
    
    for doc_key, doc_queries in tqdm(queries_by_doc.items(), desc="Processing queries"):
        doc_idx, start_idx, end_idx = doc_key
        
        if doc_key not in all_doc_embeddings:
            print(f"Warning: No embeddings for document {doc_idx}")
            continue
        
        doc_data = all_doc_embeddings[doc_key]
        candidate_embeds = doc_data['embeddings']
        candidate_indices = doc_data['indices']
        
        # Encode queries for this document
        query_texts = [q["query"] for q in doc_queries]
        query_embeds = retriever.embed_queries(query_texts)
        query_embeds = np.array(query_embeds)
        
        # Compute scores (dot product)
        scores = query_embeds @ candidate_embeds.T  # [num_queries, num_candidates]
        
        # Get top-k for each query
        for i, q in enumerate(doc_queries):
            query_scores = scores[i]
            
            # Get top-k indices
            if len(query_scores) <= top_k:
                top_indices = np.argsort(query_scores)[::-1].tolist()
            else:
                top_indices = np.argsort(query_scores)[-top_k:][::-1].tolist()
            
            top_scores = query_scores[top_indices].tolist()
            
            # Store result
            result_key = f"{q['doc_name']}_{q['q_idx']}"
            final_results[result_key] = {
                "doc_name": q["doc_name"],
                "domain": q["domain"],
                "q_idx": q["q_idx"],
                "query": q["query"],
                "answer": q["answer"],
                "page_id": q["page_id"],
                "type": q["type"],
                "layout_mapping": q["layout_mapping"],
                "start_idx": start_idx,
                "end_idx": end_idx,
                "top_k_local_indices": top_indices,
                "top_k_global_indices": [start_idx + idx for idx in top_indices],
                "top_k_scores": top_scores,
                "all_scores": query_scores.tolist(),
            }
            
            if mode == "layout":
                final_results[result_key]["top_k_layout_indices"] = [
                    candidate_indices[idx] for idx in top_indices
                ]
    
    return final_results


def run_first_stage_retrieval_single_gpu(
    annotations: List[Dict],
    parquet_df: pd.DataFrame,
    retriever,
    top_k: int = 20,
    mode: str = "page",
    layouts_df: pd.DataFrame = None,
) -> Dict:
    """
    Run first-stage retrieval using a single GPU (original implementation).
    """
    # Flatten queries
    all_queries = flatten_queries(annotations, mode)
    print(f"Total queries: {len(all_queries)}")
    
    # Group queries by document
    queries_by_doc = {}
    for q in all_queries:
        doc_key = (q["doc_idx"], q["start_idx"], q["end_idx"])
        if doc_key not in queries_by_doc:
            queries_by_doc[doc_key] = []
        queries_by_doc[doc_key].append(q)
    
    print(f"Processing {len(queries_by_doc)} documents...")
    
    results = {}
    
    for doc_key, doc_queries in tqdm(queries_by_doc.items(), desc="Processing documents"):
        doc_idx, start_idx, end_idx = doc_key
        
        # Get candidate images for this document
        if mode == "page":
            candidate_df = parquet_df.iloc[start_idx:end_idx + 1]
        else:
            candidate_df = layouts_df.iloc[start_idx:end_idx + 1]
        
        candidate_images = [
            get_image_from_binary(row['image_binary']) 
            for _, row in candidate_df.iterrows()
        ]
        
        if len(candidate_images) == 0:
            print(f"Warning: No candidates for document {doc_idx}")
            continue
        
        # Get candidate indices
        if mode == "layout":
            candidate_indices = [
                (row_idx, row['page_id'], *row['bbox'])
                for row_idx, (_, row) in enumerate(candidate_df.iterrows())
            ]
        else:
            candidate_indices = list(range(len(candidate_images)))
        
        # Encode candidates
        candidate_embeds = retriever.embed_quotes(candidate_images)
        candidate_embeds = np.array(candidate_embeds)
        
        # Encode queries
        query_texts = [q["query"] for q in doc_queries]
        query_embeds = retriever.embed_queries(query_texts)
        query_embeds = np.array(query_embeds)
        
        # Compute scores
        scores = query_embeds @ candidate_embeds.T
        
        # Get top-k for each query
        for i, q in enumerate(doc_queries):
            query_scores = scores[i]
            
            if len(query_scores) <= top_k:
                top_indices = np.argsort(query_scores)[::-1].tolist()
            else:
                top_indices = np.argsort(query_scores)[-top_k:][::-1].tolist()
            
            top_scores = query_scores[top_indices].tolist()
            
            result_key = f"{q['doc_name']}_{q['q_idx']}"
            results[result_key] = {
                "doc_name": q["doc_name"],
                "domain": q["domain"],
                "q_idx": q["q_idx"],
                "query": q["query"],
                "answer": q["answer"],
                "page_id": q["page_id"],
                "type": q["type"],
                "layout_mapping": q["layout_mapping"],
                "start_idx": start_idx,
                "end_idx": end_idx,
                "top_k_local_indices": top_indices,
                "top_k_global_indices": [start_idx + idx for idx in top_indices],
                "top_k_scores": top_scores,
                "all_scores": query_scores.tolist(),
            }
            
            if mode == "layout":
                results[result_key]["top_k_layout_indices"] = [
                    candidate_indices[idx] for idx in top_indices
                ]
    
    return results


def convert_to_official_format(results: Dict, annotations: List[Dict], mode: str = "page") -> List[Dict]:
    """
    Convert first-stage results to official MMDocIR format for evaluation.
    
    This matches the format expected by metric_eval.evaluate_page() and evaluate_layout().
    
    The official format expects:
    - scores_page: List of scores indexed by LOCAL page index within document
    - page_id: Ground truth LOCAL page indices (list)
    - domain: Domain category
    """
    # Build a mapping from (doc_name, q_idx) to result
    result_lookup = {}
    for key, result in results.items():
        result_lookup[(result["doc_name"], result["q_idx"])] = result
    
    official_results = []
    
    # Iterate through annotations in order (same order as official search.py)
    q_count = 0
    for doc in annotations:
        doc_name = doc["doc_name"]
        domain = doc["domain"]
        
        for q_idx, qa in enumerate(doc["questions"]):
            result_key = (doc_name, q_idx)
            
            if result_key not in result_lookup:
                print(f"Warning: Missing result for {doc_name} q_idx={q_idx}")
                q_count += 1
                continue
            
            result = result_lookup[result_key]
            
            official_item = {
                "domain": domain,
                "page_id": qa["page_id"],  # Ground truth LOCAL page indices
                "layout_mapping": qa.get("layout_mapping", []),
            }
            
            if mode == "page":
                # all_scores is already indexed by local page index
                official_item["scores_page"] = result["all_scores"]
            else:
                # For layout mode, need layout_indices and scores_layout
                official_item["layout_indices"] = result.get("top_k_layout_indices", [])
                official_item["scores_layout"] = result["all_scores"]
            
            official_results.append(official_item)
            q_count += 1
    
    return official_results


def evaluate_first_stage_results(results: Dict, annotations: List[Dict], mode: str = "page", model_name: str = "DSE"):
    """
    Evaluate first-stage retrieval results using official MMDocIR evaluation functions.
    """
    from utils.metric_eval import evaluate_page, evaluate_layout
    
    # Convert to official format
    official_results = convert_to_official_format(results, annotations, mode)
    
    print("\n" + "="*120)
    print(f"MMDocIR {mode.upper()}-LEVEL FIRST-STAGE EVALUATION RESULTS")
    print("="*120)
    print(f"Using OFFICIAL MMDocIR evaluation functions")
    print(f"Total queries: {len(official_results)}")
    print("-"*120)
    
    # Print domain header
    domain_list = [
        "Research", "Admin", "Tutorial", "Academic", "Brochure",
        "Financial", "Guidebook", "Government", "Laws", "News", "Avg", "Overall"
    ]
    header = " | ".join(f"{d:>8}" for d in domain_list)
    print(f"Domains: {header}")
    print("-"*120)
    
    if mode == "page":
        topk_values = [1, 3, 5]
    else:
        topk_values = [1, 5, 10]
    
    for topk in topk_values:
        if mode == "page":
            evaluate_page(official_results, model_name=model_name, topk=topk, metric="recall")
        else:
            evaluate_layout(official_results, model_name=model_name, topk=topk, metric="recall")
    
    print("="*120)


def main():
    parser = argparse.ArgumentParser(description="First-stage retrieval for MMDocIR")
    parser.add_argument(
        "--model_name",
        type=str,
        default="checkpoint/dse-phi3-v1",
        help="Path to DSE model checkpoint",
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="MMDocIR/dataset/MMDocIR_annotations.jsonl",
        help="Path to annotations file",
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
        "--output_dir",
        type=str,
        default="data/mmdocir",
        help="Output directory for cached results",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of top results to retrieve",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["page", "layout", "both"],
        default="both",
        help="Retrieval mode: page, layout, or both",
    )
    parser.add_argument(
        "--flash_attn",
        action="store_true",
        default=True,
        help="Use flash attention",
    )
    parser.add_argument(
        "--no_flash_attn",
        action="store_true",
        help="Disable flash attention",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available)",
    )
    parser.add_argument(
        "--single_gpu",
        action="store_true",
        help="Use single GPU mode (disable multi-GPU)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate cached results using official MMDocIR metrics (no retrieval)",
    )
    parser.add_argument(
        "--evaluate_only",
        action="store_true",
        help="Only evaluate cached results, skip retrieval even if cache doesn't exist",
    )
    
    args = parser.parse_args()
    
    # Handle flash attention flag
    use_flash_attn = args.flash_attn and not args.no_flash_attn
    
    # Determine number of GPUs
    if args.single_gpu:
        num_gpus = 1
    elif args.num_gpus:
        num_gpus = min(args.num_gpus, torch.cuda.device_count())
    else:
        num_gpus = torch.cuda.device_count()
    
    print(f"Using {num_gpus} GPU(s)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check for cached results
    page_cache_file = os.path.join(args.output_dir, f"first_stage_page_top{args.top_k}.pkl")
    layout_cache_file = os.path.join(args.output_dir, f"first_stage_layout_top{args.top_k}.pkl")
    
    run_page = args.mode in ["page", "both"]
    run_layout = args.mode in ["layout", "both"]
    
    # Check if already cached
    if run_page and os.path.exists(page_cache_file):
        print(f"Page retrieval results already cached at {page_cache_file}")
        run_page = False
    
    if run_layout and os.path.exists(layout_cache_file):
        print(f"Layout retrieval results already cached at {layout_cache_file}")
        run_layout = False
    
    # Load annotations (needed for both retrieval and evaluation)
    print("Loading annotations...")
    annotations = load_annotations(args.annotation_file)
    print(f"Loaded {len(annotations)} documents")
    
    if args.evaluate_only:
        # Skip retrieval, just evaluate cached results
        print("Evaluate-only mode: skipping retrieval")
        
        if args.mode in ["page", "both"] and os.path.exists(page_cache_file):
            print(f"\nLoading page results from {page_cache_file}...")
            with open(page_cache_file, 'rb') as f:
                page_results = pickle.load(f)
            evaluate_first_stage_results(page_results, annotations, mode="page", model_name="DSE-wikiss")
        elif args.mode in ["page", "both"]:
            print(f"Warning: Page cache file not found: {page_cache_file}")
        
        if args.mode in ["layout", "both"] and os.path.exists(layout_cache_file):
            print(f"\nLoading layout results from {layout_cache_file}...")
            with open(layout_cache_file, 'rb') as f:
                layout_results = pickle.load(f)
            evaluate_first_stage_results(layout_results, annotations, mode="layout", model_name="DSE-wikiss")
        elif args.mode in ["layout", "both"]:
            print(f"Warning: Layout cache file not found: {layout_cache_file}")
        
        return
    
    if not run_page and not run_layout:
        print("All results already cached.")
        if args.evaluate:
            # Run evaluation on cached results
            if args.mode in ["page", "both"] and os.path.exists(page_cache_file):
                print(f"\nLoading page results from {page_cache_file}...")
                with open(page_cache_file, 'rb') as f:
                    page_results = pickle.load(f)
                evaluate_first_stage_results(page_results, annotations, mode="page", model_name="DSE-wikiss")
            
            if args.mode in ["layout", "both"] and os.path.exists(layout_cache_file):
                print(f"\nLoading layout results from {layout_cache_file}...")
                with open(layout_cache_file, 'rb') as f:
                    layout_results = pickle.load(f)
                evaluate_first_stage_results(layout_results, annotations, mode="layout", model_name="DSE-wikiss")
        return
    
    # Change to MMDocIR directory for model loading (DSE may use relative paths)
    original_dir = os.getcwd()
    mmdocir_dir = Path(MMDOCIR_PATH)
    if not mmdocir_dir.exists():
        print(f"ERROR: MMDocIR directory not found at {mmdocir_dir}")
        print("DSE first-stage retrieval requires MMDocIR repository.")
        print("Set MMDOCIR_PATH environment variable or clone to ../MMDocIR")
        sys.exit(1)
    os.chdir(mmdocir_dir)
    
    # Determine model path
    model_path = args.model_name
    if not os.path.exists(model_path):
        model_path = f"checkpoint/{os.path.basename(args.model_name)}"
    
    try:
        if num_gpus > 1:
            # Multi-GPU mode
            if run_page:
                print("\n" + "="*80)
                print(f"Running page-level retrieval with {num_gpus} GPUs...")
                print("="*80)
                
                page_results = run_first_stage_retrieval_multigpu(
                    annotations=annotations,
                    parquet_file=os.path.join(original_dir, args.pages_parquet),
                    model_name=model_path,
                    top_k=args.top_k,
                    batch_size=args.batch_size,
                    mode="page",
                    num_gpus=num_gpus,
                    flash_attn=use_flash_attn,
                )
                
                # Save results
                os.chdir(original_dir)
                with open(page_cache_file, 'wb') as f:
                    pickle.dump(page_results, f)
                print(f"Page retrieval results saved to {page_cache_file}")
                print(f"Total queries processed: {len(page_results)}")
                os.chdir(mmdocir_dir)
            
            if run_layout:
                print("\n" + "="*80)
                print(f"Running layout-level retrieval with {num_gpus} GPUs...")
                print("="*80)
                
                layout_results = run_first_stage_retrieval_multigpu(
                    annotations=annotations,
                    parquet_file=os.path.join(original_dir, args.layouts_parquet),
                    model_name=model_path,
                    top_k=args.top_k,
                    batch_size=args.batch_size,
                    mode="layout",
                    num_gpus=num_gpus,
                    flash_attn=use_flash_attn,
                )
                
                # Save results
                os.chdir(original_dir)
                with open(layout_cache_file, 'wb') as f:
                    pickle.dump(layout_results, f)
                print(f"Layout retrieval results saved to {layout_cache_file}")
                print(f"Total queries processed: {len(layout_results)}")
                os.chdir(mmdocir_dir)
        
        else:
            # Single GPU mode
            from vision_wrapper import DSE
            
            print(f"Initializing DSE retriever from {model_path}...")
            retriever = DSE(
                model_name=model_path,
                bs=args.batch_size,
                flash_attn=use_flash_attn,
            )
            
            os.chdir(original_dir)
            
            # Load data
            pages_df = None
            layouts_df = None
            
            if run_page:
                pages_df = load_pages_parquet(args.pages_parquet)
            
            if run_layout:
                layouts_df = load_layouts_parquet(args.layouts_parquet)
            
            if run_page:
                print("\n" + "="*80)
                print("Running page-level retrieval...")
                print("="*80)
                
                page_results = run_first_stage_retrieval_single_gpu(
                    annotations=annotations,
                    parquet_df=pages_df,
                    retriever=retriever,
                    top_k=args.top_k,
                    mode="page",
                )
                
                with open(page_cache_file, 'wb') as f:
                    pickle.dump(page_results, f)
                print(f"Page retrieval results saved to {page_cache_file}")
                print(f"Total queries processed: {len(page_results)}")
            
            if run_layout:
                print("\n" + "="*80)
                print("Running layout-level retrieval...")
                print("="*80)
                
                layout_results = run_first_stage_retrieval_single_gpu(
                    annotations=annotations,
                    parquet_df=pages_df,
                    retriever=retriever,
                    top_k=args.top_k,
                    mode="layout",
                    layouts_df=layouts_df,
                )
                
                with open(layout_cache_file, 'wb') as f:
                    pickle.dump(layout_results, f)
                print(f"Layout retrieval results saved to {layout_cache_file}")
                print(f"Total queries processed: {len(layout_results)}")
    
    finally:
        os.chdir(original_dir)
    
    print("\nFirst-stage retrieval completed!")
    
    # Evaluate if requested
    if args.evaluate or args.evaluate_only:
        print("\n" + "="*80)
        print("Running evaluation...")
        print("="*80)
        
        if args.mode in ["page", "both"] and os.path.exists(page_cache_file):
            print(f"\nLoading page results from {page_cache_file}...")
            with open(page_cache_file, 'rb') as f:
                page_results = pickle.load(f)
            evaluate_first_stage_results(page_results, annotations, mode="page", model_name="DSE-wikiss")
        
        if args.mode in ["layout", "both"] and os.path.exists(layout_cache_file):
            print(f"\nLoading layout results from {layout_cache_file}...")
            with open(layout_cache_file, 'rb') as f:
                layout_results = pickle.load(f)
            evaluate_first_stage_results(layout_results, annotations, mode="layout", model_name="DSE-wikiss")


if __name__ == "__main__":
    main()
