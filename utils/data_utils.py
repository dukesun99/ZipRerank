"""
Data utilities for Qwen3-VL optical ranker training.

Adapted from rank_llm with label format changed from [A], [B], [C] to [1], [2], [3]
to work with Qwen3-VL's vision_id system.
"""

import copy
import os
from functools import partial
from typing import Dict, Sequence, List, Optional, Tuple

import torch
import transformers
from datasets import load_dataset
from ftfy import fix_text
from torch.utils.data import DataLoader, Dataset
from accelerate.logging import get_logger
from PIL import Image

logger = get_logger(__name__)

max_psg_num = 20
IGNORE_INDEX = -100


def create_ranking_prompt_for_training(query: str, num_passages: int, use_query_adapter: bool = False) -> str:
    """
    Create ranking prompt that EXACTLY matches evaluation format.
    This function is shared between training and evaluation to ensure consistency.
    
    Args:
        query: The search query text
        num_passages: Number of passages/images being ranked
        use_query_adapter: If True, append <query_emb> token after query for adapter
        
    Returns:
        Formatted prompt string
    """
    prompt_parts = []
    
    # System instruction
    system_msg = "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."
    prompt_parts.append(system_msg)
    prompt_parts.append("")
    
    # Instruction about passages (images will be added separately)
    prompt_parts.append(f"I will provide you with {num_passages} passages as images.")
    prompt_parts.append("Rank the passages based on their relevance to the search query.")
    prompt_parts.append("")
    
    # Add explicit image correspondence (matches evaluation format EXACTLY)
    image_order = ", ".join([
        f"Picture {i+1} is passage [{chr(ord('A') + i)}]"
        for i in range(num_passages)
    ])
    prompt_parts.append(f"The images are provided in order: {image_order}.")
    prompt_parts.append("")
    
    # Add query and instruction
    # When query adapter is enabled, add <query_emb> token after the query
    if use_query_adapter:
        prompt_parts.append(f"Search Query: {query}<query_emb>")
    else:
        prompt_parts.append(f"Search Query: {query}")
    prompt_parts.append("")
    prompt_parts.append("Rank the passages above based on their relevance to the search query.")
    prompt_parts.append("The passages should be listed in descending order using identifiers.")
    prompt_parts.append("The most relevant passages should be listed first.")
    prompt_parts.append("The output format should be [A] > [B], etc.")
    prompt_parts.append("Only output the ranking results, do not say anything else.")
    
    return "\n".join(prompt_parts)


def prepare_ranking_inputs(
    prompt: str,
    passage_images: list,
    processor,
):
    """
    Prepare inputs for ranking model inference.
    
    Shared between training and evaluation to ensure consistency:
    1. Create messages with text + images
    2. Apply chat template with tokenize=True (expands vision tokens)
    3. Add '[' token after prompt
    4. Return input_ids, attention_mask, pixel_values, image_grid_thw
    
    Args:
        prompt: Text prompt created by create_ranking_prompt_for_training()
        passage_images: List of PIL images (one per passage)
        processor: Qwen3VLProcessor
        
    Returns:
        dict with keys: input_ids, attention_mask, pixel_values, image_grid_thw
    """
    tokenizer = processor.tokenizer
    
    # Create messages with images for Qwen3-VL
    # Format: user role with text first, then all images
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ] + [
                {"type": "image", "image": img}
                for img in passage_images
            ]
        }
    ]

    # # Format: images first, then text
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "image": img}
    #             for img in passage_images
    #         ] + [
    #             {"type": "text", "text": prompt}
    #         ]
    #     }
    # ]

    # Use processor.apply_chat_template(tokenize=True) for vision token expansion
    prompt_inputs = processor.apply_chat_template(
        [messages],  # Wrap in list for single sample
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors='pt'
    )
    
    # Extract tokenized prompt with expanded vision tokens
    prompt_ids = prompt_inputs['input_ids'][0].tolist()
    
    # CRITICAL: Add '[' token AFTER the prompt (model predicts first letter: A, B, C)
    bracket_token = tokenizer.encode('[', add_special_tokens=False)[0]
    prompt_ids.append(bracket_token)
    
    # Test decoding the prompt
    # decoded_prompt = tokenizer.decode(prompt_ids)
    # print(f"Decoded prompt: {decoded_prompt}")
    
    # Return as dict
    # NOTE: pixel_values and image_grid_thw are already in correct shape:
    #   - pixel_values: [num_patches, features]
    #   - image_grid_thw: [num_images, 3]
    # Don't do [0] indexing - that would extract first patch/image only!
    return {
        'input_ids': prompt_ids,
        'pixel_values': prompt_inputs['pixel_values'],
        'image_grid_thw': prompt_inputs['image_grid_thw'],
    }


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Preprocess the data by tokenizing.
    
    CRITICAL: To ensure proper tokenization of ranking labels like "A ] > [ B ]",
    we tokenize prompts and labels SEPARATELY, then concatenate as token IDs.
    This prevents '[' + 'A' from being tokenized as '[A'.
    """
    # Tokenize sources (prompts) and targets (labels) separately
    sources_tokenized = _tokenize_fn(sources, tokenizer)
    
    # For targets, tokenize each component to ensure proper token boundaries
    # This prevents '[' + 'A' from becoming '[A' token
    targets_tokenized_list = []
    for target in targets:
        # Tokenize the target string
        # Since target is like "A ] > [ B ]<eos>", each letter should be separate
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        targets_tokenized_list.append(target_ids)
    
    # Concatenate source and target token IDs
    input_ids = []
    labels = []
    for source_ids, target_ids in zip(sources_tokenized["input_ids"], targets_tokenized_list):
        # Convert source_ids tensor to list if needed
        if hasattr(source_ids, 'tolist'):
            source_ids = source_ids.tolist()
        
        # Combine source + target as token IDs (both are now lists)
        combined_ids = source_ids + target_ids
        input_ids.append(combined_ids)
        
        # Create labels: mask source tokens with IGNORE_INDEX
        label = [IGNORE_INDEX] * len(source_ids) + target_ids
        labels.append(label)
    
    return input_ids, labels, sources_tokenized["input_ids_lens"]


def convert_alpha_to_numeric(text: str) -> str:
    """
    Convert [A], [B], [C] format to [1], [2], [3] format.
    
    Args:
        text: Input text with [A], [B], [C] labels
        
    Returns:
        Text with [1], [2], [3] labels
    """
    import re
    
    # Find all [A], [B], [C] patterns
    def replace_letter(match):
        letter = match.group(1)
        # A->1, B->2, C->3, etc.
        number = ord(letter) - ord('A') + 1
        return f"[{number}]"
    
    # Replace all [A-Z] patterns with [1-20]
    result = re.sub(r'\[([A-Z])\]', replace_letter, text)
    return result


def detect_msmarco_compact_format(data_sample):
    """
    Detect MS MARCO compact format.
    
    Expected fields:
    - qid (str)
    - query (str)
    - passage_ids (list[str])
    - golden_ranking (list[int])
    - view_id (int, optional)
    """
    if not isinstance(data_sample, dict):
        return False
    required = {"qid", "query", "passage_ids", "golden_ranking"}
    return required.issubset(set(data_sample.keys()))


def convert_msmarco_compact_to_conversations(training_data, passages_dict):
    """
    Convert MS MARCO compact entries to the standard conversations format.
    
    Ensures prompts match the existing datasets exactly.
    """
    converted = []
    system_msg = "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."
    
    for item in training_data:
        query = item["query"]
        passage_ids = item["passage_ids"]
        golden_ranking = item["golden_ranking"]
        
        passages = [passages_dict.get(pid, "") for pid in passage_ids]
        
        # Target ranking: order passages by golden_ranking (0 = most relevant)
        ranked_indices = sorted(range(len(golden_ranking)), key=lambda i: golden_ranking[i])
        target_ranking = " > ".join(f"[{chr(ord('A') + i)}]" for i in ranked_indices)
        
        passages_text = "\n".join(
            f"[{chr(ord('A') + i)}] {text}"
            for i, text in enumerate(passages)
        )
        
        input_context = (
            f"I will provide you with {len(passages)} passages, each indicated by a alphabetical identifier [].\n"
            f"Rank the passages based on their relevance to the search query: {query}.\n\n"
            f"{passages_text}\n\n"
            f"Search Query: {query}.\n"
            f"Rank the {len(passages)} passages above based on their relevance to the search query in descending order. "
            f"Only response the ranking results, do not say any word or explain."
        )
        
        converted.append({
            "conversations": [
                {"value": system_msg},
                {"value": input_context},
                {"value": target_ranking},
            ]
        })
    
    return converted


class MMDocIRDataset(Dataset):
    """
    Dataset for MMDocIR OpenAI-generated reranking data.
    
    Supports two formats:
    1. Self-contained format (recommended):
       - training_data.jsonl + images.parquet in the same directory
       - images.parquet has columns: doc_name, page_id, image
    
    2. Original parquet format:
       - Loads from full parquet files (ArxivQA_filter.parquet, etc.)
    
    Training data format (JSONL):
    {
        "query_id": ...,
        "query": ...,
        "doc_name": ...,
        "candidate_page_ids": [...],
        "openai_ranking": [...],  # indices into candidates (best first)
        "gt_page_ids": [...],
    }
    """
    
    def __init__(
        self,
        training_data: List[Dict],
        parquet_dir: str,
        tokenizer,
        processor=None,
        domains: List[str] = None,
        max_candidates: int = 20,
        use_query_adapter: bool = False,
        shuffle_candidates: bool = False,
        force_gt_top1: bool = False,
    ):
        self.training_data = training_data
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_candidates = max_candidates
        self.use_query_adapter = use_query_adapter
        self.shuffle_candidates = shuffle_candidates
        self.force_gt_top1 = force_gt_top1
        
        # Try to load from self-contained format first
        images_parquet = os.path.join(parquet_dir, "images.parquet")
        if os.path.exists(images_parquet):
            logger.info(f"Loading self-contained images from {images_parquet}")
            self.pages_df, self.page_lookup = self._load_self_contained(images_parquet)
        else:
            # Fall back to original parquet format
            logger.info(f"Loading from original parquet files in {parquet_dir}")
            self.pages_df, self.page_lookup = self._load_parquets(parquet_dir, domains)
        
        logger.info(f"MMDocIRDataset: {len(self.training_data)} samples, {len(self.page_lookup)} page images")
        logger.info(f"  shuffle_candidates={self.shuffle_candidates}, force_gt_top1={self.force_gt_top1}")
    
    def _load_self_contained(self, images_parquet: str):
        """Load from self-contained images.parquet format."""
        import pandas as pd
        
        df = pd.read_parquet(images_parquet)
        logger.info(f"Loaded {len(df)} images from {images_parquet}")
        
        # Build lookup: (doc_name, page_id) -> row_idx
        page_lookup = {}
        for idx, row in df.iterrows():
            key = (row['doc_name'], row['page_id'])
            page_lookup[key] = idx
        
        return df, page_lookup
    
    def _load_parquets(self, parquet_dir: str, domains: List[str] = None):
        """Load parquet files and build (doc_name, page_id) -> row_idx lookup."""
        import pandas as pd
        
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
                logger.info(f"Loading {parquet_name}...")
                df = pd.read_parquet(parquet_file)
                df['domain'] = domain
                dfs.append(df)
        
        if not dfs:
            raise ValueError(f"No parquet files found in {parquet_dir}")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Build lookup
        page_lookup = {}
        for idx, row in combined_df.iterrows():
            key = (row['file_name'], row['page'])
            page_lookup[key] = idx
        
        return combined_df, page_lookup
    
    def _resize_image_if_needed(self, img: Image.Image, max_size: int = 1024) -> Image.Image:
        """Resize image so that its largest dimension is at most max_size.
        
        This reduces the number of visual tokens generated by the vision encoder,
        significantly reducing memory usage during training.
        
        Args:
            img: PIL Image to resize
            max_size: Maximum size for the largest dimension
            
        Returns:
            Resized PIL Image (or original if already smaller)
        """
        width, height = img.size
        max_dim = max(width, height)
        
        if max_dim <= max_size:
            return img
        
        # Calculate new dimensions preserving aspect ratio
        scale = max_size / max_dim
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Use LANCZOS for high-quality downsampling
        return img.resize((new_width, new_height), Image.LANCZOS)
    
    def _get_image(self, doc_name: str, page_id: int, max_size: int = 1024) -> Optional[Image.Image]:
        """Get page image from parquet data, resized to max_size if needed.
        
        Args:
            doc_name: Document name
            page_id: Page ID
            max_size: Maximum size for largest dimension (default: 1024)
            
        Returns:
            PIL Image resized so largest dim <= max_size
        """
        import io
        key = (doc_name, page_id)
        if key not in self.page_lookup:
            return None
        
        row_idx = self.page_lookup[key]
        row = self.pages_df.iloc[row_idx]
        
        # Handle both formats: 'image' (self-contained) or 'image' (original)
        image_col = 'image' if 'image' in row else 'image'
        
        try:
            img = Image.open(io.BytesIO(row[image_col])).convert("RGB")
            # Resize to reduce visual tokens and memory usage
            img = self._resize_image_if_needed(img, max_size)
            return img
        except Exception as e:
            logger.warning(f"Failed to load image {doc_name} page {page_id}: {e}")
            return None
    
    def __len__(self):
        return len(self.training_data)
    
    def __getitem__(self, index):
        item = self.training_data[index]
        
        query = item['query']
        doc_name = item['doc_name']
        candidate_page_ids = item['candidate_page_ids'][:self.max_candidates]
        openai_ranking = item['openai_ranking'][:self.max_candidates]
        gt_page_ids_raw = set(item.get('gt_page_ids', []))
        
        # Load candidate images
        candidate_images = []
        valid_indices = []
        
        for i, page_id in enumerate(candidate_page_ids):
            img = self._get_image(doc_name, page_id)
            if img is not None:
                candidate_images.append(img)
                valid_indices.append(i)
        
        if len(candidate_images) < 2:
            # Not enough valid images - error out instead of silently skipping
            raise ValueError(
                f"Sample at index {index} has fewer than 2 valid images. "
                f"Got {len(candidate_images)} valid images out of {len(candidate_page_ids)} candidates. "
                f"Document: {doc_name}, Query: {query[:50]}..."
            )
        
        # Remap openai_ranking to valid indices only
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}
        remapped_ranking = [old_to_new[idx] for idx in openai_ranking if idx in old_to_new]
        
        # Add any missing indices at the end
        for i in range(len(valid_indices)):
            if i not in remapped_ranking:
                remapped_ranking.append(i)
        
        # Track GT indices: which final candidate positions are ground truth
        # First compute GT indices after old_to_new mapping
        gt_indices = []
        for orig_idx in valid_indices:
            page_id = candidate_page_ids[orig_idx]
            if page_id in gt_page_ids_raw:
                gt_indices.append(old_to_new[orig_idx])
        
        # SHUFFLE: Randomize candidate order to remove positional bias
        old_to_shuffled = None
        if self.shuffle_candidates:
            import random
            n_images = len(candidate_images)
            shuffle_perm = list(range(n_images))
            random.shuffle(shuffle_perm)
            
            # Reorder images according to shuffle permutation
            candidate_images = [candidate_images[i] for i in shuffle_perm]
            
            # Update ranking indices: map old index to new shuffled position
            old_to_shuffled = {old: new for new, old in enumerate(shuffle_perm)}
            remapped_ranking = [old_to_shuffled[idx] for idx in remapped_ranking]
            
            # Update GT indices to new shuffled positions
            gt_indices = [old_to_shuffled[idx] for idx in gt_indices]
        
        # FORCE GT TOP-1: Move ground truth to position 0 in target ranking
        if self.force_gt_top1 and gt_indices:
            # Use the first GT index (already computed above)
            gt_new_idx = gt_indices[0]
            if gt_new_idx in remapped_ranking:
                remapped_ranking.remove(gt_new_idx)
                remapped_ranking.insert(0, gt_new_idx)
        
        # Build target ranking string
        target_ranking = " > ".join(f"[{chr(ord('A') + idx)}]" for idx in remapped_ranking)
        
        # Build prompt
        num_passages = len(candidate_images)
        prompt_text = create_ranking_prompt_for_training(query, num_passages, use_query_adapter=self.use_query_adapter)
        
        # Create messages structure (like GenerationDataset)
        # Format: text first, then images
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text}
                ] + [
                    {"type": "image", "image": img}
                    for img in candidate_images
                ]
            }
        ]
        # # Format: images first, then text
        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image", "image": img}
        #             for img in candidate_images
        #         ] + [
        #             {"type": "text", "text": prompt_text}
        #         ]
        #     }
        # ]

        # Label text (remove first '[' as done in GenerationDataset)
        label_text = target_ranking[1:] + self.tokenizer.eos_token  # "A] > [B] > ..."
        
        # Rank labels for combined training
        rank_label = remapped_ranking
        
        # Return GT indices for GT@1 loss (list of candidate positions that are ground truth)
        # If no GT found, use position 0 as fallback (first in ranking)
        if not gt_indices:
            gt_indices = [remapped_ranking[0]] if remapped_ranking else [0]
        
        return messages, label_text, rank_label, candidate_images, gt_indices


class MSMARCOCompactDataset(Dataset):
    """
    Dataset for MS MARCO compact format with on-the-fly passage loading.
    
    Format:
    training_data.jsonl: {qid, query, passage_ids, golden_ranking, view_id}
    passages_corpus.jsonl: {pid, text}
    
    Loads passage texts on-demand and converts to conversations format dynamically.
    """
    
    def __init__(
        self,
        training_data_path,
        passages_corpus_path,
        tokenizer,
        processor=None,
        document_renderer=None
    ):
        # Load training data (compact format)
        self.training_data = load_data(training_data_path)
        logger.info(f"Loaded {len(self.training_data)} training instances")
        
        # Load passages corpus into memory for fast lookup
        logger.info(f"Loading passages corpus from {passages_corpus_path}...")
        passages_data = load_data(passages_corpus_path)
        self.passage_dict = {p['pid']: p['text'] for p in passages_data}
        logger.info(f"Loaded {len(self.passage_dict)} passages into memory")
        
        self.tokenizer = tokenizer
        self.processor = processor
        self.document_renderer = document_renderer
    
    def __len__(self):
        return len(self.training_data)
    
    def __getitem__(self, index):
        item = self.training_data[index]
        
        qid = item['qid']
        query = item['query']
        passage_ids = item['passage_ids']
        golden_ranking = item['golden_ranking']
        
        # Get passage texts
        passages = [self.passage_dict.get(pid, '') for pid in passage_ids]
        
        # Convert golden_ranking indices to letter-based ranking
        # golden_ranking[i] is the rank (0=most relevant) of passage i
        # We need to output: which passage is rank 0, rank 1, rank 2, etc.
        ranked_indices = sorted(range(len(golden_ranking)), key=lambda i: golden_ranking[i])
        target_ranking = " > ".join(f"[{chr(ord('A') + i)}]" for i in ranked_indices)
        
        # Build conversations format
        system_msg = "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."
        
        # Format passages with [A], [B], [C] labels
        passages_text = "\n".join([
            f"[{chr(ord('A') + i)}] {passage}"
            for i, passage in enumerate(passages)
        ])
        
        input_context = f"I will provide you with {len(passages)} passages, each indicated by a alphabetical identifier [].\nRank the passages based on their relevance to the search query: {query}.\n\n{passages_text}\n\nSearch Query: {query}.\nRank the {len(passages)} passages above based on their relevance to the search query in descending order. Only response the ranking results, do not say any word or explain."
        
        # Render passages to images if renderer available
        if self.document_renderer is not None:
            passages_images = [self.document_renderer.render(p) for p in passages]
            return (system_msg, input_context, target_ranking, passages_images)
        else:
            return (system_msg, input_context, target_ranking)


class RankingDataset(Dataset):
    """Dataset for ranking tasks with Qwen3-VL."""

    def __init__(
        self,
        raw_data,
        model_tokenizer,
        type,
        document_renderer=None,
        processor=None,
        use_query_adapter: bool = False,
    ) -> None:
        self.raw_data = raw_data
        self.tokenizer = model_tokenizer
        self.processor = processor  # Qwen3-VL processor for handling images
        self.tokenizer.padding_side = "left"
        self.type = type
        self.system_message_supported = "system" in self.tokenizer.chat_template
        self.document_renderer = document_renderer
        self.use_query_adapter = use_query_adapter

    def __getitem__(self, index):
        conversation = self.raw_data[index]["conversations"]

        # Validate conversation structure
        if len(conversation) < 3:
            raise ValueError(f"Invalid conversation format at index {index}")

        sys_msg = conversation[0]["value"]
        input_context = conversation[1]["value"]
        target_generation = conversation[2]["value"]

        # Handle image rendering: extract passages and render as images
        # OPTICAL RANKER: Use images only, NO passage text
        passages_images = []
        passages_data = []
        if self.document_renderer is not None:
            import re
            
            # First, extract only the passage section (before "Search Query:")
            # This avoids matching the example format "[B] > [A]" in the instructions
            # Use greedy .* to capture all passages until "Search Query:"
            passage_section_match = re.search(r'passages.+(?=Search Query:|Rank the)', input_context, re.DOTALL | re.IGNORECASE)
            if passage_section_match:
                passage_section = passage_section_match.group(0)
            else:
                # Fallback: use entire input_context but filter by length
                passage_section = input_context
            
            # Extract passages from passage section (still has [A], [B], [C] format)
            passage_pattern = r'\[([A-T])\]\s+(.*?)(?=\n\[([A-T])\]|\Z)'
            passages_found = re.findall(passage_pattern, passage_section, re.DOTALL)
            
            # Render each passage to image
            for letter, passage_text, *_ in passages_found:
                passage_text = passage_text.strip()
                # Filter out very short matches (like "> " from example format)
                if passage_text and len(passage_text) > 10:
                    # Render to PIL Image
                    img = self.document_renderer.render(passage_text)
                    passages_images.append(img)
                    passages_data.append({'letter': letter})
        
        # Extract query for compact prompt
        query_match = re.search(r'search query:\s*(.+?)\.?\s*\n', input_context, re.IGNORECASE)
        if query_match:
            query = query_match.group(1).strip()
        else:
            query_match = re.search(r'query:\s*(.+?)\.?\s*\n', input_context, re.IGNORECASE)
            query = query_match.group(1).strip() if query_match else "Unknown query"
        
        # Build prompt with EXACT evaluation format (NO passage text)
        if passages_images:
            # Use EXACT same function as evaluation
            prompt_text = create_ranking_prompt_for_training(query, len(passages_data), use_query_adapter=self.use_query_adapter)
            
            # Create messages - EXACTLY like evaluation (NO separate system role)
            # Everything goes in user message
            user_content = [{"type": "text", "text": prompt_text}]
            
            # Add all passage images
            for img in passages_images:
                user_content.append({
                    "type": "image",
                    "image": img,
                })
            
            # Create messages - EXACTLY like evaluation (only user role)
            messages = [{
                "role": "user",
                "content": user_content
            }]
        else:
            # No images rendered - this is an error for optical ranking
            raise ValueError(
                f"No images rendered for sample at index {index}. "
                f"Optical ranking requires all samples to have renderable passages. "
                f"Check document_renderer configuration and passage extraction."
            )
        
        # Apply chat template with vision support
        # add_vision_id=True assigns IDs to images
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=len(passages_images) > 0  # Only add vision IDs if images present
        )
        # Apply fix_text to clean prompt
        prompt = fix_text(prompt)
        
        # Note: RankingDataset is for ranking-only training (no text generation)
        # It only returns ranking labels, not text labels
        
        if self.type == "train":
            # Parse target generation to create label mapping
            # Format: [A] > [B] > [C] ... (alphabetic)
            import re
            letters = re.findall(r'\[([A-T])\]', target_generation)
            
            label_map = {}
            for rank, letter in enumerate(letters):
                # Convert letter to index: A=0, B=1, C=2, etc.
                letter_idx = ord(letter) - ord('A')
                label_map[letter_idx] = rank
            
            # Create label array [0, 1, 2, ...] for passages in ranking order
            label = [label_map.get(i, i) for i in range(len(label_map))]

        elif self.type == "eval":
            label = (
                [self.raw_data[index]["id"]]
                + self.raw_data[index]["docids"]
                + self.raw_data[index]["scores"]
            )
        else:
            raise Exception(
                "Invalid run type specified for Dataset. Choose from ['train', 'eval']"
            )
        
        # Always return with images (we error above if no images)
        return prompt, label, passages_images

    def __len__(self):
        return len(self.raw_data)


class GenerationDataset(Dataset):
    """Dataset for generation tasks with Qwen3-VL."""

    def __init__(
        self,
        raw_data,
        model_tokenizer,
        combined=False,
        document_renderer=None,
        processor=None,
        use_query_adapter: bool = False,
    ) -> None:
        self.raw_data = raw_data
        self.tokenizer = model_tokenizer
        self.processor = processor
        self.combined = combined
        self.system_message_supported = "system" in self.tokenizer.chat_template
        self.document_renderer = document_renderer
        self.use_query_adapter = use_query_adapter

    def __getitem__(self, index):
        conversation = self.raw_data[index]["conversations"]
        sys_msg = conversation[0]["value"]
        input_context = conversation[1]["value"]
        label = conversation[2]["value"]
        
        # Extract passages and render as images in order
        passages_images = []
        passages_data = []
        if self.document_renderer is not None:
            import re
            
            # First, extract only the passage section (before "Search Query:")
            # This avoids matching the example format "[B] > [A]" in the instructions
            # Use greedy .* to capture all passages until "Search Query:"
            passage_section_match = re.search(r'passages.+(?=Search Query:|Rank the)', input_context, re.DOTALL | re.IGNORECASE)
            if passage_section_match:
                passage_section = passage_section_match.group(0)
            else:
                # Fallback: use entire input_context but filter by length
                passage_section = input_context
            
            # Extract passages with their alphabetic indices: [A] text, [B] text, etc.
            passage_pattern = r'\[([A-T])\]\s+(.*?)(?=\n\[([A-T])\]|\Z)'
            passages_found = re.findall(passage_pattern, passage_section, re.DOTALL)
            
            for letter, passage_text, *_ in passages_found:
                passage_text = passage_text.strip()
                img = self.document_renderer.render(passage_text)
                passages_images.append(img)
                passages_data.append({
                    'letter': letter,
                    'text_preview': passage_text[:100]
                })
        
        # Build prompt with EXACT evaluation format
        # OPTICAL RANKER: Use images only, NO passage text
        if passages_images:
            # Extract query from input_context
            import re
            query_match = re.search(r'search query:\s*(.+?)\.?\s*\n', input_context, re.IGNORECASE)
            if query_match:
                query = query_match.group(1).strip()
            else:
                # Fallback: try other patterns (TODO: Remove this fallback)
                query_match = re.search(r'query:\s*(.+?)\.?\s*\n', input_context, re.IGNORECASE)
                query = query_match.group(1).strip() if query_match else "Unknown query"
            
            # Use EXACT same function as evaluation
            prompt_text = create_ranking_prompt_for_training(query, len(passages_data), use_query_adapter=self.use_query_adapter)
            
            # Create messages - EXACTLY like evaluation (NO separate system role)
            # Everything goes in user message
            # Format: text first, then images
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text}
                    ] + [
                        {"type": "image", "image": img}
                        for img in passages_images
                    ]
                }
            ]

            # # Format: images first, then text
            # messages = [
            #     {
            #         "role": "user",
            #         "content": [
            #             {"type": "image", "image": img}
            #             for img in passages_images
            #         ] + [
            #             {"type": "text", "text": prompt_text}
            #         ]
            #     }
            # ]
            
        else:
            # No images rendered - this is an error for optical ranking
            # All training samples MUST have renderable passages
            raise ValueError(
                f"No images rendered for sample at index {index}. "
                f"Optical ranking requires all samples to have renderable passages. "
                f"Found {len(passages_found)} passages but none rendered successfully. "
                f"Check document_renderer configuration and passage extraction regex."
            )
        
        # Prepare label text
        # Remove first '[' from label if present
        assert label.startswith('['), "Label must start with '['"
        label_text = label[1:]  # Remove '[' so "[A]" becomes "A]"
        label_text = label_text + self.tokenizer.eos_token
        
        # Parse ranking label if combined training
        rank_label = None
        if self.combined:
            # Labels are in format [A] > [B] > [C]
            import re
            letters = re.findall(r'\[([A-Z])\]', conversation[2]["value"])
            # Create mapping: A=0, B=1, C=2, etc. (0-indexed ranks)
            label_map = {}
            for rank, letter in enumerate(letters):
                label_map[ord(letter) - ord('A')] = rank  # A=0, B=1, ...
            # rank_label: [rank_of_passage_0, rank_of_passage_1, ...]
            rank_label = [label_map.get(i, i) for i in range(len(label_map))]
        
        # Return messages structure (not tokenized yet!)
        # Collate function will use processor.apply_chat_template() with images
        # This ensures vision tokens are properly expanded before adding labels
        if self.combined:
            return messages, label_text, rank_label, passages_images
        else:
            return messages, label_text, passages_images

    def __len__(self):
        return len(self.raw_data)


def ranking_collate_fn(data, processor):
    """
    Collate function for ranking datasets with Qwen3-VL.
    
    NOTE: This uses old approach (tokenizer only, no vision expansion).
    For vision-based ranking, use combined_collate_fn instead.
    """
    tokenizer = processor.tokenizer
    if len(data[0]) == 3:
        # Data contains (prompt, label, passages_images)
        prompts, labels, passages_images_list = list(zip(*data))
        tokenized_inputs = tokenizer(
            prompts, padding="longest", truncation=False, return_tensors="pt"
        )
        # Return images separately (processor handles them in training loop)
        return tokenized_inputs, labels, passages_images_list
    else:
        prompts, labels = list(zip(*data))
        tokenized_inputs = tokenizer(
            prompts, padding="longest", truncation=False, return_tensors="pt"
        )
        return tokenized_inputs, labels, None


def generation_collate_fn(data, processor):
    """
    Collate function for generation datasets with images.
    Uses processor.apply_chat_template() to properly handle vision token expansion.
    
    Args:
        data: List of (messages, label_text, images) tuples or None
        processor: Qwen3VL processor (not tokenizer!)
    """
    # Filter out None samples (e.g., samples that failed to render images when adapter is enabled)
    valid_data = [d for d in data if d is not None]
    if not valid_data:
        raise ValueError("All samples in batch are None (no valid images). Check data pipeline.")
    
    tokenizer = processor.tokenizer
    
    # Unpack data - messages structure with images
    if len(valid_data[0]) == 3:
        messages_list, label_texts, images_list = list(zip(*valid_data))
    else:
        raise ValueError(f"Expected 3 elements per sample, got {len(valid_data[0])}")
    
    # Process each sample
    all_input_ids = []
    all_labels = []
    all_pixel_values = []
    all_image_grid_thw = []
    
    for messages, label_text, images in zip(messages_list, label_texts, images_list):
        # Step 1: Process prompt with images using processor.apply_chat_template()
        # This properly expands vision tokens (e.g., 17 tokens → 177 tokens)
        prompt_inputs = processor.apply_chat_template(
            [messages],  # Wrap in list for single sample
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors='pt'
        )
        
        # Extract tokenized prompt (with expanded vision tokens)
        prompt_ids = prompt_inputs['input_ids'][0].tolist()
        
        # Step 2: Add '[' token at end of prompt
        bracket_token = tokenizer.encode('[', add_special_tokens=False)[0]
        prompt_ids.append(bracket_token)
        
        # Step 3: Tokenize target separately (no images)
        target_ids = tokenizer.encode(label_text, add_special_tokens=False)
        
        # Step 4: Combine prompt + target
        combined_ids = prompt_ids + target_ids
        
        # Step 5: Create labels (mask prompt tokens with IGNORE_INDEX)
        labels_for_sample = [IGNORE_INDEX] * len(prompt_ids) + target_ids
        
        all_input_ids.append(combined_ids)
        all_labels.append(labels_for_sample)
        all_pixel_values.append(prompt_inputs['pixel_values'])
        all_image_grid_thw.append(prompt_inputs['image_grid_thw'])
    
    # Convert to tensors and pad
    input_ids_tensors = [torch.tensor(ids, dtype=torch.long) for ids in all_input_ids]
    labels_tensors = [torch.tensor(labels, dtype=torch.long) for labels in all_labels]
    
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_tensors, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels_tensors, batch_first=True, padding_value=IGNORE_INDEX
    )
    
    # # Create attention mask
    # attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    # Concatenate pixel_values and image_grid_thw
    pixel_values = torch.cat(all_pixel_values, dim=0)
    image_grid_thw = torch.cat(all_image_grid_thw, dim=0)
    
    # Return dict with all required inputs
    tokenized_inputs_dict = {
        "input_ids": input_ids,
        # "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }
    
    return tokenized_inputs_dict, labels


# def combined_collate_fn(data, tokenizer):
#     """Collate function for combined ranking and generation datasets."""
#     prompts, labels, rank_labels = list(zip(*data))
#     tokenized_inputs, labels, source_lens = preprocess(prompts, labels, tokenizer)
    
#     # Convert lists to tensors before padding
#     # preprocess returns list of lists, but pad_sequence expects list of tensors
#     tokenized_inputs = [torch.tensor(ids, dtype=torch.long) for ids in tokenized_inputs]
#     labels = [torch.tensor(label, dtype=torch.long) for label in labels]
    
#     # Convert list of tensors to dict format with proper padding
#     input_ids = torch.nn.utils.rnn.pad_sequence(
#         tokenized_inputs, batch_first=True, padding_value=tokenizer.pad_token_id
#     )
#     print(input_ids.shape)
#     labels = torch.nn.utils.rnn.pad_sequence(
#         labels, batch_first=True, padding_value=IGNORE_INDEX
#     )
#     print(labels.shape)
    
#     # Create attention mask (1 for real tokens, 0 for padding)
#     attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
#     # Return as dict format expected by Qwen3-VL
#     tokenized_inputs_dict = {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#     }
    
#     return tokenized_inputs_dict, labels, rank_labels, source_lens

def mmdocir_collate_fn(data, processor):
    """
    Collate function for MMDocIRDataset.
    Uses combined_collate_fn logic directly (no filtering - errors are raised in __getitem__).
    """
    return combined_collate_fn(data, processor)


def combined_collate_fn(data, processor):
    """
    Collate function for combined ranking and generation datasets with images.
    Uses processor.apply_chat_template() to properly handle vision token expansion.
    
    Args:
        data: List of (messages, label_text, rank_label, images, gt_indices) tuples
              or (messages, label_text, rank_label, images) for backwards compatibility
        processor: Qwen3VL processor (not tokenizer!)
    """
    tokenizer = processor.tokenizer
    
    # Unpack data - messages structure with images and rank labels
    # Support both 4 elements (legacy) and 5 elements (with gt_indices)
    if len(data[0]) == 5:
        messages_list, label_texts, rank_labels, images_list, gt_indices_list = list(zip(*data))
    elif len(data[0]) == 4:
        messages_list, label_texts, rank_labels, images_list = list(zip(*data))
        # Legacy mode: no GT indices, create dummy values
        gt_indices_list = [[0] for _ in range(len(data))]
    else:
        raise ValueError(f"Expected 4 or 5 elements per sample, got {len(data[0])}")
    
    # Process each sample
    all_input_ids = []
    all_labels = []
    all_source_lens = []
    all_pixel_values = []
    all_image_grid_thw = []
    all_gt_token_masks = []  # For GT-only LM loss
    
    for messages, label_text, images, gt_indices in zip(messages_list, label_texts, images_list, gt_indices_list):
        # Step 1: Process prompt with images using processor.apply_chat_template()
        prompt_inputs = processor.apply_chat_template(
            [messages],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors='pt'
        )
        
        # Extract tokenized prompt (with expanded vision tokens)
        prompt_ids = prompt_inputs['input_ids'][0].tolist()
        
        # Step 2: Add '[' token at end of prompt
        bracket_token = tokenizer.encode('[', add_special_tokens=False)[0]
        prompt_ids.append(bracket_token)
        
        # Store source length (for ranking loss computation)
        source_len = len(prompt_ids)
        
        # Step 3: Tokenize target separately
        target_ids = tokenizer.encode(label_text, add_special_tokens=False)
        
        # Step 4: Combine
        combined_ids = prompt_ids + target_ids
        labels_for_sample = [IGNORE_INDEX] * len(prompt_ids) + target_ids
        
        # Step 5: Create GT token mask for GT-only LM loss
        # The label format is "A] > [B] > [C] > ..." (without leading '[')
        # We want to only supervise the FIRST position token (the GT letter)
        # token 0 in target is the GT letter (e.g., 'A'), token 1 is ']', etc.
        gt_token_mask = [False] * len(prompt_ids)  # No loss on prompt
        if len(target_ids) > 0:
            # Only compute loss on the first token (the GT letter)
            gt_token_mask.append(True)  # First token (GT letter)
            gt_token_mask.extend([False] * (len(target_ids) - 1))  # Rest of target
        
        all_input_ids.append(combined_ids)
        all_labels.append(labels_for_sample)
        all_source_lens.append(source_len)
        all_pixel_values.append(prompt_inputs['pixel_values'])
        all_image_grid_thw.append(prompt_inputs['image_grid_thw'])
        all_gt_token_masks.append(gt_token_mask)
    
    # Convert to tensors and pad
    input_ids_tensors = [torch.tensor(ids, dtype=torch.long) for ids in all_input_ids]
    labels_tensors = [torch.tensor(labels, dtype=torch.long) for labels in all_labels]
    gt_token_mask_tensors = [torch.tensor(mask, dtype=torch.bool) for mask in all_gt_token_masks]
    
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_tensors, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels_tensors, batch_first=True, padding_value=IGNORE_INDEX
    )
    gt_token_masks = torch.nn.utils.rnn.pad_sequence(
        gt_token_mask_tensors, batch_first=True, padding_value=False
    )
    
    # Create attention mask
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    # Concatenate pixel_values and image_grid_thw
    pixel_values = torch.cat(all_pixel_values, dim=0)
    image_grid_thw = torch.cat(all_image_grid_thw, dim=0)
    
    # Return dict with all required inputs
    tokenized_inputs_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "gt_token_masks": gt_token_masks,  # For GT-only LM loss
    }
    
    return tokenized_inputs_dict, labels, rank_labels, all_source_lens, gt_indices_list

def detect_mmdocir_openai_format(data_sample):
    """
    Detect if data is in MMDocIR OpenAI reranking format.
    
    Expected fields:
    - query_id
    - query
    - doc_name
    - candidate_page_ids
    - openai_ranking (list of indices)
    """
    if not isinstance(data_sample, dict):
        return False
    required = {"query_id", "query", "doc_name", "candidate_page_ids", "openai_ranking"}
    return required.issubset(set(data_sample.keys()))


def detect_pe_rank_format(data_sample):
    """
    Detect if data is in PE-Rank format.
    
    PE-Rank format has:
    - messages_w_content or messages_wo_content
    - ranking (list of indices)
    - extra_texts (list of passage texts)
    
    Args:
        data_sample: First item from the loaded data
        
    Returns:
        bool: True if PE-Rank format, False otherwise
    """
    if not isinstance(data_sample, dict):
        return False
    
    # Check for PE-Rank specific fields
    has_messages = "messages_w_content" in data_sample or "messages_wo_content" in data_sample
    has_ranking = "ranking" in data_sample
    has_extra_texts = "extra_texts" in data_sample
    
    return has_messages and has_ranking and has_extra_texts


def convert_pe_rank_to_conversations(pe_rank_data):
    """
    Convert PE-Rank format to conversations format expected by the trainer.
    
    PE-Rank format:
    {
        "messages_w_content": [
            {"role": "user", "content": "...Passage 1:...\nPassage 2:...Query:..."},
            {"role": "assistant", "content": "<PLACEHOLDER>..."}
        ],
        "ranking": [11, 8, 9, 4, 1, 2, ...],  # 1-indexed ranks
        "extra_texts": ["text1", "text2", ...]
    }
    
    Conversations format:
    {
        "conversations": [
            {"value": "system message"},
            {"value": "passages with [A], [B], [C] labels + query"},
            {"value": "[A] > [B] > [C] ranking"}
        ]
    }
    
    Args:
        pe_rank_data: Single PE-Rank format data item
        
    Returns:
        dict: Data in conversations format
    """
    import re
    
    # Get messages (prefer with content)
    messages = pe_rank_data.get("messages_w_content") or pe_rank_data.get("messages_wo_content", [])
    if not messages or len(messages) < 1:
        raise ValueError("PE-Rank data must have at least one message")
    
    # Extract user message content
    user_message = messages[0].get("content", "")
    
    # Extract system message (before passages)
    system_match = re.search(r'^(.*?)(?=I will provide you with \d+ passages|Passage 1:)', 
                            user_message, re.DOTALL)
    if system_match:
        system_msg = system_match.group(1).strip()
    else:
        system_msg = "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."
    
    # Extract query
    query_match = re.search(r'(?:search query|query):\s*(.+?)\.\.', user_message, re.IGNORECASE)
    if query_match:
        query = query_match.group(1).strip()
    else:
        # Fallback: try to find query between "Rank..." and passage listings
        query_match = re.search(r'query:\s*(.+?)(?:\n|$)', user_message, re.IGNORECASE)
        query = query_match.group(1).strip() if query_match else "unknown query"
    
    # Get ranking order (PE-Rank uses 1-indexed positions)
    ranking = pe_rank_data.get("ranking", [])
    extra_texts = pe_rank_data.get("extra_texts", [])
    
    if not ranking or not extra_texts:
        raise ValueError("PE-Rank data must have 'ranking' and 'extra_texts' fields")
    
    # PE-Rank format: extra_texts has all passages, but ranking only covers the first len(ranking) passages
    # ranking[i] = rank of passage i (1-indexed, lower is better)
    # Only keep passages that have rankings
    num_ranked = len(ranking)
    extra_texts = extra_texts[:num_ranked]
    
    # Truncate to top 26 passages if more than 26 (alphabetic label limit A-Z)
    if len(extra_texts) > 26:
        # Get indices of top 26 ranked passages
        # ranking[i] is the rank of passage i (1-indexed, lower is better)
        indexed_ranks = [(i, rank) for i, rank in enumerate(ranking)]
        indexed_ranks.sort(key=lambda x: x[1])  # Sort by rank
        top_26_indices = sorted([idx for idx, _ in indexed_ranks[:26]])  # Keep top 26, restore original order
        
        # Filter passages and rankings to keep only top 26
        extra_texts = [extra_texts[i] for i in top_26_indices]
        # Recalculate rankings for the subset: create new rank mapping
        old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(top_26_indices)}
        # For each kept passage, what's its new rank among the 26?
        kept_ranks = [(old_to_new_idx[idx], rank) for idx, rank in indexed_ranks if idx in top_26_indices]
        kept_ranks.sort(key=lambda x: x[1])  # Sort by original rank
        new_ranking = [0] * 26
        for new_rank, (new_idx, _) in enumerate(kept_ranks):
            new_ranking[new_idx] = new_rank + 1  # 1-indexed
        ranking = new_ranking
    
    # Build passages in [A], [B], [C] format
    # ranking[i] tells us the rank (1-indexed) of passage i
    # We want to output passages in their original order with [A], [B], [C] labels
    passages = []
    num_passages = len(extra_texts)
    
    for i, text in enumerate(extra_texts):
        letter = chr(ord('A') + i)
        passages.append(f"[{letter}] {text}")
    
    # Build input context with passages and query
    input_parts = []
    input_parts.append("I will provide you with {} passages, each indicated by a alphabetical identifier [].".format(num_passages))
    input_parts.append("Rank the passages based on their relevance to the search query: {}.".format(query))
    input_parts.append("")
    input_parts.extend(passages)
    input_parts.append("")
    input_parts.append("Search Query: {}.".format(query))
    input_parts.append("Rank the {} passages above based on their relevance to the search query in descending order. Only response the ranking results, do not say any word or explain.".format(num_passages))
    
    input_context = "\n".join(input_parts)
    
    # Build target ranking in [A] > [B] > [C] format
    # ranking contains the order: ranking[i] is the rank of passage i (1-indexed)
    # We need to reverse this: find which passage has rank 1, rank 2, etc.
    sorted_indices = sorted(range(len(ranking)), key=lambda i: ranking[i])
    target_letters = [chr(ord('A') + i) for i in sorted_indices]
    target = " > ".join(f"[{letter}]" for letter in target_letters)
    
    # Create conversations format
    conversations_data = {
        "conversations": [
            {"value": system_msg},
            {"value": input_context},
            {"value": target}
        ]
    }
    
    return conversations_data


def load_data(file_path):
    """Load data from a file (JSON or JSONL format)."""
    import json

    # Check if it's JSONL (line-delimited JSON)
    if file_path.endswith('.jsonl'):
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    else:
        # Regular JSON file
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data


def initialize_dataset_and_loader(args, tokenizer, processor=None):
    """
    Initialize dataset and dataloader.
    
    Args:
        args: Training arguments
        tokenizer: Qwen3-VL tokenizer
        processor: Qwen3-VL processor (optional, for image processing)
        
    Returns:
        (dataset, dataloader)
    """
    # Initialize document renderer for optical encoding
    document_renderer = None
    if hasattr(args, 'image_size') and args.image_size:
        from utils.rendering_utils import DocumentRenderer
        logger.info(f"Initializing document renderer with image_size={args.image_size}")
        document_renderer = DocumentRenderer(
            image_size=args.image_size,
            font_size=getattr(args, 'font_size', 12),
            fill_canvas=getattr(args, 'fill_canvas', False),
            fill_threshold=getattr(args, 'fill_threshold', 0.7),
        )
    
    # Load and merge datasets
    raw_data = []
    passages_corpus_cache = {}
    
    # Handle multiple datasets (comma-separated)
    dataset_paths = args.train_dataset_path.split(',')
    
    for dataset_path in dataset_paths:
        dataset_path = dataset_path.strip()
        
        # Check if it's a HuggingFace dataset or local file
        if os.path.exists(dataset_path):
            # Local file
            logger.info(f"Loading local dataset from {dataset_path}")
            local_data = load_data(dataset_path)
            
            # Auto-detect data format and convert if needed
            if local_data and detect_mmdocir_openai_format(local_data[0]):
                # MMDocIR format - needs special handling with parquet loading
                # Store for later, will create MMDocIRDataset instead of converting
                logger.info(f"Detected MMDocIR OpenAI format in {dataset_path}")
                logger.info(f"  Loaded {len(local_data)} MMDocIR training samples")
                # Mark this data as MMDocIR format for later special handling
                for item in local_data:
                    item['_mmdocir_format'] = True
                raw_data.extend(local_data)
            elif local_data and detect_msmarco_compact_format(local_data[0]):
                logger.info(f"Detected MS MARCO compact format in {dataset_path}, converting to conversations format...")
                
                # Filter to only keep view_id == 0
                original_count = len(local_data)
                local_data = [item for item in local_data if item.get('view_id', 0) in [0]]
                logger.info(f"Filtered MS MARCO data to view_id in [0,1]: {len(local_data)}/{original_count} entries")
                
                # TEMP HOTFIX: Truncate to 250k samples max
                if len(local_data) > 250000:
                    logger.info(f"TEMP HOTFIX: Truncating MS MARCO data from {len(local_data)} to 250000 samples")
                    local_data = local_data[:250000]
                
                passages_path = args.passages_corpus_path or os.path.join(os.path.dirname(dataset_path), "passages_corpus.jsonl")
                
                if not os.path.exists(passages_path):
                    raise ValueError(
                        f"MS MARCO compact data detected but passages corpus not found. "
                        f"Expected at {passages_path}. Provide --passages_corpus_path to override."
                    )
                
                if passages_path not in passages_corpus_cache:
                    logger.info(f"Loading passages corpus from {passages_path}...")
                    passages_corpus = load_data(passages_path)
                    passages_corpus_cache[passages_path] = {p['pid']: p['text'] for p in passages_corpus}
                    logger.info(f"Loaded {len(passages_corpus_cache[passages_path])} passages into cache")
                
                converted_data = convert_msmarco_compact_to_conversations(
                    local_data,
                    passages_corpus_cache[passages_path]
                )
                logger.info(f"Converted {len(converted_data)} MS MARCO compact examples to conversations format")
                raw_data.extend(converted_data)
            elif local_data and detect_pe_rank_format(local_data[0]):
                logger.info(f"Detected PE-Rank format in {dataset_path}, converting to conversations format...")
                converted_data = []
                for i, item in enumerate(local_data):
                    try:
                        converted_item = convert_pe_rank_to_conversations(item)
                        converted_data.append(converted_item)
                    except Exception as e:
                        logger.warning(f"Failed to convert PE-Rank item {i}: {e}")
                        continue
                logger.info(f"Converted {len(converted_data)} PE-Rank examples to conversations format")
                raw_data.extend(converted_data)
            else:
                logger.info(f"Using standard conversations format for {dataset_path}")
                raw_data.extend(local_data)
        else:
            # Try loading from HuggingFace
            logger.info(f"Loading HuggingFace dataset: {dataset_path}")
            try:
                hf_dataset = load_dataset(dataset_path, split="train")
                # Convert to list of dicts
                hf_data = [item for item in hf_dataset]
                
                # Auto-detect and convert if PE-Rank format
                if hf_data and detect_pe_rank_format(hf_data[0]):
                    logger.info(f"Detected PE-Rank format in HuggingFace dataset, converting...")
                    converted_data = []
                    for i, item in enumerate(hf_data):
                        try:
                            converted_item = convert_pe_rank_to_conversations(item)
                            converted_data.append(converted_item)
                        except Exception as e:
                            logger.warning(f"Failed to convert PE-Rank item {i}: {e}")
                            continue
                    logger.info(f"Converted {len(converted_data)} PE-Rank examples")
                    raw_data.extend(converted_data)
                else:
                    raw_data.extend(hf_data)
            except Exception as e:
                logger.warning(f"Failed to load dataset {dataset_path}: {e}")
                continue
    
    if not raw_data:
        raise ValueError(f"No data loaded from {args.train_dataset_path}")
    
    logger.info(f"Loaded {len(raw_data)} training examples total")
    
    # Check if data is MMDocIR format (needs special handling)
    is_mmdocir_format = any(item.get('_mmdocir_format', False) for item in raw_data)
    
    # Get add_query_token flag from args (default False for backward compatibility)
    # add_query_token controls whether <query_emb> token is added to prompts
    # use_query_adapter, use_trim, or use_hybrid implies add_query_token
    use_query_adapter = getattr(args, 'use_query_adapter', False)
    use_trim = getattr(args, 'use_trim', False)
    use_hybrid = getattr(args, 'use_hybrid', False)
    add_query_token = getattr(args, 'add_query_token', False) or use_query_adapter or use_trim or use_hybrid
    
    if is_mmdocir_format:
        # Filter to only MMDocIR data
        mmdocir_data = [item for item in raw_data if item.get('_mmdocir_format', False)]
        logger.info(f"Using MMDocIRDataset with {len(mmdocir_data)} samples")
        
        # Get parquet directory from args or use default
        parquet_dir = getattr(args, 'mmdocir_parquet_dir', None)
        if parquet_dir is None:
            parquet_dir = "MMDocIR/MMDocIR_Train_Dataset/parquet"
        
        train_dataset = MMDocIRDataset(
            training_data=mmdocir_data,
            parquet_dir=parquet_dir,
            tokenizer=tokenizer,
            processor=processor,
            domains=getattr(args, 'mmdocir_domains', None),
            max_candidates=getattr(args, 'mmdocir_max_candidates', 20),
            use_query_adapter=add_query_token,  # Controls <query_emb> in prompt
            shuffle_candidates=getattr(args, 'shuffle_candidates', False),
            force_gt_top1=getattr(args, 'force_gt_top1', False),
        )
        collate_fn = partial(mmdocir_collate_fn, processor=processor)
    # Create dataset based on objective (TODO: unify all classes into one)
    elif args.objective == "rank":
        train_dataset = RankingDataset(
            raw_data, tokenizer, "train",
            document_renderer=document_renderer,
            processor=processor,
            use_query_adapter=add_query_token,  # Controls <query_emb> in prompt
        )
        collate_fn = partial(ranking_collate_fn, processor=processor)
    elif args.objective == "generation":
        train_dataset = GenerationDataset(
            raw_data, tokenizer, combined=False,
            document_renderer=document_renderer,
            processor=processor,
            use_query_adapter=add_query_token,  # Controls <query_emb> in prompt
        )
        collate_fn = partial(generation_collate_fn, processor=processor)
    elif args.objective == "combined":
        train_dataset = GenerationDataset(
            raw_data, tokenizer, combined=True,
            document_renderer=document_renderer,
            processor=processor,
            use_query_adapter=add_query_token,  # Controls <query_emb> in prompt
        )
        collate_fn = partial(combined_collate_fn, processor=processor)
    else:
        raise ValueError(f"Invalid objective: {args.objective}")
    
    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    
    return train_dataset, train_dataloader
