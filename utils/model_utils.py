"""Model utilities for ZipRerank training."""

import argparse

import bitsandbytes as bnb
import torch
from accelerate.logging import get_logger
from torch import nn
from transformers import (
    AutoProcessor,
    SchedulerType,
)
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
from transformers.trainer_pt_utils import get_parameter_names

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments for ZipRerank training."""
    parser = argparse.ArgumentParser(
        description="Finetune Qwen3-VL for optical document reranking"
    )

    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        required=True,
        help="Training dataset path(s). Can be: 1) HuggingFace dataset name, 2) JSON/JSONL file path, 3) Comma-separated list of paths",
    )
    parser.add_argument(
        "--passages_corpus_path",
        type=str,
        default=None,
        help="Optional passages corpus path for MS MARCO compact format (passages_corpus.jsonl).",
    )
    parser.add_argument("--cache_dir", type=str, help="Path to cache")
    
    # Training hyperparameters
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    
    # Loss and objective
    parser.add_argument(
        "--ranking_loss",
        type=str,
        default="ranknet",
        help="Ranking loss to use: 'ranknet' (Stage 1) or 'soft_ranking' (Stage 2).",
        choices=["ranknet", "soft_ranking"],
    )
    parser.add_argument(
        "--weighted", action="store_true", help="Use weighting with RankNet"
    )
    parser.add_argument(
        "--gt_position_decay",
        type=float,
        default=0.5,
        help="Position weight decay for soft_ranking loss. Default: 0.5.",
    )
    parser.add_argument(
        "--gt_loss_weight",
        type=float,
        default=1.0,
        help="Weight for ranking loss when combined with generation loss.",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="combined",
        help="Training objective: rank, generation, or combined",
        choices=["rank", "generation", "combined"],
    )
    parser.add_argument(
        "--force_gt_top1",
        action="store_true",
        help="Force ground truth to top-1 position in target ranking.",
    )
    
    # LR scheduler
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    
    # Output and checkpointing
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Directory of a checkpoint to resume from",
    )
    
    # Logging
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help="The integration to report results and logs to.",
    )
    
    # Rendering arguments (for Stage 1)
    parser.add_argument(
        "--image_size",
        type=int,
        default=640,
        help="Image size for document rendering (default: 640).",
    )
    parser.add_argument(
        "--font_size",
        type=int,
        default=12,
        help="Font size for rendering (default: 12).",
    )
    parser.add_argument(
        "--fill_canvas",
        action="store_true",
        help="Dynamically adjust font size to fill canvas (default: False).",
    )
    parser.add_argument(
        "--fill_threshold",
        type=float,
        default=0.7,
        help="Canvas fill ratio (0.0-1.0) to skip binary search (default: 0.7 = 70%%).",
    )
    
    # Model freezing
    parser.add_argument(
        "--freeze_vision_tower",
        action="store_true",
        help="Freeze vision encoder parameters during training (default: False).",
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help="Use low cpu memory mode to load model",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow custom models from hub with custom code",
    )
    
    # MMDocIR dataset arguments (for Stage 2)
    parser.add_argument(
        "--mmdocir_parquet_dir",
        type=str,
        default="MMDocIR/MMDocIR_Train_Dataset/parquet",
        help="Directory containing MMDocIR parquet files.",
    )
    parser.add_argument(
        "--mmdocir_domains",
        type=str,
        nargs='+',
        default=None,
        help="List of MMDocIR domains to include (default: all).",
    )
    parser.add_argument(
        "--mmdocir_max_candidates",
        type=int,
        default=20,
        help="Maximum number of candidate pages per query (default: 20).",
    )
    parser.add_argument(
        "--shuffle_candidates",
        action="store_true",
        help="Shuffle candidate order during training to remove positional bias.",
    )
    
    # Evaluation during training arguments
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=0,
        help="Run evaluation every N steps. Set to 0 to disable (default: 0).",
    )
    parser.add_argument(
        "--eval_first_stage_file",
        type=str,
        default="data/mmdocir/first_stage_page_top20_colqwen.pkl",
        help="Path to first-stage retrieval results for evaluation.",
    )
    parser.add_argument(
        "--eval_pages_parquet",
        type=str,
        default="MMDocIR/dataset/MMDocIR_pages.parquet",
        help="Path to pages parquet file for evaluation.",
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="page",
        choices=["page", "layout"],
        help="Evaluation mode: page or layout (default: page).",
    )
    parser.add_argument(
        "--eval_window_size",
        type=int,
        default=20,
        help="Window size for evaluation reranking (default: 20).",
    )
    parser.add_argument(
        "--eval_sample_size",
        type=int,
        default=100,
        help="Number of samples for evaluation (default: 100).",
    )
    
    args = parser.parse_args()
    return args


def freeze_vision_tower(model):
    """
    Freeze vision encoder parameters (model.visual).
    """
    if not hasattr(model, 'visual'):
        logger.warning("Model does not have 'visual' attribute - cannot freeze vision tower")
        return
    
    # Freeze all vision tower parameters
    for param in model.visual.parameters():
        param.requires_grad = False
    
    logger.info("Vision tower (model.visual) frozen - parameters will not be updated during training")
    
    # Log how many parameters were frozen
    frozen_params = sum(p.numel() for p in model.visual.parameters())
    logger.info(f"Frozen {frozen_params:,} vision tower parameters")


def initialize_model_and_tokenizer(args):
    """
    Initialize Qwen3-VL model, processor, and tokenizer.
    
    Returns:
        (tokenizer, model, processor)
    """
    logger.info("Starting Qwen3-VL model and tokenizer initialization...")
    
    # Load processor (handles both text and images)
    if args.resume_from_checkpoint:
        processor = AutoProcessor.from_pretrained(
            args.resume_from_checkpoint,
            cache_dir=args.cache_dir,
            trust_remote_code=True,
        )
    else:
        processor = AutoProcessor.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            trust_remote_code=True,
        )
    
    # Get tokenizer from processor
    tokenizer = processor.tokenizer
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.resume_from_checkpoint,
            cache_dir=args.cache_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=True,
        )
    else:
        logger.info(f"Loading Qwen3-VL model from {args.model_name_or_path}...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=True,
        )
    
    logger.info("Qwen3-VL model loaded successfully!")
    
    # Freeze vision tower if requested
    if args.freeze_vision_tower:
        logger.info("Freezing vision tower as requested...")
        freeze_vision_tower(model)
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return tokenizer, model, processor


def initialize_optimizer(model, weight_decay, learning_rate):
    """
    Initialize the optimizer with the appropriate parameter groups.
    Only includes trainable parameters (requires_grad=True).
    """
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    
    # Filter to only include trainable parameters
    decay_params = [
        p for n, p in model.named_parameters() 
        if n in decay_parameters and p.requires_grad
    ]
    no_decay_params = [
        p for n, p in model.named_parameters() 
        if n not in decay_parameters and p.requires_grad
    ]
    
    optimizer_grouped_parameters = []
    
    # Only add parameter groups if they have parameters
    if decay_params:
        optimizer_grouped_parameters.append({
            "params": decay_params,
            "weight_decay": weight_decay,
        })
    
    if no_decay_params:
        optimizer_grouped_parameters.append({
            "params": no_decay_params,
            "weight_decay": 0.0,
        })
    
    # If no trainable parameters, raise an error
    if not optimizer_grouped_parameters:
        raise ValueError(
            "No trainable parameters found in model! "
            "Make sure at least some parameters have requires_grad=True."
        )
    
    logger.info(f"Optimizer groups: {len(optimizer_grouped_parameters)}")
    logger.info(f"  - Decay params: {len(decay_params)}")
    logger.info(f"  - No decay params: {len(no_decay_params)}")

    optimizer = bnb.optim.AdamW8bit(
        optimizer_grouped_parameters,
        lr=learning_rate,
    )

    return optimizer

