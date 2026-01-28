#!/usr/bin/env python3
"""
ZipRerank Training Script.

Two-stage training for optical document reranking:
- Stage 1: Pre-train on rank_zephyr (rendered text) with RankNet loss
- Stage 2: Fine-tune on MMDocIR (real images) with Soft Ranking loss
"""

import logging
import os
import math

import torch
import datasets
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import get_scheduler

from utils.data_utils import initialize_dataset_and_loader
from utils.model_utils import (
    initialize_model_and_tokenizer,
    initialize_optimizer,
    parse_args,
)
from utils.train_utils import (
    save_checkpoint,
    train_epoch,
)

logger = get_logger(__name__)


def main():
    args = parse_args()

    # Initialize the accelerator
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(
        cpu=False,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )

    # Set up logging
    log_format = "%(asctime)s - %(levelname)s - %(name)s - [%(process)d] - %(message)s"
    date_format = "%m/%d/%Y %H:%M:%S"
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    root_logger.addHandler(console_handler)
    
    # File handler (main process only)
    if accelerator.is_local_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        log_file = os.path.join(args.output_dir, "training.log")
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        root_logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    # Initialize model, tokenizer, and processor
    tokenizer, model, processor = initialize_model_and_tokenizer(args)
    
    # Initialize optimizer
    optimizer = initialize_optimizer(model, args.weight_decay, args.learning_rate)
    
    # Set up eval_args for periodic evaluation
    if getattr(args, 'eval_steps', 0) > 0:
        args.eval_args = {
            "first_stage_file": getattr(args, 'eval_first_stage_file', "data/mmdocir/first_stage_page_top20_colqwen.pkl"),
            "pages_parquet": getattr(args, 'eval_pages_parquet', "MMDocIR/dataset/MMDocIR_pages.parquet"),
            "mode": getattr(args, 'eval_mode', "page"),
            "window_size": getattr(args, 'eval_window_size', 20),
            "sample_size": getattr(args, 'eval_sample_size', 100),
            "use_logits": True,
        }
        logger.info(f"Periodic evaluation enabled every {args.eval_steps} steps")

    # Load training data
    train_dataset, train_dataloader = initialize_dataset_and_loader(args, tokenizer, processor)

    # Compute training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if args.max_train_steps
        else args.num_train_epochs * num_update_steps_per_epoch,
    )

    # Prepare with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Log training info
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    
    logger.info("***** Running ZipRerank training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Learning rate = {args.learning_rate}")
    logger.info(f"  Warmup steps = {args.num_warmup_steps}")
    logger.info(f"  Ranking loss = {args.ranking_loss}")
    logger.info(f"  Objective = {args.objective}")

    # Initialize trackers
    if args.with_tracking:
        experiment_config = vars(args)
        experiment_config["num_gpus"] = accelerator.num_processes
        experiment_config["total_batch_size"] = total_batch_size
        experiment_config["total_optimization_steps"] = args.max_train_steps
        experiment_config["num_training_samples"] = len(train_dataset)
        
        run_name = f"ziprerank_{args.objective}_{args.ranking_loss}_{args.num_train_epochs}ep"
        tags = [args.objective, args.ranking_loss, f"bs{args.per_device_train_batch_size}"]
        notes = f"ZipRerank training with {args.objective} objective and {args.ranking_loss} loss"
        
        accelerator.init_trackers(
            project_name="ziprerank",
            config=experiment_config,
            init_kwargs={
                "wandb": {
                    "name": run_name,
                    "tags": tags,
                    "notes": notes
                }
            }
        )
    
    # Training loop
    logger.info("Starting training...")
    
    completed_steps = 0
    
    for epoch in range(args.num_train_epochs):
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting Epoch {epoch + 1}/{args.num_train_epochs}")
        logger.info(f"{'='*80}\n")
        
        avg_loss, steps_done = train_epoch(
            epoch=epoch + 1,
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            args=args,
            tokenizer=tokenizer,
            processor=processor,
            completed_steps=completed_steps,
        )
        
        completed_steps += steps_done
        
        logger.info(f"Epoch {epoch + 1} completed with average loss: {avg_loss:.4f}")
        logger.info(f"Global steps completed: {completed_steps}/{args.max_train_steps}")
        
        # Log epoch-level metrics
        if args.with_tracking:
            epoch_metrics = {
                "epoch/loss": avg_loss,
                "epoch/number": epoch + 1,
                "epoch/learning_rate": lr_scheduler.get_last_lr()[0],
                "epoch/completed_steps": completed_steps,
            }
            accelerator.log(epoch_metrics, step=completed_steps)
        
        # Check if we've reached max_train_steps
        if completed_steps >= args.max_train_steps:
            logger.info(f"Reached max_train_steps ({args.max_train_steps}). Stopping training.")
            break

    # Save final model
    logger.info("\nTraining completed!")
    logger.info("Saving final model...")
    final_output_dir = os.path.join(args.output_dir, "final")
    save_checkpoint(model, tokenizer, processor, accelerator, final_output_dir)
    logger.info(f"Final model saved to {final_output_dir}")

    # End tracking
    if args.with_tracking:
        accelerator.end_training()

    logger.info("Training finished successfully!")


if __name__ == "__main__":
    main()

