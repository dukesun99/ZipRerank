#!/bin/bash
#
# ZipRerank Stage 2 Training Script
# Fine-tuning on MMDocIR dataset with Soft Ranking loss
#
# This stage continues training from Stage 1 checkpoint on real document images
# with GPT-generated relevance rankings as soft targets.
#

set -e

# Initialize conda
if [ -f /opt/miniforge3/etc/profile.d/conda.sh ]; then
    source /opt/miniforge3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
fi

# Activate conda environment
conda activate rank_llm_training

# Model and data paths
CHECKPOINT_PATH="models/ziprerank_stage1/final"
# GPT-5-mini distilled rankings are hosted on the HuggingFace Hub.
# (The page images are still loaded locally from the MMDocIR train parquet below.)
TRAIN_DATA_PATH="dukesunmtri/ZipRerank_GPT-5-mini_MMDocIR_Train"
MMDOCIR_PARQUET_DIR="MMDocIR/MMDocIR_Train_Dataset/parquet"
OUTPUT_DIR="models/ziprerank_stage2"

# Evaluation configuration
EVAL_STEPS=500
EVAL_FIRST_STAGE_FILE="data/mmdocir/first_stage_page_top20_colqwen.pkl"
EVAL_PAGES_PARQUET="MMDocIR/dataset/MMDocIR_pages.parquet"
EVAL_MODE="page"
EVAL_WINDOW_SIZE=20
EVAL_SAMPLE_SIZE=100

echo "========================================"
echo "ZipRerank Stage 2 Training"
echo "========================================"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Training Data: ${TRAIN_DATA_PATH}"
echo "Parquet Dir: ${MMDOCIR_PARQUET_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "========================================"
echo ""
echo "Training Configuration:"
echo "  - Dataset: MMDocIR with GPT rankings (~50k samples)"
echo "  - Epochs: 1"
echo "  - Learning Rate: 3e-6"
echo "  - Batch Size: 2 x 8 = 16 effective"
echo "  - Scheduler: Cosine with warmup"
echo "  - Ranking Loss: Soft Ranking (knowledge distillation)"
echo "  - GT Loss Weight: 1.0"
echo "  - Position Decay: 0.5"
echo "  - Objective: Combined (generation + ranking)"
echo "  - Vision Encoder: FROZEN"
echo "  - Max Candidates: 10"
echo "  - Force GT Top-1: ENABLED"
echo "========================================"
echo ""
echo "Evaluation Configuration:"
echo "  - Eval Steps: ${EVAL_STEPS}"
echo "  - First Stage File: ${EVAL_FIRST_STAGE_FILE}"
echo "  - Mode: ${EVAL_MODE}"
echo "  - Sample Size: ${EVAL_SAMPLE_SIZE}"
echo "========================================"
echo ""

# Run training
CUDA_LAUNCH_BLOCKING=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
    --config_file configs/accel_config_single_gpu.yaml \
    train.py \
    --model_name_or_path "Qwen/Qwen3-VL-8B-Instruct" \
    --resume_from_checkpoint "${CHECKPOINT_PATH}" \
    --train_dataset_path "${TRAIN_DATA_PATH}" \
    --mmdocir_parquet_dir "${MMDOCIR_PARQUET_DIR}" \
    --mmdocir_max_candidates 10 \
    --num_train_epochs 1 \
    --learning_rate 3e-6 \
    --seed 42 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 50 \
    --gradient_checkpointing \
    --output_dir "${OUTPUT_DIR}" \
    --objective combined \
    --ranking_loss soft_ranking \
    --gt_loss_weight 1.0 \
    --gt_position_decay 0.5 \
    --with_tracking \
    --report_to wandb \
    --freeze_vision_tower \
    --force_gt_top1 \
    --eval_steps "${EVAL_STEPS}" \
    --eval_first_stage_file "${EVAL_FIRST_STAGE_FILE}" \
    --eval_pages_parquet "${EVAL_PAGES_PARQUET}" \
    --eval_mode "${EVAL_MODE}" \
    --eval_window_size "${EVAL_WINDOW_SIZE}" \
    --eval_sample_size "${EVAL_SAMPLE_SIZE}"

echo ""
echo "========================================"
echo "Stage 2 Training completed!"
echo "Model saved to: ${OUTPUT_DIR}"
echo ""
echo "Next step: Evaluate with scripts/evaluate.py"
echo "========================================"

