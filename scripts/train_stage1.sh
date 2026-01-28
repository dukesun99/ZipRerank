#!/bin/bash
#
# ZipRerank Stage 1 Training Script
# Pre-training on rank_zephyr dataset with RankNet loss
#
# This stage trains the base Qwen3-VL model to understand optical document ranking
# using synthetically rendered text passages.
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
BASE_MODEL="Qwen/Qwen3-VL-8B-Instruct"
TRAIN_DATA_PATH="rryisthebest/rank_zephyr_training_data_alpha"
OUTPUT_DIR="models/ziprerank_stage1"

echo "========================================"
echo "ZipRerank Stage 1 Training"
echo "========================================"
echo "Base Model: ${BASE_MODEL}"
echo "Training Data: ${TRAIN_DATA_PATH}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "========================================"
echo ""
echo "Training Configuration:"
echo "  - Dataset: rank_zephyr (HuggingFace, ~300k samples)"
echo "  - Epochs: 3"
echo "  - Learning Rate: 3e-6"
echo "  - Batch Size: 8 x 4 = 32 effective"
echo "  - Scheduler: Cosine with warmup"
echo "  - Ranking Loss: RankNet (pairwise)"
echo "  - Objective: Combined (generation + ranking)"
echo "  - Vision Encoder: FROZEN"
echo "========================================"
echo ""

# Run training
CUDA_LAUNCH_BLOCKING=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
    --config_file configs/accel_config_single_gpu.yaml \
    train.py \
    --model_name_or_path "${BASE_MODEL}" \
    --train_dataset_path "${TRAIN_DATA_PATH}" \
    --num_train_epochs 3 \
    --learning_rate 3e-6 \
    --seed 42 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 100 \
    --gradient_checkpointing \
    --output_dir "${OUTPUT_DIR}" \
    --objective combined \
    --ranking_loss ranknet \
    --weighted \
    --with_tracking \
    --report_to wandb \
    --image_size 280 \
    --fill_canvas \
    --freeze_vision_tower

echo ""
echo "========================================"
echo "Stage 1 Training completed!"
echo "Model saved to: ${OUTPUT_DIR}"
echo ""
echo "Next step: Run Stage 2 training with train_stage2.sh"
echo "========================================"

