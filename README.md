# ZipRerank

A framework for training highly efficient list-wise multimodal rerankers for long documents.

This repository is mainly for reproducing the results in the paper. To use the pretrained model in your project, check the [HuggingFace Repository](https://huggingface.co/mtri-admin/ZipRerank).

## Overview

ZipRerank trains Qwen3-VL for document page retrieval through:

1. **Stage 1**: Pre-training on synthetically rendered text (rank_zephyr dataset) with RankNet pairwise loss
2. **Stage 2**: Fine-tuning on real document images (MMDocIR) with Soft Ranking loss using GPT-generated relevance labels

The model learns to rerank document pages by their visual relevance to a query, achieving strong performance on the MMDocIR benchmark.

For efficient inference, ZipRerank supports **single-token logits decoding** for fast ranking via a single forward pass, and **Query-Image Early Interaction (QI-EI)**, a visual token pruning method that selects query-relevant image patches early in the pipeline. 

> **🚧 Work in Progress** — We will release the stage-2 distillation data soon.

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- ~100GB GPU memory for training (H200 140GB recommended)

### Setup

```bash
# Create conda environment
conda create -n ziprerank python=3.10
conda activate ziprerank

# Install PyTorch (adjust for your CUDA version)
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt

# Optional: Install flash attention for faster training
pip install flash-attn --no-build-isolation
```

### External Dependencies

**MMDocIR Dataset & First-Stage Retrieval**: For first-stage retrieval and dataset files, clone the MMDocIR repository:

```bash
git clone https://github.com/MMDocIR/MMDocIR.git ../MMDocIR
# Or set MMDOCIR_PATH environment variable to your MMDocIR location
export MMDOCIR_PATH=/path/to/MMDocIR
```

The `first_stage_retrieval.py` script requires MMDocIR's `vision_wrapper.py` for the DSE model. Alternatively, use pre-computed first-stage results if available.

## Data Preparation

### Stage 1: Rank Zephyr Dataset

The rank_zephyr dataset is automatically downloaded from HuggingFace:
- Dataset: `rryisthebest/rank_zephyr_training_data_alpha`

### Stage 2: MMDocIR Dataset

1. Download MMDocIR dataset from the official repository
2. Generate training data with GPT relevance rankings:

```bash
python scripts/generate_training_data.py \
    --parquet_dir MMDocIR/MMDocIR_Train_Dataset/parquet \
    --output_dir data/mmdocir_train
```

3. Generate first-stage retrieval results for evaluation:

```bash
python scripts/first_stage_retrieval.py \
    --model_path MrLight/dse-qwen2-2b-mrl-v1 \
    --parquet_path MMDocIR/dataset/MMDocIR_pages.parquet \
    --annotations_path MMDocIR/dataset/MMDocIR_annotations.jsonl \
    --output_path data/mmdocir/first_stage_page_top20_dse.pkl \
    --top_k 20
```

## Training

### Stage 1: Pre-training on Rank Zephyr

```bash
bash scripts/train_stage1.sh
```

Training configuration:
- Base model: Qwen/Qwen3-VL-8B-Instruct
- Epochs: 3
- Batch size: 8 × 4 = 32 (effective)
- Learning rate: 3e-6
- Ranking loss: RankNet (weighted pairwise)
- Vision encoder: Frozen

### Stage 2: Fine-tuning on MMDocIR

```bash
bash scripts/train_stage2.sh
```

Training configuration:
- Checkpoint: Stage 1 model
- Epochs: 1
- Batch size: 2 × 8 = 16 (effective)
- Learning rate: 3e-6
- Ranking loss: Soft Ranking (knowledge distillation)
- GT Loss Weight: 1.0, Position Decay: 0.5
- Vision encoder: Frozen

## Evaluation

### Evaluate ZipRerank Model

```bash
python scripts/evaluate.py \
    --model_path models/ziprerank_stage2/final \
    --first_stage_file data/mmdocir/first_stage_page_top20_dse.pkl \
    --pages_parquet MMDocIR/dataset/MMDocIR_pages.parquet \
    --mode page \
    --window_size 20 \
    --sample_size 0 \
    --use_logits
```

### With Query-Image Early Interaction (Visual Token Pruning)

For efficient inference with reduced visual tokens:

```bash
python scripts/evaluate.py \
    --model_path models/ziprerank_stage2/final \
    --first_stage_file data/mmdocir/first_stage_page_top20_dse.pkl \
    --pages_parquet MMDocIR/dataset/MMDocIR_pages.parquet \
    --mode page \
    --use_qi_early \
    --qi_early_keep_ratio 0.5 \
    --sample_size 0 \
    --use_logits
```

### Baseline: MM-R5

```bash
python scripts/evaluate_mmr5.py \
    --model_path i2vec/MM-R5 \
    --first_stage_file data/mmdocir/first_stage_page_top20_dse.pkl \
    --pages_parquet MMDocIR/dataset/MMDocIR_pages.parquet \
    --mode page \
    --sample_size 0
```

### Baseline: GPT-5-mini (OpenAI)

```bash
export OPENAI_API_KEY=your_api_key

python scripts/evaluate_openai.py \
    --model gpt-5-mini-2025-08-07 \
    --first_stage_file data/mmdocir/first_stage_page_top20_dse.pkl \
    --pages_parquet MMDocIR/dataset/MMDocIR_pages.parquet \
    --mode page \
    --sample_size 100 \
    --num_threads 4
```

## Project Structure

```
ZipRerank/
├── train.py                    # Main training script
├── configs/
│   └── accel_config_single_gpu.yaml  # Accelerate config
├── models/
│   ├── __init__.py
│   ├── qwen3vl_with_qi_early.py      # Model with QI-EI pruning
│   └── qi_early_pruner.py            # Visual token pruner
├── utils/
│   ├── __init__.py
│   ├── data_utils.py                 # Dataset and dataloader
│   ├── loss.py                       # RankNet and Soft Ranking losses
│   ├── model_utils.py                # Model initialization
│   ├── rendering_utils.py            # Document rendering
│   └── train_utils.py                # Training loop
├── scripts/
│   ├── train_stage1.sh               # Stage 1 training script
│   ├── train_stage2.sh               # Stage 2 training script
│   ├── evaluate.py                   # ZipRerank evaluation
│   ├── evaluate_mmr5.py              # MM-R5 baseline evaluation
│   ├── evaluate_openai.py            # OpenAI baseline evaluation
│   ├── first_stage_retrieval.py      # First-stage retriever
│   └── generate_training_data.py     # Training data generation
├── requirements.txt
├── .gitignore
└── README.md
```

