# Sample Efficient Learning via Multi-Token Regularization

A PyTorch Lightning based script for pretraining and finetuning language models with multi-token prediction heads for improved sample efficiency.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
pip install -e .

# Train SmolLM-135M from scratch
python train.py \
    --model HuggingFaceTB/SmolLM-135M \
    --model_head stp \
    --lr 4e-3 \
    --scheduler cosine \
    --dataset fineweb

# Finetune Llama-3.2-3B-Instruct
python train.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset omi:1m \
    --model_head multihead \
    --lr 5e-5 \
    --scheduler cosine \
    --loss_type joint \
    --pretrained
```

> **Note:** For faster data processing, we recommend preparing the dataset in advance using additional CPU resources, as described in the [Data Preparation](#data-preparation) section. This step is optional; if not performed, the dataset will be prepared automatically during training (but may be slower).

## Data Preparation

For SLURM clusters, set environment variables:
```bash
# Set env vars if needed for wandb and huggingface
# export WANDB_CACHE_DIR=$SCRATCH/wandb
# export HF_HOME=$SCRATCH/huggingface
# Prepare dataset (requires SLURM allocation)
salloc --cpus-per-task=64 --mem=64G
python dataloaders/prepare_hf_ds.py --dataset fineweb --tokenizer HuggingFaceTB/SmolLM-135M
```

<!-- ## Hardware Requirements

| Model | Dataset | Hardware | Training Time |
|-------|---------|----------|---------------|
| SmolLM-135M | FineWeb 10B | 4× A100 | ~15 hrs |
| Llama-3.2-3B | OpenMathInstruct-2 | 4× A100 | ~12 hrs | -->
