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

## Contribute a minimal custom model to the Hugging Face Hub

This repo includes a tiny custom Transformers model to learn the full Hub contribution flow.

1) Login to the Hub (once):

```bash
huggingface-cli login
```

2) Create and push a tiny model repo (replace `your-username`):

```bash
python scripts/minimal_hf_model.py --repo-id your-username/tinymlp
```

Options:
- `--private` to create a private repo
- `--no-push` to only save locally under `artifacts/tinymlp`

3) Verify loading from the Hub with remote code:

```python
import torch
from transformers import AutoConfig, AutoModel

repo_id = "your-username/tinymlp"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)

inputs = torch.randint(0, config.vocab_size, (1, 8))
outputs = model(inputs)
print("OK:", outputs.last_hidden_state.shape)
```

This demonstrates:
- Defining a custom model and config
- Saving them with `auto_map` so `AutoModel`/`AutoConfig` work
- Hosting custom `modeling_*.py` on the Hub via `trust_remote_code=True`
