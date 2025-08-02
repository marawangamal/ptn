# Sample Efficient Learning via Lookahead Regularization

## Setup
> **Note:** Requires Python 3.10 or higher.
```bash
pip install -r requirements.txt
pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
pip install -e .
```

## Train

<!-- 
TODO:
[ ] Validate train from scratch setup [IPR]
[ ] Validate finetuning setup
 -->

### Train SmolLM-135M (from scratch)
Train `SmolLM-135M` from scratch on 10B tokens (+ evaluate on HellaSwag).

```bash
# 1. Extract data
# salloc --cpus-per-task=64 --mem=64G
# WANDB_CACHE_DIR=$SCRATCH/wandb HF_HOME=$SCRATCH/huggingface python dataloaders/prepare_hf_ds.py

# 2. Train
WANDB_CACHE_DIR=$SCRATCH/wandb HF_HOME=$SCRATCH/huggingface python train.py \
    --model HuggingFaceTB/SmolLM-135M \
    --model_head stp \  # or multihead
    --lr 4e-3 \
    --scheduler cosine
```



### Finetune Llama-3.2-3B-Instruct

Finetune `Llama-3.2-3B-Instruct` on OpenMathInstruct-2 using joint loss 
```bash
# 1. Extract data
# `salloc --cpus-per-task=64 --mem=64G`
# WANDB_CACHE_DIR=$SCRATCH/wandb HF_HOME=$SCRATCH/huggingface python dataloaders/prepare_hf_ds.py \
#     --tokenizer meta-llama/Llama-3.2-3B-Instruct \
#     --dataset nvidia/OpenMathInstruct-2 \
#     --split train \
#     --subset "" \
#     --column_names problem generated_solution 

# 2. Finetune
# NOTE: remove limits
WANDB_CACHE_DIR=$SCRATCH/wandb HF_HOME=$SCRATCH/huggingface python train.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset omi \
    --model_head multihead \
    --lr 4e-3 \
    --scheduler cosine \
    --loss_type joint \
    --pretrained \
    --max_len 512 \
    --batch_size 1 \
    --epochs 5 \
    --limit_train_batches 5 \ 
    --limit_val_batches 1 \
    --val_check_interval 5
```

<!-- 
DEBUG::
--dataset wikitext --subset wikitext-2-raw-v1  --split "train[:10000]" \ 
-->

| Training Setup                | Dataset         | Hardware      | Time    |
|-------------------------------|-----------------|--------------|---------|
| SmolLM-135M (next-token/multihead) | FineWeb 10B     | 4× A100 GPUs | ~15 hrs |
| Llama-3.2-3B finetune (joint loss) | OpenMathInstruct-2 | 4× A100 GPUs | ~12 hrs |