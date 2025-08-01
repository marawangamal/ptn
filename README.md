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

### Training from scratch
Train `HuggingFaceTB/SmolLM-135M` model from scratch on 10B token subset of fineweb, and occasionally evaluate on HellaSwag.

First, prepare the data using (CPU intensive use `salloc --cpus-per-task=64 --mem=64G`)
```bash
>> HF_HOME=$SCRATCH/huggingface python dataloaders/prepare_hf_ds.py --tokenizer HuggingFaceTB/SmolLM-135M
```

Train `HuggingFaceTB/SmolLM-135M` model on 10B token subset of fineweb using next-token-prediction
```bash
>> python train.py --model HuggingFaceTB/SmolLM-135M --model_head stp --lr 4e-3 --scheduler cosine
```

> **Note:** The `--model_head` option can be set to `stp` (standard next-token prediction) or `multihead` (multi-task/joint prediction).


### Finetune 

First, prepare the data using (CPU intensive use `salloc --cpus-per-task=64 --mem=64G`)
```bash
>> HF_HOME=$SCRATCH/huggingface python dataloaders/prepare_hf_ds.py \
    --tokenizer meta-llama/Llama-3.2-3B-Instruct \
    --dataset nvidia/OpenMathInstruct-2 \
    --split train \
    --subset "" \
    --column_names problem generated_solution
```

Finetune `meta-llama/Llama-3.2-3B-Instruct` on OpenMathInstruct-2 using joint loss 
```bash
>> HF_HOME=$SCRATCH/huggingface python train.py \
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