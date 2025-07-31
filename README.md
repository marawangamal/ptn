# Sample Efficient Learning via Lookahead Regularization

## Setup
> **Note:** Requires Python 3.10 or higher.
```bash
pip install -r requirements.txt
pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
pip install -e .
```

## Train
Train `HuggingFaceTB/SmolLM-135M` model on 10B token subset of fineweb, and occasionally evaluate on HellaSwag.


First, prepare the data using:
```bash
>> python dataloaders/prepare_fineweb.py --max_length 2048 # CPU intensive: `salloc --cpus-per-task=64 --mem=64G`
```

Train `HuggingFaceTB/SmolLM-135M` model on 10B token subset of fineweb using next-token-prediction
```bash
>> python train.py --model HuggingFaceTB/SmolLM-135M --model_head stp --lr 4e-3 --scheduler cosine
```

Train `HuggingFaceTB/SmolLM-135M` model on 10B token subset of fineweb using multi-token-prediction
```bash
>> python train.py --model HuggingFaceTB/SmolLM-135M --model_head multihead --lr 4e-3 --scheduler cosine
```

Finetune `meta-llama/Llama-3.2-3B-Instruct` on OpenMathInstruct-2 using joint loss 
```bash
>> python train.py --model meta-llama/Llama-3.2-3B-Instruct --model_head multihead --lr 4e-3 --scheduler cosine --loss_type joint --pretrained
```

| Training Setup                | Dataset         | Hardware      | Time    |
|-------------------------------|-----------------|--------------|---------|
| SmolLM-135M (next-token/multihead) | FineWeb 10B     | 4× A100 GPUs | ~15 hrs |
| Llama-3.2-3B finetune (joint loss) | OpenMathInstruct-2 | 4× A100 GPUs | ~12 hrs |