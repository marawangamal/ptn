# Sample Efficient Learning via Lookahead Regularization

## Setup
> **Note:** Requires Python 3.10 or higher.
```bash
pip install -r requirements.txt
pip install -e .
```

## Train
Train `HuggingFaceTB/SmolLM-135M` model on 10B token subset of fineweb, and occasionally evaluate on HellaSwag.
```python
python train.py --model_head multihead --horizon 2
```

Training should take apporximately 15h on 4x A100L GPUs.