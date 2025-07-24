# Sample Efficient Learning Via Lookahead regularization

## Setup
> **Note:** Requires Python 3.10 or higher.
```bash
pip install -r requirements.txt
pip install -e .
```

## Train
Train SmolLM model on 10B token subset of fineweb
```python
python train_smol.py --model_head cp --horizon 4 --rank 8
```


## Eval
Run eval using lm-evaluation-harness
```python
python scripts/eval.py --ckpt path/to/checkpoint
```