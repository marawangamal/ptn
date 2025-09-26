# Efficient Probabilistic Tensor Networks

Modelling multi-variate discrete distributions using tensor networks.

## Installation

**Requirements:** Python 3.10+

```bash
# Install dependencies
# (optional) python -m venv .venv
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Training
Train $\mathrm{MPS}_{\sigma+\mathrm{LSF}}$
```bash

# Train model on MNIST
python scripts/train_mnist.py --model mps --rank 8 --pos_func eps

# Train on UCLA datasets
python scripts/train_ucla.py  --lr 5e-3 --model mps --rank 32 --pos_func abs
```

Train $\mathrm{MPS}_{\mathrm{BM+LSF}}$ models
```bash

# Train model on MNIST
python scripts/train_mnist.py --model bmnc --rank 8

# Train on UCLA datasets
python scripts/train_ucla.py  --lr 5e-3 --model bmnc --rank 32
```

## Evaluation

All training and validation metrics are automatically tracked and logged to [Weights & Biases (Wandb)](https://wandb.ai/).

```bash
pip install wandb && wandb login
```
