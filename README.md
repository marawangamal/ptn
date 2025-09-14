# Conditional Tensor Networks

Modelling high dimensional conditional distributions using tensor networks.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Train MNIST model with MPS architecture
python scripts/train_mnist.py --model mps --rank 32 --pos_func eps
```

## Available Models

The framework supports various tensor network architectures including:
- MPS (Matrix Product States)
- CP (Canonical Polyadic)

## Example Usage

Train a model on MNIST dataset with MPS architecture:
```bash
python scripts/train_mnist.py --model mps --rank 32 --pos_func exp
```

This command trains a Matrix Product State model with rank 32 using the exponential function for positivity on the MNIST dataset.

