<h1 align="center">Efficient Probabilistic Tensor Networks</h1>

<p align="center">
  Train <strong>probabilistic tensor networks (PTNs)</strong> with
  <strong>logarithmic scale factors (LSF)</strong> for
  <strong>stable</strong> and <strong>fast</strong> learning.
</p>


<div align="center" style="display:flex; justify-content:center; gap:32px; align-items:center;">
  <img src="images/mps-ptn.png"    alt="Model Overview" style="width:40%; height:auto; max-width:100%; display:block;">
  <img src="images/mps-profile.png"  alt="MPS Profile"   style="width:20%; height:auto; max-width:100%; display:block;">
</div>


## Installation

**Requirements:** Python 3.10+

```bash
# Install dependencies
# (optional) python -m venv .venv
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
# optional logging
pip install wandb && wandb login
```

## Training

### MNIST Dataset

Train $\mathrm{MPS\_\mathrm{\sigma+LSF}$ and $\mathrm{MPS}\_\mathrm{BM+LSF}$ on MNIST
```bash
python scripts/train_mnist.py --model mps_sigma_lsf --rank 8 --pos_func exp
python scripts/train_mnist.py --model mps_bm_lsf --rank 8
```

### UCLA Datasets

Train $\mathrm{MPS\_\mathrm{\sigma+LSF}$ and $\mathrm{MPS}\_\mathrm{BM+LSF}$ on UCLA density estimation benchmarks
```bash
python scripts/train_ucla.py --dataset nltcs --model mps_sigma_lsf --lr 5e-3  --rank 32 --pos_func abs
python scripts/train_ucla.py --dataset nltcs --model mps_bm_lsf --lr 5e-3  --rank 32
```

## Evaluation

All training and validation metrics are automatically tracked and logged to [Weights & Biases (Wandb)](https://wandb.ai/).


## Reproducing Figures
The figures from the paper can be reproduced using the Jupyter notebook in [`scripts/results.ipynb`](scripts/results.ipynb).