#!/bin/bash
#SBATCH --array=1-20%10
#SBATCH --cpus-per-task=2
#SBATCH --output=orion/logs/hpo-exps.%A.%a.out
#SBATCH --error=orion/logs/hpo-exps.%A.%a.err
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --job-name=hpo-exps
#SBATCH --mem=128GB

echo "Activating virtual environment..."
source /home/mila/m/marawan.gamal/scratch/mtl-dev/.venv/bin/activate
export HF_DATA_DIR=/home/mila/m/marawan.gamal/scratch/mtl/data
export HF_HOME=$SCRATCH/huggingface

echo "Running Orion Hunt..."
orion hunt -n hpo-exps \
  python train_smol.py \
    --disable_evals \
    --limit_batches 1000 \
    --val_check_interval 1000 \
    --lr~"loguniform(1e-8, 1e-3)" \
    --scheduler~"choices(['none', 'cosine'])"



# This script will run HPO using Orion.