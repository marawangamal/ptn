# # Installation
# ------------------------------------------------------------------
# module load cuda/12.6 arrow python/3.12 httpproxy
# python -m venv .venv
# source .venv/bin/activate
# pip install -r requirements.txt
# pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
# pip install -e .

# # Usage
# ------------------------------------------------------------------
module load cuda/12.6 arrow python/3.12 httpproxy
source .venv/bin/activate
export HF_HOME=$SCRATCH/huggingface


# Extra:
# - if you get an error about ssl run `export SSL_CERT_DIR=/etc/ssl/certs`