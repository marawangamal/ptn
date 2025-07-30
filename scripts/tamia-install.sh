module load cuda/12.6 arrow python/3.12 httpproxy
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
pip install -e .

# export SSL_CERT_DIR=/etc/ssl/certs if you get an error about ssl