module load cuda/12.6 arrow python/3.12 httpproxy
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt