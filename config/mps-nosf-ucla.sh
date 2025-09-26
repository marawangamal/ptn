# nltcs
python scripts/train_ucla.py --lr 5e-3 --rank 32 --model mps --dataset nltcs --sf 0 --epochs 20 --tags scale_factors
# msnbc
python scripts/train_ucla.py --lr 5e-3 --rank 8 --model mps --dataset nltcs --sf 0 --epochs 20 --tags scale_factors

# kdd
# python scripts/train_ucla.py --lr 5e-3 --rank 1 --model mps --dataset kdd --sf 0 --epochs 20 --tags scale_factors [FAIL]
python scripts/train_ucla.py --lr 5e-4 --rank 1 --model mps --dataset kdd --sf 0 --epochs 20 --tags scale_factors --max_samples 30000

# plants
python scripts/train_ucla.py --lr 5e-3 --rank 1 --model mps --dataset plants --sf 0 --epochs 20 --tags scale_factors
python scripts/train_ucla.py --lr 5e-4 --rank 1 --model mps --dataset plants --sf 0 --epochs 20 --tags scale_factors


# --lr 5e-4 --rank 1 --model mps --dataset plants --sf 0 --epochs 20 --tags scale_factors --max_samples 1000