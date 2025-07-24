orion hunt -n hpo-exps --max-trials 5 python train_smol.py \
    --disable_evals \
    --limit_batches 100 \
    --val_check_interval 10 \
    --scheduler~'categorical(none, cosine)'
