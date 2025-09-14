import time

import torch
from ctn.mheads import MHEADS
from ctn.mheads._abc import AbstractDisributionHeadConfig


def test_latency(fn, n_warmup, n_iters, **kwargs):
    for _ in range(n_warmup):
        fn(**kwargs)

    start_time = time.time()
    for _ in range(n_iters):
        fn(**kwargs)
    end_time = time.time()
    return (end_time - start_time) / n_iters  # seconds


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    B, T, D, R, H, V = 32, 128, 32, 4, 4, 1024
    model_kwargs = {
        "d_model": 32,
        "d_output": V,
        "horizon": H,
        "rank": R,
    }
    fw_kwargs = {
        "x": torch.randn(B, T, D, device=device),
        "y": torch.randint(0, V, (B, T), device=device),
    }
    for key in ["moe", "cp"]:
        model = MHEADS[key](AbstractDisributionHeadConfig(**model_kwargs))
        model.to(device)
        results[key] = test_latency(model.forward_seq, 10, 100, **fw_kwargs)
    print(results)
