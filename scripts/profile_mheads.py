"""
profile_mheads.py

Script for profiling the latency and memory usage mheads.

Results:
    B, T, Do, Di, R, H, V = 2, 32, 9, 9, 8, 32, 2  (CPU)

            latency  peak_memory_MB name
    0  1.407217e-01        0.104656           cp
    1  1.167424e-01        0.510219          mps
    2  8.013570e-03        0.003728          moe
    3  3.838539e-06        0.000145   cp_decoder
    4  5.316734e-07        0.000092  mps_decoder
    5  8.119421e-03        0.004225  moe_decoder

"""

import torch
import pandas as pd
from tqdm import tqdm

import time
import pandas as pd
import torch
from mtp.mheads import MHEADS
from mtp.mheads._abc import AbstractDisributionHeadConfig


def test_latency(fn, n_warmup, n_iters, *args, **kwargs):
    results = {}
    # Warmup
    for _ in range(n_warmup):
        fn(*args, **kwargs)

    if device is not None and device.type == "cuda":
        torch.cuda.synchronize()
        # Simple memory tracking: just use peak memory before and after
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        for _ in range(n_iters):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        end_time = time.time()
        peak_mem = torch.cuda.max_memory_allocated()
        results["latency"] = (end_time - start_time) / n_iters  # seconds
        results["peak_memory_MB"] = peak_mem / (1024**2)
    else:
        import tracemalloc

        tracemalloc.start()
        start_time = time.time()
        for _ in range(n_iters):
            fn(*args, **kwargs)
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results["latency"] = (end_time - start_time) / n_iters  # seconds
        results["peak_memory_MB"] = peak / (1024**2)

    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    B, T, Do, Di, R, H, V = 2, 32, 9, 9, 8, 32, 2  # MNIST like
    # B, T, Do, Di, R, H, V = 2, 32, 1096, 1096, 8, 4, 10_000  # Shakespeare like

    results = []
    for model_name in tqdm(
        ["cp", "mps", "moe", "cp_decoder", "mps_decoder", "moe_decoder"]
    ):
        x = torch.randn(B, Di, device=device)
        y = torch.randint(0, V, (B, H), device=device)
        model = MHEADS[model_name](
            AbstractDisributionHeadConfig(
                d_model=Do,
                d_output=V,
                horizon=H,
                rank=R,
            )
        )
        r = test_latency(model.generate, 10, 100, x)
        r["name"] = model_name
        results.append(r)

    df = pd.DataFrame(results)
    print(df)
