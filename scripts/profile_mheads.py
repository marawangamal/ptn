"""
profile_mheads.py

Script for profiling the latency and memory usage mheads.

Example results:
    B, T, Do, Di, R, H, V = 2, 32, 100, 100, 8, 32, 1000  (CPU)

        latency  peak_memory_MB name
    0  0.177569        0.099307   cp
    1  0.173181        0.492130  mps
    2  0.032299        0.003414  moe

"""

import torch
import pandas as pd


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
    B, T, Do, Di, R, H, V = 2, 32, 100, 100, 8, 128, 1000

    results = []
    for model_name in ["cp", "mps", "moe"]:
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
        # r = test_latency(model, 10, 100, x, y)
        r = test_latency(model.generate, 10, 100, x)
        r["name"] = model_name
        results.append(r)

    df = pd.DataFrame(results)
    print(df)
