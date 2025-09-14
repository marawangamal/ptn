"""
profile_mheads.py

Script for profiling the latency and memory usage mheads.

Results:
    B, T, Do, Di, R, H, V = 2, 32, 9, 9, 8, 32, 2  (CPU)

    (forward)

        latency  peak_memory_MB         name
    0  0.008942        0.076008           cp
    1  0.005066        0.060666          mps
    2  0.002637        0.066278          moe
    3  0.004792        0.068810   cp_decoder
    4  0.010727        0.079746  mps_decoder
    5  0.004128        0.058426  moe_decoder

    (generate)
            latency  peak_memory_MB         name
    0  1.461174e-01        0.095435           cp
    1  1.155332e-01        0.514679          mps
    2  8.528681e-03        0.004478          moe
    3  3.600121e-06        0.000145   cp_decoder
    4  5.722046e-07        0.000092  mps_decoder
    5  8.140678e-03        0.004823  moe_decoder


    latency  peak_memory_MB name
0  2.132869        0.286308   cp (select_margin_cp_batched)
1  0.032112        0.004693  moe

"""

import torch
import pandas as pd
from tqdm import tqdm

import time
import pandas as pd
import torch
from ctn.mheads import MHEADS
from ctn.mheads._abc import AbstractDisributionHeadConfig


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
    B, T, Do, Di, R, H, V = 2, 32, 9, 9, 8, 128, 2  # MNIST like
    # B, T, Do, Di, R, H, V = 2, 32, 1096, 1096, 8, 4, 10_000  # Shakespeare like

    results = []
    pbar = tqdm(["cp", "moe"])
    for model_name in pbar:
        pbar.set_description(f"Testing {model_name}")
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
        r = test_latency(model.generate, 5, 10, x)
        r["name"] = model_name
        results.append(r)

    df = pd.DataFrame(results)
    print(df)
