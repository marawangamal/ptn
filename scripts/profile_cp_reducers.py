"""
profile_cp_reducers.py

Script for profiling the latency and memory usage of CP reducers,
including the log_prob_moe_batched function.

Example results:

B, T, Do, R, H, V = 32, 512, 1024, 32, 4, 30_000

        latency  peak_memory_MB                             name
0  0.001008      610.937988                  batch_cp_reduce
1  0.005106      728.915527  select_margin_cp_tensor_batched
2  0.000769      627.912598          batch_cp_reduce_decoder  <---- Fastest
3  0.041925     1250.639160   batch_cp_reduce_decoder_einlse
4  0.040477      846.895508             log_prob_moe_batched
"""

import torch
import pandas as pd
from tqdm import tqdm


import time
import pandas as pd
from tqdm import tqdm
import torch
from mtp.mheads._tensorops import (
    batch_cp_reduce,
    batch_cp_reduce_decoder,
    batch_cp_reduce_decoder_einlse,
    select_margin_cp_tensor_batched,
)
from mtp.mheads.moe_proj import log_prob_moe_batched


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
    B, T, Do, R, H, V = 32, 512, 1024, 32, 4, 30_000

    cp_params_tilde = torch.randn(B, R, H, Do, device=device)
    decoder = torch.randn(Do, V, device=device)
    cp_params = torch.einsum("brhd,dv->brhv", cp_params_tilde, decoder)
    alpha_tilde = torch.randn(B, R, device=device)
    ops = torch.randint(0, V, (B, H), device=device)
    fw_args = {
        "base": [cp_params, ops],
        "decoder": [cp_params_tilde, ops, decoder],
        "moe": [ops, alpha_tilde, cp_params_tilde, decoder.T],
    }

    results = []
    for fn_name, fw_args_name, fn in tqdm(
        [
            ("batch_cp_reduce", "base", batch_cp_reduce),
            (
                "select_margin_cp_tensor_batched",
                "base",
                select_margin_cp_tensor_batched,
            ),
            ("batch_cp_reduce_decoder", "decoder", batch_cp_reduce_decoder),
            (
                "batch_cp_reduce_decoder_einlse",
                "decoder",
                batch_cp_reduce_decoder_einlse,
            ),
            ("log_prob_moe_batched", "moe", log_prob_moe_batched),
        ]
    ):

        r = test_latency(fn, 10, 100, *fw_args[fw_args_name])
        r["name"] = fn_name
        results.append(r)

    df = pd.DataFrame(results)
    print(df)
