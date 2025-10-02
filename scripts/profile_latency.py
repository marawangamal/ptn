import argparse
import torch
import pandas as pd
from tqdm import tqdm

import time
import pandas as pd
from tqdm import tqdm
import torch
from ptn.dists import dists
from ptn.dists._abc import AbstractDisributionHeadConfig


def test_latency(fn, n_warmup, n_iters, device, *args, **kwargs):
    results = {}
    # Warmup
    for _ in range(n_warmup):
        fn(*args, **kwargs)

    if device is not None and device.type == "cuda":
        torch.cuda.synchronize()
        start_time = time.time()
        try:
            for _ in range(n_iters):
                if hasattr(fn, "train_example"):
                    fn.train_example(*args, **kwargs)
                else:
                    fn(*args, **kwargs)
        except Exception as e:
            start_time = float("inf")
            print(e)
        torch.cuda.synchronize()
        end_time = time.time()
        results["latency"] = (end_time - start_time) / n_iters  # seconds
    else:
        start_time = time.time()
        for _ in range(n_iters):
            if hasattr(fn, "train_example"):
                fn.train_example(*args, **kwargs)
            else:
                fn(*args, **kwargs)
        end_time = time.time()
        results["latency"] = (end_time - start_time) / n_iters  # seconds

    return results


def sweep(
    horizons=[8, 16, 32],
    d_outputs=[8, 16, 32],
    device=None,
    models=["mps_bm_lsf", "mps_bm_dmrg"],
    n_warmup=10,
    n_iters=100,
    out="results/sweep.csv",
    d_output=8,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    # FIXED PARAMETERS (baseline config)
    B, H0, R0, Do0, Di0 = 32, 8, 32, d_output, 1

    results = []

    # Helper to run one config
    def run_config(R, H, Di, Do, sweep_type, sweep_value):

        # Create input data
        x = torch.randn(B, Di, device=device)
        y = torch.randint(0, Do, (B, H), device=device)

        # Measure latency
        for m in models:
            model = dists[m](
                config=AbstractDisributionHeadConfig(
                    rank=R,
                    d_model=Di,
                    d_output=Do,
                    horizon=H,
                )
            ).to(device)
            r = test_latency(model, n_warmup, n_iters, device, x, y)
            r.update(
                dict(
                    name=m,
                    R=R,
                    H=H,
                    Do=Do,
                    Di=Di,
                    B=B,
                    sweep_type=sweep_type,
                    sweep_value=sweep_value,
                )
            )
            results.append(r)

            if torch.cuda.is_available():
                del model
                torch.cuda.empty_cache()

    # --- Sweep horizon ---
    for H in tqdm(horizons, desc="Sweep horizon", leave=False):
        run_config(R0, H, Di0, Do0, sweep_type="H", sweep_value=H)

    # --- Sweep d_output ---
    for Do in tqdm(d_outputs, desc="Sweep d_output", leave=False):
        run_config(R0, H0, Di0, Do, sweep_type="Do", sweep_value=Do)

    df = pd.DataFrame(results)
    print(df)
    df.to_csv(out, index=False)
    print(f"Saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mps_bm_lsf", "mps_bm_dmrg"],
    )
    parser.add_argument("--horizons", nargs="+", default=[], type=int)
    parser.add_argument("--d_outputs", nargs="+", default=[], type=int)
    parser.add_argument("--d_output", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=str, default="results/latency_sweep.csv")
    args = parser.parse_args()
    sweep(**args.__dict__)
