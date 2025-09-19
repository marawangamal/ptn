import torch
import pandas as pd
from tqdm import tqdm


import time
import pandas as pd
from tqdm import tqdm
import torch
from ctn.mheads import MHEADS
from ctn.mheads._abc import AbstractDisributionHeadConfig


def test_latency(fn, n_warmup, n_iters, device, *args, **kwargs):
    results = {}
    # Warmup
    for _ in range(n_warmup):
        fn(*args, **kwargs)

    if device is not None and device.type == "cuda":
        torch.cuda.synchronize()
        # Simple memory tracking: just use peak memory before and after
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        try:
            for _ in range(n_iters):
                fn(*args, **kwargs)
        except Exception as e:
            print(e)
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


def profile_bm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    B, H, R, Do, Di = 32, 128, 2, 2, 1  # Config used for Fig. 1

    mps_sigma = MHEADS["mps"](
        config=AbstractDisributionHeadConfig(
            rank=R,
            d_model=Di,
            d_output=Do,
            horizon=H,
        )
    )

    mps_bm = MHEADS["bm"](
        config=AbstractDisributionHeadConfig(
            rank=R,
            d_model=Di,
            d_output=Do,
            horizon=H,
        )
    )

    # B, H, D, V = 32, 5, 9, 2
    # mt_head = BM(
    #     AbstractDisributionHeadConfig(
    #         d_model=D,
    #         d_output=V,
    #         horizon=H,
    #         rank=8,
    #     ),
    # )

    alpha_tilde = torch.randn(B, R, device=device)
    ops = torch.randint(0, Do, (B, H), device=device)
    results = []
    for fn_name, fn, fn_args, fn_kwargs in tqdm(
        [
            (
                "mps_sigma",
                mps_sigma,
                [
                    torch.randn(B, Di, device=device),
                    torch.randint(0, Do, (B, H), device=device),
                ],
                {},
            ),
            (
                "mps_bm",
                mps_bm.train_example,
                [
                    torch.randn(B, Di, device=device),
                    torch.randint(0, Do, (B, H), device=device),
                ],
                {},
            ),
        ]
    ):

        r = test_latency(fn, 10, 100, device, *fn_args, **fn_kwargs)
        r["name"] = fn_name
        results.append(r)

    df = pd.DataFrame(results)
    print(df)


def sweep():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # FIXED PARAMETERS (baseline config)
    B, H0, R0, Do0, Di0 = 32, 8, 2, 2, 1

    # SWEEP PARAMETERS
    horizons = [8, 16, 32]
    d_outputs = [8, 16, 32]
    # DEBUG SWEEP PARAMETERS
    horizons = [1024]
    d_outputs = []

    results = []

    # Helper to run one config
    def run_config(R, H, Di, Do, sweep_type, sweep_value):
        mps_sigma = (
            MHEADS["mps"](
                config=AbstractDisributionHeadConfig(
                    rank=R, d_model=Di, d_output=Do, horizon=H
                )
            )
            .to(device)
            .eval()
        )

        mps_bm = (
            MHEADS["bm"](
                config=AbstractDisributionHeadConfig(
                    rank=R, d_model=Di, d_output=Do, horizon=H
                )
            )
            .to(device)
            .eval()
        )

        x = torch.randn(B, Di, device=device)
        y = torch.randint(0, Do, (B, H), device=device)

        for fn_name, fn, fn_args, fn_kwargs in [
            ("mps_sigma", mps_sigma, [x, y], {}),
            ("mps_bm", mps_bm.train_example, [x, y], {}),
        ]:
            r = test_latency(fn, 10, 100, device, *fn_args, **fn_kwargs)
            r.update(
                dict(
                    name=fn_name,
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
            del mps_sigma, mps_bm, x, y
            torch.cuda.empty_cache()

    # --- Sweep rank ---
    # for R in tqdm(ranks, desc="Sweep rank", leave=False):
    #     run_config(R, H0, Di0, Do0, sweep_type="R", sweep_value=R)

    # --- Sweep horizon ---
    for H in tqdm(horizons, desc="Sweep horizon", leave=False):
        run_config(R0, H, Di0, Do0, sweep_type="H", sweep_value=H)

    # --- Sweep d_output ---
    for Do in tqdm(d_outputs, desc="Sweep d_output", leave=False):
        run_config(R0, H0, Di0, Do, sweep_type="Do", sweep_value=Do)

    df = pd.DataFrame(results)
    print(df)
    df.to_csv("results/sweep.csv", index=False)
    print("Saved to results/sweep.csv")


if __name__ == "__main__":
    sweep()
