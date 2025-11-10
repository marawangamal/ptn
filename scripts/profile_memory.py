import argparse
import torch
from ptn.dists import dists
from ptn.dists._abc import AbstractDisributionHeadConfig
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def main(models, d_outputs, out_file="results/memory_sweep.csv"):
    B, H, R, Di = 1, 5, 128, 1
    rows = []
    for m in models:
        for Do in d_outputs:
            x = torch.randn(B, Di, device=device)
            y = torch.randint(0, Do, (B, H), device=device)
            model = dists[m](
                config=AbstractDisributionHeadConfig(
                    rank=R, d_model=Di, d_output=Do, horizon=H
                )
            ).to(device)

            # Reset peak memory stats after model creation but before forward pass
            torch.cuda.reset_peak_memory_stats()

            if hasattr(model, "train_example"):
                model.train_example(x, y, max_bond_dim=2)
            else:
                model(x, y)

            mem_after = torch.cuda.max_memory_allocated()
            rows.append(
                {
                    "model": m,
                    "d": Do,
                    "mem_mb": mem_after / (1024**2),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_file, index=False)
    print(df)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=["mps_sigma_lsf", "hmm_sigma_lsf"]
    )
    parser.add_argument("--d_outputs", nargs="+", default=[256], type=int)
    args = parser.parse_args()
    main(**args.__dict__)
