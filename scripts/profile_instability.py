# Instability
import argparse
import torch
from tqdm import tqdm
from ptn.dists import dists
from ptn.dists._abc import AbstractDisributionHeadConfig
import pandas as pd


def main(horizons):

    # Common HPs
    BATCH_SIZE = 8
    N_ITERS = 100
    LR = 1e-3

    # Model hps
    R, Do = 2, 2

    rows = []
    for name, m, kwargs in zip(
        ["LSF", "SGD"], ["mps", "mps"], [{}, {"use_scale_factors": False}]
    ):
        for H in horizons:
            model = dists[m](
                config=AbstractDisributionHeadConfig(
                    rank=R, d_model=1, d_output=Do, horizon=H, **kwargs
                )
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
            for i in tqdm(range(N_ITERS)):
                x = torch.randn(BATCH_SIZE, 1)
                y = torch.randint(0, Do, (BATCH_SIZE, H))
                try:
                    if hasattr(model, "train_example"):
                        model.train_example(x, y, lr=LR)
                    else:
                        output = model(x, y)
                        loss = output.loss
                        loss.backward()
                        optimizer.step()
                except Exception as e:
                    break

            rows.append(
                {
                    "name": name,
                    "horizon": H,
                    "iters": i + 1,
                }
            )

    df = pd.DataFrame(rows)
    print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="+", default=[100], type=int)
    args = parser.parse_args()
    main(**args.__dict__)
