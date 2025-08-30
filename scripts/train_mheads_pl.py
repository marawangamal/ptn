from collections import defaultdict
import itertools
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mtp.mheads._abc import AbstractDisributionHeadConfig
from mtp.mheads import MHEADS
from tqdm import tqdm

# set theme
sns.set_theme()


def make_dataset(n_samples, d_model, d_output, horizon):
    x = torch.randn(n_samples, d_model)
    y = torch.randint(0, 2, (n_samples, horizon)) * (d_output - 1)
    return x, y


if __name__ == "__main__":
    lr = 1e-3
    d_model = 4096
    d_output = 10_00
    experiments = (
        [
            {
                "mt_name": "moe",
                "batch_size": 32,
                "horizon": h,
                "rank": r,
                "d_model": d_model,
                "d_output": d_output,
                "seed": seed,
            }
            # for r, h, seed in itertools.product([8], [8], [0, 42, 84])
            for r, h, seed in itertools.product([8], [8], [0, 42, 84])
        ]
        + [
            {
                "mt_name": "moe_proj",
                "batch_size": 32,
                "horizon": h,
                "rank": r,
                "d_model": d_model,
                "d_output": d_output,
                "seed": seed,
            }
            # for r, h, seed in itertools.product([8], [8], [0, 42, 84])
            for r, h, seed in itertools.product([8], [8], [0, 42, 84])
        ]
        + [
            {
                "mt_name": "cp",
                "batch_size": 32,
                "horizon": h,
                "rank": r,
                "d_model": 512,
                "d_output": 100,
                "seed": seed,
            }
            for r, h, seed in itertools.product([8], [8], [0, 42, 84])
        ]
    )

    log_dicts = []
    pbar = tqdm(experiments)
    for config in experiments:
        pbar.set_description(
            f"{config['mt_name']} | R: {config['rank']} | H: {config['horizon']}"
        )
        log_dicts.append(log_dict)
