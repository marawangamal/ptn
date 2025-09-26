# Instability
import torch
from tqdm import tqdm
from ctn.mheads import MHEADS
from ctn.mheads._abc import AbstractDisributionHeadConfig
import pandas as pd

# Common HPs
BATCH_SIZE = 8
LR = 1e-3
N_ITERS = 1000
SEED = 0

# Model hps
H, R, Do = 100, 8, 2

models = [
    {"name": r"LSF", "model": "mps", "iters": 0, "kwargs": {}},
    {
        "name": r"SGD",
        "model": "mps",
        "iters": 0,
        "kwargs": {"use_scale_factors": False},
    },
]

for mdict in models:
    model = MHEADS[mdict["model"]](
        config=AbstractDisributionHeadConfig(
            rank=R, d_model=1, d_output=Do, horizon=H, **mdict["kwargs"]
        )
    )
    for i in tqdm(range(N_ITERS)):
        x = torch.randn(BATCH_SIZE, 1)
        y = torch.randint(0, Do, (BATCH_SIZE, H))
        try:
            if hasattr(model, "train_example"):
                model.train_example(x, y)
            else:
                model(x, y)
            mdict["iters"] += 1
        except Exception as e:
            break


df = pd.DataFrame(models)[["name", "iters"]]

print("Max iterations: reached for each model")
print(df)
