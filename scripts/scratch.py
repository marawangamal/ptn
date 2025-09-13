import wandb, numpy as np, pandas as pd
from tqdm import tqdm

api = wandb.Api(timeout=299)
PROJECT = "marawan-gamal/ctn-mnist"


def get_min_val(run, key="val/loss"):
    vals = []
    for row in run.scan_history(keys=[key], page_size=10000):
        v = row.get(key)
        if isinstance(v, (int, float)):
            vals.append(float(v))
        elif isinstance(v, str):
            try:
                vals.append(float(v))
            except ValueError:
                pass
    return np.min(vals) if vals else float("nan")


rows = []
for run in api.runs(PROJECT):
    min_val_loss = get_min_val(run, key="val/loss")
    rows.append(
        {
            "model": run.config["model"],
            "pos_func": run.config["pos_func"],
            "rank": run.config["rank"],
            "val/loss": min_val_loss,
            "horizon": 28 * 28,
            "name": run.name,
        }
    )
    break

df = pd.DataFrame(rows)
df["val/loss"] = df["val/loss"] * df["horizon"]
df
