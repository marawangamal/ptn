#!/usr/bin/env python3

import wandb
import pandas as pd
import numpy as np

def load_wandb_runs_fixed(
    project: str,
    log_specs=None,
    config_attrs=None,
    entity=None,
):
    """
    Load runs from a wandb project, extracting logs and config attributes.
    Fixed version that handles the key filtering issue.
    """
    api = wandb.Api()
    runs = api.runs(project if entity is None else f"{entity}/{project}")

    rows = []
    for run in runs:
        row = {"run_id": run.id, "name": run.name, "state": run.state}

        # pull configs
        if config_attrs:
            for attr in config_attrs:
                row[attr] = run.config.get(attr, None)

        # pull logs - get full history first, then filter
        if log_specs:
            # Get full history instead of filtering by keys
            history = run.history()
            
            for spec in log_specs:
                key = spec["name"]
                reduce = spec.get("reduce", "last")

                if key not in history.columns:
                    row[f"{key}_{reduce}"] = None
                    continue

                series = history[key].dropna().to_numpy()
                if len(series) == 0:
                    row[f"{key}_{reduce}"] = None
                else:
                    if reduce == "last":
                        row[f"{key}_{reduce}"] = series[-1]
                    elif reduce == "min":
                        row[f"{key}_{reduce}"] = np.min(series)
                    elif reduce == "max":
                        row[f"{key}_{reduce}"] = np.max(series)
                    elif reduce == "mean":
                        row[f"{key}_{reduce}"] = np.mean(series)
                    else:
                        raise ValueError(f"Unknown reducer: {reduce}")

        rows.append(row)

    return pd.DataFrame(rows)

if __name__ == "__main__":
    print("Testing with horizon using 'last' instead of 'min':")
    df_ptn = load_wandb_runs_fixed('ptn-ucla', 
                                  log_specs=[{'name': 'val/loss', 'reduce': 'min'}, 
                                           {'name': 'horizon', 'reduce': 'last'}], 
                                  config_attrs=['dataset', 'model'])
    print(df_ptn)
