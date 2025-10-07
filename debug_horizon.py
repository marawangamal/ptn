#!/usr/bin/env python3

import wandb
import pandas as pd
import numpy as np

def debug_horizon_issue(run_id):
    """Debug horizon logging for a specific run."""
    api = wandb.Api()
    run = api.run(f"ptn-ucla/{run_id}")
    
    print(f"Run ID: {run_id}")
    print(f"Run Name: {run.name}")
    
    # Get full history
    history = run.history()
    print(f"\nFull history shape: {history.shape}")
    print(f"Columns: {list(history.columns)}")
    
    if 'horizon' in history.columns:
        horizon_series = history['horizon'].dropna()
        print(f"\nHorizon series:")
        print(f"  Length: {len(horizon_series)}")
        print(f"  Values: {horizon_series.tolist()}")
        print(f"  Min: {horizon_series.min() if len(horizon_series) > 0 else 'N/A'}")
        print(f"  Max: {horizon_series.max() if len(horizon_series) > 0 else 'N/A'}")
    else:
        print("\nHorizon column not found!")
    
    # Check if horizon is logged as a config value instead
    print(f"\nConfig horizon: {run.config.get('horizon', 'Not in config')}")
    
    return run

if __name__ == "__main__":
    # Debug the first run
    debug_horizon_issue("7udq7qz9")
