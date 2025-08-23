from collections import defaultdict
import itertools
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mtp.mheads._abc import AbstractDisributionHeadConfig
from mtp.mheads import MHEADS

# set theme
sns.set_theme()


def run_train(
    mt_name, batch_size, horizon, rank, d_model, d_output, add_loss_dict=False
):
    """Test if CP distribution can recover a target distribution on small scale."""
    import torch.optim as optim

    # set seed
    torch.manual_seed(42)
    random.seed(42)

    # Training parameters
    n_iters = 100
    B, H, D, V = batch_size, horizon, d_model, d_output

    config = AbstractDisributionHeadConfig(
        d_model=d_model, d_output=d_output, horizon=horizon, rank=rank
    )
    mt_head = MHEADS[mt_name](config)

    log_dict = defaultdict(list)

    optimizer = optim.AdamW(mt_head.parameters(), lr=1e-3)
    for i in range(n_iters):
        optimizer.zero_grad()

        x = torch.randn(B, D)
        y = torch.randint(0, 2, (B, H)) * (V - 1)
        out = mt_head(x, y)
        out.loss.backward()
        optimizer.step()

        if out.loss_dict is not None and add_loss_dict:
            for k, v in out.loss_dict.items():
                # Add or create new key if not exists
                if k not in log_dict:
                    log_dict[k] = []
                log_dict[k].append(v)

        # also add grad_norm and loss to log_dict
        log_dict["loss"].append(out.loss.item())
        log_dict["grad_norm"].append(
            torch.nn.utils.clip_grad_norm_(
                mt_head.parameters(), max_norm=float("inf")
            ).item()
        )

        if i % 100 == 0:
            # Print latest values for each metric
            latest_values = {k: v[-1] if v else 0 for k, v in log_dict.items()}
            print(
                f"[{i}] "
                + " | ".join([f"{k}: {v:.4f}" for k, v in latest_values.items()])
            )
            if out.loss.isnan():
                print("Loss is NaN!")
                break

    print("Training test completed!")

    # Add mt_name to log_dict for tracking
    log_dict["mt_name"] = mt_name
    log_dict["iteration"] = list(range(n_iters))
    log_dict["horizon"] = horizon

    # Plot training metrics using seaborn
    return log_dict


def plot_training_metrics(
    log_dicts: list[dict], save_path: str = "results/plots/train_mheads.png"
):
    """Plot training metrics using seaborn."""
    df = pd.concat([pd.DataFrame(log_dict) for log_dict in log_dicts])

    # Melt excluding mt_name from value columns
    melt_df = df.melt(
        id_vars=["iteration", "mt_name", "horizon"],
        var_name="metric",
        value_name="value",
    )

    sns.relplot(
        data=melt_df,
        kind="line",
        x="iteration",
        y="value",
        hue="mt_name",
        col="metric",
        row="horizon",
    )
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    configs = [
        {
            "mt_name": "moe",
            "batch_size": 32,
            "horizon": h,
            "rank": r,
            "d_model": 512,
            "d_output": 100,
        }
        for r, h in itertools.product([2, 4, 8], [2, 4, 8])
    ] + [
        {
            "mt_name": "cp",
            "batch_size": 32,
            "horizon": h,
            "rank": r,
            "d_model": 512,
            "d_output": 100,
        }
        for r, h in itertools.product([2, 4, 8], [2, 4, 8])
    ]

    log_dicts = []
    for config in configs:
        log_dict = run_train(**config)
        log_dicts.append(log_dict)

    # concat all log_dicts into a single dataframe
    plot_training_metrics(log_dicts)
