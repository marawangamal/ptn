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


def run_train(
    mt_name,
    batch_size,
    horizon,
    rank,
    d_model,
    d_output,
    add_loss_dict=False,
    lr=1e-3,
    seed=42,
    device="cuda" if torch.cuda.is_available() else "cpu",
    n_iters=100,
):
    """Test if CP distribution can recover a target distribution on small scale."""
    import torch.optim as optim

    # set seed
    torch.manual_seed(seed)
    random.seed(seed)

    # Training parameters
    B, H, D, V = batch_size, horizon, d_model, d_output

    config = AbstractDisributionHeadConfig(
        d_model=d_model, d_output=d_output, horizon=horizon, rank=rank
    )
    mt_head = MHEADS[mt_name](config)

    log_dict = defaultdict(list)

    optimizer = optim.AdamW(mt_head.parameters(), lr=lr)
    mt_head.to(device)
    for i in range(n_iters):
        optimizer.zero_grad()

        x = torch.randn(B, D, device=device)
        y = torch.randint(0, 2, (B, H), device=device) * (V - 1)
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
            # latest_values = {k: v[-1] if v else 0 for k, v in log_dict.items()}
            # print(
            #     f"[{i}] "
            #     + " | ".join([f"{k}: {v:.4f}" for k, v in latest_values.items()])
            # )
            if out.loss.isnan():
                print("Loss is NaN!")
                break

    print("Training test completed!")

    # Add mt_name to log_dict for tracking
    log_dict["mt_name"] = mt_name
    log_dict["iteration"] = list(range(n_iters))
    log_dict["horizon"] = horizon
    log_dict["rank"] = rank

    # Plot training metrics using seaborn
    return log_dict


def plot_training_metrics(
    log_dicts: list[dict],
    metric_key="loss",
    save_path: str = "results/plots/train_mhead_moe.png",
):
    """Plot training metrics using seaborn."""
    df = pd.concat([pd.DataFrame(log_dict) for log_dict in log_dicts])

    # Melt excluding mt_name from value columns
    # melt_df = df.melt(
    #     id_vars=["iteration", "mt_name", "horizon"],
    #     var_name="metric",
    #     value_name="value",
    # )

    sns.relplot(
        data=df,
        kind="line",
        x="iteration",
        y=metric_key,
        hue="mt_name",
        col="rank",
        row="horizon",
    )
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    lr = 1e-3
    # d_model = 4096
    # d_output = 10_00

    d_model = 8
    d_output = 16
    configs = (
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
            for r, h, seed in itertools.product([2, 8], [2, 8], [0, 42, 84])
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
            for r, h, seed in itertools.product([2, 8], [2, 8], [0, 42, 84])
        ]
        # + [
        #     {
        #         "mt_name": "cp",
        #         "batch_size": 32,
        #         "horizon": h,
        #         "rank": r,
        #         "d_model": 512,
        #         "d_output": 100,
        #         "seed": seed,
        #     }
        #     # for r, h, seed in itertools.product([8], [8], [0, 42, 84])
        #     for r, h, seed in itertools.product([8], [32], [0, 42, 84])
        # ]
        # + [
        #     {
        #         "mt_name": "multihead",
        #         "batch_size": 32,
        #         "horizon": h,
        #         "rank": 1,
        #         "d_model": 512,
        #         "d_output": 100,
        #     }
        #     for h in [2, 4, 8]
        # ]
    )

    log_dicts = []
    pbar = tqdm(configs)
    for config in configs:
        pbar.set_description(
            f"{config['mt_name']} | R: {config['rank']} | H: {config['horizon']}"
        )
        log_dict = run_train(**config, lr=lr)
        log_dicts.append(log_dict)
        pbar.update()
        # add description to pbar
    # concat all log_dicts into a single dataframe
    plot_training_metrics(
        log_dicts,
        metric_key="loss",
        save_path=f"results/plots/train_mhead_loss_lr{lr}.png",
    )
    plot_training_metrics(
        log_dicts,
        metric_key="grad_norm",
        save_path=f"results/plots/train_mhead_grad_norm_lr{lr}.png",
    )
