from collections import defaultdict
import itertools
import random
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mtp.mheads._abc import AbstractDisributionHeadConfig
from mtp.mheads import MHEADS
from tqdm import tqdm

# set theme
sns.set_theme()


class SyntheticDataset(data.Dataset):
    """Synthetic dataset for multi-head training."""

    def __init__(self, num_samples, d_model, horizon, d_output, seed=42):
        """
        Args:
            num_samples: Number of samples in the dataset
            d_model: Input dimension
            horizon: Output sequence length
            d_output: Output vocabulary size
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.d_model = d_model
        self.horizon = horizon
        self.d_output = d_output

        # Set seed for reproducibility
        torch.manual_seed(seed)

        # Generate all data upfront
        self.x = torch.randn(num_samples, d_model)
        self.y = torch.randint(0, 2, (num_samples, horizon)) * (d_output - 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_grad_norm(params):
    return torch.norm(
        torch.cat([p.grad.view(-1) for p in params if p.grad is not None]), p=2
    ).item()


def get_param_norm(params):
    return torch.norm(
        torch.cat([p.view(-1) for p in params if p is not None]), p=2
    ).item()


def run_train(
    model,
    dataloader,
    add_loss_dict=False,
    lr=1e-3,
    seed=42,
    device="cuda" if torch.cuda.is_available() else "cpu",
    epochs=10,
    **kwargs,
):
    """Test if CP distribution can recover a target distribution on small scale."""
    import torch.optim as optim

    # set seed
    torch.manual_seed(seed)
    random.seed(seed)

    # Training parameters

    mt_head = model

    log_dict = defaultdict(list)

    optimizer = optim.AdamW(mt_head.parameters(), lr=lr)
    mt_head.to(device)

    iteration = 0
    for epoch in range(epochs):
        mt_head.train()

        for batch_idx, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()

            x = x.to(device)
            y = y[:, : model.config.horizon].to(device)

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
            log_dict["grad_norm"].append(get_grad_norm(mt_head.parameters()))
            log_dict["param_norm"].append(get_param_norm(mt_head.parameters()))
            log_dict["logits_norm"].append(out.logits.norm().item())
            log_dict["epoch"].append(epoch)
            log_dict["batch"].append(batch_idx)

            iteration += 1

            if out.loss.isnan():
                print(f"Loss is NaN at epoch {epoch}, batch {batch_idx}!")
                break

        # Print epoch summary
        if epoch % 5 == 0:
            epoch_losses = [
                log_dict["loss"][i]
                for i, e in enumerate(log_dict["epoch"])
                if e == epoch
            ]
            avg_epoch_loss = (
                sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            )

    # Add mt_name to log_dict for tracking
    log_dict["iteration"] = list(range(iteration))

    # Plot training metrics using seaborn
    return log_dict


def plot_training_metrics(
    log_dicts: list[dict],
    metric_key="loss",
    save_path: str = "results/plots/train_mhead_moe.png",
):
    """Plot training metrics using seaborn."""
    df = pd.concat([pd.DataFrame(log_dict) for log_dict in log_dicts])
    sns.relplot(
        data=df,
        kind="line",
        x="iteration",
        y=metric_key,
        hue="name",
        col="rank",
        row="horizon",
    )
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    lr = 1e-3
    d_model = 1024
    d_output = 2_000
    batch_size = 32
    num_samples = 10000

    # ---------------
    # Debug config
    d_model = 4
    d_output = 8
    num_samples = 100
    batch_size = 8
    # ---------------

    # Create dataloader once to be reused
    dataset = SyntheticDataset(
        num_samples=num_samples,
        d_model=d_model,
        horizon=32,  # Use max horizon for dataset
        d_output=d_output,
        seed=42,
    )

    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    configs = (
        [
            {
                "name": "moe",
                "mt_name": "moe",
                "mt_kwargs": {
                    "horizon": h,
                    "rank": r,
                    "d_model": d_model,
                    "d_output": d_output,
                },
                "seed": seed,
            }
            # for r, h, seed in itertools.product([8], [8], [0, 42, 84])
            for r, h, seed in itertools.product(
                [2, 8, 16, 32], [2, 8, 16, 32], [0, 42, 84]
            )
        ]
        + [
            {
                "name": "moe_proj",
                "mt_name": "moe_proj",
                "mt_kwargs": {
                    "horizon": h,
                    "rank": r,
                    "d_model": d_model,
                    "d_output": d_output,
                },
                "seed": seed,
            }
            # for r, h, seed in itertools.product([8], [8], [0, 42, 84])
            for r, h, seed in itertools.product(
                [2, 8, 16, 32], [2, 8, 16, 32], [0, 42, 84]
            )
        ]
        + [
            {
                "name": "cp",
                "mt_name": "cp",
                "mt_kwargs": {
                    "horizon": h,
                    "rank": r,
                    "d_model": d_model,
                    "d_output": d_output,
                },
                "seed": seed,
            }
            # for r, h, seed in itertools.product([8], [8], [0, 42, 84])
            for r, h, seed in itertools.product(
                [2, 8, 16, 32], [2, 8, 16, 32], [0, 42, 84]
            )
        ]
    )

    log_dicts = []
    pbar = tqdm(configs)
    for config in configs:
        pbar.set_description(
            f"{config['mt_name']} | R: {config['mt_kwargs']['rank']} | H: {config['mt_kwargs']['horizon']}"
        )

        model = MHEADS[config["mt_name"]](
            AbstractDisributionHeadConfig(**config["mt_kwargs"])
        )

        log_dict = run_train(
            model=model,
            dataloader=dataloader,
            lr=lr,
            epochs=1,
            **config,
        )

        log_dict["name"] = config["name"]
        log_dict["horizon"] = config["mt_kwargs"]["horizon"]
        log_dict["rank"] = config["mt_kwargs"]["rank"]

        log_dicts.append(log_dict)
        pbar.update()
        # add description to pbar

    # Make plots
    for metric_key in ["loss", "grad_norm", "param_norm", "logits_norm"]:
        plot_training_metrics(
            log_dicts,
            metric_key=metric_key,
            save_path=f"results/plots/train_mhead_{metric_key}_lr{lr}_dm{d_model}_do{d_output}.png",
        )

    # concat all log_dicts into a single dataframe
    # plot_training_metrics(
    #     log_dicts,
    #     metric_key="loss",
    #     save_path=f"results/plots/train_mhead_loss_lr{lr}_dm{d_model}_do{d_output}.png",
    # )
    # plot_training_metrics(
    #     log_dicts,
    #     metric_key="grad_norm",
    #     save_path=f"results/plots/train_mhead_grad_norm_lr{lr}_dm{d_model}_do{d_output}.png",
    # )
    # plot_training_metrics(
    #     log_dicts,
    #     metric_key="param_norm",
    #     save_path=f"results/plots/train_mhead_param_norm_lr{lr}_dm{d_model}_do{d_output}.png",
    # )
    # plot_training_metrics(
    #     log_dicts,
    #     metric_key="logits_norm",
    #     save_path=f"results/plots/train_mhead_logits_norm_lr{lr}_dm{d_model}_do{d_output}.png",
    # )
