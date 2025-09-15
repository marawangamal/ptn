import argparse
import os
import re
import certifi
from datasets import load_dataset
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import wandb
import numpy as np

from ctn.mheads._abc import AbstractDisributionHeadConfig
from ctn.mheads import MHEADS


import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-GUI backend
import io
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset


import torch
from sklearn.cluster import KMeans


os.environ["SSL_CERT_FILE"] = certifi.where()


def disc(
    points: torch.Tensor, n_bins: int = 128, min_val: float = -1.0, max_val: float = 1.0
) -> torch.Tensor:
    """
    Discretize 3D point cloud into integer bins [0, n_bins-1] along each axis.

    Args:
        points (torch.Tensor): (N, 3) float tensor of coordinates.
        n_bins (int): Number of bins per axis.
        min_val (float): Minimum value of the range.
        max_val (float): Maximum value of the range.

    Returns:
        torch.Tensor: (N, 3) integer tensor of bin indices in [0, n_bins-1].
    """
    scale = (n_bins - 1) / (max_val - min_val)
    out = torch.round((points - min_val) * scale).to(torch.int32)
    return torch.clamp(out, 0, n_bins - 1)


def invdisc(
    indices: torch.Tensor,
    n_bins: int = 128,
    min_val: float = -1.0,
    max_val: float = 1.0,
) -> torch.Tensor:
    """
    Map discretized integer bins back to continuous coordinates (bin centers).

    Args:
        indices (torch.Tensor): (N, 3) integer tensor of bin indices.
        n_bins (int): Number of bins per axis.
        min_val (float): Minimum value of the range.
        max_val (float): Maximum value of the range.

    Returns:
        torch.Tensor: (N, 3) float tensor of bin center coordinates.
    """
    step = (max_val - min_val) / (n_bins - 1)
    return min_val + indices.to(torch.float32) * step


def cluster(x: torch.Tensor, K: int, seed: int = 0) -> torch.Tensor:
    """
    Run KMeans on x and return cluster centers.

    Args:
        x (torch.Tensor): (N, D) float tensor on CPU or GPU.
        K (int): number of clusters.
        seed (int): random seed for reproducibility.

    Returns:
        torch.Tensor: (K, D) float tensor of cluster centers (on same device as x).
    """
    device = x.device
    x_np = x.detach().cpu().numpy()

    km = KMeans(n_clusters=K, n_init="auto", random_state=seed)
    km.fit(x_np)

    centers = torch.from_numpy(km.cluster_centers_).to(device=device, dtype=x.dtype)
    return centers


def collate_fn(batch):
    # x, y = batch["inputs"], batch["label"]
    x = torch.stack([torch.tensor(b["inputs"]) for b in batch])
    y = torch.stack([torch.tensor(b["label"]) for b in batch])
    return x, y


def preprocess(example):
    # For each point: (Li, 3) -> (1024, 3)
    pts_clustered = cluster(torch.tensor(example["inputs"]), K=2048)
    x = disc(pts_clustered).reshape(-1)  # (B, L, 3)
    y = torch.tensor(example["label"])
    return {"inputs": x, "label": y}


def get_data_loaders(batch_size=32, test_size=0.1):
    ds = load_dataset("jxie/modelnet40")  # has train/test splits in parquet
    # limit to 100 for testing
    # ds["train"] = ds["train"].select(range(100))
    # ds["test"] = ds["test"].select(range(100))
    ds["train"] = ds["train"].map(preprocess)
    ds["test"] = ds["test"].map(preprocess)

    # map to tensors
    train_loader = torch.utils.data.DataLoader(
        ds["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        ds["test"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, device, wandb_logger):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0
    num_batches = 0
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # for gen modelling: x is the class, y is the pixel values
        y, x = batch
        B = y.shape[0]

        # Convert to one-hot encoding
        x = F.one_hot(x, num_classes=40).reshape(B, -1).float()
        x, y = x.to(device), y.to(device)

        # Forward pass
        output = model(x, y.reshape(B, -1))
        loss = output.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        g = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        p = sum(torch.linalg.norm(p) for p in model.parameters())
        optimizer.step()

        wandb_logger.log(
            {
                "train/batch_loss": loss.item(),
                "train/grad_norm": g,
                "train/param_norm": p,
            }
        )

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"train/loss": loss.item()})

        if loss.isnan():
            raise ValueError("Loss is NaN!")

    return total_loss / num_batches


def evaluate(model, val_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for y, x in val_loader:
            B = x.shape[0]
            x = F.one_hot(x, num_classes=40).reshape(B, -1).float()
            x, y = x.to(device), y.to(device)

            output = model(x, y.reshape(B, -1))
            total_loss += output.loss.item()
            num_batches += 1

    return total_loss / num_batches


def generate_images(model, device, num_images=8, image_size=(2048, 3)):
    generated_images = []
    with torch.no_grad():
        for digit in range(min(num_images, 10)):
            # Create one-hot encoding for the digit
            z = F.one_hot(torch.tensor([digit], device=device), num_classes=10).float()

            # Generate image
            generated = model.generate(z)[0].detach().cpu().view(*image_size).numpy()

            # Convert to 0-255 range for better visualization
            generated = (generated * 255).astype(np.uint8)
            generated_images.append(generated)

    images = []
    for i in range(len(test)):
        pts = np.array(test[i]["inputs"], dtype=np.float32)  # (N, 3)
        label = int(test[i]["label"])

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1)
        ax.set_title(f"ModelNet40 sample (label={label})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=20, azim=45)
        plt.tight_layout()

        # --- Convert figure to HxWx3 uint8 array ---
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        images.append(img)
        plt.close(fig)

    return images


def build_exp_name(args: argparse.Namespace):
    ignore_keys = ["seed", "sample", "debug"]
    abbrev_map = {}
    parts = []
    for k, v in args.__dict__.items():
        if k not in ignore_keys:
            # Use abbreviation if it exists, otherwise use first letter
            abbrev = abbrev_map.get(k, k[:1])
            parts.append(f"{abbrev}{v}")

    parts = [re.sub(r"[^a-zA-Z0-9]", "", p) for p in parts]
    return "_".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Train MHEADS on ModelNet40")
    parser.add_argument("--model", default="cp")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--pos_func", type=str, default="abs", help="Position function")
    parser.add_argument(
        "--num_gen_images",
        type=int,
        default=10,
        help="Number of images to generate per epoch",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of training samples for quick testing",
    )
    parser.add_argument(
        "--save_checkpoint", action="store_true", help="Save checkpoint"
    )

    args = parser.parse_args()

    # Initialize wandb
    wandb.init(project="ctn-modelnet", name=build_exp_name(args), config=vars(args))

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(f"checkpoints/{build_exp_name(args)}", exist_ok=True)
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader = get_data_loaders(args.batch_size)

    # Limit training data for quick testing
    if args.max_samples is not None:
        train_dataset = train_loader.dataset
        train_dataset.data = train_dataset.data[: args.max_samples]
        train_dataset.targets = train_dataset.targets[: args.max_samples]
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Model
    test_batch = next(iter(train_loader))
    model = MHEADS[args.model](
        AbstractDisributionHeadConfig(
            horizon=test_batch[0].shape[1],  # 3*1024 for ModelNet40
            d_model=40,  # 40 classes
            d_output=128,  # 128 bins
            rank=args.rank,
            pos_func=args.pos_func,
        )
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, wandb)
        val_loss = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # Generate and log images
        generated_images = generate_images(model, device, args.num_gen_images)

        # Log to wandb
        wandb.log(
            {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "generated_images": [
                    wandb.Image(img, caption=f"3D Sample {i} - Epoch {epoch+1}")
                    for i, img in enumerate(generated_images)
                ],
            }
        )

        if val_loss < best_val_loss and args.save_checkpoint:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "epoch": epoch,
                },
                f"checkpoints/{build_exp_name(args)}/model_best.pt",
            )

    wandb.finish()


if __name__ == "__main__":
    main()
