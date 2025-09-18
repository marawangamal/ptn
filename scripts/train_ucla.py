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
from urllib.request import urlretrieve, urlopen

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


def collate_fn(batch):
    # x, y = batch["inputs"], batch["label"]
    x = torch.stack([torch.tensor(b["inputs"]) for b in batch])
    y = torch.stack([torch.tensor(b["label"]) for b in batch])
    return x, y


def get_data_loaders(batch_size=32, data_dir="./data", dataset="nltcs"):
    """Create MNIST data loaders with binary thresholding."""

    URI = "https://raw.githubusercontent.com/UCLA-StarAI/Density-Estimation-Datasets/refs/heads/master/datasets/"

    URLS = {
        "nltcs": {
            "train": URI + "nltcs/nltcs.train.data",
            "val": URI + "nltcs/nltcs.test.data",
        },
        "msnbc": {
            "train": URI + "msnbc/msnbc.train.data",
            "val": URI + "msnbc/msnbc.test.data",
        },
        "kdd": {
            "train": URI + "kdd/kdd.train.data",
            "val": URI + "kdd/kdd.test.data",
        },
        "plants": {
            "train": URI + "plants/plants.train.data",
            "val": URI + "plants/plants.test.data",
        },
        "baudio": {
            "train": URI + "baudio/baudio.train.data",
            "val": URI + "baudio/baudio.test.data",
        },
        "jester": {
            "train": URI + "jester/jester.train.data",
            "val": URI + "jester/jester.test.data",
        },
        "bnetflix": {
            "train": URI + "bnetflix/bnetflix.train.data",
            "val": URI + "bnetflix/bnetflix.test.data",
        },
        "accidents": {
            "train": URI + "accidents/accidents.train.data",
            "val": URI + "accidents/accidents.test.data",
        },
        "retail": {
            "train": URI + "tretail/tretail.train.data",
            "val": URI + "tretail/tretail.test.data",
        },
        "pumsb_star": {
            "train": URI + "pumsb_star/pumsb_star.train.data",
            "val": URI + "pumsb_star/pumsb_star.test.data",
        },
        "dna": {
            "train": URI + "dna/dna.train.data",
            "val": URI + "dna/dna.test.data",
        },
        "kosarek": {
            "train": URI + "kosarek/kosarek.train.data",
            "val": URI + "kosarek/kosarek.test.data",
        },
        "msweb": {
            "train": URI + "msweb/msweb.train.data",
            "val": URI + "msweb/msweb.test.data",
        },
        "book": {
            "train": URI + "book/book.train.data",
            "val": URI + "book/book.test.data",
        },
        "eachmovie": {
            "train": URI + "tmovie/tmovie.train.data",
            "val": URI + "tmovie/tmovie.test.data",
        },
        "webkb": {
            "train": URI + "webkb/webkb.train.data",
            "val": URI + "webkb/webkb.test.data",
        },
        "reuters_52": {
            "train": URI + "reuters_52/reuters_52.train.data",
            "val": URI + "reuters_52/reuters_52.test.data",
        },
        "20news": {
            "train": URI + "c20ng/c20ng.train.data",
            "val": URI + "c20ng/c20ng.test.data",
        },
        "bbc": {
            "train": URI + "bbc/bbc.train.data",
            "val": URI + "bbc/bbc.test.data",
        },
        "ad": {
            "train": URI + "ad/ad.train.data",
            "val": URI + "ad/ad.test.data",
        },
    }

    train_path = os.path.join(data_dir, dataset, f"{dataset}.train.data")
    val_path = os.path.join(data_dir, dataset, f"{dataset}.test.data")
    os.makedirs(os.path.join(data_dir, dataset), exist_ok=True)

    # Download if missing
    if not os.path.exists(train_path):
        urlretrieve(URLS[dataset]["train"], train_path)
    if not os.path.exists(val_path):
        urlretrieve(URLS[dataset]["val"], val_path)

    with urlopen(URLS[dataset]["train"]) as f:
        x_train = np.loadtxt(f, dtype=int, delimiter=",")

    with urlopen(URLS[dataset]["val"]) as f:
        x_val = np.loadtxt(f, dtype=int, delimiter=",")

    x_train = torch.from_numpy(x_train)
    x_val = torch.from_numpy(x_val)

    D = x_train.shape[1]
    cols = torch.randperm(D)
    x_train, y_train = x_train[:, cols[: D // 2]], x_train[:, cols[D // 2 :]]
    x_val, y_val = x_val[:, cols[: D // 2]], x_val[:, cols[D // 2 :]]

    train_set = torch.utils.data.TensorDataset(x_train, y_train)
    val_set = torch.utils.data.TensorDataset(x_val, y_val)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, device, wandb_logger, num_classes):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0
    num_batches = 0
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # for gen modelling: x is the class, y is the pixel values
        x, y = batch
        B = y.shape[0]

        # Convert to one-hot encoding
        x = x.float()
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


def evaluate(model, val_loader, device, num_classes):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for x, y in val_loader:
            B = x.shape[0]
            x = x.float()
            x, y = x.to(device), y.to(device)

            output = model(x, y.reshape(B, -1))
            total_loss += output.loss.item()
            num_batches += 1

    return total_loss / num_batches


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


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Train MHEADS on ModelNet40")
    parser.add_argument("--model", default="mps")
    parser.add_argument("--epochs", type=int, default=50)
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
    parser.add_argument("--dataset", type=str, default="nltcs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(project="ctn-ucla", name=build_exp_name(args), config=vars(args))
    set_seeds(args.seed)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(f"checkpoints/{build_exp_name(args)}", exist_ok=True)
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader = get_data_loaders(args.batch_size, dataset=args.dataset)

    # Limit training data for quick testing
    if args.max_samples is not None:
        train_dataset = train_loader.dataset
        train_dataset.data = train_dataset.data[: args.max_samples]
        train_dataset.targets = train_dataset.targets[: args.max_samples]
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )

    # Model
    test_batch = next(iter(train_loader))
    horizon = test_batch[0].shape[1]
    num_classes = 2
    model = MHEADS[args.model](
        AbstractDisributionHeadConfig(
            horizon=test_batch[1].shape[1],  # horizon is num bits to predict
            d_model=test_batch[0].shape[1],  # input is bit-vector
            d_output=2,  # binary
            rank=args.rank,
            pos_func=args.pos_func,
        )
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Horizon: {horizon}")
    wandb.log(
        {
            "train/samples": len(train_loader.dataset),
            "val/samples": len(val_loader.dataset),
            "horizon": horizon,
        }
    )
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, wandb, num_classes
        )
        val_loss = evaluate(model, val_loader, device, num_classes)

        print(
            f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # Log to wandb
        wandb.log(
            {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
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
