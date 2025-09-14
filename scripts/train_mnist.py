import argparse
import os
import re
import certifi
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import wandb
import numpy as np

from mtp.mheads._abc import AbstractDisributionHeadConfig
from mtp.mheads import MHEADS

os.environ["SSL_CERT_FILE"] = certifi.where()


def get_data_loaders(batch_size=32, data_dir="./data"):
    """Create MNIST data loaders with binary thresholding."""
    transform = transforms.Compose([transforms.ToTensor(), lambda x: (x > 0.5).long()])

    train_set = torchvision.datasets.MNIST(
        data_dir, train=True, transform=transform, download=True
    )
    val_set = torchvision.datasets.MNIST(
        data_dir, train=False, transform=transform, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, device, wandb_logger):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0
    num_batches = 0
    pbar = tqdm(train_loader, desc="Training")
    for i, batch in enumerate(pbar):
        # if i == 114:
        #     print(f"Batch {i} is {batch}")
        y, x = batch  # for generative modeling, reverse x, y
        B = x.shape[0]

        # Convert to one-hot encoding
        z = F.one_hot(x, num_classes=10).reshape(B, -1).float()
        z, y = z.to(device), y.to(device)

        # Forward pass
        output = model(z, y.reshape(B, -1))
        loss = output.loss

        if loss.isnan():
            # Save model state, input, and output for debugging
            debug_dir = "debug_nan"
            os.makedirs(debug_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(debug_dir, f"model_nan_{i}.pt"))
            torch.save(z.cpu(), os.path.join(debug_dir, f"input_z_nan_{i}.pt"))
            torch.save(y.cpu(), os.path.join(debug_dir, f"input_y_nan_{i}.pt"))
            torch.save(output, os.path.join(debug_dir, f"output_nan_{i}.pt"))
            raise ValueError(f"Loss is NaN! Saved model and inputs to {debug_dir}")

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

    return total_loss / num_batches


def evaluate(model, val_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for y, x in val_loader:
            B = x.shape[0]
            z = F.one_hot(x, num_classes=10).reshape(B, -1).float()
            z, y = z.to(device), y.to(device)

            output = model(z, y.reshape(B, -1))
            total_loss += output.loss.item()
            num_batches += 1

    return total_loss / num_batches


def generate_images(model, device, num_images=10, image_size=(28, 28)):
    """Generate images from the model and return as wandb-compatible format."""
    model.eval()
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

    return generated_images


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


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Train MNIST model with MHEADS")
    parser.add_argument("--model", default="moe_proj")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--pos_func", type=str, default="abs", help="Position function")
    parser.add_argument("--lambda_ortho", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=None)
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
    wandb.init(project="ctn-mnist", name=build_exp_name(args), config=vars(args))

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(f"checkpoints/{build_exp_name(args)}", exist_ok=True)
    print(f"Using device: {device}")
    set_seed()

    # Data
    train_loader, val_loader = get_data_loaders(args.batch_size, "./data")

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
    model = MHEADS[args.model](
        AbstractDisributionHeadConfig(
            horizon=(28 * 28),  # 28x28 for MNIST
            d_model=10,
            d_output=2,
            rank=args.rank,
            pos_func=args.pos_func,
            lambda_ortho=args.lambda_ortho,
        )
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    best_val_loss = float("inf")
    wandb.watch(model, log="all", log_freq=20)  # log_freq = steps between logging
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, wandb)
        val_loss = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # Generate and log images
        try:
            generated_images = generate_images(model, device, args.num_gen_images)
        except Exception as e:
            print(f"Error generating images: {e}")
            generated_images = []

        # Log to wandb
        wandb.log(
            {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "generated_images": [
                    wandb.Image(img, caption=f"Digit {i} - Epoch {epoch+1}")
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
