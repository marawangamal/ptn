#!/usr/bin/env python3
"""
Example script demonstrating second-order optimization for tensor networks.

This script shows how to use different optimization methods (first-order and second-order)
with MPS tensor network models and compares their performance.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# Import your tensor network modules
from ptn.regressors.mps import MPS_REGRESSOR
from ptn.regressors._abc import AbstractRegressorHeadConfig
from ptn.optimizers import (
    create_optimizer,
    get_optimizer_recommendations,
    get_optimizer_info,
)


def generate_synthetic_data(
    num_samples=1000, num_features=64, rank=16, noise_level=0.1
):
    """Generate synthetic regression data."""
    torch.manual_seed(42)

    # Generate random input data
    x = torch.randn(num_samples, num_features).abs()

    # Define a polynomial target function
    def target_fn(x):
        return (x.pow(2).sum(dim=1) + x.sum(dim=1) * 0.5).unsqueeze(1)

    y = target_fn(x)
    y += torch.randn_like(y) * noise_level  # Add noise

    return x, y


def create_dataloader(x, y, batch_size=32):
    """Create a dataloader with bias term addition."""

    def add_bias(batch):
        x_batch, y_batch = zip(*batch)
        x_batch = torch.stack(x_batch, dim=0)
        y_batch = torch.stack(y_batch, dim=0)

        # Reshape and add bias
        n_cores = 1
        d_in = x_batch.size(1) // n_cores
        x_batch = x_batch.reshape(x_batch.size(0), n_cores, d_in)
        x_batch = torch.cat(
            [torch.ones(x_batch.size(0), x_batch.size(1), 1), x_batch], dim=-1
        )

        return x_batch, y_batch

    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=add_bias
    )


def train_model(model, dataloader, optimizer_name, num_epochs=100, lr=1e-3):
    """Train a model with the specified optimizer."""
    model = MPS_REGRESSOR(
        AbstractRegressorHeadConfig(
            d_in=dataloader.dataset[0][0].size(1) + 1,  # +1 for bias
            d_out=1,
            horizon=1,
            rank=16,
        )
    )

    # Create optimizer
    optimizer = create_optimizer(model, optimizer_name, lr=lr)

    losses = []
    pbar = tqdm(range(num_epochs), desc=f"Training with {optimizer_name}")

    for epoch in pbar:
        if optimizer_name.lower() == "lbfgs":
            # L-BFGS requires special handling
            def lbfgs_closure():
                optimizer.zero_grad()
                total_loss = 0
                num_batches = 0

                for batch in dataloader:
                    x, y = batch
                    output = model(x, y)
                    loss = output.loss
                    loss.backward()
                    total_loss += loss.item()
                    num_batches += 1

                return total_loss / num_batches if num_batches > 0 else 0.0

            loss = optimizer.step(lbfgs_closure)
            losses.append(loss)
            pbar.set_postfix(loss=loss)

            # L-BFGS converges faster, so we can stop earlier
            if epoch > 10 and abs(losses[-1] - losses[-2]) < 1e-6:
                break
        else:
            # Standard first-order or AdaHessian training
            epoch_losses = []
            for batch in dataloader:
                optimizer.zero_grad()
                x, y = batch
                output = model(x, y)
                loss = output.loss
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_loss)
            pbar.set_postfix(loss=avg_loss)

    return losses


def compare_optimizers(
    x, y, optimizers=["AdamW", "LBFGS", "AdaHessian"], num_epochs=100
):
    """Compare different optimizers on the same problem."""
    dataloader = create_dataloader(x, y)

    results = {}
    for optimizer_name in optimizers:
        print(f"\n{'='*50}")
        print(f"Training with {optimizer_name}")
        print(f"{'='*50}")

        # Get optimizer info
        info = get_optimizer_info(optimizer_name)
        print(f"Type: {info['type']}")
        print(f"Description: {info['description']}")
        print(f"Best for: {', '.join(info['best_for'])}")

        # Train model
        losses = train_model(None, dataloader, optimizer_name, num_epochs)
        results[optimizer_name] = losses

    return results


def plot_results(results):
    """Plot comparison results."""
    plt.figure(figsize=(12, 8))

    # Plot 1: Loss curves
    plt.subplot(2, 2, 1)
    for optimizer_name, losses in results.items():
        plt.plot(losses, label=optimizer_name, alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.yscale("log")
    plt.grid(True, alpha=0.3)

    # Plot 2: Final losses
    plt.subplot(2, 2, 2)
    final_losses = {name: losses[-1] for name, losses in results.items()}
    names = list(final_losses.keys())
    values = list(final_losses.values())
    plt.bar(names, values)
    plt.ylabel("Final Loss")
    plt.title("Final Loss Comparison")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)

    # Plot 3: Convergence speed
    plt.subplot(2, 2, 3)
    convergence_speed = {}
    for name, losses in results.items():
        # Find epoch where loss drops below 10% of initial loss
        initial_loss = losses[0]
        target_loss = initial_loss * 0.1
        convergence_epoch = next(
            (i for i, loss in enumerate(losses) if loss <= target_loss), len(losses)
        )
        convergence_speed[name] = convergence_epoch

    names = list(convergence_speed.keys())
    values = list(convergence_speed.values())
    plt.bar(names, values)
    plt.ylabel("Epochs to 10% Loss")
    plt.title("Convergence Speed")
    plt.grid(True, alpha=0.3)

    # Plot 4: Summary table
    plt.subplot(2, 2, 4)
    plt.axis("off")

    # Create summary table
    summary_data = []
    for name in results.keys():
        info = get_optimizer_info(name)
        summary_data.append(
            [
                name,
                info["type"],
                f"{results[name][-1]:.2e}",
                f"{convergence_speed[name]}",
                info["memory_usage"],
            ]
        )

    table = plt.table(
        cellText=summary_data,
        colLabels=["Optimizer", "Type", "Final Loss", "Conv. Speed", "Memory"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title("Summary", pad=20)

    plt.tight_layout()
    plt.show()

    # Print recommendations
    print("\n" + "=" * 60)
    print("OPTIMIZER RECOMMENDATIONS")
    print("=" * 60)

    dataset_size = len(results[list(results.keys())[0]])  # Approximate
    recommendations = get_optimizer_recommendations(
        dataset_size=dataset_size,
        model_complexity="medium",
        memory_limited=False,
        convergence_speed_priority=True,
    )

    print(f"Primary recommendation: {recommendations['primary']['name']}")
    print(f"Reason: {recommendations['primary']['reason']}")
    print(f"Fallback: {recommendations['fallback']['name']}")
    print(f"Reason: {recommendations['fallback']['reason']}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare optimization methods for tensor networks"
    )
    parser.add_argument(
        "--num_samples", type=int, default=500, help="Number of training samples"
    )
    parser.add_argument(
        "--num_features", type=int, default=32, help="Number of input features"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=["AdamW", "LBFGS"],
        choices=["AdamW", "SGD", "LBFGS", "AdaHessian"],
        help="Optimizers to compare",
    )
    parser.add_argument(
        "--noise_level", type=float, default=0.1, help="Noise level in target data"
    )

    args = parser.parse_args()

    print("Second-Order Optimization for Tensor Networks")
    print("=" * 50)
    print(f"Dataset: {args.num_samples} samples, {args.num_features} features")
    print(f"Optimizers: {', '.join(args.optimizers)}")
    print(f"Epochs: {args.num_epochs}")

    # Generate synthetic data
    x, y = generate_synthetic_data(
        args.num_samples, args.num_features, noise_level=args.noise_level
    )
    print(f"Generated data shape: x={x.shape}, y={y.shape}")

    # Compare optimizers
    results = compare_optimizers(x, y, args.optimizers, args.num_epochs)

    # Plot results
    plot_results(results)


if __name__ == "__main__":
    main()
