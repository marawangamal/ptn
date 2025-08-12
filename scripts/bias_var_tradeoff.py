import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from tqdm import tqdm

from nanogpt.modelling_nanogpt import NanoGPT, ModelOutput


def get_dataset(
    seed: int, N_samples: int, N_vocab: int, T_prefix: int, T_suffix: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # X: (N_samples, T_prefix) - prefix sequences
    # Y: (N_samples, T_suffix) - suffix sequences
    torch.manual_seed(seed)  # For reproducibility
    X = torch.randint(0, N_vocab, (N_samples, T_prefix))
    Y = torch.randint(0, N_vocab, (N_samples, T_suffix))
    return X, Y


def compute_bias_variance(P: torch.Tensor, Y: torch.Tensor) -> Tuple[float, float]:
    """Compute bias and variance from predictions P and targets Y."""
    # P: (N_models, N_samples, N_vocab)
    # Y: (N_samples, T_suffix)

    # # ---- Compute bias ------
    # p_bar = P.log().mean(dim=0).softmax(dim=-1)  # (N_samples, N_vocab)
    # bias = -(P.log().gather(dim=-1, index=Y).mean())  # (1,)

    # ---- Compute variance ----
    # var = 0
    # for i, j in itertools.product(range(N_models), range(N_samples)):
    #     var += torch.nn.functional.kl_div(p_bar[j], P[i, j])
    # var = var / (N_models * N_samples)

    # Average prediction across models
    p_bar = P.mean(dim=0)  # (N_samples, N_vocab)

    # Bias: difference between average prediction and true target
    # Convert Y to one-hot for comparison
    Y_onehot = F.one_hot(
        Y.squeeze(), num_classes=P.shape[-1]
    ).float()  # (N_samples, N_vocab)
    bias = F.mse_loss(p_bar, Y_onehot)

    # Variance: average squared difference between individual predictions and average prediction
    variance = torch.mean((P - p_bar.unsqueeze(0)) ** 2)

    return bias.item(), variance.item()


def train_model(
    model: NanoGPT,
    X: torch.Tensor,
    Y: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 32,
) -> NanoGPT:
    """Train a model on the given data using mini-batch gradient descent."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    N_samples = X.shape[0]
    model.train()
    for epoch in range(epochs):
        # Shuffle indices for each epoch
        perm = torch.randperm(N_samples)
        X_shuffled = X[perm]
        Y_shuffled = Y[perm]

        for start in range(0, N_samples, batch_size):
            end = min(start + batch_size, N_samples)
            X_batch = X_shuffled[start:end]
            Y_batch = Y_shuffled[start:end]

            optimizer.zero_grad()

            # Forward pass
            output: ModelOutput = model(X_batch)  # (batch_size, N_vocab)
            logits = output.logits  # (batch_size, N_vocab)
            loss = criterion(logits[:, -1], Y_batch.squeeze())

            # Backward pass
            loss.backward()
            optimizer.step()

    return model


def main():
    # ---- Parameters ----
    N_models = 10
    N_vocab = 100
    N_samples = 200  # Number of training samples
    N_test_samples = 50  # Number of test samples
    T_prefix = 10  # Length of prefix sequence
    T_suffix = 1  # Length of suffix sequence
    d_model = 100
    complexities = [1, 2, 4]  # Different model complexities

    # Create training and test datasets
    X_test, Y_test = get_dataset(
        seed=123,
        N_samples=N_test_samples,
        N_vocab=N_vocab,
        T_prefix=T_prefix,
        T_suffix=T_suffix,
    )

    bias_values = []
    variance_values = []
    total_error_values = []

    print("Computing bias-variance trade-off for different model complexities...")

    for complexity in complexities:
        print(f"Complexity: {complexity}")

        # Create P tensor: (N_models, N_test_samples, N_vocab)
        P = torch.zeros(N_models, N_test_samples, N_vocab)

        # Train multiple models with different seeds
        for i in tqdm(range(N_models), desc=f"Models (complexity={complexity})"):
            model = NanoGPT(
                n_layers=complexity, d_model=d_model, d_vocab=N_vocab, d_block=T_prefix
            )
            X_train, Y_train = get_dataset(
                seed=i,
                N_samples=N_samples,
                N_vocab=N_vocab,
                T_prefix=T_prefix,
                T_suffix=T_suffix,
            )
            model = train_model(model, X_train, Y_train, epochs=50)

            # Get predictions on test set
            model.eval()
            with torch.no_grad():
                for j in range(N_test_samples):
                    x = X_test[j]  # Shape: (T_prefix,)
                    output: ModelOutput = model(x.unsqueeze(0))
                    logits = output.logits
                    P[i, j] = F.softmax(logits[0, -1], dim=-1)

        # Compute bias and variance
        bias, variance = compute_bias_variance(P, Y_test)
        total_error = bias + variance

        bias_values.append(bias)
        variance_values.append(variance)
        total_error_values.append(total_error)

        print(
            f"  Bias: {bias:.4f}, Variance: {variance:.4f}, Total Error: {total_error:.4f}"
        )

    # ---- Plotting ----
    plt.figure(figsize=(12, 8))

    # Plot bias, variance, and total error
    plt.subplot(2, 2, 1)
    plt.plot(complexities, bias_values, "b-o", label="Bias", linewidth=2, markersize=8)
    plt.plot(
        complexities,
        variance_values,
        "r-s",
        label="Variance",
        linewidth=2,
        markersize=8,
    )
    plt.plot(
        complexities,
        total_error_values,
        "g-^",
        label="Total Error (Bias + Variance)",
        linewidth=2,
        markersize=8,
    )
    plt.xlabel("Model Complexity")
    plt.ylabel("Error")
    plt.title("Bias-Variance Trade-off")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale("log")

    # Plot bias vs variance
    plt.subplot(2, 2, 2)
    plt.scatter(
        bias_values, variance_values, c=complexities, cmap="viridis", s=100, alpha=0.7
    )
    plt.colorbar(label="Complexity")
    plt.xlabel("Bias")
    plt.ylabel("Variance")
    plt.title("Bias vs Variance")
    plt.grid(True, alpha=0.3)

    # Plot individual components
    plt.subplot(2, 2, 3)
    plt.bar(
        range(len(complexities)), bias_values, alpha=0.7, label="Bias", color="blue"
    )
    plt.bar(
        range(len(complexities)),
        variance_values,
        bottom=bias_values,
        alpha=0.7,
        label="Variance",
        color="red",
    )
    plt.xlabel("Model Complexity")
    plt.ylabel("Error")
    plt.title("Bias and Variance Components")
    plt.xticks(range(len(complexities)), complexities)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot total error with optimal complexity highlighted
    plt.subplot(2, 2, 4)
    optimal_idx = np.argmin(total_error_values)
    plt.plot(complexities, total_error_values, "g-o", linewidth=2, markersize=8)
    plt.plot(
        complexities[optimal_idx],
        total_error_values[optimal_idx],
        "ro",
        markersize=12,
        label=f"Optimal Complexity: {complexities[optimal_idx]}",
    )
    plt.xlabel("Model Complexity")
    plt.ylabel("Total Error")
    plt.title("Total Error vs Complexity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale("log")

    plt.tight_layout()
    plt.savefig("bias_variance_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nOptimal model complexity: {complexities[optimal_idx]}")
    print(f"Minimum total error: {total_error_values[optimal_idx]:.4f}")


if __name__ == "__main__":
    main()
