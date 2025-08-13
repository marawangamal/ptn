import copy
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Tuple
from tqdm import tqdm

from nanogpt.modelling_nanogpt import NanoGPT, ModelOutput


def get_dataset(
    seed: int, N_samples: int, N_vocab: int, T_prefix: int, T_suffix: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a simple structured synthetic dataset using one discrete rule.

    Pattern: arithmetic progression modulo `N_vocab`.

    Returns:
      X: (N_samples, T_prefix) prefix tokens
      Y: (N_samples, T_suffix) next-token target(s)
    """

    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)

    total_len = T_prefix + T_suffix
    X = torch.empty((N_samples, T_prefix), dtype=torch.long)
    Y = torch.empty((N_samples, T_suffix), dtype=torch.long)

    for i in range(N_samples):
        start = int(torch.randint(0, N_vocab, (1,), generator=rng).item())
        step_upper = max(2, min(7, N_vocab))
        step = int(torch.randint(1, step_upper, (1,), generator=rng).item())
        seq = [int((start + step * t) % N_vocab) for t in range(total_len)]

        seq_tensor = torch.tensor(seq, dtype=torch.long)
        X[i] = seq_tensor[:T_prefix]
        Y[i] = seq_tensor[T_prefix : T_prefix + T_suffix]

    return X, Y


def compute_bias_variance(P: torch.Tensor, Y: torch.Tensor) -> Tuple[float, float]:
    """Compute bias and variance from predictions P and targets Y.

    Bias variance decomposition computed from NLL loss (https://pubmed.ncbi.nlm.nih.gov/9698350)

    Args:
        P (torch.Tensor): Probability tensor. Shape: (N_models, N_samples, N_vocab)
        Y (torch.Tensor): Target tensor. Shape: (N_samples, T_suffix)

    Returns:
        bias (float): Bias
        var (float): Variance
    """
    N_models, N_samples, _ = P.shape

    # ---- Compute bias ------
    p_bar = P.log().mean(dim=0).softmax(dim=-1)  # (N_samples, N_vocab)
    bias = -(
        P.log()
        .gather(dim=-1, index=Y.reshape(1, N_samples, -1).repeat(N_models, 1, 1))
        .mean()
    ).item()  # (1,)

    # ---- Compute variance ----
    var = 0.0
    for i, j in itertools.product(range(N_models), range(N_samples)):
        var += torch.nn.functional.kl_div(p_bar[j].log(), P[i, j]).item()
    var = var / (N_models * N_samples)
    return bias, var


def evaluate(model, X: torch.Tensor, Y: torch.Tensor, max_samples=None) -> float:
    """Evaluate next-token accuracy on a subset of (X, Y)."""
    model.eval()
    with torch.no_grad():
        N = X.shape[0]
        correct = 0
        for j in range(N):
            x = X[j]
            output: ModelOutput = model(x.unsqueeze(0))
            logits = output.logits
            pred = torch.argmax(logits[0, -1], dim=-1)
            correct += int(pred == Y[j].squeeze())
    return correct / max(1, N)


def train_model(
    model: NanoGPT,
    X: torch.Tensor,
    Y: torch.Tensor,
    batch_size: int = 32,
    lr: float = 0.01,
) -> NanoGPT:
    """Train a model on the given data using mini-batch gradient descent."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    N_samples = X.shape[0]
    model.train()
    num_batches = (N_samples + batch_size - 1) // batch_size
    total_steps = num_batches

    # move to GPU if available
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
        X = X.to(torch.device("cuda"))
        Y = Y.to(torch.device("cuda"))

    with tqdm(total=total_steps, leave=False, desc="train", dynamic_ncols=True) as pbar:
        for start in range(0, N_samples, batch_size):
            end = min(start + batch_size, N_samples)
            X_batch = X[start:end]
            Y_batch = Y[start:end]

            optimizer.zero_grad()

            # Forward pass
            output: ModelOutput = model(X_batch)
            logits = output.logits
            logits_last = logits[:, -1]
            targets = Y_batch.view(-1)
            loss = criterion(logits_last, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Accuracy for this batch
            preds = torch.argmax(logits_last, dim=-1)
            acc = (preds == targets).float().mean().item()

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{acc:.3f}",
                }
            )
            pbar.update(1)

    return model


def find_best_lr(
    model: NanoGPT,
    X: torch.Tensor,
    Y: torch.Tensor,
    batch_size: int = 32,
    n_train: int = 1000,
    n_eval: int = 128,
    lrs: list[float] = [1e-2, 1e-3, 1e-4, 5e-5],
) -> float:

    accs: list[float] = []
    for lr in lrs:
        # Train a fresh copy for each LR so we don't mutate the original or
        # carry training across different learning rates.
        m = copy.deepcopy(model)
        m = train_model(m, X[:n_train], Y[:n_train], batch_size=batch_size, lr=lr)
        accs.append(evaluate(m, X[:n_eval], Y[:n_eval], max_samples=n_eval))

    best_idx = int(torch.argmax(torch.tensor(accs)).item())
    return lrs[best_idx]


def main():
    # ---- Parameters ----
    N_models = 2
    N_vocab = 100
    N_train = 20000  # Number of training samples
    N_test = 50  # Number of test samples
    T_prefix = 10  # Length of prefix sequence
    T_suffix = 1  # Length of suffix sequence
    model_kwargs = [
        # {"d_model": 2**i}
        # for i in [1, 2, 3, 4, 5, 6, 7, 8]
        {"d_model": 4},
        {"d_model": 8},
        {"d_model": 16},
        {"d_model": 32},
        {"d_model": 64},
        {"d_model": 128},
        {"d_model": 256},
        {"d_model": 512},
        {"d_model": 1024},
        {"d_model": 2048},
        {"d_model": 4096},
        {"d_model": 8192},
        # i.e. 2-256
    ]  # Different model complexities

    # Create training and test datasets
    X_test, Y_test = get_dataset(
        seed=123,
        N_samples=N_test,
        N_vocab=N_vocab,
        T_prefix=T_prefix,
        T_suffix=T_suffix,
    )

    bias_values = []
    variance_values = []
    total_error_values = []

    print("Computing bias-variance trade-off for different model complexities...")

    for i, m_kwargs in enumerate(model_kwargs):
        print(f"Model kwargs: {m_kwargs}")

        # Create P tensor: (N_models, N_test_samples, N_vocab)
        P = torch.zeros(N_models, N_test, N_vocab)

        # Train multiple models with different seeds
        for i in tqdm(range(N_models), desc=f"Models (complexity={i})"):
            model = NanoGPT(
                n_layers=1,
                d_vocab=N_vocab,
                d_block=T_prefix,
                **m_kwargs,
            )
            X_train, Y_train = get_dataset(
                seed=i,
                N_samples=N_train,
                N_vocab=N_vocab,
                T_prefix=T_prefix,
                T_suffix=T_suffix,
            )
            lr = find_best_lr(model, X_train, Y_train, n_train=N_train, n_eval=1000)
            model = train_model(model, X_train, Y_train, lr=lr)

            # Get predictions on test set
            model.eval()
            with torch.no_grad():
                for j in range(N_test):
                    x = X_test[j]  # Shape: (T_prefix,)
                    output: ModelOutput = model(x.unsqueeze(0))
                    logits = output.logits
                    P[i, j] = F.softmax(logits[0, -1], dim=-1)

        # Compute bias and variance
        bias, variance = compute_bias_variance(P, Y_test)
        total_error = bias + variance

        # Evaluate on a small subset of the training set for quick feedback
        train_acc = evaluate(model, X_train, Y_train, max_samples=128)
        test_acc = evaluate(model, X_test, Y_test, max_samples=128)

        bias_values.append(bias)
        variance_values.append(variance)

        print(
            f"  Bias: {bias:.4f}, Variance: {variance:.4f}, Total Error: {total_error:.4f} | Train Acc (subset): {train_acc:.4f}, Test Acc (subset): {test_acc:.4f} | LR: {lr:.4f}"
        )

    # ---- Plotting ----
    plt.figure(figsize=(12, 8))
    complexities = list(range(len(model_kwargs)))

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
    plt.xticks(range(len(complexities)), [str(c) for c in complexities])
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot total error with optimal complexity highlighted
    plt.subplot(2, 2, 4)
    optimal_idx = total_error_values.index(min(total_error_values))
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
    # plt.show()

    print(f"\nOptimal model complexity: {complexities[optimal_idx]}")
    print(f"Minimum total error: {total_error_values[optimal_idx]:.4f}")


if __name__ == "__main__":
    main()
