import itertools
import torch


def get_opt_model(seed: int) -> torch.nn.Module:
    raise NotImplementedError("Not implemented")


def get_dataset(
    seed: int, N_samples: int, N_vocab: int
) -> tuple[torch.Tensor, torch.Tensor]:
    T_prefix = 10  # Length of prefix sequence
    T_suffix = 1  # Length of suffix sequence
    # X: (N_samples, T_prefix) - prefix sequences
    # Y: (N_samples, T_suffix) - suffix sequences
    torch.manual_seed(42)  # For reproducibility
    X = torch.randint(0, N_vocab, (N_samples, T_prefix))
    Y = torch.randint(0, N_vocab, (N_samples, T_suffix))
    return X, Y


def main():
    # ---- Train models ----
    N_models = 10
    N_vocab = 100
    N_samples = 100  # Number of test samples

    # Create dataset
    X, Y = get_dataset(seed=42, N_samples=N_samples, N_vocab=N_vocab)

    # Create P tensor: (N_models, N_samples, N_vocab)
    P = torch.zeros(N_models, N_samples, N_vocab)  # Probs tensor

    for i in range(N_models):
        model = get_opt_model(seed=i)
        for j in range(N_samples):
            x = X[j]  # Shape: (T_prefix,)
            logits = model(x)
            P[i, j] = torch.softmax(logits, dim=-1)

    # ---- Compute bias and variance ----
    p_bar = P.log().mean(dim=0).softmax(dim=-1)  # (N_samples, N_vocab)
    bias = -(P.log().gather(dim=-1, index=Y).mean())  # (1,)

    # ---- Compute variance ----
    var = 0
    for i, j in itertools.product(range(N_models), range(N_samples)):
        var += torch.nn.functional.kl_div(p_bar[j], P[i, j])
    var = var / (N_models * N_samples)


if __name__ == "__main__":
    pass
