import torch


def get_opt_model(seed: int) -> torch.nn.Module:
    raise NotImplementedError("Not implemented")


def main():
    # ---- train models ----
    N_models = 10
    N_vocab = 100
    test_samples = [torch.randint(0, N_vocab, (10,))]
    N_test_samples = len(test_samples)
    P = torch.zeros(N_models, N_test_samples, N_vocab)  # Probs tensor

    for i in range(N_models):
        model = get_opt_model(seed=i)
        for j in range(N_test_samples):
            x, y = test_samples[j]
            logits = model(x)
            P[i, j] = torch.softmax(logits, dim=-1)

    # ---- compute bias and variance ----
    # GPT generated this part
    # P_mean = P.mean(dim=0)
    # P_var = P.var(dim=0)

    # bias = (P_mean - P_true).abs().mean()
    # variance = P_var.mean()

    p_bar = P.log().mean(dim=0).softmax(dim=-1)

    # bias =

    # ---- compute bias and variance ----


if __name__ == "__main__":
    pass
