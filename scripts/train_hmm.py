import torch
import wandb
import argparse
from transformers import AutoTokenizer


def log_domain_matmul(log_A, log_B):
    """
    log_A : m x n
    log_B : n x p
    output : m x p matrix

    Normally, a matrix multiplication
    computes out_{i,j} = sum_k A_{i,k} x B_{k,j}

    A log domain matrix multiplication
    computes out_{i,j} = logsumexp_k log_A_{i,k} + log_B_{k,j}
    """
    m = log_A.shape[0]
    n = log_A.shape[1]
    p = log_B.shape[1]

    log_A_expanded = torch.reshape(log_A, (m, n, 1))
    log_B_expanded = torch.reshape(log_B, (1, n, p))

    elementwise_sum = log_A_expanded + log_B_expanded
    out = torch.logsumexp(elementwise_sum, dim=1)

    return out


class TransitionModel(torch.nn.Module):
    def __init__(self, N):
        super(TransitionModel, self).__init__()
        self.N = N
        self.unnormalized_transition_matrix = torch.nn.Parameter(torch.randn(N, N))

    def forward(self, log_alpha):
        log_transition_matrix = torch.nn.functional.log_softmax(
            self.unnormalized_transition_matrix, dim=0
        )

        out = log_domain_matmul(
            log_transition_matrix, log_alpha.transpose(0, 1)
        ).transpose(0, 1)
        return out


class EmissionModel(torch.nn.Module):
    def __init__(self, N, M):
        super(EmissionModel, self).__init__()
        self.N = N
        self.M = M
        self.unnormalized_emission_matrix = torch.nn.Parameter(torch.randn(N, M))

    def forward(self, x_t):
        log_emission_matrix = torch.nn.functional.log_softmax(
            self.unnormalized_emission_matrix, dim=1
        )
        out = log_emission_matrix[:, x_t].transpose(0, 1)
        return out


class HMM(torch.nn.Module):
    def __init__(self, M, N):
        super(HMM, self).__init__()
        self.M = M
        self.N = N

        self.transition_model = TransitionModel(self.N)
        self.emission_model = EmissionModel(self.N, self.M)
        self.unnormalized_state_priors = torch.nn.Parameter(torch.randn(self.N))
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda:
            self.cuda()

    def sample(self, T=32):
        state_priors = torch.nn.functional.softmax(
            self.unnormalized_state_priors, dim=0
        )
        transition_matrix = torch.nn.functional.softmax(
            self.transition_model.unnormalized_transition_matrix, dim=0
        )
        emission_matrix = torch.nn.functional.softmax(
            self.emission_model.unnormalized_emission_matrix, dim=1
        )

        z_t = int(
            torch.distributions.categorical.Categorical(state_priors).sample().item()
        )
        z = []
        x = []
        z.append(z_t)
        for t in range(0, T):
            x_t = int(
                torch.distributions.categorical.Categorical(emission_matrix[z_t])
                .sample()
                .item()
            )
            x.append(x_t)

            z_t = int(
                torch.distributions.categorical.Categorical(transition_matrix[:, z_t])
                .sample()
                .item()
            )
            if t < T - 1:
                z.append(z_t)

        return x, z

    def forward(self, x, T):
        if self.is_cuda:
            x = x.cuda()
            T = T.cuda()

        batch_size = x.shape[0]
        T_max = x.shape[1]
        log_state_priors = torch.nn.functional.log_softmax(
            self.unnormalized_state_priors, dim=0
        )
        log_alpha = torch.zeros(batch_size, T_max, self.N)
        if self.is_cuda:
            log_alpha = log_alpha.cuda()

        log_alpha[:, 0, :] = self.emission_model(x[:, 0]) + log_state_priors
        for t in range(1, T_max):
            log_alpha[:, t, :] = self.emission_model(x[:, t]) + self.transition_model(
                log_alpha[:, t - 1, :]
            )

        log_sums = log_alpha.logsumexp(dim=2)
        log_probs = torch.gather(log_sums, 1, T.view(-1, 1) - 1)
        return log_probs


class ShakespeareDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        seq_len=256,
        max_samples=None,
        file_path="data/shakespeare/main.txt",
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        tokens = self.tokenizer.encode(text)
        n_batches = len(tokens) // seq_len
        self.sequences = torch.tensor(
            tokens[: n_batches * seq_len], dtype=torch.long
        ).reshape(n_batches, seq_len)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {"input_ids": seq, "length": len(seq)}


def main():
    parser = argparse.ArgumentParser(
        description="Train a Hidden Markov Model on Shakespeare text"
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--seq_len", type=int, default=32, help="Sequence length for training"
    )
    parser.add_argument(
        "--n_hidden", type=int, default=256, help="Number of hidden states in HMM"
    )
    parser.add_argument(
        "--epochs", type=int, default=250, help="Number of epochs to train"
    )
    parser.add_argument(
        "--accum_grad",
        type=int,
        default=1,
        help="Number of gradients to accumulate",
    )
    args = parser.parse_args()

    # Initialize wandb
    exp_name = f"h{args.n_hidden}_b{args.batch_size}_sl{args.seq_len}_lr{args.lr:g}_ag{args.accum_grad}"
    wandb.init(project="hmm-shakespeare", name=exp_name, config=vars(args))

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = ShakespeareDataset(tokenizer, seq_len=args.seq_len)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True
    )

    model = HMM(M=len(tokenizer), N=args.n_hidden)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.00001)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params/1e6:.2f}M")
    for epoch in range(args.epochs):
        for idx, batch in enumerate(dataloader):
            x, T = batch["input_ids"], batch["length"]
            logp = model(x, T)
            loss = -logp.mean()
            loss.backward()

            if (idx + 1) % args.accum_grad == 0:
                optimizer.step()
                optimizer.zero_grad()

            wandb.log({"train/batch_loss": loss.item()})

            if idx % 10 == 0:
                sample = tokenizer.decode(model.sample(args.seq_len)[0])
                print(
                    f"[Epoch {epoch}][{idx}/{len(dataloader)}] Loss: {loss.item():.2f} | {repr(sample)}"
                )


if __name__ == "__main__":
    main()
