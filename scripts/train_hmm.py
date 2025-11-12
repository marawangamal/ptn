import torch


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

    # log_A_expanded = torch.stack([log_A] * p, dim=2)
    # log_B_expanded = torch.stack([log_B] * m, dim=0)
    # fix for PyTorch > 1.5 by egaznep on Github:
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
        """
        log_alpha : Tensor of shape (batch size, N)
        Multiply previous timestep's alphas by transition matrix (in log domain)
        """
        log_transition_matrix = torch.nn.functional.log_softmax(
            self.unnormalized_transition_matrix, dim=0
        )

        # Matrix multiplication in the log domain
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
    """
    Hidden Markov Model with discrete observations.
    """

    def __init__(self, M, N):
        super(HMM, self).__init__()
        self.M = M  # number of possible observations
        self.N = N  # number of states

        # A
        self.transition_model = TransitionModel(self.N)

        # b(x_t)
        self.emission_model = EmissionModel(self.N, self.M)

        # pi
        self.unnormalized_state_priors = torch.nn.Parameter(torch.randn(self.N))

        # use the GPU
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

        # sample initial state
        z_t = torch.distributions.categorical.Categorical(state_priors).sample().item()
        z = []
        x = []
        z.append(z_t)
        for t in range(0, T):
            # sample emission
            x_t = (
                torch.distributions.categorical.Categorical(emission_matrix[z_t])
                .sample()
                .item()
            )
            x.append(x_t)

            # sample transition
            z_t = (
                torch.distributions.categorical.Categorical(transition_matrix[:, z_t])
                .sample()
                .item()
            )
            if t < T - 1:
                z.append(z_t)

        return x, z

    def forward(self, x, T):
        """
        x : IntTensor of shape (batch size, T_max)
        T : IntTensor of shape (batch size)

        Compute log p(x) for each example in the batch.
        T = length of each example
        """
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

        # Select the sum for the final timestep (each x may have different length).
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

        # Read Shakespeare text
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


# Train loop
import torch
from transformers import AutoTokenizer

# Hyperparameters
lr = 1e-2
batch_size = 32
seq_len_train = 32
seq_len_test = 32
n_hidden = 256

# Data
tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset = ShakespeareDataset(tokenizer, seq_len=seq_len_train)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
model = HMM(M=len(tokenizer), N=n_hidden)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)

# Train loop
for epoch in range(250):
    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        x, T = batch["input_ids"], batch["length"]
        logp = model(x, T)
        loss = -logp.mean()
        loss.backward()
        optimizer.step()

        if idx % 10 == 0:
            sample = tokenizer.decode(model.sample(seq_len_test)[0])
            print(
                f"[Epoch {epoch}][{idx}/{len(dataloader)}] Loss: {loss.item():.2f} | {repr(sample)}"
            )
