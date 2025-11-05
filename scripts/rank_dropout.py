from typing import Optional
import torch
from dataclasses import dataclass
import numpy as np
import pandas as pd

from ptn.dists._abc import AbstractDisributionHeadConfig
from ptn.dists import dists


# Converts flat indices to multi-index
def torch_unravel_index(indices, shape):
    id_tuples = np.unravel_index(indices, shape)
    return torch.stack([torch.tensor(t) for t in id_tuples], dim=-1)


class RandomDiagonalDataset(torch.utils.data.Dataset):
    def __init__(self, vocab_size, seq_length, num_samples):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.samples = torch.randint(
            low=0, high=self.vocab_size, size=(self.num_samples,)
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Each element in the sequence is identical, e.g., [0,0] or [1,1], etc.
        value = self.samples[idx].item()
        input_ids = torch.full((self.seq_length,), fill_value=value, dtype=torch.long)
        target = input_ids.clone()
        return {"input_ids": input_ids, "labels": target}


@dataclass
class SimpleModelOutput:
    loss: torch.Tensor


class SimpleModel(torch.nn.Module):
    def __init__(
        self,
        d_output: int,
        rank: int,
        pos_func: str = "abs",
        mode: str = "factorized",
        pos_func_mode: str = "factorized",
        rank_dropout: Optional[
            float
        ] = None,  # probability of a rank dim being dropped out
    ):
        super().__init__()
        self.d_output = d_output
        self.rank = rank
        self.rank_dropout = rank_dropout
        self.mode = mode
        self.pos_func = pos_func
        self.pos_func_mode = pos_func_mode

        if self.mode == "factorized":
            # Decompose into two rank-r matrices
            self._p1 = torch.nn.Parameter(torch.randn(d_output, rank))
            self._p2 = torch.nn.Parameter(torch.randn(rank, d_output))
        elif self.mode == "full":
            # Model full matrix
            self._p_tilde = torch.nn.Parameter(torch.randn(d_output, d_output))
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    @property
    def p_tilde(self):  # Materializes unnormalized dist (d_output, d_output)
        pos_func = {
            "abs": torch.abs,
            "exp": torch.exp,
            "sigmoid": torch.sigmoid,
        }[self.pos_func]
        ids = torch.arange(self.rank)
        if self.rank_dropout is not None:
            n_rank = int(self.rank * self.rank_dropout or 1)
            ids = torch.randperm(self.d_output)
            ids = ids[:n_rank]
        if self.mode == "factorized":
            if self.pos_func_mode == "factorized":
                p_tilde = pos_func(self._p1[:, ids]) @ pos_func(self._p2[ids, :])
            else:
                p_tilde = pos_func(self._p1[:, ids] @ self._p2[ids, :])
        elif self.mode == "full":
            p_tilde = pos_func(self._p_tilde)
        return p_tilde

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape of x: (B, 2)
        ids = x[:, 0] * self.d_output + x[:, 1]  # (B,)
        p = self.p_tilde
        loss = (
            p.sum().clamp(min=1e-12).log() - p.reshape(-1)[ids].clamp(min=1e-12).log()
        )
        return loss.sum()

    def sample(self, n_samples: int) -> torch.Tensor:
        p_tilde = self.p_tilde
        p = p_tilde.reshape(-1)
        ids = torch.multinomial(p, n_samples, replacement=True)  # (n_samples,)
        return torch_unravel_index(ids, (self.d_output, self.d_output))


def train_simple_model(
    # --- Data configuration ---
    dataloader: torch.utils.data.DataLoader,
    # --- Model configuration ---
    d_output: int = 16,
    pos_func: str = "sigmoid",
    mode: str = "factorized",
    pos_func_mode: str = "factorized",
    horizon: int = 2,
    rank: int = 16,
    rank_dropout: Optional[float] = None,
    # --- Training configuration ---
    n_epochs: int = 20_000,
    lr: float = 1e-3,
    device: str = "cuda",
    **kwargs,
):
    # model = SimpleModel(
    #     d_output=d_output,
    #     rank=rank,
    #     pos_func=pos_func,
    #     mode=mode,
    #     pos_func_mode=pos_func_mode,
    #     # rank_dropout=rank_dropout,
    # )
    model = dists["mps_sigma_lsf"](
        config=AbstractDisributionHeadConfig(
            rank=rank,
            rank_dropout=rank_dropout,
            d_model=1,
            d_output=d_output,
            horizon=horizon,
        )
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        train_losses = []
        for i, batch in enumerate(dataloader):
            x = batch["input_ids"].to(device)

            # Simple model
            # loss = model(x)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # Distribution head
            output = model(torch.ones(x.shape[0], 1, device=device), x)
            loss = output.loss
            train_losses.append(loss.item())

        loss_avg = sum(train_losses) / len(train_losses)
        if epoch % 5_000 == 0:
            print(f"[Epoch {epoch}/{n_epochs}] Loss: {loss_avg:.2f}")
    return model


def main():

    # Setup
    torch.manual_seed(12)
    num_samples = 512
    batch_size = 128
    d_output = 256
    device = "cuda"
    rank = 128
    horizon = 8
    dataset = RandomDiagonalDataset(
        d_output, num_samples=num_samples, seq_length=horizon
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    rows = []
    exps = [
        {
            "rank_dropout": 1.0,
            "n_epochs": 1,
            "device": device,
            "rank": rank,
            "d_output": d_output,
            "horizon": horizon,
        },
        {
            "rank_dropout": 0.1,
            "n_epochs": 1,
            "device": device,
            "rank": rank,
            "d_output": d_output,
            "horizon": horizon,
        },
    ]
    for exp in exps:
        # Reset peak memory stats after model creation but before forward pass
        # torch.cuda.reset_peak_memory_stats()

        # ***** Memory measurement start *****

        train_simple_model(dataloader, **exp)

        # ***** Memory measurement end *****

        mem_after = torch.cuda.max_memory_allocated()
        rows.append({"mem_mb": mem_after / (1024**2), **exp})

    df = pd.DataFrame(rows)
    print(df)


if __name__ == "__main__":
    main()
