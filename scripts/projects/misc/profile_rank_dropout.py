from typing import List
import torch
import pandas as pd

from ptn.tensorops.mps import select_margin_mps_tensor_batched


def reconstruct_cores(glist: List[List[torch.Tensor]], rank: int) -> torch.Tensor:
    slct = lambda g, ids: [g[k] for k in ids]
    ids = torch.randperm(len(glist))[:rank]
    outlist = []
    for glist_i in slct(glist, ids):
        glist_ii = slct(glist_i, ids)
        outlist.append(torch.stack(glist_ii))
    gtens = torch.stack(outlist)  # (R, R, H, Di, Do)
    # reshape to (H, R, Di, R, Do)
    gtens = gtens.permute(2, 0, 3, 1, 4)
    return gtens


class RDModel(torch.nn.Module):
    def __init__(
        self,
        rank: int,
        rank_dropout: float,
        horizon: int,
        d_output: int,
        d_input: int = 1,
    ):
        super().__init__()
        self.rank = rank
        self.rank_dropout = rank_dropout

        # Create parameter list
        self._g = torch.nn.ParameterList()
        for _ in range(rank):
            plist = torch.nn.ParameterList()
            for _ in range(rank):
                plist.append(
                    torch.nn.Parameter(torch.randn(horizon, d_output, d_input))
                )
            self._g.append(plist)
        self._a = torch.nn.Parameter(torch.randn(rank))
        self._b = torch.nn.Parameter(torch.randn(rank))

    @property
    def g(self) -> torch.Tensor:
        rank_eff = int(self.rank * self.rank_dropout)
        return reconstruct_cores(self._g, rank_eff)

    @property
    def a(self) -> torch.Tensor:
        n_rank = int(self.rank * self.rank_dropout)
        ids = torch.randperm(self.rank)[:n_rank]
        return self._a[ids]

    @property
    def b(self) -> torch.Tensor:
        n_rank = int(self.rank * self.rank_dropout)
        ids = torch.randperm(self.rank)[:n_rank]
        return self._b[ids]

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (B, d_input)
        # y: (B, d_output)
        B = x.size(0)
        g = torch.einsum("hpoqi,bi->bhpoq", self.g, x)
        a = self.a.reshape(1, -1).expand(B, -1)
        b = self.b.reshape(1, -1).expand(B, -1)
        res = select_margin_mps_tensor_batched(a, b, g, y)
        return res


def forward_pass(
    rank: int = 32,
    rank_dropout: float = 1.0,
    horizon: int = 8,
    d_output: int = 8,
    batch_size: int = 128,
    d_input: int = 1,
    device: str = "cuda",
    **kwargs,
):
    device = torch.device(device)
    model = RDModel(rank, rank_dropout, horizon, d_output, d_input)
    model.to(device)
    x = torch.randn(batch_size, d_input, device=device)
    y = torch.randint(0, d_output, (batch_size, horizon), device=device)
    res = model(x, y)
    return res


def main():

    # Setup
    torch.manual_seed(12)
    # HPs
    rank = 32
    horizon = 8
    d_output = 8

    rows = []
    exps = [
        {
            "rank_dropout": 1.0,
            "rank": rank,
            "d_output": d_output,
            "horizon": horizon,
        },
        {
            "rank_dropout": 0.1,
            "rank": rank,
            "d_output": d_output,
            "horizon": horizon,
        },
    ]
    for exp in exps:
        # Reset peak memory stats after model creation but before forward pass
        # torch.cuda.reset_peak_memory_stats()

        # ***** Memory measurement start *****

        forward_pass(**exp)

        # ***** Memory measurement end *****

        mem_after = torch.cuda.max_memory_allocated()
        rows.append({"mem_mb": mem_after / (1024**2), **exp})

    df = pd.DataFrame(rows)
    print(df)


if __name__ == "__main__":
    main()
