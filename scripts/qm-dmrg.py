from typing import List
import torch

from ptn.dists.tensorops.mps import mps_norm


def mps_select(g: List[torch.Tensor], x: torch.Tensor, merged: dict) -> torch.Tensor:
    B, N, D = x.size(0), x.size(1), g[0].size(1)
    psi_tilde = torch.ones(B, 1)
    i = 0
    dv, dt = x.device, g[0].dtype
    while i < N:
        # g[i] is (R, D, R)
        # x[:, i, :] is (B, D)
        # x_slct is (R, B, R)
        if str(i) in merged:
            g_tilde = merged[str(i)]
            Rl, Rr = g_tilde.size(0), g_tilde.size(-1)
            # g_tilde is (Rl, D, D, Rr)
            xl = torch.nn.functional.one_hot(x[:, i], num_classes=D).to(dv, dt)
            xr = torch.nn.functional.one_hot(x[:, i + 1], num_classes=D).to(dv, dt)
            gi_y = torch.einsum("idvj, bd, bv -> ibj", g_tilde, xl, xr)  # (Rl, B, Rr)
            i += 2
        else:
            Rl, Rr = g[i].size(0), g[i].size(-1)
            x_slct = x[:, i].reshape(1, B, 1).expand(Rl, -1, Rr)
            gi_y = g[i].gather(dim=1, index=x_slct)  # (Rl, B, Rr)
            i += 1
        psi_tilde = torch.einsum("bi, ibj -> bj", psi_tilde, gi_y)  # (B, Rr)
    return psi_tilde


def mps_norm(
    g: List[torch.Tensor],
    merged: dict,
    use_scale_factors: bool = True,
    norm: str = "l2",
):
    H = len(g)
    scale_factors = []
    norm_fn = {
        "l2": torch.linalg.norm,
        "linf": torch.amax,
    }[norm]
    L = torch.einsum("pdq, rds->qs", g[0], g[0])
    for h in range(1, H - 1):
        if str(h) in merged:
            g_tilde = merged[str(h)]
            L = torch.einsum("pdvq,pr,rdvs ->qs", g_tilde, L, g_tilde)
        else:
            L = torch.einsum("pdq,pr,rds ->qs", g[h], L, g[h])
        if use_scale_factors:
            sf = norm_fn(L.abs())
            scale_factors.append(sf)
            L = L / sf
    L = torch.einsum("pq,pdr,qds ->", L, g[-1], g[-1])
    if not use_scale_factors:
        scale_factors = [torch.tensor(1.0)]
    return L, torch.stack(scale_factors)  # (1,), (H,)


class MPS(torch.nn.Module):
    def __init__(
        self,
        n_cores: int,
        physical_dim: int,
        bond_dim: int,
    ):
        super().__init__()
        self.n_cores = n_cores
        self.physical_dim = physical_dim
        self.bond_dim = bond_dim
        self.merged = torch.nn.ParameterDict()

        # Initialize parameters
        R, D = bond_dim, physical_dim
        self.g = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(1, D, R), requires_grad=False)]
            + [
                torch.nn.Parameter(torch.randn(R, D, R), requires_grad=False)
                for _ in range(n_cores - 2)
            ]
            + [torch.nn.Parameter(torch.randn(R, D, 1), requires_grad=False)]
        )
        bond_dims = [(g.size(0), g.size(-1)) for g in self.g]
        print("Bond dimensions per core:")
        for idx, (left, right) in enumerate(bond_dims):
            print(f"  Core {idx}: left = {left}, right = {right}")

    def merge_block(self, block_position: int):
        # position range: 0: (0, 1), 1: (1, 2), ..., n_cores - 2: (n_cores - 2, n_cores - 1)
        assert (
            0 <= block_position < self.n_cores - 1
        ), f"Block position {block_position} is out of range [0, {self.n_cores - 1})"
        g_tilde = torch.einsum(
            "idj, jvk -> idvk", self.g[block_position], self.g[block_position + 1]
        )
        # g_tilde.detach().requires_grad_()  # mark merged core as trainable
        # self.merged[block_position] = g_tilde
        # self.merged[block_position].requires_grad_()
        self.merged.update(
            {str(block_position): torch.nn.Parameter(g_tilde, requires_grad=True)}
        )

    def unmerge_block(
        self, block_position: int, cum_percentage: float, side: str = "right"
    ):
        """Unmerge a block at a given position.

        Args:
            block_position (int): Position of the block to unmerge.
            cum_percentage (float): Cumulative percentage of the singular values to keep.
            side (str, optional):
                Indicates the side to which the diagonal matrix should be contracted.
                If "left", the first resultant node's tensor will be U @ S and the other node's tensor will be V^T.
                If "right", their tensors will be U and S @ V^T, respectively.
        """
        # position range: 0: (0, 1), 1: (1, 2), ..., n_cores - 2: (n_cores - 2, n_cores - 1)
        assert (
            0 <= block_position < self.n_cores - 1
        ), f"Block position {block_position} is out of range [0, {self.n_cores - 1})"
        merged = self.merged.pop(str(block_position))
        p, d, v, r = merged.shape
        u, s, vt = torch.linalg.svd(merged.reshape(p * d, v * r))

        # Truncate singular values
        n_components = (
            ((torch.cumsum(s, dim=-1) / s.sum()) >= cum_percentage)
            .float()
            .argmax()
            .item()
        ) + 1
        s = s[:n_components]
        u = u[:, :n_components]
        vt = vt[:n_components]

        # Compute new cores
        if side == "left":
            gl = torch.einsum(
                "idj, jk -> idk", u.reshape(p, d, n_components), torch.diag(s)
            )
            gr = vt.reshape(n_components, v, r)
        else:
            gl = u.reshape(p, d, n_components)
            gr = torch.einsum(
                "ij, jdk -> idk", torch.diag(s), vt.reshape(n_components, v, r)
            )

        # Reset cores
        self.g[block_position].data = gl
        self.g[block_position + 1].data = gr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute loss w.r.t active merged core.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N)

        Returns:
            loss (torch.Tensor): Loss of shape (1,)
        """
        assert x.ndim == 2, "Input must be 3D (B, N)"
        assert (
            x.size(1) == self.n_cores
        ), "Input must have the same number of cores as the model"

        psi_tilde = mps_select(self.g, x, self.merged)
        z_tilde, gammas_z = mps_norm(self.g, self.merged)
        loss = z_tilde.log() - psi_tilde.log().nanmean() + gammas_z.log().sum(dim=-1)
        return loss


if __name__ == "__main__":
    mps = MPS(n_cores=4, physical_dim=2, bond_dim=3)
    mps.merge_block(0)
    # mps.unmerge_block(0, 0.5, "left")
    x = torch.randint(0, 2, (32, 4))
    # loss = mps(torch.randint(0, 2, (32, 4)))
    # print(loss)
    optimizer = torch.optim.Adam(mps.parameters(), lr=1e-3)
    for i in range(10000):
        optimizer.zero_grad()
        loss = mps(x)
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print(f"Iteration {i}: loss = {loss.item():.4f}")
