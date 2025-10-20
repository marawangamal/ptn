import torch


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
        self.merged = dict()

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
        self.merged[block_position] = g_tilde
        self.merged[block_position].requires_grad_()

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
        merged = self.merged.pop(block_position)
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
        B, N, D = x.size(0), x.size(1), self.physical_dim
        psi_tilde = torch.ones(B, 1)
        i = 0
        dv, dt = x.device, next(self.parameters()).dtype
        while i < N:
            # g[i] is (R, D, R)
            # x[:, i, :] is (B, D)
            # x_slct is (R, B, R)
            if i in self.merged:
                g_tilde = self.merged[i]
                Rl, Rr = g_tilde.size(0), g_tilde.size(-1)
                # g_tilde is (Rl, D, D, Rr)
                xl = torch.nn.functional.one_hot(x[:, i], num_classes=D).to(dv, dt)
                xr = torch.nn.functional.one_hot(x[:, i + 1], num_classes=D).to(dv, dt)
                gi_y = torch.einsum(
                    "idvj, bd, bv -> ibj", g_tilde, xl, xr
                )  # (Rl, B, Rr)
                i += 2
            else:
                Rl, Rr = self.g[i].size(0), self.g[i].size(-1)
                x_slct = x[:, i].reshape(1, B, 1).expand(Rl, -1, Rr)
                gi_y = self.g[i].gather(dim=1, index=x_slct)  # (Rl, B, Rr)
                i += 1

            psi_tilde = torch.einsum("bi, ibj -> bj", psi_tilde, gi_y)  # (B, Rr)


if __name__ == "__main__":
    mps = MPS(n_cores=4, physical_dim=2, bond_dim=3)
    mps.merge_block(0)
    # mps.unmerge_block(0, 0.5, "left")
    mps(torch.randint(0, 2, (32, 4)))
