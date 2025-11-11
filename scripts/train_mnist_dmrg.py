from typing import List, Optional
import math as m
import torch
import torchvision
from torchvision import transforms

from ptn.dists.tensorops.mps import mps_norm

torch.set_default_dtype(torch.float64)


def mps_select(
    g: List[torch.Tensor],
    x: torch.Tensor,
    merged: dict,
    use_scale_factors: bool = True,
    norm: str = "l2",
):
    """Select operator for MPS.
    Args:
        g (List[torch.Tensor]): List of core tensors.
        x (torch.Tensor): Input tensor.
        merged (dict): Dictionary of merged core tensors.
        use_scale_factors (bool): Whether to use scale factors.

    Returns:
        psi_tilde (torch.Tensor): Selected tensor of shape (B, 1).
    """
    B, N, D = x.size(0), x.size(1), g[0].size(1)
    dv, dt = x.device, g[0].dtype
    psi_tilde = torch.ones(B, 1, device=dv, dtype=dt)
    i = 0
    scale_factors = []
    norm_fn = {
        "l2": torch.linalg.norm,
        "linf": torch.amax,
    }[norm]
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
        if use_scale_factors:
            sf = norm_fn(psi_tilde.abs())
            scale_factors.append(sf)
            psi_tilde = psi_tilde / sf
    return psi_tilde, torch.stack(scale_factors)


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
    # for h in range(1, H - 1):
    h = 0
    while h < H:
        if str(h) in merged:
            g_tilde = merged[str(h)]
            L = torch.einsum("pdvq,pr,rdvs ->qs", g_tilde, L, g_tilde)
            h += 2
        else:
            L = torch.einsum("pdq,pr,rds ->qs", g[h], L, g[h])
            h += 1
        if use_scale_factors:
            sf = norm_fn(L.abs())
            scale_factors.append(sf)
            L = L / sf
    L = torch.einsum("pq,pdr,qds ->", L, g[-1], g[-1])
    if not use_scale_factors:
        scale_factors = [torch.tensor(1.0)]
    return L, torch.stack(scale_factors)  # (1,), (H,)


def get_dataloaders(batch_size=32, data_dir="./data", scale=None):
    """Create MNIST data loaders with binary thresholding."""
    transform = torchvision.transforms.Compose(
        [transforms.ToTensor(), lambda x: (x > 0.5).long()]
    )
    if scale is not None:
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (scale, scale)
                ),  # rescale from 28x28 -> (scale, scale)
                transforms.ToTensor(),
                lambda x: (x > 0.5).long(),  # binarize
            ]
        )
    train_set = torchvision.datasets.MNIST(
        data_dir, train=True, transform=transform, download=True
    )
    val_set = torchvision.datasets.MNIST(
        data_dir, train=False, transform=transform, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader


def rando(d, num):
    "Make `num` d-dim orthogonal vectors. If `num` is greater than `d`, make copies then scale."
    q = torch.linalg.qr(torch.randn(d, d))[0]
    if num > d:
        reps = num // d
        q = q.repeat(1, reps)
    return q[:, :num]


class MPS(torch.nn.Module):
    def __init__(
        self,
        n_cores: int,
        physical_dim: int,
        bond_dim: int,
        init_method: str = "ortho",  # "ortho" or "randn"
        verbose: bool = False,
    ):
        super().__init__()
        self.n_cores = n_cores
        self.physical_dim = physical_dim
        self.bond_dim = bond_dim
        self.merged = torch.nn.ParameterDict()
        self.init_method = init_method
        self.verbose = verbose

        # Initialize parameters
        self.g = torch.nn.ParameterList(self._get_init_params())

        if verbose:
            bond_dims = [(g.size(0), g.size(-1)) for g in self.g]
            print("Bond dimensions per core:")
            for idx, (left, right) in enumerate(bond_dims):
                print(f"  Core {idx}: left = {left}, right = {right}")

    def _get_init_params(self):
        N, D, R = self.n_cores, self.physical_dim, self.bond_dim
        plist = []
        if self.init_method == "ortho":
            # NOTE: At initialization, we are in right-canonical format, so we can go rightwards.
            # NOTE: ortho init must start w/ rank = physical dim
            R = D
            for h in range(N):
                if h == 0:
                    w = rando(d=D, num=R)  # will always have w.T @ w = I
                    plist.append(torch.nn.Parameter(w.reshape(1, D, R)))
                elif h == N - 1:
                    w = rando(d=D, num=R)  # will always have w.T @ w = I
                    plist.append(torch.nn.Parameter(w.T.reshape(R, D, 1)))
                else:
                    w = rando(d=D * R, num=R)  # will always have w.T @ w = I
                    plist.append(torch.nn.Parameter(w.T.reshape(R, D, R)))
        elif self.init_method == "randn":
            plist = (
                [torch.nn.Parameter(torch.randn(1, D, R), requires_grad=False)]
                + [
                    torch.nn.Parameter(torch.randn(R, D, R), requires_grad=False)
                    for _ in range(N - 2)
                ]
                + [torch.nn.Parameter(torch.randn(R, D, 1), requires_grad=False)]
            )
        return plist

    def __repr__(self):
        core_shapes = [tuple(g.shape) for g in self.g]
        merged = {int(k): tuple(v.shape) for k, v in self.merged.items()}
        out = []
        i = 0
        while i < self.n_cores:
            if str(i) in self.merged:
                out.append(f"|{merged[i]}|")
                i += 2
            else:
                out.append(str(core_shapes[i]))
                i += 1
        return ",".join(out)

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
        self,
        block_position: int,
        cum_percentage: float,
        side: str = "right",
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

        psi_tilde, gammas_psi = mps_select(self.g, x, self.merged)  # (B, 1)
        z_tilde, gammas_z = mps_norm(self.g, self.merged)  # (1,), (N,)
        loss = (
            z_tilde.log()
            - 2 * psi_tilde.abs().log().nanmean()
            + gammas_z.log().sum(dim=-1)
            - 2 * gammas_psi.log().sum(dim=-1)
        )

        # Health checks
        if not psi_tilde.isfinite().all():
            print(f"NaN in psi_tilde: {psi_tilde}")
            raise ValueError(f"NaN in psi_tilde: {psi_tilde}")
        if not z_tilde.isfinite().all():
            print(f"NaN in z_tilde: {z_tilde}")
            raise ValueError(f"NaN in z_tilde: {z_tilde}")
        if not gammas_psi.isfinite().all():
            print(f"NaN in gammas_psi: {gammas_psi}")
            raise ValueError(f"NaN in gammas_psi: {gammas_psi}")
        if not gammas_z.isfinite().all():
            print(f"NaN in gammas_z: {gammas_z}")
            raise ValueError(f"NaN in gammas_z: {gammas_z}")
        return loss


if __name__ == "__main__":

    # HPs
    N_EPOCHS_PER_CORE = 5  # num epoch per core
    N_SWEEPS = 4
    LR = 1e-3
    RANK = 2
    PHYSICAL_DIM = 2
    SCALE = 28
    N_CORES = SCALE**2
    BATCH_SIZE = 8192
    PRINT_FREQ = 10
    SITE_LENGTH = 2  # number of cores in a merged block (do not change this)
    CUM_PERCENTAGE = 0.99  # cumulative percentage of singular values to keep

    # Data
    # x = torch.randint(0, PHYSICAL_DIM, (BATCH_SIZE, N_CORES))
    train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE, scale=SCALE)

    # Model
    mps = MPS(n_cores=N_CORES, physical_dim=PHYSICAL_DIM, bond_dim=RANK)
    mps.merge_block(0)

    # Optimizer
    optimizer = torch.optim.AdamW(mps.parameters(), lr=LR)
    dv = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mps.to(dv)

    # Training loop
    direction = 1
    block_position = 0
    get_bond_dims = lambda mps: list(set(g.size(0) for g in mps.g))
    print(f"Training for {N_EPOCHS_PER_CORE * N_CORES} epochs on device {dv}")
    for epoch in range(N_EPOCHS_PER_CORE * N_CORES * N_SWEEPS):
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            x, _ = batch
            loss = mps(x.to(dv).reshape(-1, N_CORES))
            loss.backward()
            optimizer.step()

        print(
            f"[Epoch {epoch}] loss: {loss.item():.2f} | Dir: {direction} | Block: {block_position} | Bond Dims: {get_bond_dims(mps)}"
        )

        if epoch > 0 and epoch % N_EPOCHS_PER_CORE == 0:
            dir_old = direction

            if block_position + direction + SITE_LENGTH > N_CORES:
                direction *= -1
            if block_position + direction < 0:
                direction *= -1
            if SITE_LENGTH == N_CORES:
                direction = 0

            # print(f"Direction changed from {dir_old} to {direction}. Model state: {mps}")

            if direction >= 0:
                mps.unmerge_block(
                    block_position, cum_percentage=CUM_PERCENTAGE, side="left"
                )
            else:
                mps.unmerge_block(
                    block_position, cum_percentage=CUM_PERCENTAGE, side="right"
                )

            block_position += direction
            mps.merge_block(block_position)
            optimizer = torch.optim.AdamW(mps.parameters(), lr=LR)
