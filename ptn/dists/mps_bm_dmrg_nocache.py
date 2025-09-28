import torch
from ptn.dists._abc import (
    AbstractDisributionHead,
    AbstractDisributionHeadConfig,
    AbstractDisributionHeadOutput,
)
import torch.nn.functional as F


def rando(d, num):
    "Make `num` d-dim orthogonal vectors. If `num` is greater than `d`, make copies then scale."
    q = torch.linalg.qr(torch.randn(d, d))[0]
    if num > d:
        reps = num // d
        q = q.repeat(1, reps)
    return q[:, :num]


class MPS_BM_DMRG(AbstractDisributionHead):
    """Born Machine (Canonical Form w/ DMRG).

    MPS based Born Machine probabilistic.

    .. math::
        p(y1, y2, ..., yH) = \\left(G0[y1] G1[y2] ... GH[yH] \\right) **2 / Z

    where :math:`G0, G1, ..., GH` are the MPS cores and :math:`Z` is the partition function.

    .. math::
        Z = \\sum_{y1, y2, ..., yH} \\left(G0[y1] G1[y2] ... GH[yH] \\right) **2


    """

    def __init__(self, config: AbstractDisributionHeadConfig):
        super().__init__(config)
        if not config.ignore_canonical:
            config.rank = 2  # initially set to rank 2
        H, R, Di, Do = (
            config.horizon,
            config.rank,
            config.d_model,
            config.d_output,
        )
        plist = []
        # NOTE: At initialization, we are in right-canonical format, so we can go rightwards.
        for h in range(H):
            if h == 0:
                w = rando(d=Do, num=R)  # will always have w.T @ w = I
                plist.append(torch.nn.Parameter(w.reshape(1, Do, R)))
            elif h == H - 1:
                w = rando(d=Do, num=R)  # will always have w.T @ w = I
                plist.append(torch.nn.Parameter(w.T.reshape(R, Do, 1)))
            else:
                w = rando(d=Do * R, num=R)  # will always have w.T @ w = I
                # plist.append(torch.nn.Parameter(rando(R, Do * R).reshape(R, Do, R)))
                plist.append(torch.nn.Parameter(w.T.reshape(R, Do, R)))
        self.g = torch.nn.ParameterList(plist)

        self.assert_right_canonical()

    def __repr__(self):
        bond_dims = [self.g[h].shape[0] for h in range(len(self.g))]
        bond_dims.append(self.g[-1].shape[-1])
        return f"MPS_BM_DMRG(bond_dims={bond_dims})"

    def assert_right_canonical(self):
        if self.config.ignore_canonical:
            return
        for h in range(len(self.g) - 1, 0, -1):
            w = self.g[h].reshape(self.g[h].size(0), -1)  # (R, Do*R)
            assert torch.allclose(w @ w.T, torch.eye(w.size(0)), atol=1e-6)
        # print("[PASS] MPS is in right-canonical format")

    def forward(
        self,
        x,
        y=None,
        ignore_index: int = -100,
        return_logits: bool = False,
        eps_clamp: float = 1e-10,
    ):
        # Input validation
        assert x.ndim == 2, "x must be 2D (B, D)"
        assert y is None or y.ndim == 2, "y must be 2D (B, H)"
        assert (
            y is None or y.size(1) == self.config.horizon
        ), f"Incorrect y horizon, must of shape (B, {self.config.horizon}) but got {y.shape}"
        dt, dv = x.dtype, x.device

        B, R, H, V = (
            x.size(0),
            self.config.rank,
            self.config.horizon,
            self.config.d_output,
        )
        loss = None
        self.sig = (
            lambda x: x
        )  # left for user to override (i.e. when using born machine)
        sl, sr, ml, mr = self.build_cache(x, y)
        g = self.g

        if y is not None:
            h = H - 1

            # Map: (R, D, R) -> (R, B, R)
            yh = y[:, h].reshape(1, -1, 1).expand(g[h].size(0), -1, g[h].size(-1))
            gh_yh = g[h].gather(dim=1, index=yh)  # (R, B, R)
            psi = torch.einsum("bi,ibj,bj->b", sl[h], gh_yh, sr[h])
            z = torch.einsum("ip,idj,pdq,jq->", ml[h], g[h], g[h], mr[h])

            # if ((psi**2) > z).any():
            #     print("WARNING: psi > z")
            #     print(f"psi > z: {psi} > {z}")

            loss = (
                z.clamp(min=eps_clamp).log()
                - 2 * (psi).abs().clamp(min=eps_clamp).log().mean()
            )

            return AbstractDisributionHeadOutput(loss=loss, logits=torch.randn(B, H, V))

        return AbstractDisributionHeadOutput(loss=loss, logits=torch.randn(B, H, V))

    def build_cache(self, x, y):
        # Precompute L and R
        B, R, Do, H = (
            x.shape[0],
            self.config.rank,
            self.config.d_output,
            self.config.horizon,
        )
        dt, dv = x.dtype, x.device
        g = self.g  # MPS cores list H x (R, Do, R)

        # Precomputes left/right selection and marginalization caches
        # NEW:
        sl, sl_mask, sr, sr_mask = batch_compute_lr_selection_terms(
            g, y
        )  # (B, H, Rmax), (B, H, Rmax), (B, H, Rmax), (B, H, Rmax)
        ml, ml_mask, mr, mr_mask = compute_lr_marginalization_cache_bm(
            g
        )  # (H, R', R'), (H, R', R'), (H, R', R'), (H, R', R')

        # Convert caches to lists as ranks can change throughout sweep and break tensor shapes
        sl = [
            sl[:, h, : sl_mask[0, h].sum().long()] for h in range(H)
        ]  # (H, B, R) -> H x (B, R)
        sr = [
            sr[:, h, : sr_mask[0, h].sum().long()] for h in range(H)
        ]  # (H, B, R) -> H x (B, R)
        ml = [
            ml[h, : ml_mask[h][:, 0].sum().long(), : ml_mask[h][:, 0].sum().long()]
            for h in range(H)
        ]  # (H, R) -> H x (R,)
        mr = [
            mr[h, : mr_mask[h][:, 0].sum().long(), : mr_mask[h][:, 0].sum().long()]
            for h in range(H)
        ]  # (H, R) -> H x (R,)

        # Boundary Corrections:
        sl[0] = sl[0][:, :1]
        ml[0] = ml[0][:1]
        sr[-1] = sr[-1][:, :1]
        mr[-1] = mr[-1][:1]

        return sl, sr, ml, mr

    def get_environment(self, h):
        pass

    def train_example(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_sweeps: int = 1,
        eps_clamp: float = 1e-10,
        lr: float = 1e-3,
        eps_trunc: float = 1e-32,
        max_bond_dim: int = 32,
    ):
        """Train the model for one step.

        Args:
            x (torch.Tensor): Input tensor. Shape: (B, H).
            y (torch.Tensor): Target tensor. Shape: (B, H).
            optimizer (torch.optim.Optimizer): Optimizer.
            n_sweeps (int, optional): Number of sweeps. Defaults to 1.

        Note: we don't use x since this is an unconditional model.

        Returns:
            torch.Tensor: Loss.
        """
        # Precompute L and R
        B, R, Do, H = (
            x.shape[0],
            self.config.rank,
            self.config.d_output,
            self.config.horizon,
        )
        dt, dv = x.dtype, x.device
        g = self.g  # MPS cores list H x (R, Do, R)
        # sl, sr, ml, mr = self.build_cache(x, y)

        going_right = True
        losses = []
        # BUG: m[k] = 0 for all k, m[0] should never be 0 and always be 1
        for n_sweeps in range(n_sweeps):
            for h in range(1, H - 1) if going_right else range(H - 1, -1, -1):
                Rl, Rr = g[h].size(0), g[h + 1].size(-1)

                # Freeze all cores except h
                for k in range(H):
                    g[k].requires_grad = True if k == h else False

                # Map: (R, D, R) -> (R, B, R)
                yh = y[:, h].reshape(1, -1, 1).expand(g[h].size(0), -1, g[h].size(-1))
                gh_yh = g[h].gather(dim=1, index=yh)  # (R, B, R)
                psi = torch.einsum("bi,ibj,bj->b", sl_h, gh_yh, sr_h)
                z = torch.einsum("ip,idj,pdq,jq->", ml_h, g[h], g[h], mr_h)
                g_tilde = torch.einsum("ivj,jdk->ivdk", g[h], g[h + 1])

                loss = (
                    z.clamp(min=eps_clamp).log()
                    - 2 * (psi).abs().clamp(min=eps_clamp).log().mean()
                )
                losses.append(loss)

                z_prime = 2 * g_tilde  # (Rl, Do, Do Rr)
                i_h = torch.eye(self.config.d_output, device=dv)[y[:, h]]  # (B,)
                i_hp1 = torch.eye(self.config.d_output, device=dv)[y[:, h + 1]]  # (B,)
                psi_prime = torch.einsum(
                    "bi,bd,bv,bj->bidvj", sl[h], i_h, i_hp1, sr[h + 1]
                )  # (B, Rl, Do, Do, Rr)

                # Shape:  (Rl, Do, Do, Rr)
                dldg_tilde = (z_prime / z.clamp(min=eps_clamp)) - 2 * (
                    psi_prime / psi.view(-1, 1, 1, 1, 1).clamp(min=eps_clamp)
                ).mean(dim=0)
                g_tilde = g_tilde - dldg_tilde * lr  # (Rl, Do, Do Rr)
                u, s, vt = torch.linalg.svd(
                    g_tilde.reshape(Rl * Do, Do * Rr), full_matrices=True
                )

                if not g_tilde.isfinite().all():
                    print(f"NaN in g_tilde")
                    raise ValueError("NaN in g_tilde")

                # Now we need to re-canonicalize and update left / right caches
                with torch.no_grad():
                    # NOTE: We are going right, so everything in front is right canonicalized. However, we need to
                    # leave our wake left canonicalized as we pass through so that we are ready to go leftwards at the
                    # end of the rightwards sweep.
                    if going_right:
                        Rtrunc = min(
                            max_bond_dim, (s / s.abs().max() > eps_trunc).sum() or 1
                        )
                        g[h] = u.reshape(Rl, Do, u.size(-1))[
                            :, :, :Rtrunc
                        ]  # (R, Do, R)
                        g[h + 1] = torch.einsum(
                            "ir,rvj->ivj",
                            torch.diag(s[:Rtrunc]),
                            vt[:Rtrunc].reshape(Rtrunc, Do, Rr),
                        )
                        # normalize g[h+1]
                        g[h + 1] = g[h + 1] / g[h + 1].norm(dim=1, keepdim=True).clamp(
                            min=eps_clamp
                        )
        return losses


def train_example():
    B, H, D, V = 32, 32, 9, 2
    mt_head = MPS_BM_DMRG(
        AbstractDisributionHeadConfig(
            d_model=D,
            d_output=V,
            horizon=H,
            rank=2,
        ),
    )
    x = torch.randn(B, D)
    y = torch.randint(0, V, (B, H))
    for i in range(1000):
        losses = mt_head.train_example(x, y, lr=1e-4)
        print(f"loss: {torch.stack(losses).mean():.2f}")


if __name__ == "__main__":
    # set seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    train_example()
