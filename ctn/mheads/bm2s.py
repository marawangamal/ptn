import torch
from ctn.mheads._abc import AbstractDisributionHead, AbstractDisributionHeadConfig


def compute_lr_marginalization_cache(g: torch.nn.ParameterList):
    """Compute the left and right marginalization caches for the Born machine.

    Args:
        g (torch.Tensor): MPS cores. Shape: (Rh-1, Do, Rh) x H

    Returns:
        torch.Tensor: Left term. Shape: (H, R).
        torch.Tensor: Right term. Shape: (H, R).
    """
    _, H = g[1].shape[0], len(g)

    # Map: (1, Do, R) -> (R,)
    lh = g[0].sum(dim=1).squeeze(0)  # (R,)
    left = [torch.ones_like(lh), lh]
    for h in range(1, H - 1):
        # Map: (R, Do, R) -> (R, R)
        gh_y = g[h].sum(dim=1)  # (R, R)
        lh = torch.einsum("i,ij->j", lh, gh_y)
        left.append(lh)

    # Map: (R, Do, 1) -> (R,)
    rh = g[-1].sum(dim=1).squeeze(-1)  # (R,)
    right = [torch.ones_like(rh), rh]
    for h in range(H - 2, 0, -1):
        gh_y = g[h].sum(dim=1)  # (R, R)
        rh = torch.einsum("ij,j->i", gh_y, rh)
        right.append(rh)
    right.reverse()

    return torch.stack(left), torch.stack(right)  # (H, R)


def compute_lr_selection_cache(g: torch.nn.ParameterList, y: torch.Tensor):
    """Compute the left and right selection caches for the Born machine.

    Args:
        g (torch.Tensor): MPS cores. Shape: (Rh-1, Do, Rh) x H
        y (torch.Tensor): Target tensor. Shape: (H,).

    Returns:
        torch.Tensor: Left term. Shape: (H, R).
        torch.Tensor: Right term. Shape: (H, R).
    """
    R, H = g[1].shape[0], len(g)
    y_slct = y.reshape(1, -1, 1).expand(R, -1, R)  # (R, H, R)

    # Map: (R, Do, R) -> (R, 1, R) -> (R, R)
    lh = g[0].gather(dim=1, index=y_slct[:1, :1]).squeeze(0, 1)  # (R,)
    left = [torch.ones_like(lh), lh]
    for h in range(1, H - 1):
        gh_y = g[h].gather(dim=1, index=y_slct[:, h : h + 1, :]).squeeze(1)  # (R, R)
        lh = torch.einsum("i,ij->j", lh, gh_y)
        left.append(lh)

    # Map: (R, Do, R) -> (R, 1, R) -> (R, R)
    rh = g[-1].gather(dim=1, index=y_slct[:, -1:, :1]).squeeze(1, 2)  # (R,)
    right = [torch.ones_like(rh), rh]
    for h in range(H - 2, 0, -1):
        gh_y = g[h].gather(dim=1, index=y_slct[:, h : h + 1, :]).squeeze(1)  # (R, R)
        rh = torch.einsum("ij,j->i", gh_y, rh)
        right.append(rh)
    right.reverse()

    return torch.stack(left), torch.stack(right)  # (H, R)


batch_compute_lr_selection_terms = torch.vmap(
    compute_lr_selection_cache, in_dims=(None, 0)
)


def rando(d, num):
    "Make `num` d-dim orthogonal vectors. If `num` is greater than `d`, make copies then scale."
    q = torch.linalg.qr(torch.randn(d, d))[0]
    if num > d:
        reps = num // d
        q = q.repeat(1, reps)
    return q[:, :num]


class BM(AbstractDisributionHead):
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

    def assert_right_canonical(self):
        for h in range(len(self.g) - 1, 0, -1):
            w = self.g[h].reshape(self.g[h].size(0), -1)  # (R, Do*R)
            assert torch.allclose(w @ w.T, torch.eye(w.size(0)), atol=1e-6)
        print("[PASS] MPS is in right-canonical format")

    def g_dot(self, h: int):
        return self.g[h].sum(dim=1)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ):
        raise NotImplementedError("Forward does not exist for BornMachineUnconditional")

    def train_example(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        n_sweeps: int = 1,
        eps: float = 1e-10,
        step_size: float = 1e-3,
        eps_trunc: float = 1e-3,
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

        # Precomputes left/right selection and marginalization caches
        sl, sr = batch_compute_lr_selection_terms(g, y)  # (B, H, R), (B, H, R)
        ml, mr = compute_lr_marginalization_cache(g)  # (H, R), (H, R)

        # Convert caches to lists as ranks can change throughout sweep and break tensor shapes
        sl = [x for x in sl.permute(1, 0, 2)]  # (H, B, R) -> H x (B, R)
        sr = [x for x in sr.permute(1, 0, 2)]  # (H, B, R) -> H x (B, R)
        ml = [x for x in ml]  # (H, R) -> H x (R,)
        mr = [x for x in mr]  # (H, R) -> H x (R,)

        going_right = True
        for n_sweeps in range(n_sweeps):
            for h in range(H - 1) if going_right else range(H - 1, -1, -1):
                Rl, Rr = g[h].size(0), g[h].size(-1)

                optimizer.zero_grad()
                # Freeze all cores except h
                for k in range(H):
                    g[k].requires_grad = True if k == h else False

                # Map: (R, D, R) -> (R, B, R)
                yh = y[:, h].reshape(1, -1, 1).expand(g[h].size(0), -1, g[h].size(-1))
                gh_yh = g[h].gather(dim=1, index=yh)  # (R, B, R)
                psi = torch.einsum("bi,ibj,bj->b", sl[h], gh_yh, sr[h])
                z = torch.einsum("i,ij,j->", ml[h], self.g_dot(h), mr[h])
                g_tilde = torch.einsum("ivj,jdk->ivdj", g[h], g[h + 1])

                z_prime = 2 * g_tilde  # (Rl, Do, Do Rr)
                i_h = torch.eye(self.config.d_output)[y[:, h]]  # (B,)
                i_hp1 = torch.eye(self.config.d_output)[y[:, h + 1]]  # (B,)
                if h == 0:
                    psi_prime = torch.einsum(
                        "bi, bd,bv,bj->bidvj",
                        torch.ones(B, 1, dtype=dt, device=dv),
                        i_h,
                        i_hp1,
                        sr[h + 1],
                    )
                else:
                    psi_prime = torch.einsum(
                        "bi,bd,bv,bj->bidvj", sl[h], i_h, i_hp1, sr[h + 1]
                    )  # (B, Rl, Do, Do, Rr)

                # Shape:  (Rl, Do, Do, Rr)
                dldg_tilde = (z_prime / z) - 2 * (psi_prime / psi).mean(dim=0)
                g_tilde = g_tilde - dldg_tilde * step_size  # (Rl, Do, Do Rr)
                u, s, vt = torch.linalg.svd(
                    g_tilde.reshape(Rl * Do, Do * Rr), full_matrices=True
                )

                # Now we need to re-canonicalize and update left / right caches
                with torch.no_grad():
                    # NOTE: We are going right, so everything in front is right canonicalized. However, we need to
                    # leave our wake left canonicalized as we pass through so that we are ready to go leftwards at the
                    # end of the rightwards sweep.
                    if going_right:
                        # Rl, Rr = g[h].size(0), g[h].size(-1)
                        # U, s, Vt = torch.linalg.svd(g[h].reshape(Rl * Do, Rr))
                        # R_prime = U.size(-1)
                        # g[h] = U.reshape(Rl, Do, R_prime)[:, :, :Rr]  # (R, Do, R)
                        # g[h + 1] = torch.einsum(
                        #     "ir,rp,pvj->ivj", torch.diag(s), Vt[: len(s)], g[h + 1]
                        # )[:Rr]

                        Rtrunc = (s / s.abs().max() > eps_trunc).sum()
                        g[h] = u.reshape(Rl, Do, u.size(-1))[
                            :, :, :Rtrunc
                        ]  # (R, Do, R)
                        g[h + 1] = torch.einsum(
                            "ir,rvj->ivj",
                            torch.diag(s[:Rtrunc]),
                            vt[:Rtrunc].reshape(Rtrunc, Do, Rr),
                        )

                        # NOTE: we changed gh and gh+1 so left[h+1:], right[:h+1] are invalid for both select/margin caches.
                        # But, we only need to use h+1 in the next iteration.
                        # So we can update sl[h+1] and ml[h+1] only. At then end of the sweep sl, ml will be valid for all h.
                        # But sr, mr will be invalid everywhere. Thankfully we wont need sr, mr on initial leftwards sweep.
                        sl[h + 1] = torch.einsum("bi,idj->bj", sl[h], g[h])
                        ml[h + 1] = torch.einsum("i,ij->j", ml[h], g[h].sum(dim=1))

                    # TODO: left sweep
                    # else:  # going left
                    #     U, s, Vt = torch.linalg.svd(g[h].reshape(R, Do * R))
                    #     R_prime = Vt.size(0)
                    #     g[h] = Vt.reshape(R_prime, Do, R)[:R]  # (R, Do, R)
                    #     g[h - 1] = torch.einsum("ivk,kr,rj->ivj", g[h - 1], U, s)[
                    #         :, :, :R
                    #     ]
                    #     # sl[:, h + 1] = torch.einsum("bi,idj->bj", sl[:, h], self._g[h])
                    #     # ml[h + 1] = torch.einsum(
                    #     #     "i,ij->j", ml[h], self._g[h].sum(dim=1)
                    #     # )


def train_example():
    B, H, D, V = 1, 5, 9, 2
    mt_head = BM(
        AbstractDisributionHeadConfig(
            d_model=D,
            d_output=V,
            horizon=H,
            rank=8,
        ),
    )
    x = torch.randn(B, D)
    y = torch.randint(0, V, (B, H))
    optimizer = torch.optim.AdamW(mt_head.parameters(), lr=1e-3)
    mt_head.train_example(x, y, optimizer)


if __name__ == "__main__":
    train_example()
