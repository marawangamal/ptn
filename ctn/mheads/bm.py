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
    left = [torch.zeros_like(lh), lh]
    for h in range(1, H - 1):
        # Map: (R, Do, R) -> (R, R)
        gh_y = g[h].sum(dim=1)  # (R, R)
        lh = torch.einsum("i,ij->j", lh, gh_y)
        left.append(lh)

    # Map: (R, Do, 1) -> (R,)
    rh = g[-1].sum(dim=1).squeeze(-1)  # (R,)
    right = [torch.zeros_like(rh), rh]
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
    left = [torch.zeros_like(lh), lh]
    for h in range(1, H - 1):
        gh_y = g[h].gather(dim=1, index=y_slct[:, h : h + 1, :]).squeeze(1)  # (R, R)
        lh = torch.einsum("i,ij->j", lh, gh_y)
        left.append(lh)

    # Map: (R, Do, R) -> (R, 1, R) -> (R, R)
    rh = g[-1].gather(dim=1, index=y_slct[:, -1:, :1]).squeeze(1, 2)  # (R,)
    right = [torch.zeros_like(rh), rh]
    for h in range(H - 2, 0, -1):
        gh_y = g[h].gather(dim=1, index=y_slct[:, h : h + 1, :]).squeeze(1)  # (R, R)
        rh = torch.einsum("ij,j->i", gh_y, rh)
        right.append(rh)
    right.reverse()

    return torch.stack(left), torch.stack(right)  # (H, R)


batch_compute_lr_selection_terms = torch.vmap(
    compute_lr_selection_cache, in_dims=(None, 0)
)


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
        H, R, Di, Do = (
            config.horizon,
            config.rank,
            config.d_model,
            config.d_output,
        )
        plist = []
        # Right-canonical format on init
        for h in range(H):
            if h == 0:
                plist.append(torch.nn.Parameter(torch.eye(1, Do * R).reshape(1, Do, R)))
            elif h == H - 1:
                plist.append(torch.nn.Parameter(torch.eye(R, Do * 1).reshape(R, Do, 1)))
            else:
                plist.append(torch.nn.Parameter(torch.eye(R, Do * R).reshape(R, Do, R)))
        self.g = torch.nn.ParameterList(plist)

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
        Do, H = self.config.d_output, self.config.horizon
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
            for h in range(H) if going_right else range(H - 1, -1, -1):
                optimizer.zero_grad()
                # Freeze all cores except h
                for k in range(H):
                    g[k].requires_grad = True if k == h else False

                # Map: (R, D, R) -> (R, B, R)
                yh = y[:, h].reshape(1, -1, 1).expand(g[h].size(0), -1, g[h].size(-1))
                gh_yh = g[h].gather(dim=1, index=yh)  # (R, B, R)
                psi_y = torch.einsum("bi,ibj,bj->b", sl[h], gh_yh, sr[h])
                z = torch.einsum("i,ij,j->", ml[h], self.g_dot(h), mr[h])
                loss = z.clamp(min=eps).log() - psi_y.pow(2).clamp(min=eps).log()
                loss.backward()
                optimizer.step()

                # Now we need to re-canonicalize and update left / right caches
                with torch.no_grad():
                    # NOTE: We are going right, so everything in front is right canonicalized. However, we need to
                    # leave our wake left canonicalized as we pass through so that we are ready to go leftwards at the
                    # end of the rightwards sweep.
                    if going_right:
                        Rl, Rr = g[h].size(0), g[h].size(-1)
                        U, s, Vt = torch.linalg.svd(g[h].reshape(Rl * Do, Rr))
                        R_prime = U.size(-1)
                        g[h] = U.reshape(Rl, Do, R_prime)[:, :, :Rr]  # (R, Do, R)
                        g[h + 1] = torch.einsum(
                            "ir,rp,pvj->ivj", torch.diag(s), Vt[: len(s)], g[h + 1]
                        )[:Rr]
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
    B, H, D, V = 1, 28 * 28, 9, 2
    mt_head = BM(
        AbstractDisributionHeadConfig(
            d_model=D,
            d_output=V,
            horizon=H,
            rank=8,
        ),
    )
    x = torch.randn(B, D, dtype=torch.float64)
    y = torch.randint(0, V, (B, H))
    optimizer = torch.optim.AdamW(mt_head.parameters(), lr=1e-3)
    mt_head.train_example(x, y, optimizer)


if __name__ == "__main__":
    train_example()
