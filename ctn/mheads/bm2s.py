import torch
from tqdm import tqdm
from ctn.mheads._abc import (
    AbstractDisributionHead,
    AbstractDisributionHeadConfig,
    AbstractDisributionHeadOutput,
)
import torch.nn.functional as F

from ctn.mheads.tensorops.mps import select_margin_mps_tensor_batched


def pad_and_stack(seq, max_len):
    """Pad and stack a sequence of tensors.

    Args:
        seq (list): Sequence of tensors. Shape: (Di,) x T
        max_len (int): Maximum length of the sequence.

    Returns:
        torch.Tensor: Padded and stacked sequence.
    """
    padded = []
    mask = []
    for x in seq:
        pad_len = max_len - x.size(0)
        padded.append(F.pad(x, (0, pad_len)))
        mask.append(
            torch.cat(
                [
                    torch.ones_like(x),
                    torch.zeros(pad_len, device=x.device, dtype=x.dtype),
                ]
            )
        )
    return torch.stack(padded), torch.stack(mask)


import torch
import torch.nn.functional as F


def pad_and_stack2d(seq, max_len=None, pad_value=0.0):
    """Pad and stack a list of 2D (Di x Di) tensors.

    Args:
        seq: list[Tensor], each (Di, Di)
        max_len: int or None — if None, inferred from seq
        pad_value: scalar fill value for padding

    Returns:
        padded: (T, max_len, max_len)
        mask:   (T, max_len, max_len) bool (True=valid)
    """
    if not seq:
        raise ValueError("seq must be non-empty.")

    if max_len is None:
        max_len = max(x.size(0) for x in seq)

    padded, masks = [], []
    for x in seq:
        h, w = x.shape
        if h > max_len or w > max_len:
            raise ValueError(f"item {h}x{w} exceeds max_len={max_len}")
        pad_h, pad_w = max_len - h, max_len - w

        # data
        padded.append(F.pad(x, (0, pad_w, 0, pad_h), value=pad_value))
        # mask: ones on data region, zeros on padding
        m = torch.ones_like(x, dtype=torch.bool)
        masks.append(F.pad(m, (0, pad_w, 0, pad_h), value=False))

    return torch.stack(padded), torch.stack(masks)


def compute_lr_marginalization_cache(g: torch.nn.ParameterList, pad: bool = True):
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
        # Map: (R, Do, R') -> (R, R')
        gh_y = g[h].sum(dim=1)  # (R, R')
        lh = torch.einsum("i,ij->j", lh, gh_y)
        left.append(lh)

    # Map: (R, Do, 1) -> (R,)
    rh = g[-1].sum(dim=1).squeeze(-1)  # (R,)
    right = [torch.ones_like(rh), rh]
    for h in range(H - 2, 0, -1):
        gh_y = g[h].sum(dim=1)  # (R, R')
        rh = torch.einsum("ij,j->i", gh_y, rh)
        right.append(rh)
    right.reverse()

    if pad:
        r_max = max([x.size(0) for x in left + right])
        left_m, left_mask = pad_and_stack(left, max_len=r_max)
        right_m, right_mask = pad_and_stack(right, max_len=r_max)

        return (
            left_m,
            left_mask,
            right_m,
            right_mask,
        )  # (H, Rmax), (H, Rmax), (H, Rmax), (H, Rmax)
    return torch.stack(left), torch.stack(right)


def compute_lr_marginalization_cache_bm(g: torch.nn.ParameterList, pad: bool = True):
    """Compute the left and right marginalization caches for the Born machine.

    Args:
        g (torch.Tensor): MPS cores. Shape: (Rh-1, Do, Rh) x H

    Returns:
        torch.Tensor: Left term. Shape: (H, R).
        torch.Tensor: Right term. Shape: (H, R).
    """
    _, H = g[1].shape[0], len(g)
    dt, dv = g[0].dtype, g[0].device

    # Map: (1, Do, R) -> (R,)
    lh = torch.einsum("idj,pdq->jq", g[0], g[0])
    R0 = g[0].shape[0]
    left = [
        torch.einsum(
            "i,j->ij",
            torch.ones(R0, dtype=dt, device=dv),
            torch.ones(R0, dtype=dt, device=dv),
        ),
        lh,
    ]
    for h in range(1, H - 1):
        lh = torch.einsum("ip,idj,pdq->jq", lh, g[h], g[h])
        left.append(lh)

    # Map: (R, Do, 1) -> (R,)
    rh = torch.einsum("idj,pdq->ip", g[-1], g[-1])
    RH = g[-1].shape[0]
    right = [
        torch.einsum(
            "i,j->ij",
            torch.ones(RH, dtype=dt, device=dv),
            torch.ones(RH, dtype=dt, device=dv),
        ),
        rh,
    ]
    for h in range(H - 2, 0, -1):
        rh = torch.einsum("idj,pdq,jq->ip", g[h], g[h], rh)
        right.append(rh)
    right.reverse()

    if pad:
        r_max = max([x.size(0) for x in left + right])
        left_m, left_mask = pad_and_stack2d(left, max_len=r_max)
        right_m, right_mask = pad_and_stack2d(right, max_len=r_max)

        return (
            left_m,
            left_mask,
            right_m,
            right_mask,
        )  # (H, Rmax, Rmax), (H, Rmax, Rmax), (H, Rmax, Rmax), (H, Rmax, Rmax)
    return torch.stack(left), torch.stack(right)


def compute_lr_selection_cache(
    g: torch.nn.ParameterList, y: torch.Tensor, pad: bool = True
):
    """Compute the left and right selection caches for the Born machine.

    Args:
        g (torch.Tensor): MPS cores. Shape: (Rh-1, Do, Rh) x H
        y (torch.Tensor): Target tensor. Shape: (H,).

    Returns:
        torch.Tensor: Left term. Shape: (H, R).
        torch.Tensor: Right term. Shape: (H, R).
    """
    R0, R1, H = g[0].shape[0], g[0].shape[-1], len(g)
    y_slct = y.reshape(1, -1, 1).expand(R0, -1, R1)  # (1, H, R1)

    # Map: (R, Do, R) -> (R, 1, R) -> (R, R)
    lh = g[0].gather(dim=1, index=y_slct[:1, :1]).squeeze(0, 1)  # (R,)
    left = [torch.ones_like(lh), lh]
    for h in range(1, H - 1):
        Rh, Rhp1 = g[h].shape[0], g[h + 1].shape[0]
        y_slct = y.reshape(1, -1, 1).expand(Rh, -1, Rhp1)  # (R, H, R)
        gh_y = (
            g[h].gather(dim=1, index=y_slct[:, h : h + 1, :]).squeeze(1)
        )  # (Rh, Rh+1)
        lh = torch.einsum("i,ij->j", lh, gh_y)  # (Rh,)
        left.append(lh)

    # Map: (R, Do, R) -> (R, 1, R) -> (R, R)
    RHm1, RH = g[-1].shape[0], g[-1].shape[-1]
    y_slct = y.reshape(1, -1, 1).expand(RHm1, -1, RH)  # (1, H, R1)
    rh = g[-1].gather(dim=1, index=y_slct[:, -1:, :1]).squeeze(1, 2)  # (R,)
    right = [torch.ones_like(rh), rh]
    for h in range(H - 2, 0, -1):
        Rh, Rhp1 = g[h].shape[0], g[h + 1].shape[0]
        y_slct = y.reshape(1, -1, 1).expand(Rh, -1, Rhp1)  # (1, H, R1)
        gh_y = (
            g[h].gather(dim=1, index=y_slct[:, h : h + 1, :]).squeeze(1)
        )  # (Rh, Rhp1)
        rh = torch.einsum("ij,j->i", gh_y, rh)
        right.append(rh)
    right.reverse()

    if pad:
        r_max = max([x.size(0) for x in left + right])
        left_m, left_mask = pad_and_stack(left, max_len=r_max)
        right_m, right_mask = pad_and_stack(right, max_len=r_max)

        return (
            left_m,
            left_mask,
            right_m,
            right_mask,
        )  # (H, Rmax), (H, Rmax), (H, Rmax), (H, Rmax)
    return torch.stack(left), torch.stack(right)


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

    def train_example(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_sweeps: int = 1,
        eps_clamp: float = 1e-10,
        lr: float = 1e-3,
        eps_trunc: float = 1e-10,
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
        sl, sr, ml, mr = self.build_cache(x, y)

        going_right = True
        losses = []
        # BUG: m[k] = 0 for all k, m[0] should never be 0 and always be 1
        for n_sweeps in range(n_sweeps):
            for h in range(H - 1) if going_right else range(H - 1, -1, -1):
                Rl, Rr = g[h].size(0), g[h + 1].size(-1)

                # Freeze all cores except h
                for k in range(H):
                    g[k].requires_grad = True if k == h else False

                # Map: (R, D, R) -> (R, B, R)
                yh = y[:, h].reshape(1, -1, 1).expand(g[h].size(0), -1, g[h].size(-1))
                gh_yh = g[h].gather(dim=1, index=yh)  # (R, B, R)
                psi = torch.einsum("bi,ibj,bj->b", sl[h], gh_yh, sr[h])
                z = torch.einsum("ip,idj,pdq,jq->", ml[h], g[h], g[h], mr[h])
                g_tilde = torch.einsum("ivj,jdk->ivdk", g[h], g[h + 1])

                # if ((psi**2) > z).any():
                #     print("WARNING: psi > z")
                #     print(f"psi > z: {psi} > {z}")

                loss = (
                    z.clamp(min=eps_clamp).log()
                    - 2 * (psi).abs().clamp(min=eps_clamp).log().mean()
                )
                losses.append(loss)
                # if loss < 0:
                #     print(f"Loss is negative: {loss}")
                #     print(f"z: {z}")
                #     print(f"psi: {psi}")
                #     print(f"g_tilde: {g_tilde}")
                #     print(f"sl: {sl}")
                #     print(f"sr: {sr}")
                #     print(f"ml: {ml}")
                #     print(f"mr: {mr}")

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
                        # NOTE: we changed gh and gh+1 so left[h+1:], right[:h+1] are invalid for both select/margin caches.
                        # But, we only need to use h+1 in the next iteration.
                        # So we can update sl[h+1] and ml[h+1] only. At then end of the sweep sl, ml will be valid for all h.
                        # But sr, mr will be invalid everywhere. Thankfully we wont need sr, mr on initial leftwards sweep.
                        sl[h + 1] = torch.einsum("bi,idj->bj", sl[h], g[h])
                        ml[h + 1] = torch.einsum("ip,idj,pdq->jq", ml[h], g[h], g[h])

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

        return losses


def train_example():
    B, H, D, V = 32, 5, 9, 2
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
    for i in range(32):
        losses = mt_head.train_example(x, y, lr=1e-3)
        print(torch.stack(losses).mean())


if __name__ == "__main__":
    # set seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    train_example()
