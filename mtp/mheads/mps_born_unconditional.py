from typing import List
import torch

from mtp.mheads._abc import (
    AbstractDisributionHead,
    AbstractDisributionHeadConfig,
    AbstractDisributionHeadOutput,
)
from mtp.mheads.tensorops.mps import select_margin_mps_tensor_batched
from mtp.mheads.tensorops.mps_born import (
    batch_born_mps_marginalize,
    batch_born_mps_select,
)


# def compute_lr_marginal_terms(g: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
#     """Compute the left and right terms for the Born machine loss."""
#     H, R, Do, _ = g.shape  # (H, R, Do, R)
#     dt, dv = g.dtype, g.device
#     left, right = torch.zeros(H, R, dtype=dt, device=dv), torch.ones(
#         H, R, dtype=dt, device=dv
#     )
#     left[0], right[-1] = a, b

#     # Map: (H, R, Do, R) -> (H, R, 1, R) -> (H, R, R)
#     g_dot = g.sum(dim=2)
#     for h in range(1, H):
#         left[h] = torch.einsum("i,ij->j", left[h - 1], g_dot[h - 1])

#     for h in range(H - 2, -1, -1):
#         right[h] = torch.einsum("ij,j->i", g_dot[h + 1], right[h + 1])

#     return left, right


def compute_lr_marginal_terms(g: torch.nn.ParameterList):
    """Compute the left and right terms for the Born machine loss."""
    R, H = g[1].shape[0], len(g)
    dt, dv = g[1].dtype, g[1].device
    left, right = torch.zeros(H, R, dtype=dt, device=dv), torch.ones(
        H, R, dtype=dt, device=dv
    )
    left[0], right[-1] = g[0].sum(dim=1), g[-1].sum(dim=1)

    # Map: (H, R, Do, R) -> (H, R, 1, R) -> (H, R, R)
    for h in range(1, H):
        left[h] = torch.einsum("i,ij->j", left[h - 1], g[h].sum(-2))

    for h in range(H - 2, -1, -1):
        right[h] = torch.einsum("ij,j->i", g[h].sum(-2), right[h + 1])

    return left, right


def gather_g(g: torch.Tensor, y: torch.Tensor):
    """_summary_

    Args:
        g (torch.Tensor): MPS core. Shape: (Rh-1, Do, Rh)
        y (torch.Tensor): Target tensor. Shape: (1,).
    """


def compute_lr_terms(g: torch.nn.ParameterList, y: torch.Tensor):
    """Compute the left and right terms for the Born machine loss.

    Args:
        g (torch.Tensor): MPS cores. Shape: (Rh-1, Do, Rh) x H
        y (torch.Tensor): Target tensor. Shape: (H,).

    Returns:
        torch.Tensor: Left term. Shape: (H, R).
        torch.Tensor: Right term. Shape: (H, R).
    """
    R, H = g[1].shape[0], len(g)
    y_slct = y.reshape(1, -1, 1).expand(R, -1, R)  # (R, H, R)

    # Map: (H, R, Do, R) -> (H, R, 1, R) -> (H, R, R)
    lh = g[0].gather(dim=1, index=y_slct[:1, :1]).squeeze(0, 1)  # (R,)
    left = [torch.zeros_like(lh), lh]
    for h in range(1, H - 1):
        # Map: (R, Do, R) -> (R, 1, R) -> (R, R)
        gh_y = g[h].gather(dim=1, index=y_slct[:, h : h + 1, :]).squeeze(1)  # (R, R)
        lh = torch.einsum("i,ij->j", lh, gh_y)
        left.append(lh)

    rh = g[-1].gather(dim=1, index=y_slct[:, -1:, :1]).squeeze(1, 2)  # (R,)
    right = [torch.zeros_like(rh), rh]
    for h in range(H - 2, 0, -1):
        gh_y = g[h].gather(dim=1, index=y_slct[:, h : h + 1, :]).squeeze(1)  # (R, R)
        rh = torch.einsum("ij,j->i", gh_y, rh)
        right.append(rh)
    right.reverse()

    return torch.stack(left), torch.stack(right)  # (H, R)


# def batch_compute_lr_terms(g: torch.nn.ParameterList, y: torch.Tensor):
#     """Compute the left and right terms for the Born machine loss.

#     Args:
#         g (torch.Tensor): MPS cores. Shape: (Rh-1, Do, Rh) x H
#         y (torch.Tensor): Target tensor. Shape: (B, H).

#     Returns:
#         torch.Tensor: Left term. Shape: (H, R).
#         torch.Tensor: Right term. Shape: (H, R).
#     """
#     B, R, H = y.shape[0], g[1].shape[0], y.shape[1]
#     dt, dv = g[1].dtype, g[1].device
#     left, right = torch.zeros(B, H, R, dtype=dt, device=dv), torch.ones(
#         B, H, R, dtype=dt, device=dv
#     )
#     # left[0], right[-1] = g[0][:, y[0]], g[-1][:, y[-1]]
#     y_slct = y.reshape(B, 1, -1, 1).expand(-1, R, -1, R)  # (B, R, H, R)
#     # Map: (1, Do, R) -> (1, 1, R) -> (1, R)
#     left[:1] = g[0].gather(dim=1, index=y_slct[:1, :1]).squeeze(1)  # (R,)
#     # Map: (R, Do, R) -> (R, 1, 1) -> (R, 1)
#     right[-1:] += g[-1].gather(dim=1, index=y_slct[:, -1:, :1]).squeeze(1)  # (R,)

#     # Map: (H, R, Do, R) -> (H, R, 1, R) -> (H, R, R)
#     for h in range(1, H):
#         # Map: (R, Do, R) -> (R, 1, R) -> (R, R)
#         ghm1_y = g[h - 1].gather(dim=1, index=y_slct[:, h : h + 1, :])  # (R, R)
#         left[h] = torch.einsum("i,ij->j", left[h - 1], ghm1_y)

#     for h in range(H - 2, -1, -1):
#         ghp1_y = g[h + 1].gather(dim=1, index=y_slct[:, h + 1 : h + 2, :])  # (R, R)
#         right[h] = torch.einsum("ij,j->i", ghp1_y, right[h + 1])

#     return left, right


# def compute_lr_terms(
#     g: torch.Tensor, a: torch.Tensor, b: torch.Tensor, y: torch.Tensor
# ):
#     """Compute the left and right terms for the Born machine loss.

#     Args:
#         g (torch.Tensor): MPS cores. Shape: (H, R, Do, R).
#         a (torch.Tensor): Left contraction factor. Shape: (R,).
#         b (torch.Tensor): Right contraction factor. Shape: (R,).
#         y (torch.Tensor): Target tensor. Shape: (H,).

#     Returns:
#         torch.Tensor: Left term. Shape: (H, R).
#         torch.Tensor: Right term. Shape: (H, R).
#     """
#     H, R, Do, _ = g.shape
#     dt, dv = g.dtype, g.device
#     left, right = torch.ones(H, R, dtype=dt, device=dv), torch.ones(
#         H, R, dtype=dt, device=dv
#     )
#     left[0], right[-1] = a, b

#     # Map: (H, R, Do, R) -> (H, R, 1, R) -> (H, R, R)
#     g_y = g.gather(dim=2, index=y.reshape(-1, 1, 1, 1).expand(-1, R, Do, R)).squeeze(2)
#     for h in range(1, H):
#         left[h] = torch.einsum("i,ij->j", left[h - 1], g_y[h - 1])

#     for h in range(H - 2, -1, -1):
#         right[h] = torch.einsum("ij,j->i", g_y[h + 1], right[h + 1])

#     return left, right


batch_compute_lr_select_terms = torch.vmap(compute_lr_terms, in_dims=(None, 0))


class BornMachineUnconditional(AbstractDisributionHead):
    def __init__(self, config: AbstractDisributionHeadConfig):
        """Simple multi-head distribution with independent linear heads for each position."""
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

    def train_example(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        n_sweeps: int = 1,
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
        B, H = y.shape
        R, Do = self.config.rank, self.config.d_output
        # g, a, b = self.g, self.alpha, self.beta  # (H, R, Do, R), (R,), (R,)
        g = self.g
        sl, sr = batch_compute_lr_select_terms(g, y)  # (B, H, R), (B, H, R)
        ml, mr = compute_lr_marginal_terms(g)  # (H, R), (H, R)

        going_right = True
        for n_sweeps in range(n_sweeps):
            for h in range(H) if going_right else range(H - 1, -1, -1):
                for h in range(H):
                    g[h].requires_grad = False if h != h else True  # single site update

                optimizer.zero_grad()
                psi_y = torch.einsum("bi,ij,bj->b", sl[:, h], g[h][:, y[h]], sr[h])
                z = torch.einsum("bi,ij,bj->b", ml[h], g[h].sum(dim=1), mr[h])
                loss = z.log() - psi_y.pow(2).log()
                loss.backward()
                optimizer.step()

                # Now we need to re-canonicalize and update left or right terms
                with torch.no_grad():
                    if going_right:
                        U, s, Vt = torch.linalg.svd(g[h].reshape(R * Do, R))
                        R_prime = U.size(-1)
                        g[h] = U.reshape(R, Do, R_prime)[:, :, :R]  # (R, Do, R)
                        g[h + 1] = torch.einsum("ir,rp,pvj->ivj", s, Vt, g[h + 1])[:R]
                        # NOTE: we changed gh and gh+1 so al[h+1:], ar[:h+1] are invalid for both s,m mats
                        # But, we only need to use h+1 in the next iteration.
                        # So we can update sl[h+1] and ml[h+1] only. At then end of the sweep sl, ml will be valid for all h.
                        # But sr, mr will be invalid everywhere.
                        sl[:, h + 1] = torch.einsum("bi,idj->bj", sl[:, h], g[h])
                        ml[h + 1] = torch.einsum("i,ij->j", ml[h], g[h].sum(dim=1))
                        # sr[:, h] = torch.einsum(
                        #     "ij, bj->bj", self._g[h + 1][:, y[h + 1]], sr[:, h + 1]
                        # )
                        # mr[h] = torch.einsum(
                        #     "ij, bj->bj", self._g[h + 1].sum(dim=1), mr[h + 1]
                        # )

                    else:  # going left
                        U, s, Vt = torch.linalg.svd(g[h].reshape(R, Do * R))
                        R_prime = Vt.size(0)
                        g[h] = Vt.reshape(R_prime, Do, R)[:R]  # (R, Do, R)
                        g[h - 1] = torch.einsum("ivk,kr,rj->ivj", g[h - 1], U, s)[
                            :, :, :R
                        ]
                        # sl[:, h + 1] = torch.einsum("bi,idj->bj", sl[:, h], self._g[h])
                        # ml[h + 1] = torch.einsum(
                        #     "i,ij->j", ml[h], self._g[h].sum(dim=1)
                        # )

    def forward(
        self,
        x,
        y=None,
        ignore_index: int = -100,
        return_logits: bool = False,
    ):
        # Input validation
        assert x.ndim == 2, "x must be 2D (B, D)"
        assert y is None or y.ndim == 2, "y must be 2D (B, H)"
        assert (
            y is None or y.size(1) == self.config.horizon
        ), f"Incorrect y horizon, must of shape (B, {self.config.horizon}) but got {y.shape}"

        B, R, H, V = (
            x.size(0),
            self.config.rank,
            self.config.horizon,
            self.config.d_output,
        )
        loss = None
        loss_dict = {}

        if y is not None:

            if not self.is_canonical:
                self.canonicalize()

            g = self.w_mps.unsqueeze(0).expand(B, -1, -1, -1, -1)
            a = self.alpha.unsqueeze(0).expand(B, -1)
            b = self.beta.unsqueeze(0).expand(B, -1)
            p_tilde, gamma_p = batch_born_mps_select(g, a, b, y, use_scale_factors=True)
            z, gamma_z = batch_born_mps_marginalize(
                g,
                a,
                b,
                # self.canonical_index,  only for cano variant
                use_scale_factors=True,
            )

            loss = (1 / H) * (  # avg across seq dimension
                -torch.log(p_tilde)  # (B,)
                + torch.log(z)  # (B,)
                # Contraction Stability Scale Factors
                - (gamma_p.log().sum(dim=-1))  # (B, H)
                + (gamma_z.log().sum(dim=-1))  # (B, H)
            ).mean()  # avg across batch dimension

            self.is_canonical = False

        return AbstractDisributionHeadOutput(
            logits=torch.randn(B, H, V, dtype=torch.float64),
            loss=loss,
            loss_dict=loss_dict,
        )

    def generate(self, x: torch.Tensor):
        raise NotImplementedError("generate not implemented")


def left_cano(theta_mps: torch.Tensor, canonical_index: List[int]):
    # Shape(theta_mps): (H, R, D, R)
    h = 0
    while h < theta_mps.shape[0] - 1:
        if h not in canonical_index:
            a_4oder = torch.einsum(
                "idj,jvk->idvk", theta_mps[h], theta_mps[h + 1]
            )  # merged tensor A^{h,h+1}
            R1, V1, V2, R2 = a_4oder.shape
            Rm = theta_mps[h].shape[2]  # middle rank
            u, s, vt = torch.linalg.svd(
                a_4oder.reshape(R1 * V1, V2 * R2), full_matrices=True
            )
            u, s, vt = u[:, :Rm], s[:Rm], vt[:Rm]  # truncate
            if h < min(canonical_index):  # left cano
                theta_mps[h] = u.reshape(R1, V1, Rm)
                theta_mps[h + 1] = (s[:, None] * vt).reshape(Rm, V2, R2)
            else:
                theta_mps[h + 1] = vt.reshape(Rm, V2, R2)
                theta_mps[h] = (u * s[None, :]).reshape(R1, V1, Rm)

        h += 1
    return theta_mps


def train_example():
    B, H, D, V = 1, 28 * 28, 9, 2
    mt_head = BornMachineUnconditional(
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


def run_test():
    B, H, D, V = 1, 28 * 28, 9, 2
    mt_head = BornMachineUnconditional(
        AbstractDisributionHeadConfig(
            d_model=D,
            d_output=V,
            horizon=H,
            rank=8,
        ),
    )
    x = torch.randn(B, D, dtype=torch.float64)
    y = torch.randint(0, V, (B, H))
    loss = mt_head(x, y).loss
    print(f"loss: {loss}")
    # out = mt_head.generate(x)
    # print(f"generated: {out}")


if __name__ == "__main__":
    train_example()
