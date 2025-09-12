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


# TODO: instead of a separate generate, maybe handle the case when y is not given as generate
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

        # std_fan_in = torch.sqrt(torch.tensor(2.0)) / Do**0.5
        self.w_mps = torch.nn.Parameter(torch.randn(H, R, Do, R))
        self.is_canonical = False
        self.canonical_index = [config.horizon // 2]
        if self.canonical_index[0] % 2 == 0:
            self.canonical_index.insert(0, self.canonical_index[0] - 1)

        dtype = self.w_mps.dtype
        self.alpha = torch.nn.Parameter(
            torch.nn.functional.one_hot(torch.tensor(0), num_classes=R).to(dtype),
            requires_grad=False,
        )
        self.beta = torch.nn.Parameter(
            torch.nn.functional.one_hot(torch.tensor(0), num_classes=R).to(dtype),
            requires_grad=False,
        )

        self.canonicalize()

    def set_output_embeddings(self, embeddings: torch.nn.Parameter):
        raise NotImplementedError("set_output_embeddings not implemented")

    def get_output_embeddings(self):
        raise NotImplementedError("get_output_embeddings not implemented")

    def freeze_decoder(self):
        raise NotImplementedError("freeze_decoder not implemented")

    def canonicalize(self):
        with torch.no_grad():
            h = 0
            while h < self.config.horizon - 1:
                if h not in self.canonical_index:
                    a_4oder = torch.einsum(
                        "idj,jvk->idvk", self.w_mps[h], self.w_mps[h + 1]
                    )
                    R1, V1, V2, R2 = a_4oder.shape
                    Rm = self.w_mps[h].shape[2]  # middle rank
                    u, s, vt = torch.svd(a_4oder.reshape(R1 * V1, V2 * R2))
                    u, s, vt = u[:, :Rm], s[:Rm], vt[:Rm]  # truncate
                    if h < min(self.canonical_index):  # left cano
                        self.w_mps[h] = u.reshape(R1, V1, Rm)
                        self.w_mps[h + 1] = (s.unsqueeze(-1) * vt).reshape(Rm, V2, R2)
                    else:
                        self.w_mps[h] = (u * s.unsqueeze(0)).reshape(R1, V1, Rm)
                        self.w_mps[h + 1] = vt.reshape(Rm, V2, R2)
                h += 1
            self.is_canonical = True

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

            g = self.w_mps
            a = self.alpha.unsqueeze(0).expand(B, -1)
            b = self.beta.unsqueeze(0).expand(B, -1)
            p_tilde = batch_born_mps_select(
                g.unsqueeze(0).expand(B, -1, -1, -1, -1), a, b, y
            )
            if len(self.canonical_index) == 1:
                z_tilde = torch.einsum(
                    "idj,idj->", g[self.canonical_index[0]], g[self.canonical_index[0]]
                )
            else:
                i0, i1 = self.canonical_index
                z_tilde = torch.einsum("idj,jvr,idq,qvr->", g[i0], g[i1], g[i0], g[i1])

            loss = (1 / H) * (  # avg across seq dimension
                -torch.log(p_tilde) + torch.log(z_tilde)  # (B,)  # (B,)
            ).mean()  # avg across batch dimension

            self.is_canonical = False

        return AbstractDisributionHeadOutput(
            logits=torch.randn(B, H, V),
            loss=loss,
            loss_dict=loss_dict,
        )

    def generate(self, x: torch.Tensor):
        pass


def run_test():
    B, H, D, V = 1, 28, 9, 2
    mt_head = BornMachineUnconditional(
        AbstractDisributionHeadConfig(
            d_model=D,
            d_output=V,
            horizon=H,
            rank=8,
        ),
    )
    x = torch.randn(B, D)
    y = torch.randint(0, V, (B, H))
    loss = mt_head(x, y).loss
    # print(f"loss: {loss}")
    # out = mt_head.generate(x)
    # print(f"generated: {out}")


if __name__ == "__main__":
    run_test()
