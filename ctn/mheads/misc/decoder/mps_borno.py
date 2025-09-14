import torch

from ctn.mheads._abc import (
    AbstractDisributionHead,
    AbstractDisributionHeadConfig,
    AbstractDisributionHeadOutput,
)
from ctn.mheads.mps import MPS
from ctn.mheads.mps_born import BornMachine
from ctn.mheads.tensorops.mps import select_margin_mps_tensor_batched
from ctn.mheads.tensorops.mps_born import (
    batch_born_mps_select,
    batch_ortho_born_mps_marginalize,
)


def print_tens_stats(t: torch.Tensor, name: str):
    """Prints one line of stats for a tensor."""
    print(
        f"{name}: mean: {t.mean():.2f} ± {t.std():.2f}, min: {t.min():.2f}, max: {t.max():.2f}"
    )


# TODO: instead of a separate generate, maybe handle the case when y is not given as generate
class MPSBO(MPS):
    def __init__(self, config: AbstractDisributionHeadConfig):
        """Simple multi-head distribution with independent linear heads for each position."""
        super().__init__(config)
        H, R, Di, Do = (
            config.horizon,
            config.rank,
            config.d_model,
            config.d_output,
        )

        # std_fan_in = torch.sqrt(torch.tensor(2.0)) / Di**0.5
        # self.w_mps = torch.nn.Parameter(torch.randn(H, R, Do, R, Di) * std_fan_in)
        self.w_hr = torch.nn.Parameter(torch.randn(H, R, Do, Di))
        self.b_mps = torch.nn.Parameter(torch.zeros(H, R, Do, R))

        dtype = self.w_hr.dtype
        self.alpha = torch.nn.Parameter(
            torch.nn.functional.one_hot(torch.tensor(0), num_classes=R).to(dtype),
            requires_grad=False,
        )
        self.beta = torch.nn.Parameter(
            torch.nn.functional.one_hot(torch.tensor(0), num_classes=R).to(dtype),
            requires_grad=False,
        )

        self.sig = lambda x: x**2

    def set_output_embeddings(self, embeddings: torch.nn.Parameter):
        raise NotImplementedError("set_output_embeddings not implemented")

    def get_output_embeddings(self):
        raise NotImplementedError("get_output_embeddings not implemented")

    def freeze_decoder(self):
        raise NotImplementedError("freeze_decoder not implemented")

    def w_mps(self, x: torch.Tensor):
        # (H, R, Do, Di), (*, Di) -> v: (..., H, R, Do)
        v = torch.einsum("...i,hroi->...hro", x, self.w_hr)

        # ||v||^2 along R, per (…, h, o): shape (..., H, 1, Do, 1) for clean broadcasting
        v_norm2 = (v**2).sum(dim=-2, keepdim=True).clamp_min(1e-8).unsqueeze(-1)

        # vv^T keeping Do in the middle: (..., H, R, Do, R)
        vvT = torch.einsum("...hro,...hso->...hros", v, v) / v_norm2

        R = v.size(-2)
        # Identity with Do in the middle: (..., H, R, Do, R)
        I = (
            torch.eye(R, device=v.device, dtype=v.dtype)
            .view(*([1] * (vvT.ndim - 3)), R, 1, R)
            .expand_as(vvT)
        )

        # Householder: H = I - 2 * vv^T / ||v||^2
        g = I - 2.0 * vvT
        return g

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

            # g = torch.einsum("bi,hpoqi->bhpoq", x, self.w_mps) + self.b_mps
            g = self.w_mps(x)
            a = self.alpha.unsqueeze(0).expand(B, -1)
            b = self.beta.unsqueeze(0).expand(B, -1)
            p_tilde = batch_born_mps_select(g, a, b, y)
            z_tilde = batch_ortho_born_mps_marginalize(g, a, b)

            loss = (1 / H) * (  # avg across seq dimension
                -torch.log(p_tilde) + torch.log(z_tilde)  # (B,)  # (B,)
            ).mean()  # avg across batch dimension

        return AbstractDisributionHeadOutput(
            logits=torch.randn(B, H, V),
            loss=loss,
            loss_dict=loss_dict,
        )


def run_test():
    B, H, D, V = 4, 28 * 28, 9, 2
    mt_head = MPSBO(
        AbstractDisributionHeadConfig(
            d_model=D,
            d_output=V,
            horizon=H,
            rank=8,
            pos_func="abs",
        ),
    )
    x = torch.randn(B, D)
    y = torch.randint(0, V, (B, H))
    loss = mt_head(x, y).loss
    print(f"loss: {loss}")
    out = mt_head.generate(x)
    print(f"generated: {out}")


if __name__ == "__main__":
    run_test()
