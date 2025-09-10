import torch

from mtp.mheads._abc import AbstractDisributionHeadConfig
from mtp.mheads.mps import MPS
from mtp.mheads.tensorops.mps import select_margin_mps_tensor_batched


def print_tens_stats(t: torch.Tensor, name: str):
    """Prints one line of stats for a tensor."""
    print(
        f"{name}: mean: {t.mean():.2f} ± {t.std():.2f}, min: {t.min():.2f}, max: {t.max():.2f}"
    )


# TODO: instead of a separate generate, maybe handle the case when y is not given as generate
class MPSHR(MPS):
    def __init__(self, config: AbstractDisributionHeadConfig):
        """Simple multi-head distribution with independent linear heads for each position."""
        super().__init__(config)
        H, R, Di, Do = (
            config.horizon,
            config.rank,
            config.d_model,
            config.d_output,
        )

        std_fan_in = torch.sqrt(torch.tensor(2.0)) / Di**0.5
        # self.w_mps = torch.nn.Parameter(torch.randn(H, R, Do, R, Di) * std_fan_in)
        self.w_hr = torch.nn.Parameter(torch.randn(H, R, Do, Di) * std_fan_in)
        self.b_mps = torch.nn.Parameter(torch.zeros(H, R, Do, R) * std_fan_in)

        dtype = self.w_hr.dtype
        self.alpha = torch.nn.Parameter(
            torch.nn.functional.one_hot(torch.tensor(0), num_classes=R).to(dtype),
            requires_grad=False,
        )
        self.beta = torch.nn.Parameter(
            torch.nn.functional.one_hot(torch.tensor(0), num_classes=R).to(dtype),
            requires_grad=False,
        )

    def w_mps(self, x: torch.Tensor):
        # (H, R, Do, Di), (*, Di) -> (*, H, R, Do, R)
        v = torch.einsum("...i,hroi->...hro", x, self.w_hr)  # (*, H, R, Do, R)
        vvt = torch.einsum("...hpo,...hro->...hpor", v, v)  # (*, H, R, D, R)
        R = vvt.size(-1)
        I = (
            torch.eye(R, R)
            .to(vvt.device, vvt.dtype)
            .reshape(1, 1, R, 1, R)
            .expand(*list(vvt.shape))
        )
        g = I - vvt
        return g


def run_test():
    B, H, D, V = 4, 28 * 28, 9, 2
    mt_head = MPSHR(
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
