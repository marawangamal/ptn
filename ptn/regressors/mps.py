import torch

from ptn.regressors._abc import (
    AbstractRegressorHead,
    AbstractRegressorHeadConfig,
    AbstractRegressorHeadOutput,
)


def assert_all(*args):
    for arg in args:
        assert arg[0], arg[1]


def mps_gamma_contract(
    g: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    norm: str = "l2",
    **kwargs,
):
    """Contract a MPS tensor using scale factors.

    Args:
        g (torch.Tensor): MPS tensor of shape (H, R, Din, R)
        a (torch.Tensor): Left boundary tensor of shape (R,)
        b (torch.Tensor): Right boundary tensor of shape (R,)
        x (torch.Tensor): Input tensor of shape (H, Din)
    """
    norm_fn = {
        "l2": torch.linalg.norm,
        "linf": torch.amax,
    }[norm]
    gammas = []
    result = a  # (R,)
    for i in range(g.size(0)):
        result = torch.einsum("i,idj,d->j", result, g[i], x[i])  # (R,)
        gammas.append(norm_fn(result, dim=-1))
        result = result / gammas[i]  # (R,)
    result = result @ b  # (R,)
    return result, torch.stack(gammas, dim=-1)  # (1,), (H,)


mps_gamma_contract_batch = torch.vmap(mps_gamma_contract, in_dims=(0, 0, 0, 0))


class MPS_REGRESSOR(AbstractRegressorHead):
    def __init__(self, config: AbstractRegressorHeadConfig):
        super().__init__(config)
        H, R, Di, Do = (
            config.horizon,
            config.rank,
            config.d_in,
            config.d_out,
        )
        self.eps = 1e-12

        std_fan_in = torch.sqrt(torch.tensor(2.0)) / Di**0.5
        self.g = torch.nn.Parameter(torch.randn(Do, H, R, Di, R) * std_fan_in)

        dtype = self.g.dtype
        self.alpha = torch.nn.Parameter(
            torch.nn.functional.one_hot(torch.tensor(0), num_classes=R).to(dtype),
            requires_grad=False,
        )
        self.beta = torch.nn.Parameter(
            torch.nn.functional.one_hot(torch.tensor(0), num_classes=R).to(dtype),
            requires_grad=False,
        )

    def forward(
        self,
        x,  # (B, H, D_in)
        y=None,  # (B, D_out)
        **kwargs,
    ):
        assert_all(
            (x.ndim == 3, f"Expected x to be 3D (B, H, D_in), but got {x.shape}"),
            (
                y is None or y.ndim == 2,
                f"Expected y to be 2D (B, D_out), but got {y.shape}",
            ),
            (
                x.size(1) == self.config.horizon,
                f"Expected x horizon to be {self.config.horizon}, but got {x.shape}",
            ),
            (
                y is None or (y >= 0).all(),
                "Expected y to be non-negative",
            ),
        )

        B, H, R, Di, Do = (
            x.size(0),
            self.config.horizon,
            self.config.rank,
            self.config.d_in,
            self.config.d_out,
        )
        loss = None
        y_tilde, gammas_y = mps_gamma_contract_batch(  # type: ignore
            self.g.reshape(1, Do, H, R, Di, R)
            .expand(B, -1, -1, -1, -1, -1)
            .reshape(B * Do, H, R, Di, R),
            self.alpha.reshape(1, -1).expand(B * Do, -1),
            self.beta.reshape(1, -1).expand(B * Do, -1),
            x,
        )  # (B*Do,), (B*Do, H)

        # Non-negative regression
        y_tilde = y_tilde.abs().clamp(min=self.eps)
        gammas_y = gammas_y.clamp(min=self.eps)

        if y is not None:
            loss = (1 / H) * (  # avg across seq dimension
                y_tilde.log()
                + (gammas_y.log().sum(dim=-1))  # (B*Do, H) => (B*Do,)
                + y.reshape(-1).log()  # (B*Do,)
            ).mean()  # avg across batch dimension

            if loss.isnan() or loss < 0:
                print(f"[MPS_REGRESSOR] Loss is NaN or negative: {loss}")
                raise ValueError("[MPS_REGRESSOR] Loss is NaN or negative")

        return AbstractRegressorHeadOutput(
            logits=torch.randn(B, H, Do),
            loss=loss,
        )


def run_test():
    B, H, R, Di, Do = 4, 20000, 8, 4, 1
    mt_head = MPS_REGRESSOR(
        AbstractRegressorHeadConfig(
            d_in=Di,
            d_out=Do,
            horizon=H,
            rank=R,
        ),
    )
    x = torch.randn(B, H, Di)
    y = torch.randn(B, Do).abs()
    loss = mt_head(x, y).loss
    print(f"loss: {loss}")


if __name__ == "__main__":
    run_test()
