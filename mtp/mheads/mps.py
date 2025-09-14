import torch

from mtp.mheads._abc import (
    AbstractDisributionHead,
    AbstractDisributionHeadConfig,
    AbstractDisributionHeadOutput,
)
from mtp.mheads.tensorops.mps import select_margin_mps_tensor_batched


def print_tens_stats(t: torch.Tensor, name: str):
    """Prints one line of stats for a tensor."""
    print(
        f"{name}: mean: {t.mean():.2f} ± {t.std():.2f}, min: {t.min():.2f}, max: {t.max():.2f}"
    )


def rbf_activation(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    # Elementwise RBF around 0: exp(-x^2 / (2*sigma^2))
    return torch.exp(-x.pow(2) / (2 * (sigma**2)))


POS_FUNC_MAP = {
    "sigmoid": torch.nn.functional.sigmoid,
    "relu": torch.nn.functional.relu,
    "exp": torch.exp,
    "square": torch.square,
    "abs": torch.abs,
    "rbf": rbf_activation,
}


# TODO: instead of a separate generate, maybe handle the case when y is not given as generate
class MPS(AbstractDisributionHead):
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
        self._w_mps = torch.nn.Parameter(torch.randn(H, R, Do, R, Di) * std_fan_in)
        self.b_mps = torch.nn.Parameter(torch.zeros(H, R, Do, R) * std_fan_in)

        dtype = self._w_mps.dtype
        self.alpha = torch.nn.Parameter(
            torch.nn.functional.one_hot(torch.tensor(0), num_classes=R).to(dtype),
            requires_grad=False,
        )
        self.beta = torch.nn.Parameter(
            torch.nn.functional.one_hot(torch.tensor(0), num_classes=R).to(dtype),
            requires_grad=False,
        )

        self.sig = (
            lambda x: x
        )  # left for user to override (i.e. when using born machine)
        self._init_weights()

        self.eps = 1e-12

    def w_mps(self, x: torch.Tensor):
        return POS_FUNC_MAP[self.config.pos_func](
            torch.einsum("bi,hpoqi->bhpoq", x, self._w_mps) + self.b_mps
        )  # (B, H, R, V, R)

    def _init_weights(self):
        H, R, Do, Di = (
            self.config.horizon,
            self.config.rank,
            self.config.d_output,
            self.config.d_model,
        )
        self._w_mps.data = (
            torch.eye(R, R).reshape(1, R, 1, R, 1).repeat(H, 1, Do, 1, Di)
        )

    def compute_orthogonal_reg(self):
        H, R, Do, Di = (
            self.config.horizon,
            self.config.rank,
            self.config.d_output,
            self.config.d_model,
        )
        dt, dv = self._w_mps.dtype, self._w_mps.device
        w = torch.einsum("hpoqi->hoipq", self._w_mps)
        I = (
            torch.eye(R, R, dtype=dt, device=dv)
            .reshape(1, 1, 1, R, R)
            .expand(H, Do, Di, -1, -1)
        )
        I_hat = torch.einsum("hoipq,hoipz->hoipz", w, w)
        loss = (I - I_hat).pow(2).mean()
        return loss

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
        self.sig = (
            lambda x: x
        )  # left for user to override (i.e. when using born machine)

        if y is not None:

            theta_mps = self.w_mps(x)
            p_tilde, gammas_p = select_margin_mps_tensor_batched(
                self.alpha.unsqueeze(0).expand(B, -1),
                self.beta.unsqueeze(0).expand(B, -1),
                theta_mps,
                y.reshape(B, H),
                use_scale_factors=True,
            )  # (B,), (B, H)
            z_tilde, gammas_z = select_margin_mps_tensor_batched(
                self.alpha.unsqueeze(0).expand(B, -1),
                self.beta.unsqueeze(0).expand(B, -1),
                theta_mps,
                torch.full(
                    (B, H),
                    -2,  # marginalize
                    dtype=torch.long,
                    device=x.device,
                ),
                use_scale_factors=True,
            )

            gammas_p = torch.stack(gammas_p, dim=-1)
            gammas_z = torch.stack(gammas_z, dim=-1)  # (B, H)

            # --- clamp before logs ---
            p_tilde = p_tilde.clamp(min=self.eps)
            z_tilde = z_tilde.clamp(min=self.eps)
            gammas_p = gammas_p.clamp(min=self.eps)
            gammas_z = gammas_z.clamp(min=self.eps)

            # --nan filtering--
            fmask = (
                p_tilde.isfinite()
                & z_tilde.isfinite()
                & gammas_p.isfinite().all(dim=-1)
                & gammas_z.isfinite().all(dim=-1)
            )  # (B,)
            p_tilde = p_tilde[fmask]
            z_tilde = z_tilde[fmask]
            gammas_p = gammas_p[fmask]
            gammas_z = gammas_z[fmask]

            loss = (1 / H) * (  # avg across seq dimension
                -torch.log(p_tilde)  # (B,)
                + torch.log(z_tilde)  # (B,)
                # Contraction Stability Scale Factors
                - (gammas_p.log().sum(dim=-1))  # (B, H)
                + (gammas_z.log().sum(dim=-1))  # (B, H)
            ).mean()  # avg across batch dimension

            if loss.isnan() or loss < 0:
                print(f"[MPS] Loss is NaN or negative: {loss}")
                raise ValueError("[MPS] Loss is NaN or negative")

            if self.config.lambda_ortho > 0:
                loss += self.config.lambda_ortho * self.compute_orthogonal_reg()

        return AbstractDisributionHeadOutput(
            logits=torch.randn(B, H, V),
            loss=loss,
            loss_dict=loss_dict,
        )

    def generate(self, x: torch.Tensor):
        """Generate a sequence of length H from the model.

        Args:
            x (torch.Tensor): Input features. Shape: (B, D)

        Returns:
            y (torch.Tensor): Generated sequence. Shape: (B, H)
        """
        with torch.no_grad():
            B, D, H, R = x.shape[0], x.shape[1], self.config.horizon, self.config.rank

            # Get MPS tensor
            g = self.w_mps(x)  # (B, H, R, V, R)

            # Build marginals dp array
            y_out = torch.full((B, H), -1, dtype=torch.long, device=x.device)
            g_dot = g.sum(dim=-2)  # (B, H, R, R)
            m = torch.ones(B, R, H + 1, device=x.device, dtype=x.dtype)
            res = self.beta.unsqueeze(0).expand(B, -1)
            m[:, :, H] = res
            for h in range(H - 1, -1, -1):
                res = torch.einsum("bqr,br->bq", g_dot[:, h], res)  # (B, )
                res = res / torch.linalg.norm(res, dim=-1, keepdim=True)[0]
                m[:, :, h] = res

            g_y = self.alpha.unsqueeze(0).expand(B, -1)  # (B, R)

            for h in range(H):
                y_slct_h = (
                    y_out[:, min(0, h - 1) : min(0, h - 1) + 1]
                    .reshape(B, 1, min(max(0, h), 1), 1)
                    .expand(B, R, -1, R)
                )  # (B, R, 1/0, R)
                gh_yh = g[:, min(0, h - 1)].gather(  # (B, R, V, R) => (B, R, 1/0, R)
                    -2,
                    y_slct_h,
                )  # (B, R, 1/0, R)
                if gh_yh.shape[2] > 0:
                    g_y = torch.einsum("br, brdq->bq", g_y, gh_yh)  # (B, R)
                    g_y = g_y / torch.linalg.norm(g_y, dim=1, keepdim=True)[0]  # (B, R)
                g_ = g[:, h]  # (B, R, V, R)  free leg
                p_tilde = self.sig(
                    torch.einsum("br,brdq,bq->bd", g_y, g_, m[:, :, h + 1])
                )
                probs = p_tilde / p_tilde.sum(-1, keepdim=True)  # (B, V)
                dist = torch.distributions.Categorical(probs=probs)
                yi = dist.sample()  # (B,1)
                y_out[:, h] = yi
            return y_out


def run_test():
    B, H, D, V = 4, 28 * 28, 9, 2
    mt_head = MPS(
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
    # out = mt_head.generate(x)
    # print(f"generated: {out}")


if __name__ == "__main__":
    run_test()
