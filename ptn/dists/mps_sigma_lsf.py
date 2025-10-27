from typing import Optional
import torch

from ptn.dists._abc import (
    AbstractDisributionHead,
    AbstractDisributionHeadConfig,
    AbstractDisributionHeadGenerateOutput,
    AbstractDisributionHeadOutput,
)
from ptn.dists.tensorops.mps import select_margin_mps_tensor_batched


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


class MPS_SIGMA_LSF(AbstractDisributionHead):
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

        if config.init_method == "ortho":
            self._ortho_init()

        self.eps = 1e-12

        # Residual params
        self.mu = torch.eye(R, R).reshape(1, R, 1, R).repeat(H, 1, Do, 1)
        if self.config.mode == "residual":
            self._w_mps.data[0] = torch.zeros(R, Do, R, Di)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(horizon={self.config.horizon}, "
            f"rank={self.config.rank}, d_model={self.config.d_model}, "
            f"d_output={self.config.d_output}, pos_func='{self.config.pos_func}', "
            f"init_method='{getattr(self.config, 'init_method', None)}')"
        )

    def w_mps(self, x: torch.Tensor):
        theta = POS_FUNC_MAP[self.config.pos_func](
            torch.einsum("bi,hpoqi->bhpoq", x, self._w_mps) + self.b_mps
        )

        if self.config.mode == "direct":
            return theta  # (B, H, R, V, R)
        elif self.config.mode == "residual":
            dvc = x.device
            return self.mu.unsqueeze(0).to(dvc) + theta  # (B, H, R, V, R)

    def _ortho_init(self):
        H, R, Do, Di = (
            self.config.horizon,
            self.config.rank,
            self.config.d_output,
            self.config.d_model,
        )
        self._w_mps.data = (
            torch.eye(R, R).reshape(1, R, 1, R, 1).repeat(H, 1, Do, 1, Di)
        )

    def _compute_orthogonal_reg(self):
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
                use_scale_factors=self.config.use_scale_factors,
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
                use_scale_factors=self.config.use_scale_factors,
            )

            if len(gammas_p) == 0:
                gammas_p = [torch.ones(B, dtype=x.dtype, device=x.device)]
            if len(gammas_z) == 0:
                gammas_z = [torch.ones(B, dtype=x.dtype, device=x.device)]

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
                loss += self.config.lambda_ortho * self._compute_orthogonal_reg()

        return AbstractDisributionHeadOutput(
            logits=torch.randn(B, H, V),
            loss=loss,
            loss_dict=loss_dict,
        )

    def generate(
        self,
        x: torch.Tensor,
        do_sample: bool = True,
        y: Optional[torch.Tensor] = None,
        debug: bool = False,
    ):
        """Generate a sequence of length H from the model.

        Args:
            x (torch.Tensor): Input features. Shape: (B, D)
            y (torch.Tensor): Target sequence. Shape: (B, H'). If provided, generation will start from the last timestep of y.

        Returns:
            y (torch.Tensor): Generated sequence. Shape: (B, H)
        """
        B, D, H = x.shape[0], x.shape[1], self.config.horizon
        y_out = torch.empty(B, H, dtype=torch.long, device=x.device)
        start_idx = 0
        if y is not None:
            start_idx = y.shape[1]
            y_out[:, :start_idx] = y

        theta_mps = self.w_mps(x)

        left_cache, right_cache = None, None
        p_tilde_seq = []
        for h in range(start_idx, H):
            y_mrgn = torch.full(
                (B, H - h - 1),
                -2,
                dtype=torch.long,
                device=x.device,
            )
            y_free = torch.full(
                (B, 1),
                -1,
                dtype=torch.long,
                device=x.device,
            )
            ops = torch.cat([y_out[:, :h], y_free, y_mrgn], dim=-1)  # (B, H)
            p_tilde, gammas_p, left_cache, right_cache = (
                select_margin_mps_tensor_batched(
                    self.alpha.unsqueeze(0).expand(B, -1),
                    self.beta.unsqueeze(0).expand(B, -1),
                    theta_mps,  # (B, R, H, V)
                    ops,
                    use_scale_factors=True,
                    build_cache=True if h == 0 else False,
                    left_cache=left_cache,
                    right_cache=right_cache,
                )
            )  # (B, V), (B, H), (B, H, R), (B, H, R)
            p_tilde_seq.append(p_tilde)

            if do_sample:
                p = p_tilde.clamp(min=self.eps).log()
                if debug:
                    print(f"p: {p}")
                dist = torch.distributions.Categorical(logits=p)
                yi = dist.sample()  # (B,1)
            else:
                yi = p_tilde.argmax(dim=-1)  # (B,1)
            y_out[:, h] = yi

        # return y_out
        return AbstractDisributionHeadGenerateOutput(
            y=y_out,
            p_tilde=torch.stack(p_tilde_seq, dim=1),  # (B, H, V)
        )

    # USED FOR TESTING THE CACHE IMPLEMENTATION
    def generate_without_cache(self, x: torch.Tensor, do_sample: bool = True):
        """Generate a sequence of length H from the model.

        Args:
            x (torch.Tensor): Input features. Shape: (B, D)

        Returns:
            y (torch.Tensor): Generated sequence. Shape: (B, H)
        """
        B, D, H = x.shape[0], x.shape[1], self.config.horizon
        y_out = torch.empty(B, H, dtype=torch.long, device=x.device)

        theta_mps = self.w_mps(x)

        for h in range(H):
            y_mrgn = torch.full(
                (B, H - h - 1),
                -2,
                dtype=torch.long,
                device=x.device,
            )
            y_free = torch.full(
                (B, 1),
                -1,
                dtype=torch.long,
                device=x.device,
            )
            ops = torch.cat([y_out[:, :h], y_free, y_mrgn], dim=-1)  # (B, H)
            p_tilde, gammas_p, _, _ = select_margin_mps_tensor_batched(
                self.alpha.unsqueeze(0).expand(B, -1),
                self.beta.unsqueeze(0).expand(B, -1),
                theta_mps,  # (B, R, H, V)
                ops,
                use_scale_factors=True,
            )  # (B, V), (B, H), (B, H, R), (B, H, R)

            if do_sample:
                dist = torch.distributions.Categorical(logits=p_tilde)
                yi = dist.sample()  # (B,1)
            else:
                yi = p_tilde.argmax(dim=-1)  # (B,1)
            y_out[:, h] = yi

        return y_out


def test_generate():
    B, H, D, V = 8, 32, 9, 2
    mt_head = MPS_SIGMA_LSF(
        AbstractDisributionHeadConfig(
            d_model=D,
            d_output=V,
            horizon=H,
            rank=8,
            pos_func="abs",
        ),
    )
    x = torch.randn(B, D)
    y_1 = mt_head.generate_fast(x, do_sample=False)
    y_2 = mt_head.generate_slow(x, do_sample=False)
    assert torch.allclose(y_1, y_2)
    print("[PASS] generate and generate_slow match")


def run_test():
    B, H, D, V = 4, 28 * 28, 9, 2
    mt_head = MPS_SIGMA_LSF(
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


if __name__ == "__main__":
    test_generate()
