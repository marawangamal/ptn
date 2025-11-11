from typing import List, Optional
import warnings
import torch

from ptn.dists._abc import (
    AbstractDisributionHead,
    AbstractDisributionHeadConfig,
    AbstractDisributionHeadOutput,
)
from ptn.dists.tensorops.mps import (
    select_margin_hmm_tensor_batched,
    select_margin_mps_tensor_batched,
)


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

# TODO:
# [ ] remove bias? or replace with pre & post bias and make both trainable/non-trainable
# [ ] remove self.sig?


class HMM_SIGMA_LSF(AbstractDisributionHead):
    def __init__(self, config: AbstractDisributionHeadConfig):
        """Simple multi-head distribution with independent linear heads for each position."""
        super().__init__(config)
        self.eps = 1e-12
        H, R, Di, Do = (
            config.horizon,
            config.rank,
            config.d_model,
            config.d_output,
        )

        std_fan_in = torch.sqrt(torch.tensor(2.0)) / Di**0.5
        self._wt_mps = torch.nn.Parameter(torch.randn(R, R, Di) * std_fan_in)
        self._we_mps = torch.nn.Parameter(torch.randn(Do, R, Di) * std_fan_in)
        self.b_mps = None

        dtype = self._wt_mps.dtype
        self.alpha = torch.nn.Parameter(
            torch.nn.functional.one_hot(torch.tensor(0), num_classes=R).to(dtype),
            requires_grad=False,
        )
        self.beta = torch.nn.Parameter(
            torch.nn.functional.one_hot(torch.tensor(0), num_classes=R).to(dtype),
            requires_grad=False,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(horizon={self.config.horizon}, "
            f"rank={self.config.rank}, d_model={self.config.d_model}, "
            f"d_output={self.config.d_output}, pos_func='{self.config.pos_func}', "
            f"init_method='{getattr(self.config, 'init_method', None)}')"
        )

    def w_mps(self, x: torch.Tensor):
        H = self.config.horizon
        wt = self._wt_mps.unsqueeze(0).expand(
            H, -1, -1, -1
        )  # (H, R, R, Di) or (H, R, Do, R, Di)
        we = self._we_mps.unsqueeze(0).expand(H, -1, -1)  # (H, Do, Di) or None
        theta_t = POS_FUNC_MAP[self.config.pos_func](
            torch.einsum("bi,hpqi->bhpq", x, wt)
        )
        theta_e = POS_FUNC_MAP[self.config.pos_func](
            torch.einsum("bi,hori->bhor", x, we)
        )
        return theta_t, theta_e  # (B, H, R, R), (B, H, Do)

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

        B, H, V = (
            x.size(0),
            self.config.horizon,
            self.config.d_output,
        )
        loss = None
        loss_dict = {}
        self.sig = (
            lambda x: x
        )  # left for user to override (i.e. when using born machine)

        if y is not None:
            theta_t, theta_e = self.w_mps(x)  # (B, H, R, V, R), (B, H, Do)
            rank = theta_t.size(-1)
            alpha = self.alpha.unsqueeze(0).expand(B, -1)[:, :rank]
            beta = self.beta.unsqueeze(0).expand(B, -1)[:, :rank]

            p_tilde, gammas_p = select_margin_hmm_tensor_batched(  # type: ignore
                alpha=alpha,
                beta=beta,
                core=theta_t,
                emission=theta_e,
                ops=y.reshape(B, H),
                use_scale_factors=self.config.use_scale_factors,
            )  # (B,), (B, H)
            z_tilde, gammas_z = select_margin_hmm_tensor_batched(  # type: ignore
                alpha=alpha,
                beta=beta,
                core=theta_t,
                emission=theta_e,
                ops=torch.full(
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
                print(f"[HMM] Loss is NaN or negative: {loss}")
                raise ValueError("[HMM] Loss is NaN or negative")

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

        theta_t, theta_e = self.w_mps(x)
        rank = theta_t.size(-1)
        alpha = self.alpha.unsqueeze(0).expand(B, -1)[:, :rank]
        beta = self.beta.unsqueeze(0).expand(B, -1)[:, :rank]

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
                select_margin_hmm_tensor_batched(  # type: ignore
                    alpha=alpha,
                    beta=beta,
                    core=theta_t,  # (B, R, H, V)
                    emission=theta_e,
                    ops=ops,
                    build_cache=True if h == 0 else False,
                    left_cache=left_cache,
                    right_cache=right_cache,
                )
            )  # (B, V), (B, H), (B, H, R), (B, H, R)
            p_tilde_seq.append(p_tilde)

            if do_sample:
                p = p_tilde / p_tilde.sum(-1, keepdim=True)
                yi = torch.multinomial(p, num_samples=1).reshape(-1)  # (B,1)
            else:
                yi = p_tilde.argmax(dim=-1)  # (B,1)
            y_out[:, h] = yi

        return y_out

    def materialize(self, x: torch.Tensor):
        """Materialize probabilities into a tensor.

        Args:
            x (torch.Tensor): Input features. Shape: (B, Di)

        Returns:
            p (torch.Tensor): Materialized probabilities. Shape: (B, V**H)
        """
        theta_mps = self.w_mps(x)  # (B, H, R, V, R)
        esum = []
        H = theta_mps.size(1)
        cores = (
            [torch.einsum("i,bidj->bdj", self.alpha, theta_mps[:, 0]).unsqueeze(1)]
            + [theta_mps[:, h] for h in range(1, H - 1)]
            + [
                torch.einsum("bidj,j->bid", theta_mps[:, H - 1], self.beta).unsqueeze(
                    -1
                )
            ]
        )
        for h in range(H):
            esum.append(cores[h])
            esum.append([0, h + 1, h + H + 2, h + 2])
        esum.append([0] + [h + H + 2 for h in range(H)])
        p_tilde = torch.einsum(*esum)  # (B, V**H)
        return p_tilde


def run_test():
    B, H, R, Di, Do = 1, 5, 2, 1, 256
    mt_head = HMM_SIGMA_LSF(
        AbstractDisributionHeadConfig(
            d_model=Di,
            d_output=Do,
            horizon=H,
            rank=R,
            pos_func="abs",
        ),
    )
    x = torch.randn(B, Di)
    y = torch.randint(0, Do, (B, H))
    loss = mt_head(x, y).loss
    print(f"loss: {loss}")
    # test generate
    y_out = mt_head.generate(x)
    print(f"y_out: {y_out}")


def test_materialize():
    B, H, Di, Do, R = 1, 2, 1, 32, 2
    mt_head = HMM_SIGMA_LSF(
        AbstractDisributionHeadConfig(
            d_model=Di,
            d_output=Do,
            horizon=H,
            rank=R,
            pos_func="exp",
            use_bias=False,
        ),
    )
    x = torch.ones(B, Di)
    p = mt_head.materialize(x)
    print(f"p: {p}")


if __name__ == "__main__":
    run_test()
