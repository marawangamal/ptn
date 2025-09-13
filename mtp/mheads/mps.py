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


POS_FUNC_MAP = {
    "sigmoid": torch.nn.functional.sigmoid,
    "relu": torch.nn.functional.relu,
    "exp": torch.exp,
    "square": torch.square,
    "abs": torch.abs,
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
        # epsilon for numerical stability
        # self.register_buffer("eps", torch.tensor(1e-12))

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

    def w_mps(self, x: torch.Tensor):
        return POS_FUNC_MAP[self.config.pos_func](
            torch.einsum("bi,hpoqi->bhpoq", x, self._w_mps) + self.b_mps
        )  # (B, H, R, V, R)

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

            # theta_mps = POS_FUNC_MAP[self.config.pos_func](
            #     torch.einsum("bi,hpoqi->bhpoq", x, self.w_mps) + self.b_mps
            # )
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

            # # --- clamp before logs ---
            # p_tilde = p_tilde.clamp(min=self.eps)
            # z_tilde = z_tilde.clamp(min=self.eps)
            # gammas_p = gammas_p.clamp(min=self.eps)
            # gammas_z = gammas_z.clamp(min=self.eps)

            loss = (1 / H) * (  # avg across seq dimension
                -torch.log(p_tilde)  # (B,)
                + torch.log(z_tilde)  # (B,)
                # Contraction Stability Scale Factors
                - (gammas_p.log().sum(dim=-1))  # (B, H)
                + (gammas_z.log().sum(dim=-1))  # (B, H)
            ).mean()  # avg across batch dimension

        return AbstractDisributionHeadOutput(
            logits=torch.randn(B, H, V),
            loss=loss,
            loss_dict=loss_dict,
        )

    # USED FOR DEBUGGING
    # def generate_v1(self, x: torch.Tensor):
    #     B, D, H, R = x.shape[0], x.shape[1], self.config.horizon, self.config.rank

    #     y_out = torch.empty(B, H, dtype=torch.long, device=x.device)

    #     # theta_mps = POS_FUNC_MAP[self.config.pos_func](
    #     #     torch.einsum("bi,hpoqi->bhpoq", x, self.w_mps) + self.b_mps
    #     # )
    #     g = self.w_mps(x)  # (B, H, R, V, R)

    #     g_dot = g.sum(dim=-2)  # (B, H, R, R)
    #     m = torch.ones(B, R, H + 1, device=x.device, dtype=x.dtype)
    #     res = self.beta.unsqueeze(0).expand(B, -1)
    #     m[:, :, H] = res
    #     for h in range(H - 1, -1, -1):
    #         res = torch.einsum("bqr,br->bq", g_dot[:, h], res)  # (B, )
    #         res = res / torch.linalg.norm(res, dim=-1, keepdim=True)[0]
    #         m[:, :, h] = res

    #     for h in range(H):

    #         # 0:h-1 -- select
    #         # h -- free
    #         # h+1:H -- marginalize
    #         # gh_yh -- g_{h, yh}
    #         # g_y -- stack(g_{0, y0}, g_{1, y1}, ..., g_{h-1, yh-1})

    #         # **** DEBUG ****
    #         g_y = self.alpha.unsqueeze(0).expand(B, -1)  # (B, R)

    #         y_slct_h = (
    #             y_out[:, min(0, h - 1) : min(0, h - 1) + 1]
    #             .reshape(B, 1, min(max(0, h), 1), 1)
    #             .expand(B, R, -1, R)
    #         )  # (B, R, 1/0, R)
    #         gh_yh = g[
    #             :, min(0, h - 1)
    #         ].gather(  # (B, H, R, V, R) => (B, 1/0, R, 1/0, R)
    #             2,
    #             y_slct_h,
    #         )  # (B, R, 1/0, R)
    #         if gh_yh.shape[2] > 0:
    #             g_y = torch.einsum("br, brdq->bq", g_y, gh_yh)  # (B, R)
    #             g_y = g_y / torch.linalg.norm(g_y, dim=1, keepdim=True)[0]  # (B, R)
    #         g_ = g[:, h]  # (B, R, V, R)  free leg
    #         p_tilde = self.sig(torch.einsum("br,brdq,bq->bd", g_y, g_, m[:, :, h + 1]))
    #         # probs = p_tilde / p_tilde.sum(-1, keepdim=True)  # (B, V)
    #         # dist = torch.distributions.Categorical(probs=probs)
    #         # yi = dist.sample()  # (B,1)
    #         # y_out[:, h] = yi
    #         # **** DEBUG ****

    #         y_mrgn = torch.full(
    #             (B, H - h - 1),
    #             -2,
    #             dtype=torch.long,
    #             device=x.device,
    #         )
    #         y_free = torch.full(
    #             (B, 1),
    #             -1,
    #             dtype=torch.long,
    #             device=x.device,
    #         )
    #         ops = torch.cat([y_out[:, :h], y_free, y_mrgn], dim=-1)  # (B, H)
    #         p_tilde, gammas_p = select_margin_mps_tensor_batched(
    #             self.alpha.unsqueeze(0).expand(B, -1),
    #             self.beta.unsqueeze(0).expand(B, -1),
    #             g,  # (B, R, H, V)
    #             ops,
    #             use_scale_factors=True,
    #             m=m,
    #             g_y=g_y,
    #             g_=g_,
    #             p_tilde=p_tilde,
    #             y_slct_h=y_slct_h,
    #             y_out=y_out,
    #             gh_yh=gh_yh,
    #         )  # (B, V), (B, H)

    #         dist = torch.distributions.Categorical(logits=p_tilde)
    #         yi = dist.sample()  # (B,1)
    #         y_out[:, h] = yi

    #     return y_out

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
