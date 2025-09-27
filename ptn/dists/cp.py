import torch
from tqdm import tqdm

from ptn.dists._abc import (
    AbstractDisributionHead,
    AbstractDisributionHeadConfig,
    AbstractDisributionHeadOutput,
)
from ptn.dists._tensorops import select_margin_cp_tensor_batched


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


class CP(AbstractDisributionHead):
    def __init__(self, config: AbstractDisributionHeadConfig):
        """Simple multi-head distribution with independent linear heads for each position."""
        super().__init__(config)
        H, R, Di, V = (
            config.horizon,
            config.rank,
            config.d_model,
            config.d_output,
        )

        std_fan_in = torch.sqrt(torch.tensor(2.0)) / Di**0.5
        self.w_cp = torch.nn.Parameter(torch.randn(R, H, Di, V) * std_fan_in)
        self.b_cp = torch.nn.Parameter(torch.zeros(R, H, V) * std_fan_in)

    def set_output_embeddings(self, embeddings: torch.nn.Parameter):
        raise NotImplementedError("set_output_embeddings not implemented")

    def get_output_embeddings(self):
        raise NotImplementedError("get_output_embeddings not implemented")

    def freeze_decoder(self):
        raise NotImplementedError("freeze_decoder not implemented")

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

            theta_cp = POS_FUNC_MAP[self.config.pos_func](
                torch.einsum("bi,rhiv->brhv", x, self.w_cp) + self.b_cp
            )

            p_tilde, gammas_p = select_margin_cp_tensor_batched(
                theta_cp,
                y.reshape(B, H),
                use_scale_factors=True,
            )  # (B,), (B, H)
            z_tilde, gammas_z = select_margin_cp_tensor_batched(
                theta_cp,
                torch.full(
                    (B, H),
                    -2,  # marginalize
                    dtype=torch.long,
                    device=x.device,
                ),
                use_scale_factors=True,
                # margin_index=-2,
            )

            gammas_p = torch.stack(gammas_p, dim=-1)
            gammas_z = torch.stack(gammas_z, dim=-1)  # (B, H)

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

    def generate(self, x: torch.Tensor):
        """Generate a sequence of length H from the model.

        Args:
            x (torch.Tensor): Input features. Shape: (B, D)

        Returns:
            y (torch.Tensor): Generated sequence. Shape: (B, H)
        """
        with torch.no_grad():
            B, D, H, R = x.shape[0], x.shape[1], self.config.horizon, self.config.rank
            # y_out = torch.empty(B, H, dtype=torch.long, device=x.device)

            theta_cp = POS_FUNC_MAP[self.config.pos_func](
                torch.einsum("bi,rhiv->brhv", x, self.w_cp) + self.b_cp
            )

            # -1: marginalize
            y_out = torch.full((B, H), -1, dtype=torch.long, device=x.device)
            cp_mrgn = theta_cp.sum(dim=-1)  # (B, R, H)
            cp_mrgn_reduced = torch.ones(B, R, H + 1, device=x.device, dtype=x.dtype)

            # DP: cp_mrgn_reduced_h =  \prod_{i=h}^{H} \sum_v \theta_cp_{iv}
            res = torch.ones(B, R, device=x.device, dtype=x.dtype)
            for h in range(H - 1, -1, -1):
                res = res * cp_mrgn[:, :, h]  # (B, R)
                res = res / torch.max(res, dim=-1, keepdim=True)[0]
                cp_mrgn_reduced[:, :, h] = res

            cp_slct = torch.empty(B, R, 0, device=x.device, dtype=x.dtype)

            for h in tqdm(range(H), desc="Generating"):
                y_slct_h = (
                    y_out[:, min(0, h - 1) : min(0, h - 1) + 1]
                    .reshape(B, 1, min(max(0, h), 1))
                    .expand(B, R, -1)
                )  # (B, R, 1/``)
                cp_slct_h = theta_cp[:, :, h].gather(  # (B, R, 1)
                    -1,
                    y_slct_h,
                )  # (B, R, 1/0)
                cp_slct = torch.cat([cp_slct, cp_slct_h], dim=-1)  # (B, R, 2)
                cp_slct = cp_slct.prod(dim=-1, keepdim=True)  # (B, R, 1)
                cp_slct = (
                    cp_slct / torch.max(cp_slct, dim=1, keepdim=True)[0]
                )  # (B, R, 1)

                cp_free = theta_cp[:, :, h]  # (B, R, V)
                p_tilde = (
                    cp_slct
                    * cp_mrgn_reduced[:, :, h + 1 : h + 2]
                    * cp_free
                    # (B, R, V) -> (B, V)
                ).sum(dim=1)
                probs = p_tilde / p_tilde.sum(-1, keepdim=True)
                dist = torch.distributions.Categorical(probs=probs)
                yi = dist.sample()  # (B,1)
                y_out[:, h] = yi
            return y_out


def run_test():
    B, H, D, V = 4, 28 * 28, 9, 2
    mt_head = CP(
        AbstractDisributionHeadConfig(
            d_model=D,
            d_output=V,
            horizon=H,
            rank=8,
            pos_func="exp",
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
