import itertools
import random
import torch
import torch.nn as nn

from mtp.mheads._abc import (
    AbstractDisributionHead,
    AbstractDisributionHeadConfig,
    AbstractDisributionHeadOutput,
)
from mtp.mheads._tensorops import batch_cp_reduce


def print_tens_stats(t: torch.Tensor, name: str):
    """Prints one line of stats for a tensor."""
    print(
        f"{name}: mean: {t.mean():.2f} ± {t.std():.2f}, min: {t.min():.2f}, max: {t.max():.2f}"
    )


class CP(AbstractDisributionHead):
    def __init__(self, config: AbstractDisributionHeadConfig):
        """Simple multi-head distribution with independent linear heads for each position."""
        super().__init__(config)
        H, R, D, V = (
            config.horizon,
            config.rank,
            config.d_model,
            config.d_output,
        )

        std_fan_in = torch.sqrt(torch.tensor(2.0)) / D**0.5
        self.w = torch.nn.Parameter(
            torch.randn(config.rank, config.horizon, config.d_model, config.d_model)
            * std_fan_in
        )
        self.decoder = nn.Linear(config.d_model, config.d_output)

    def get_cp_params(self, x: torch.Tensor, **kwargs):
        # Mapping: (B, D) -> (B, R, H, V)

        # first: (B, D) -> (B, R, H, D)
        theta = torch.einsum("be,rhde->brhd", x, self.w)

        # second: (B, R, H, D) -> (B, R, H, V)
        theta = self.decoder(theta)

        # make positive
        theta = torch.nn.functional.sigmoid(theta)

        return theta

    def set_output_embeddings(self, embeddings: torch.nn.Parameter):
        assert (
            embeddings.shape == self.decoder.weight.shape
        ), f"embeddings must be of shape {self.decoder.weight.shape} but got {embeddings.shape}"
        self.decoder.weight = embeddings

    def get_output_embeddings(self):
        return self.decoder.weight

    def freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        x,
        y=None,
        ignore_index: int = -100,
        return_logits: bool = False,
    ):
        assert x.ndim == 2, "x must be 2D (B, D)"
        assert y.ndim == 2 if y is not None else True, "y must be 2D (B, H)"

        # Get indexed distribution
        H_ = 1 if y is None else self.config.horizon
        B, R, V = x.size(0), self.config.rank, self.config.d_output
        loss = None
        logits = torch.zeros(B, 1, self.config.d_output, device=x.device)

        if y is not None:
            params = self.get_cp_params(x)  # (B, R, H, V)
            p_tilde, gammas_p = batch_cp_reduce(
                params.reshape(B, R, H_, V),
                y.reshape(B, H_),
                margin_index=ignore_index,
                use_scale_factors=True,
            )  # (B,), (B, H)
            z_tilde, gammas_z = batch_cp_reduce(
                params.reshape(B, R, H_, V),
                torch.full(
                    (B, H_),
                    -1,
                    dtype=torch.long,
                    device=x.device,
                ),
                margin_index=-1,
                use_scale_factors=True,
            )

            # loss = (
            #     -torch.log(p_tilde)  # (B, T')
            #     + torch.log(z_tilde)  # (B, T')
            #     # Contraction Stability Scale Factors
            #     - sum([torch.log(z) for z in gammas_p])  # (B, T')
            #     + sum([torch.log(z) for z in gammas_z])
            # )  # (B, T-H)

            loss = (1 / H_) * (  # avg across seq dimension
                -torch.log(p_tilde)  # (B,)
                + torch.log(z_tilde)  # (B,)
                # Contraction Stability Scale Factors
                - (gammas_p.log().sum(dim=-1))  # (B, H)
                + (gammas_z.log().sum(dim=-1))  # (B, H)
            ).mean()  # avg across batch dimension

            def get_stat_str(v):
                return f"{v.mean():.2f} ± {v.std():.2f}"

            # loss_dict = {
            #     # after log
            #     "p": get_stat_str(p_tilde),
            #     "z": get_stat_str(z_tilde),
            #     "g_p": get_stat_str(gammas_p),
            #     "g_z": get_stat_str(gammas_z),
            #     # params
            #     "params": get_stat_str(params),
            # }

            loss_dict = {
                # after log
                "p": p_tilde.mean().item(),
                "z": z_tilde.mean().item(),
                "g_p": gammas_p.mean().item(),
                "g_z": gammas_z.mean().item(),
                "params": params.mean().item(),
            }

            if return_logits:
                pass

        return AbstractDisributionHeadOutput(
            logits=logits,
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
        pass


def run_test():
    B, H, D, V = 8, 4, 4096, 32000
    mt_head = CP(
        AbstractDisributionHeadConfig(
            d_model=D,
            d_output=V,
            horizon=H,
            rank=8,
        ),
    )
    x = torch.randn(B, D)
    y = torch.randint(0, V, (B, H))
    # set some 50% of y to ignore_index
    for i, j in itertools.product(range(B), range(H)):
        if random.random() < 0.5:
            y[i, j] = -100

    out = mt_head(x, y)
    print(f"loss: {out.loss}")


if __name__ == "__main__":
    run_test()
