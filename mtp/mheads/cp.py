import torch
import torch.nn as nn

from mtp.mheads._abc import (
    AbstractDisributionHead,
    AbstractDisributionHeadConfig,
    AbstractDisributionHeadOutput,
)
from mtp.mheads._utils import select_margin_cp_tensor_batched


class CP(AbstractDisributionHead):
    def __init__(self, config: AbstractDisributionHeadConfig):
        """Simple multi-head distribution with independent linear heads for each position."""
        super().__init__(config)
        self.a = torch.nn.Parameter(torch.randn(config.rank))
        self.w = torch.nn.Parameter(
            torch.randn(config.rank, config.horizon, config.d_model, config.d_model)
        )
        self.decoder = nn.Linear(config.d_model, config.d_output)

    def get_cp_params(self, x: torch.Tensor, **kwargs):
        # Mapping: (B, D) -> (B, R, H, V)

        # first: (B, D) -> (B, R, H, D)
        theta = torch.einsum("be,rhde->brhd", x, self.w)

        # second: (B, R, H, D) -> (B, R, H, V)
        theta = self.decoder(theta)

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

    def forward(self, x, y=None):
        assert x.ndim == 2, "x must be 2D (B, D)"
        assert y.ndim == 2 if y is not None else True, "y must be 2D (B, H)"

        # Get indexed distribution
        H_ = 1 if y is None else self.config.horizon
        B, R, V = x.size(0), self.config.rank, self.config.d_output
        loss = None
        logits = torch.zeros(B, self.config.d_output, device=x.device)
        if y is not None:
            params = self.get_cp_params(x)  # (B, R, H, V)
            p_tilde, gammas_p = select_margin_cp_tensor_batched(
                cp_params=params.reshape(B, R, H_, V),
                ops=y.reshape(B, H_),
            )  # (B,), (B, H)
            z_tilde, gammas_z = select_margin_cp_tensor_batched(
                cp_params=params.reshape(B, R, H_, V),
                ops=torch.full(
                    (B, H_),
                    -2,
                    dtype=torch.long,
                    device=x.device,
                ),
            )
            loss = (
                -torch.log(p_tilde)  # (B, T')
                + torch.log(z_tilde)  # (B, T')
                # Contraction Stability Scale Factors
                - sum([torch.log(z) for z in gammas_p])  # (B, T')
                + sum([torch.log(z) for z in gammas_z])
            )  # (B, T-H)
        return AbstractDisributionHeadOutput(logits=logits, loss=loss)


def run_test():
    B, H, D, V = 8, 2, 4096, 32000
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
    out = mt_head(x, y)
    print(f"loss: {out.loss}")


if __name__ == "__main__":
    run_test()
