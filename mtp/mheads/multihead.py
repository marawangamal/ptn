import torch
import torch.nn as nn

from mtp.mheads._abc import (
    AbstractDisributionHead,
    AbstractDisributionHeadConfig,
    AbstractDisributionHeadOutput,
)


class Multihead(AbstractDisributionHead):
    def __init__(self, config: AbstractDisributionHeadConfig):
        """Simple multi-head distribution with independent linear heads for each position."""
        super().__init__(config)

        # Separate linear heads for each position
        self.heads = nn.ModuleList(
            [
                nn.Linear(config.d_model, config.d_model)
                for _ in range(self.config.horizon)
            ]
        )
        self.decoder = nn.Linear(config.d_model, config.d_output)

    def set_output_embeddings(self, embeddings: torch.Tensor):
        V, D = embeddings.shape
        assert embeddings.shape == (
            self.config.d_output,
            self.config.d_model,
        ), "embeddings shape must be (V, D)"
        u, s, vt = torch.svd(embeddings)  # (V, R), (R,), (R, D)
        self.decoder.weight = u[:, :D]
        self.heads[0].weight = s[:D] * vt[:D]  # (D, D)

    def get_output_embeddings(self):
        return torch.einsum("vo,oi->vi", self.decoder.weight, self.heads[0].weight)

    def forward(self, x, y=None):
        # if y is none (eval), only compute logits for the first head
        H_ = 1 if y is None else self.config.horizon
        logits = torch.stack(
            [self.decoder(self.heads[h](x)) for h in range(H_)], dim=1
        )  # (B, H_, V)
        loss = None
        if y is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, self.config.d_output), y.reshape(-1)
            )
        return AbstractDisributionHeadOutput(logits=logits[:, 0], loss=loss)


if __name__ == "__main__":
    B, H, D, V = 1, 5, 10, 32
    config = AbstractDisributionHeadConfig(d_model=D, d_output=V, horizon=H, rank=8)
    head = Multihead(config)
    x = torch.randn(B, D)
    y = torch.randint(0, V, (B, H))
    assert head(x, y).logits.shape == (B, V), "logits should be (B, V)"
    assert head(x, y).loss is not None, "loss should be not None"
