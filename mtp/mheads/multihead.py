from typing import Optional
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
        self.decoder = nn.Linear(config.d_model, config.d_output, bias=False)

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
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ):
        assert x.ndim == 2, "x must be 2D (B, D)"
        assert y.ndim == 2 if y is not None else True, "y must be 2D (B, H)"

        # if y is none (eval), only compute logits for the first head
        H_ = 1 if y is None else self.config.horizon
        logits = torch.stack(
            [self.decoder(self.heads[h](x)) for h in range(H_)], dim=1
        )  # (B, H_, V)
        loss = None
        if y is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, self.config.d_output),  # (BH, V)
                y.reshape(-1),  # (BH,)
                ignore_index=ignore_index,
            )
        else:
            logits = logits[:, 0]  # (B, V)
        return AbstractDisributionHeadOutput(logits=logits, loss=loss)


if __name__ == "__main__":
    B, H, D, V = 1, 5, 10, 32
    config = AbstractDisributionHeadConfig(d_model=D, d_output=V, horizon=H, rank=8)
    head = Multihead(config)
    x = torch.randn(B, D)
    y = torch.randint(0, V, (B, H))
    assert head(x, y).logits.shape == (B, H, V), "logits should be (B, H, V)"
    assert head(x, y).loss is not None, "loss should be not None"
    assert head(x).logits.shape == (B, V), "logits should be (B, V)"
    print("✅ Test passed!")
