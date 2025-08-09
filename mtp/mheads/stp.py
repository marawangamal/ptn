from typing import Optional
import torch

from ._abc import (
    AbstractDisributionHead,
    AbstractDisributionHeadConfig,
    AbstractDisributionHeadOutput,
)


class STP(AbstractDisributionHead):
    def __init__(self, config: AbstractDisributionHeadConfig):
        super().__init__(config)
        assert config.horizon == 1, "STP only supports horizon=1"
        self.head = torch.nn.Linear(config.d_model, config.d_output)

    def set_output_embeddings(self, new_embeddings):
        self.head.weight = new_embeddings

    def get_output_embeddings(self):
        return self.head.weight

    def freeze_decoder(self):
        for param in self.head.parameters():
            param.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ):
        assert x.ndim == 2, "x must be 2D (B, D)"
        assert y.ndim == 2 if y is not None else True, "y must be 2D (B, H)"

        logits = self.head(x)  # (B, V)
        loss = None
        if y is not None:
            loss = torch.nn.functional.cross_entropy(
                logits, y, ignore_index=ignore_index
            )
            logits = logits.unsqueeze(1)  # (B, 1, V)
        return AbstractDisributionHeadOutput(logits=logits, loss=loss)
