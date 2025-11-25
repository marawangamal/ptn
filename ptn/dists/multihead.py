from typing import Callable, List, Optional
import torch
import torch.nn as nn

from ._abc import (
    AbstractDisributionHead,
    AbstractDisributionHeadConfig,
    AbstractDisributionHeadOutput,
)


class MultiHeadDist(AbstractDisributionHead):
    """Simple multi-head distribution with independent linear heads for each position."""

    def __init__(self, config: AbstractDisributionHeadConfig):
        super().__init__(config)
        self.vocab_size = config.d_output
        self.horizon = config.horizon
        self.embedding_dim = config.d_model

        # Separate linear heads for each position
        self.heads = nn.ModuleList(
            [nn.Linear(config.d_model, config.d_output) for _ in range(config.horizon)]
        )

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError("from_pretrained not implemented")

    def forward(self, x: torch.Tensor, y: torch.Tensor, ignore_index: int = -100):
        """Compute cross-entropy loss for each position, or for one random head if partial=True."""
        total_loss = 0.0
        for h in range(self.horizon):
            logits = self.heads[h](x)
            loss = torch.nn.functional.cross_entropy(
                logits, y[:, h], ignore_index=ignore_index
            )
            total_loss += loss
        return AbstractDisributionHeadOutput(logits=torch.ones_like(y), loss=total_loss)
