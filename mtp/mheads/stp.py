from typing import Optional
import torch

from mtp.mheads._abc import (
    AbstractDisributionHead,
    AbstractDisributionHeadConfig,
    AbstractDisributionHeadOutput,
)


class STP(AbstractDisributionHead):
    def __init__(self, config: AbstractDisributionHeadConfig):
        super().__init__(config)
        assert config.horizon == 1, "STP only supports horizon=1"
        self.decoder = torch.nn.Linear(config.d_model, config.d_output)

    def set_output_embeddings(self, new_embeddings):
        self.decoder.weight = new_embeddings

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
        assert y.size(1) == 1 if y is not None else True, "y must have 1 dimension"

        logits = self.decoder(x)  # (B, V)
        loss = None
        if y is not None:
            loss = torch.nn.functional.cross_entropy(
                logits, y.reshape(-1), ignore_index=ignore_index
            )
            logits = logits.unsqueeze(1)  # (B, 1, V)
        return AbstractDisributionHeadOutput(logits=logits, loss=loss)


if __name__ == "__main__":
    B, H, D, V = 1, 1, 10, 32
    config = AbstractDisributionHeadConfig(d_model=D, d_output=V, horizon=H, rank=1)
    head = STP(config)
    x = torch.randn(B, D)
    y = torch.randint(0, V, (B, H))
    assert head(x, y).logits.shape == (B, H, V), "logits should be (B, H, V)"
    assert head(x, y).loss is not None, "loss should be not None"
    assert head(x).logits.shape == (B, V), "logits should be (B, V)"
    print("✅ Test passed!")
