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

    def forward(self, x, y=None, ignore_index=-1):
        logits = self.head(x)
        loss = None
        if y is not None:
            loss = torch.nn.functional.cross_entropy(
                logits, y, ignore_index=ignore_index
            )
        return AbstractDisributionHeadOutput(logits=logits, loss=loss)
