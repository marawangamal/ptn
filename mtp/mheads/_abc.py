from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class AbstractDisributionHeadConfig:
    d_model: int
    d_output: int  # e.g. vocab size
    horizon: int
    rank: int


@dataclass
class AbstractDisributionHeadOutput:
    # TODO: logits output to always be (B, H*, V)
    logits: torch.Tensor  # (B, H, V) or (B, V)
    loss: Optional[torch.Tensor] = None  # (1,)
    loss_dict: Optional[dict] = None  # (1,)


class AbstractDisributionHead(ABC, torch.nn.Module):
    def __init__(self, config: AbstractDisributionHeadConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        pass

    @abstractmethod
    def get_output_embeddings(self) -> torch.Tensor:
        pass

    @abstractmethod
    def freeze_decoder(self):
        pass

    @abstractmethod
    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> AbstractDisributionHeadOutput:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input features. Shape: (B, D)
            y (Optional[torch.Tensor], optional): Target tensor. Shape: (B, H). Defaults to None.

        Returns:
            AbstractDisributionHeadOutput: Output of the head.
        """
        pass
