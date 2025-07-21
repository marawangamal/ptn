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
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


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
    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> AbstractDisributionHeadOutput:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input features. Shape: (B, D)
            y (Optional[torch.Tensor], optional): Target tensor. Shape: (B, V). Defaults to None.

        Returns:
            torch.Tensor: _description_
        """
        pass
