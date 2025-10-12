from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class AbstractRegressorHeadConfig:
    d_in: int  # i.e., x is (d_in)-dimensional
    d_out: int  # i.e., y is (d_out)-dimensional
    horizon: int  # i.e., num cores
    rank: int
    use_scale_factors: bool = True
    norm: str = "linf"  # "linf" or "l2"


@dataclass
class AbstractRegressorHeadOutput:
    logits: torch.Tensor  # (B, D_out)
    loss: Optional[torch.Tensor] = None  # (1,)


class AbstractRegressorHead(ABC, torch.nn.Module):
    """Abstract base class for regressor heads.

    A regressor head is a module that takes in a latent representation z (B, d_emb) and outputs `n_model` MPS models with
    `horizon` cores, each with dimension (R, d_out, R).
    """

    def __init__(self, config: AbstractRegressorHeadConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> AbstractRegressorHeadOutput:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input features. Shape: (B, H, D_in)
            y (Optional[torch.Tensor], optional): Target tensor. Shape: (B, D_out). Defaults to None.

        Returns:
            AbstractRegressorHeadOutput: Output of the head.
        """
        pass
