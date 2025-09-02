from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import torch
import warnings

from mtp.mheads._utils import window_input_ids


@dataclass
class AbstractDisributionHeadConfig:
    d_model: int
    d_output: int  # e.g. vocab size
    horizon: int
    rank: int
    n_feats: int = 1
    d_hidden: Optional[int] = None
    pool_method: str = "mean"  # "mean" or "linear"
    pos_func: str = "sigmoid"


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
        self.feat_proj = None
        if config.n_feats > 1:
            if config.pool_method == "linear":
                self.feat_proj = torch.nn.Linear(
                    config.d_model * config.n_feats, config.d_model
                )
                # init such that first col is ones and rest are zeros
                self.feat_proj.weight.data[:, config.d_model] = torch.eye(
                    config.d_model, config.d_model
                )

            else:
                # issue a warning
                warnings.warn(
                    f"Using pool_method='{config.pool_method}' with n_feats={config.n_feats}. "
                    "This may not be the intended behavior. Consider using pool_method='linear' "
                    "for better feature projection control.",
                    UserWarning,
                    stacklevel=2,
                )

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

    def forward_seq(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        use_memory_efficient_loss: bool = True,
        window_shift: int = 1,
        ignore_index: int = -100,
    ) -> AbstractDisributionHeadOutput:
        """Forward pass for sequence data.

        Args:
            x (torch.Tensor): Input features. Shape: (B, T, D)
            y (torch.Tensor): Target tensor of shape (B, T). Note: this should be the unshifted target.

        Returns:
            - loss: (1,)
            - logits: (B, T, H, V)
        """

        # Input validation
        assert len(x.shape) == 3, "z should be (B, T, D)"
        assert y is None or len(y.shape) == 2, "y should be (B, T)"

        H_ = min(self.config.horizon, x.size(1))
        B_, T_, D_ = x.shape

        # Create targets
        # Shape: (B, T) -> (B, T, H)
        yw = None
        if y is not None:
            yw = window_input_ids(
                y,
                horizon=H_,
                shift=window_shift,
                ignore_index=-100,  # used to mask positions beyond seq length
            )

            # Sub-sample for memory efficiency
            if use_memory_efficient_loss and self.config.horizon > 1:
                offset = torch.randint(0, H_, (1,)).item()
                x = x[:, offset::H_]
                yw = yw[:, offset::H_]

        # Merge batch and sequence dims
        B, T, D = x.shape

        x = x.reshape(-1, D)  # (BT, D)
        yw = yw.reshape(-1, H_) if yw is not None else None  # (BT, H)
        output = self(x, yw, ignore_index=ignore_index)
        loss = output.loss.mean() if output.loss is not None else None
        logits = output.logits.reshape(B, T, H_, -1)
        return AbstractDisributionHeadOutput(loss=loss, logits=logits)

    def get_loss_and_logits(
        self,
        y,
        z,
        use_memory_efficient_loss: bool = True,
        window_shift: int = 1,
        ignore_index: int = -100,
    ):
        """Compute mhead loss.

        Args:
            y (torch.Tensor): Target tensor of shape (B, T). Note: this should be the unshifted target.
            z (torch.Tensor): Hidden state tensor. Should be of shape (B, T, D).
            use_memory_efficient_loss (bool, optional): Whether to use memory efficient loss. Defaults to True.
            window_shift (int, optional): The number of steps to shift the window. Defaults to 1. See example in `window_input_ids` docstring.

        Returns:
            torch.Tensor: loss
        """

        B, T = y.shape
        H_, D = min(self.config.horizon, z.size(1)), z.size(-1)

        # Create targets
        # Shape: (B, T) -> (B, T, H)
        yw = window_input_ids(
            y,
            horizon=H_,
            shift=window_shift,
            ignore_index=-100,  # used to mask positions beyond seq length
        )

        # Sub-sample for memory efficiency
        if use_memory_efficient_loss and self.config.horizon > 1:
            offset = torch.randint(0, H_, (1,)).item()
            z = z[:, offset::H_]
            yw = yw[:, offset::H_]

        # Merge batch and sequence dims
        z = z.reshape(-1, D)  # (BT, D)
        yw = yw.reshape(-1, H_)  # (BT, H)
        output = self(z, yw, ignore_index=ignore_index)
        loss = output.loss.mean()
        return loss, output.logits

    def get_loss_and_logits_from_hidden_state(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        use_memory_efficient_loss: bool = True,
        window_shift: int = 1,
        ignore_index: int = -100,
    ):
        """Get loss from hidden states.

        Args:
            z (torch.Tensor): Hidden states. Shape: (B, T, L, D). Note: L is the hidden state layer index.
            y (torch.Tensor): Target tensor of shape (B, T). Note: this should be the unshifted target.
                If you want to use shifted targets, use `window_input_ids` to create the targets.
            use_memory_efficient_loss (bool, optional): Whether to use memory efficient loss. Defaults to True.
            window_shift (int, optional): The number of steps to shift the window. Defaults to 1. See example in `window_input_ids` docstring.
            ignore_index (int, optional): The index to ignore. Defaults to -100.
        """
        B, T, L, D = z.shape
        assert (
            L == self.config.n_feats
        ), f"Incorrect number of hidden state layers. Expected {self.config.n_feats} but got {L}."

        if self.feat_proj is not None:
            z_prime = self.feat_proj(z.reshape(B, T, -1))  # (B, T, D)
        else:
            z_prime = z.mean(dim=-2)  # (B, T, D)
        return self.get_loss_and_logits(
            y,
            z_prime,
            use_memory_efficient_loss=use_memory_efficient_loss,
            window_shift=window_shift,
            ignore_index=ignore_index,
        )
