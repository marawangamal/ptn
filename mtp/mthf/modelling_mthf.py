import copy
from typing import Literal, Optional
import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

from mtp.mheads import MHEADS
from mtp._types import ModelHeadType
from mtp.mheads._abc import AbstractDisributionHeadConfig
from mtp.mheads._utils import get_windowed_input_ids, window_input_ids


# Should use mhead as auxiliary loss with H-1 heads in addtion to basic head:
# Multi-Token [Gloeckle et al., 2024], adapted for finetuning
# by adding dmax − 1 auxiliary prediction heads and applying a loss-weighting strategy for these heads,
# similar to our method. To ensure a fair comparison, for both Multi-Token and MuToR, we tune the
# number of predicted future tokens, dmax, alongside the auxiliary loss coefficient. Implementation
# details are provided in Appendix A.1.


class MultiTokenHFConfig(PretrainedConfig):
    model_type = "multi_token_hfmodel"

    def __init__(
        self,
        model_name: str = "gpt2",
        horizon: int = 1,
        rank: int = 1,
        model_head: ModelHeadType = "stp",
        loss_type: Literal["joint", "mhead"] = "mhead",
        pretrained: bool = False,
        lambda_mhead: float = 1.0,  # weight for mhead loss
        vocab_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.horizon = horizon
        self.model_head = model_head
        self.rank = rank
        self.pretrained = pretrained
        self.lambda_mhead = lambda_mhead
        self.vocab_size = vocab_size
        self.loss_type = loss_type


# TODO: standardize naming scheme (vocab_size vs d_output)
class MultiTokenHF(PreTrainedModel, GenerationMixin):
    config_class = MultiTokenHFConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: MultiTokenHFConfig):
        """Multi-token HuggingFace model.

        Args:
            config (MultiTokenHFConfig): Config for MultiTokenHF.

        """
        super().__init__(config)

        # Set dims
        vocab_size, embedding_dim = get_model_dims(config.model_name)
        self.embedding_dim = embedding_dim
        self.horizon = config.horizon
        self.vocab_size = config.vocab_size or vocab_size  # override if provided
        self.lambda_mhead = config.lambda_mhead
        self.loss_type = config.loss_type

        # Set backbone
        self.backbone, lm_head = get_backbone(
            config.model_name,
            config.pretrained,
            vocab_size=self.vocab_size,
        )

        # Set lm_head
        self.lm_head = None
        if self.loss_type == "joint":
            self.lm_head = lm_head

        # Set mhead
        self.mhead_config = AbstractDisributionHeadConfig(
            d_model=self.embedding_dim,
            d_output=self.vocab_size,
            horizon=config.horizon,
            rank=config.rank,
        )
        self.mhead = MHEADS[config.model_head](self.mhead_config)

        # Init both mhead and lm_head from pretrained
        if config.pretrained:
            self.mhead.set_output_embeddings(lm_head.weight)
            if self.lm_head is not None:
                self.lm_head.weight = lm_head.weight

        # NOTE: ensure new mhead has same dtype as backbone
        self.mhead.to(next(self.backbone.parameters()).dtype)

    def get_output_embeddings(self):
        if self.lm_head is not None:
            return self.lm_head.weight  # for forward_joint
        return self.mhead.get_output_embeddings()

    def get_input_embeddings(self):
        # Try to get input embeddings from the backbone
        if hasattr(self.backbone, "wte"):
            return self.backbone.wte
        elif hasattr(self.backbone, "embed_tokens"):
            return self.backbone.embed_tokens
        else:
            raise NotImplementedError("Input embeddings not found in backbone.")

    def set_output_embeddings(self, new_embeddings):
        if self.lm_head is not None:
            self.lm_head.weight = new_embeddings  # for forward_joint
        else:
            self.mhead.set_output_embeddings(new_embeddings)

    def set_input_embeddings(self, new_embeddings):
        if hasattr(self.backbone, "wte"):
            self.backbone.wte = new_embeddings
        elif hasattr(self.backbone, "embed_tokens"):
            self.backbone.embed_tokens = new_embeddings
        else:
            raise NotImplementedError("Input embeddings not found in backbone.")

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # Standard for decoder-only models: just return input_ids and any attention_mask
        model_inputs = {"input_ids": input_ids}
        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
            model_inputs["attention_mask"] = kwargs["attention_mask"]
        return model_inputs

    def adjust_logits_during_generation(self, logits, **kwargs):
        # No adjustment by default
        return logits

    def _get_mhead_loss(
        self, x, z, use_memory_efficient_loss: bool = True, window_shift: int = 1
    ):
        """Compute mhead loss.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T).
            z (torch.Tensor): Hidden state tensor of shape (B, T-H, D).
            use_memory_efficient_loss (bool, optional): Whether to use memory efficient loss. Defaults to True.
            window_shift (int, optional): The number of steps to shift the window. Defaults to 1. See example in `window_input_ids` docstring.

        Returns:
            _type_: _description_
        """
        H_, D = min(self.horizon, z.size(1)), z.size(-1)

        # Create targets
        # Shape: (B, T) -> (B, T, H)
        y = window_input_ids(
            x,
            horizon=H_,
            shift=window_shift,
            ignore_index=-1,  # used to mask positions beyond seq length
        )

        # Sub-sample for memory efficiency
        if use_memory_efficient_loss and self.horizon > 1:
            offset = torch.randint(0, H_, (1,)).item()
            z = z[:, offset::H_]
            y = y[:, offset::H_]

        # Merge batch and sequence dims
        z = z.reshape(-1, D)  # (BT, D)
        y = y.reshape(-1)  # (BT,)
        output = self.mhead(z, y, ignore_index=-1)
        return output.loss.mean(), output.logits

    # Combined loss lm_head + mhead
    def forward_joint(
        self, input_ids, labels=None, use_memory_efficient_loss=False, **kwargs
    ):
        # Get hidden states from model
        outputs = self.backbone(input_ids=input_ids, **kwargs)
        hidden_state = outputs.last_hidden_state  # (B, T-H, D)

        # Compute loss if labels provided
        if labels is not None:
            mhead_loss, _ = self._get_mhead_loss(
                input_ids, hidden_state, use_memory_efficient_loss
            )
            logits = self.lm_head(hidden_state)  # (B, T, V)
            lm_head_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
            loss = self.lambda_mhead * mhead_loss + lm_head_loss
            return CausalLMOutput(loss=loss, logits=logits)

        # For inference: return logits from last position
        logits = self.lm_head(hidden_state[:, -1:, :])
        return CausalLMOutput(logits=logits)

    # Single mhead loss
    def forward_mhead(
        self, input_ids, labels=None, use_memory_efficient_loss=False, **kwargs
    ):
        # Get hidden states from model
        outputs = self.backbone(input_ids=input_ids, **kwargs)
        hidden_state = outputs.last_hidden_state  # (B, T, D)

        # Compute loss if labels provided
        if labels is not None:
            loss, logits = self._get_mhead_loss(
                input_ids, hidden_state, use_memory_efficient_loss
            )
            return CausalLMOutput(loss=loss, logits=logits)

        # For inference: return logits from last position
        output = self.mhead(hidden_state)
        return CausalLMOutput(logits=output.logits)

    def forward(
        self, input_ids, labels=None, use_memory_efficient_loss=False, **kwargs
    ):
        if self.loss_type == "joint":
            return self.forward_joint(
                input_ids, labels, use_memory_efficient_loss, **kwargs
            )
        elif self.loss_type == "mhead":
            return self.forward_mhead(
                input_ids, labels, use_memory_efficient_loss, **kwargs
            )
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")


def get_model_dims(model_name: str) -> tuple[int, int]:
    """Get vocabulary size and embedding dimension from model config."""
    hf_config = AutoConfig.from_pretrained(model_name)
    vocab_size = hf_config.vocab_size
    embedding_dim = [  # ad-hoc to support both GPT and Llama
        getattr(hf_config, k)
        for k in ["n_embd", "hidden_size"]
        if hasattr(hf_config, k)
    ][0]
    return vocab_size, embedding_dim


def get_backbone(
    model_name: str, pretrained: bool = False, **kwargs
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Get a randomly initialized backbone of a HuggingFace model."""
    if pretrained:
        hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        config = AutoConfig.from_pretrained(model_name)
        for k, v in kwargs.items():
            setattr(config, k, v)
        hf_model = AutoModelForCausalLM.from_config(config)

    if hasattr(hf_model, "transformer"):
        return hf_model.transformer, hf_model.lm_head
    elif hasattr(hf_model, "model"):  # e.g., Llama
        return hf_model.model, hf_model.lm_head
    else:
        raise ValueError(f"Cannot find transformer/model backbone in {type(hf_model)}")
