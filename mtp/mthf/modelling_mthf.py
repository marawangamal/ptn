import copy
from dataclasses import dataclass
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
from mtp.mheads._utils import window_input_ids


# Should use mhead as auxiliary loss with H-1 heads in addtion to basic head:
# Multi-Token [Gloeckle et al., 2024], adapted for finetuning
# by adding dmax − 1 auxiliary prediction heads and applying a loss-weighting strategy for these heads,
# similar to our method. To ensure a fair comparison, for both Multi-Token and MuToR, we tune the
# number of predicted future tokens, dmax, alongside the auxiliary loss coefficient. Implementation
# details are provided in Appendix A.1.


@dataclass
class MultiTokenHFOutput(CausalLMOutput):
    loss_main: Optional[torch.Tensor] = None
    loss_aux: Optional[torch.Tensor] = None
    past_key_values: Optional[tuple] = None


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
        lambda_mhead: float = 0.1,  # weight for mhead loss
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.horizon = horizon
        self.model_head = model_head
        self.rank = rank
        self.pretrained = pretrained
        self.lambda_mhead = lambda_mhead
        self.loss_type = loss_type


# TODO:
# [ ] standardize naming scheme (vocab_size vs d_output)
# [ ] remove vocab_size customization
class MultiTokenHF(PreTrainedModel, GenerationMixin):
    config_class = MultiTokenHFConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: MultiTokenHFConfig):
        """Multi-token HuggingFace model.

        Args:
            config (MultiTokenHFConfig): Config for MultiTokenHF.

        """
        super().__init__(config)
        self._tp_plan = []  # added for compatibility with HF

        # Set dims
        self.vocab_size, self.embedding_dim = get_model_dims(config.model_name)
        self.horizon = config.horizon
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

        # NOTE: Important for compatibility with HF
        self.name_or_path = self.backbone.name_or_path

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
        # Enable KV caching: when cache exists, feed only the last token
        past_key_values = kwargs.get("past_key_values", None)
        attention_mask = kwargs.get("attention_mask", None)

        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
        }
        return model_inputs

    def adjust_logits_during_generation(self, logits, **kwargs):
        # No adjustment by default
        return logits

    def _get_mhead_loss(
        self, y, z, use_memory_efficient_loss: bool = True, window_shift: int = 1
    ):
        """Compute mhead loss.

        Args:
            y (torch.Tensor): Target tensor of shape (B, T). Note: this should be the unshifted target.
            z (torch.Tensor): Hidden state tensor of shape (B, T-H, D).
            use_memory_efficient_loss (bool, optional): Whether to use memory efficient loss. Defaults to True.
            window_shift (int, optional): The number of steps to shift the window. Defaults to 1. See example in `window_input_ids` docstring.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - loss (torch.Tensor): Loss tensor of shape (1,).
                - logits (torch.Tensor): Logits tensor of shape (B, T, V). Note: this is the logits from the first head only, we discard the rest.
        """
        B, T = y.shape
        H_, D = min(self.horizon, z.size(1)), z.size(-1)

        # Create targets
        # Shape: (B, T) -> (B, T, H)
        yw = window_input_ids(
            y,
            horizon=H_,
            shift=window_shift,
            ignore_index=-100,  # used to mask positions beyond seq length
        )

        # Sub-sample for memory efficiency
        if use_memory_efficient_loss and self.horizon > 1:
            offset = torch.randint(0, H_, (1,)).item()
            z = z[:, offset::H_]
            yw = yw[:, offset::H_]

        # Merge batch and sequence dims
        z = z.reshape(-1, D)  # (BT, D)
        yw = yw.reshape(-1, H_)  # (BT, H)
        output = self.mhead(z, yw, ignore_index=-100)
        loss = output.loss.mean()
        logits = output.logits[:, 0].reshape(B, T, -1)  # (BT, H, V) -> (B, T, V)
        return loss, logits

    # Combined loss lm_head + mhead
    def forward_joint(
        self,
        input_ids,
        labels=None,
        use_memory_efficient_loss=False,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if self.lm_head is None:
            raise ValueError("LM head is not initialized")

        # Get hidden states from model
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        hidden_state = outputs.last_hidden_state  # (B, T, D)

        # Compute loss if labels provided
        if labels is not None:
            mhead_loss, _ = self._get_mhead_loss(
                labels,
                hidden_state,
                use_memory_efficient_loss,
                window_shift=2,  # lm_head predicts first tok, mhead predicts next H tokens
            )
            logits = self.lm_head(hidden_state)  # (B, T, V)
            loss_main = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
            loss_aux = self.lambda_mhead * mhead_loss
            loss = loss_main + loss_aux
            return MultiTokenHFOutput(
                loss=loss,
                logits=logits,
                loss_main=loss_main,
                loss_aux=loss_aux,
            )

        # For inference: return logits from last position
        logits = self.lm_head(hidden_state[:, -1:])
        return MultiTokenHFOutput(
            logits=logits, past_key_values=outputs.past_key_values
        )

    # Single mhead loss
    def forward_mhead(
        self,
        input_ids,
        labels=None,
        use_memory_efficient_loss=False,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Get hidden states from model
        B, T = input_ids.shape
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        hidden_state = outputs.last_hidden_state  # (B, T, D)

        # Compute loss if labels provided
        if labels is not None:
            loss, logits = self._get_mhead_loss(
                labels,
                hidden_state,
                use_memory_efficient_loss,
                window_shift=1,  # mhead predicts next H tokens
            )
            return MultiTokenHFOutput(loss=loss, logits=logits.reshape(B, T, -1))

        # For inference: return logits from last position
        last_hidden = hidden_state[:, -1]  # (B, D)
        output = self.mhead(last_hidden)
        logits = output.logits.unsqueeze(1)  # (B, 1, V)
        return MultiTokenHFOutput(
            logits=logits, past_key_values=outputs.past_key_values
        )

    def forward(
        self,
        input_ids,
        labels=None,
        use_memory_efficient_loss=False,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> MultiTokenHFOutput:
        if self.loss_type == "joint":
            return self.forward_joint(
                input_ids, labels, use_memory_efficient_loss, attention_mask, **kwargs
            )
        elif self.loss_type == "mhead":
            return self.forward_mhead(
                input_ids, labels, use_memory_efficient_loss, attention_mask, **kwargs
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
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    else:
        config = AutoConfig.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        for k, v in kwargs.items():
            setattr(config, k, v)
        hf_model = AutoModelForCausalLM.from_config(config)

    if hasattr(hf_model, "transformer"):
        return hf_model.transformer, hf_model.lm_head
    elif hasattr(hf_model, "model"):  # e.g., Llama
        return hf_model.model, hf_model.lm_head
    else:
        raise ValueError(f"Cannot find transformer/model backbone in {type(hf_model)}")
