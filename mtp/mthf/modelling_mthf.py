import copy
from typing import Optional
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
from mtp.mheads._utils import get_windowed_input_ids


class MultiTokenHFConfig(PretrainedConfig):
    model_type = "multi_token_hfmodel"

    def __init__(
        self,
        model_name: str = "gpt2",
        horizon: int = 1,
        rank: int = 1,
        model_head: ModelHeadType = "stp",
        pretrained: bool = False,
        lambda_mhead: float = 1.0,  # weight for mhead loss
        vocab_size: Optional[int] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.horizon = horizon
        self.model_head = model_head
        self.rank = rank
        self.pretrained = pretrained
        self.lambda_mhead = lambda_mhead
        self.vocab_size = vocab_size


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

        # Set backbone
        self.backbone, lm_head = get_backbone(
            config.model_name,
            config.pretrained,
            vocab_size=self.vocab_size,
        )
        del lm_head

        # Set multi-token head
        self.mhead_config = AbstractDisributionHeadConfig(
            d_model=self.embedding_dim,
            d_output=self.vocab_size,
            horizon=config.horizon,
            rank=config.rank,
        )
        self.mhead = MHEADS[config.model_head](self.mhead_config)
        self.mhead.to(next(self.backbone.parameters()).dtype)

        # Compatibility with HF
        self.name_or_path = self.backbone.name_or_path

    def get_output_embeddings(self):
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

    def forward(
        self, input_ids, labels=None, use_memory_efficient_loss=False, **kwargs
    ):
        # Get hidden states from model
        outputs = self.backbone(input_ids=input_ids, **kwargs)
        z = outputs.last_hidden_state  # (B, T-H, D)

        # Compute loss if labels provided
        if labels is not None:
            seq_len = input_ids.shape[1]
            if seq_len <= self.horizon:
                raise ValueError(
                    f"Input sequence length ({seq_len}) must be greater than horizon ({self.horizon}) for loss computation."
                )
            # Remove last H positions since we can't predict them
            x = z[:, : -self.horizon, :]  # (B, T-H, D)

            # Create targets: (B*(T-H), H)
            # TODO: move shift logic to mhead to support simultaneous training of conid/joint dists
            y = get_windowed_input_ids(input_ids, self.horizon)
            if use_memory_efficient_loss and self.horizon > 1:
                shift = torch.randint(0, self.horizon, (1,)).item()
                x = x[:, shift :: self.horizon]
                y = y[:, shift :: self.horizon]

            # Merge batch and sequence dims
            x = x.reshape(-1, self.embedding_dim)  # (B*(T-H), D)
            y = y.reshape(-1)  # (B*(T-H),)
            output = self.mhead(x, y)
            loss = output.loss.mean()
            logits = output.logits
            return CausalLMOutput(loss=loss, logits=logits)

        # For inference: return logits from last position
        output = self.mhead(z)
        return CausalLMOutput(logits=output.logits)


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
