from typing import Callable, Optional

import torch

from ctn.mheads._abc import (
    AbstractDisributionHead,
    AbstractDisributionHeadConfig,
    AbstractDisributionHeadOutput,
)


from transformers.models.gemma.modeling_gemma import GemmaForCausalLM, GemmaConfig


class EAGLE(AbstractDisributionHead):
    def __init__(self, config: AbstractDisributionHeadConfig):
        super().__init__(config)
        # === dims
        self.config = config
        H, R, D, V = (
            self.config.horizon,
            self.config.rank,
            self.config.d_model,
            self.config.d_output,
        )
        self.model = GemmaForCausalLM(
            GemmaConfig(
                vocab_size=V,
                hidden_size=D,
                intermediate_size=D,
                num_hidden_layers=1,
            ),
        )

    def freeze_decoder(self):
        raise NotImplementedError("freeze_decoder not implemented")

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, new_embeddings):
        self.decoder = new_embeddings

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ):
        """
        Args:
            x (torch.Tensor): (B, D)
            y (Optional[torch.Tensor], optional): (B, H). Defaults to None.
            ignore_index (int, optional): Index to ignore in the loss calculation. Defaults to -100.

        Returns:
            AbstractDisributionHeadOutput:
        """
        B = x.shape[0]
        loss = None
        if y is not None:
            z = torch.cat(
                [x.reshape(B, 1, -1), self.model.model.embed_tokens(y)], dim=1
            )
            l = torch.cat(
                [torch.ones(B, 1, dtype=y.dtype, device=y.device) * -100, y], dim=-1
            )
            output = self.model(inputs_embeds=z, labels=l, ignore_index=ignore_index)
            loss = output.loss
            logits = output.logits
        else:
            output = self.model(x)
            logits = output.logits

        return AbstractDisributionHeadOutput(
            logits=logits,
            loss=loss,
            loss_dict={},
        )


if __name__ == "__main__":
    B, H, R, D, V = 2, 1, 1, 512, 30000
    moe = EAGLE(AbstractDisributionHeadConfig(horizon=H, rank=R, d_model=D, d_output=V))
    x = torch.randn(B, D)
    y = torch.randint(0, V, (B, H))
    print(f"loss: {moe(x, y).loss}")
