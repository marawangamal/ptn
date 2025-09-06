from typing import Callable, Optional

import torch

from mtp.mheads._abc import (
    AbstractDisributionHead,
    AbstractDisributionHeadConfig,
    AbstractDisributionHeadOutput,
)


def log_prob_moe(
    y: torch.Tensor,
    alpha_tilde: torch.Tensor,
    p_dists_tilde: torch.Tensor,
    decoder: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Compute log probability of MoE distribution.

    Args:
        y (torch.Tensor): Indices. Shape: (H).
        alpha_tilde (torch.Tensor): Mixture weights. Shape: (R).
        p_dists_tilde (torch.Tensor): Unnormalized logits. Shape: (R, H, D).
        decoder (torch.Tensor): Decoder weights. Shape: (D, V).

    Returns:
        torch.Tensor: Log probability. Shape: (,).
    """
    R, H, D = p_dists_tilde.shape
    lsm_alpha = torch.log_softmax(alpha_tilde, dim=-1)  # (R)

    # NOTE: Can trade-off memory/speed here. Instead of loop can do in parallel but
    # would use O(BRDV) memory vs O(BRV)
    # Map: (R, H, D) -> (R, H, V) -> (R, H, 1) -> (R,)
    lsm_cp_cores = torch.stack(
        [
            torch.log_softmax(
                torch.einsum("rd,vd->rv", p_dists_tilde[:, i], decoder),
                dim=-1,
            )
            .gather(dim=-1, index=y[i : i + 1].unsqueeze(0).repeat(R, 1))
            .squeeze(-1)
            for i in range(H)
        ],
        dim=1,
    ).sum(dim=-1)
    return torch.logsumexp(lsm_alpha + lsm_cp_cores, dim=-1)


log_prob_moe_batched = torch.func.vmap(log_prob_moe, in_dims=(0, 0, 0, None))  # type: ignore


class MoEProjector(AbstractDisributionHead):
    """Variant of MoE that projects input onto CP parameters.

    This is a MoE that projects the input onto the CP parameters, rather than
    using the CP parameters directly.

    Args:
        config (AbstractDisributionHeadConfig): Configuration for the MoEProjector.

    Attributes:
        w_alpha (torch.nn.Linear): Linear layer to project the input onto the CP parameters.
        cp_params (torch.nn.Parameter): CP parameters.
    """

    def __init__(self, config: AbstractDisributionHeadConfig):
        super().__init__(config)
        """CP parameterized MoE distribution.

        TN:
              a
            / |   \
           /  |    \
          θ₁  θ₂ .. θₕ
          |   |     |
          D   D     D
          |   |     |
          y₁  y₂ .. yₕ
        """

        # === dims
        self.config = config
        H, R, Di, Do, V = (
            self.config.horizon,
            self.config.rank,
            self.config.d_model,
            self.config.d_hidden or self.config.d_model,
            self.config.d_output,
        )

        # since w_cp: (D) -> (R, H, D) i.e. fan in is D
        std_fan_in = torch.sqrt(torch.tensor(2.0)) / Di**0.5

        # === params
        self.w_alpha = torch.nn.Linear(Di, R)
        self._w_cp_params = torch.nn.Parameter(torch.randn(R, H, Di, Do) * std_fan_in)
        self._b_cp_params = torch.nn.Parameter(torch.randn(R, H, Do))
        self.decoder = torch.nn.Parameter(torch.randn(V, Do) * std_fan_in)

    def w_cp_params(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "be,rhed->brhd", x, self._w_cp_params
        ) + self._b_cp_params.unsqueeze(0)

    def freeze_decoder(self):
        raise NotImplementedError("freeze_decoder not implemented")

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, new_embeddings):
        self.decoder = new_embeddings

    def generate(self, x: torch.Tensor, do_sample: bool = True):
        """Generate a sequence of length H from the model.

        Args:
            x (torch.Tensor): Input features. Shape: (B, D)

        Returns:
            y (torch.Tensor): Generated sequence. Shape: (B, H)
        """
        cp_params_tilde = torch.einsum(
            "brhd,vd->brhv", self.w_cp_params(x), self.decoder
        )  # (B, R, H, D)
        alphas_tilde = self.w_alpha(x)  # (B, R)

        B, R, H, D = cp_params_tilde.shape

        y_out = torch.empty(B, H, dtype=torch.long, device=x.device)

        for h in range(H):
            # Compute log P(y_h | x, y_1, ..., y_h-1)
            lsm_a = torch.log_softmax(alphas_tilde, dim=-1).unsqueeze(-1)  # (B, R, 1)
            lsm_cp = cp_params_tilde.log_softmax(dim=-1)  # (B, R, H, V)
            lsm_select = lsm_cp.gather(  # (B, R, H, V) before gather
                dim=-1,
                index=y_out[:, :h].unsqueeze(1).unsqueeze(-1).expand(-1, R, -1, -1),
            ).sum(
                dim=-2
            )  # (B, R, 1)
            lsm_free = lsm_cp[:, :, h]  # (B, R, V)
            log_p = torch.logsumexp(lsm_a + lsm_select + lsm_free, dim=1)  # (B, V)

            dist = torch.distributions.Categorical(logits=log_p)
            yi = dist.sample()  # (B,)
            y_out[:, h] = yi

            # exponentiate and sample
            # prob_y_bar_xy = torch.exp(log_p)  # (B, V)
            # prob_y_bar_xy = prob_y_bar_xy / prob_y_bar_xy.sum(-1, keepdim=True)
            # if do_sample:
            #     dist =
            #     y_out[:, h] = torch.multinomial(prob_y_bar_xy, 1).squeeze(-1)
            # else:
            #     y_out[:, h] = torch.argmax(prob_y_bar_xy, dim=-1)

        return y_out

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ):
        # Input validation
        assert x.ndim == 2, "x must be 2D (B, D)"
        assert y is None or y.ndim == 2, "y must be 2D (B, H)"
        assert (
            y is None or y.size(1) == self.config.horizon
        ), f"Incorrect y horizon, must of shape (B, {self.config.horizon}) but got {y.shape}"

        B, V = x.shape[0], self.config.d_output
        H = self.config.horizon
        loss = None
        loss_dict = {}
        if y is not None:
            # NOTE: this is not optimal, as it will over-filter samples
            # filter out entire sample if any of the H y vals are ignore_index
            mask = (y != ignore_index).all(dim=-1)  # (B,)
            alpha_tilde = self.w_alpha(x[mask])  # (B, R)
            p_dists_tilde = self.w_cp_params(x[mask])  # (B, R, H, D)
            if y[mask].shape[0] > 0:
                loss = -log_prob_moe_batched(
                    y[mask],  # (B, H)
                    alpha_tilde,
                    p_dists_tilde,
                    self.decoder,
                ).mean() * (1 / H)

            # DEBUGGING / ANALYSIS
            # logging the following list: softmax(--r,h--|p_dists_tilde|--|decoder|--)  (R,H,V)
            if self.debug:
                alphas = torch.softmax(
                    torch.einsum("brhd,vd->brhv", p_dists_tilde, self.decoder), dim=-1
                )
                loss_dict["alphas"] = alphas.reshape(-1).detach().cpu()

        return AbstractDisributionHeadOutput(
            logits=torch.randn(B, H, V),
            loss=loss,
            loss_dict=loss_dict,
        )


if __name__ == "__main__":
    H, R, D, V = 28 * 28, 1, 10, 2
    moe = MoEProjector(
        AbstractDisributionHeadConfig(horizon=H, rank=R, d_model=D, d_output=V)
    )

    # # Sample
    # model = MHEADS["moe_proj"](
    #     AbstractDisributionHeadConfig(
    #         horizon=28*28,
    #         d_model=10,  # 9 digits
    #         d_output=2,  # 2 classes
    #         rank=1,
    #     )
    # )
    # z = torch.nn.functional.one_hot(torch.tensor([0]), num_classes=10).to(torch.float32).to(device)
    # y = model.generate(z)

    # horizon=300,
    # d_model=10,  # 9 digits
    # d_output=2,  # 2 classes
    # rank=1,

    x = torch.randn(2, D)
    y = torch.randint(0, V, (2, H))
    # print(f"loss: {moe(x, y).loss}")
    print(f"generated: {moe.generate(x)}")
