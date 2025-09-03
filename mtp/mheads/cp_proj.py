import itertools
import random
import torch
import torch.nn as nn

from mtp.mheads._abc import (
    AbstractDisributionHead,
    AbstractDisributionHeadConfig,
    AbstractDisributionHeadOutput,
)
from mtp.mheads._tensorops import (
    batch_cp_reduce_decoder_einlse_margin_only,
    batch_cp_reduce,
    batch_cp_reduce_decoder,
    batch_cp_reduce_decoder_einlse,
    batch_cp_reduce_decoder_einlse_select_only,
    batch_cp_reduce_decoder_einlse_margin_only,
    select_margin_cp_tensor_batched,
)


def print_tens_stats(t: torch.Tensor, name: str):
    """Prints one line of stats for a tensor."""
    print(
        f"{name}: mean: {t.mean():.2f} ± {t.std():.2f}, min: {t.min():.2f}, max: {t.max():.2f}"
    )


POS_FUNC_MAP = {
    "sigmoid": torch.nn.functional.sigmoid,
    "relu": torch.nn.functional.relu,
    "exp": torch.exp,
    "square": torch.square,
}


class CPProjector(AbstractDisributionHead):
    def __init__(self, config: AbstractDisributionHeadConfig):
        """Simple multi-head distribution with independent linear heads for each position."""
        super().__init__(config)
        H, R, Di, Do, V = (
            config.horizon,
            config.rank,
            config.d_model,
            config.d_hidden or config.d_model,
            config.d_output,
        )

        std_fan_in = torch.sqrt(torch.tensor(2.0)) / Di**0.5
        self.w_cp = torch.nn.Parameter(torch.randn(R, H, Di, Do) * std_fan_in)
        self.b_cp = torch.nn.Parameter(torch.zeros(R, H, Do) * std_fan_in)
        self.decoder = torch.nn.Parameter(torch.randn(V, Do) * std_fan_in)

    def set_output_embeddings(self, embeddings: torch.nn.Parameter):
        assert (
            embeddings.shape == self.decoder.shape
        ), f"embeddings must be of shape {self.decoder.shape} but got {embeddings.shape}"
        self.decoder = embeddings

    def get_output_embeddings(self):
        return self.decoder

    def freeze_decoder(self):
        self.decoder.requires_grad = False

    def forward(
        self,
        x,
        y=None,
        ignore_index: int = -100,
        return_logits: bool = False,
    ):
        # Input validation
        assert x.ndim == 2, "x must be 2D (B, D)"
        assert y is None or y.ndim == 2, "y must be 2D (B, H)"
        assert (
            y is None or y.size(1) == self.config.horizon
        ), f"Incorrect y horizon, must of shape (B, {self.config.horizon}) but got {y.shape}"

        B, R, H, V = (
            x.size(0),
            self.config.rank,
            self.config.horizon,
            self.config.d_output,
        )
        loss = None
        loss_dict = {}

        if y is not None:

            # *****************************************************************************************
            #  ORIGINAL VERSION
            # *****************************************************************************************

            # cp_params = POS_FUNC_MAP[self.config.pos_func](
            #     torch.einsum("bi,rhio->brho", x, self.w_cp) + self.b_cp
            # )
            # cp_decoder = POS_FUNC_MAP[self.config.pos_func](self.decoder)

            # p_tilde, gammas_p = select_margin_cp_tensor_batched(
            #     torch.einsum("brho,vo->brhv", cp_params, cp_decoder),
            #     y.reshape(B, H),
            #     use_scale_factors=True,
            # )  # (B,), (B, H)
            # z_tilde, gammas_z = select_margin_cp_tensor_batched(
            #     torch.einsum("brho,vo->brhv", cp_params, cp_decoder),
            #     torch.full(
            #         (B, H),
            #         -2,  # marginalize
            #         dtype=torch.long,
            #         device=x.device,
            #     ),
            #     use_scale_factors=True,
            # )

            # gammas_p = torch.stack(gammas_p, dim=-1)
            # gammas_z = torch.stack(gammas_z, dim=-1)  # (B, H)

            # loss = (1 / H) * (  # avg across seq dimension
            #     -torch.log(p_tilde)  # (B,)
            #     + torch.log(z_tilde)  # (B,)
            #     # Contraction Stability Scale Factors
            #     - (gammas_p.log().sum(dim=-1))  # (B, H)
            #     + (gammas_z.log().sum(dim=-1))  # (B, H)
            # ).mean()  # avg across batch dimension

            if self.config.pos_func == "exp":

                # *****************************************************************************************
                #  LOGSUMEXP VERSION
                # *****************************************************************************************

                # NOTE: this is not optimal, as it will over-filter samples
                # filter out entire sample if any of the H y vals are ignore_index
                # maybe we can absorb this functionality in the reducers
                mask = (y != ignore_index).all(dim=-1)  # (B,)

                cp_params = (
                    torch.einsum("bi,rhio->brho", x[mask], self.w_cp) + self.b_cp
                )  # (B', R, H, V)
                cp_decoder = self.decoder

                log_p_tilde = batch_cp_reduce_decoder_einlse_select_only(
                    cp_params,
                    y.reshape(B, H)[mask],
                    cp_decoder.T,
                    # margin_index=ignore_index, not accepted arg for select_only
                )  # (B,)
                log_z = batch_cp_reduce_decoder_einlse_margin_only(
                    cp_params,
                    cp_decoder.T,
                    # margin_index=ignore_index, not accepted arg for margin_only
                )  # (B,)

                loss = (1 / H) * (log_z - log_p_tilde).mean()
            else:

                # *****************************************************************************************
                #  VMAP VERSION
                # *****************************************************************************************

                cp_params = POS_FUNC_MAP[self.config.pos_func](
                    torch.einsum("bi,rhio->brho", x, self.w_cp) + self.b_cp
                )
                cp_decoder = POS_FUNC_MAP[self.config.pos_func](self.decoder)

                p_tilde, gammas_p = batch_cp_reduce_decoder(
                    cp_params,
                    y.reshape(B, H),
                    cp_decoder.T,
                    margin_index=ignore_index,
                    use_scale_factors=True,
                )  # (B,), (B, H)
                z_tilde, gammas_z = batch_cp_reduce_decoder(
                    cp_params,
                    torch.full(
                        (B, H),
                        -1,
                        dtype=torch.long,
                        device=x.device,
                    ),
                    cp_decoder.T,
                    margin_index=-1,
                    use_scale_factors=True,
                )

                loss = (1 / H) * (  # avg across seq dimension
                    -torch.log(p_tilde)  # (B,)
                    + torch.log(z_tilde)  # (B,)
                    # Contraction Stability Scale Factors
                    - (gammas_p.log().sum(dim=-1))  # (B, H)
                    + (gammas_z.log().sum(dim=-1))  # (B, H)
                ).mean()  # avg across batch dimension

        return AbstractDisributionHeadOutput(
            logits=torch.randn(B, H, V),
            loss=loss,
            loss_dict=loss_dict,
        )

    def generate(self, x: torch.Tensor):
        """Generate a sequence of length H from the model.

        Args:
            x (torch.Tensor): Input features. Shape: (B, D)

        Returns:
            y (torch.Tensor): Generated sequence. Shape: (B, H)
        """
        pass


def run_test():
    B, H, D, V = 8, 4, 4096, 32000
    mt_head = CPProjector(
        AbstractDisributionHeadConfig(
            d_model=D,
            d_output=V,
            horizon=H,
            rank=8,
        ),
    )
    x = torch.randn(B, D)
    y = torch.randint(0, V, (B, H))
    # set some 50% of y to ignore_index
    for i, j in itertools.product(range(B), range(H)):
        if random.random() < 0.5:
            y[i, j] = -100

    out = mt_head(x, y)
    print(f"loss: {out.loss}")


if __name__ == "__main__":
    run_test()
