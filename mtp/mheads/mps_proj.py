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
    batch_mps_reduce_decoder_einlse_margin_only,
    batch_mps_reduce_decoder_einlse_select_only,
    batch_mps_reduce_decoder_margin_only,
    batch_mps_reduce_decoder_select_only,
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
    "abs": torch.abs,
}


class MPSProj(AbstractDisributionHead):
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
        self.w_mps = torch.nn.Parameter(torch.randn(H, R, Do, R, Di) * std_fan_in)
        self.b_mps = torch.nn.Parameter(torch.zeros(H, R, Do, R))
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

        # methods: exp, abs, square, relu, softmax*, exp_lse*

        if y is not None:

            # NOTE: this is not optimal, as it will over-filter samples
            # filter out entire sample if any of the H y vals are ignore_index
            # maybe we can absorb this functionality in the reducers
            mask = (y != ignore_index).all(dim=-1)  # (B,)

            mps_params = (
                torch.einsum("bi,hroqi->bhroq", x[mask], self.w_mps) + self.b_mps
            )  # (B', R, H, V)
            mps_decoder = self.decoder

            if self.config.pos_func in POS_FUNC_MAP:
                mps_params = POS_FUNC_MAP[self.config.pos_func](mps_params)
                mps_decoder = POS_FUNC_MAP[self.config.pos_func](mps_decoder)
                slct_func = batch_mps_reduce_decoder_select_only
                mrgn_func = batch_mps_reduce_decoder_margin_only

            elif self.config.pos_func == "exp_lse":
                mps_params = torch.log_softmax(mps_params, dim=-1)
                mps_decoder = torch.log_softmax(mps_decoder, dim=-1)

                slct_func = batch_mps_reduce_decoder_einlse_select_only
                mrgn_func = batch_mps_reduce_decoder_einlse_margin_only

            else:
                raise ValueError(f"Invalid position function: {self.config.pos_func}")

            log_p_tilde = slct_func(
                mps_params,
                y.reshape(B, H)[mask],
                mps_decoder.T,
            )  # (B,)
            log_z = mrgn_func(
                mps_params,
                mps_decoder.T,
            )  # (B,)

            loss = (1 / H) * (log_z - log_p_tilde).mean()

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
    B, H, R, D, V = 32, 2, 4, 1024, 30_000
    mt_head = MPSProj(
        AbstractDisributionHeadConfig(
            d_model=D,
            d_output=V,
            horizon=H,
            rank=R,
            pos_func="square",
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
