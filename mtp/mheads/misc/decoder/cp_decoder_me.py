"""
CPD: CP parameterized distribution with decoder.

Note: this is the memory efficient version.

Runtime memory complexity: O(BRV)
"""

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
    "abs": torch.abs,
}


class CPDME(AbstractDisributionHead):
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

            if self.config.pos_func in POS_FUNC_MAP:


                # *****************************************************************************************
                #  RUNTIME RECONSTRUCTION VERSION
                # *****************************************************************************************

                # # *****************************************************************************************
                # #  VMAP VERSION
                # # *****************************************************************************************

                # cp_params = POS_FUNC_MAP[self.config.pos_func](
                #     torch.einsum("bi,rhio->brho", x, self.w_cp) + self.b_cp
                # )
                # cp_decoder = POS_FUNC_MAP[self.config.pos_func](self.decoder)

                # p_tilde, gammas_p = batch_cp_reduce_decoder(
                #     cp_params,
                #     y.reshape(B, H),
                #     cp_decoder.T,
                #     margin_index=ignore_index,
                #     use_scale_factors=True,
                # )  # (B,), (B, H)
                # z_tilde, gammas_z = batch_cp_reduce_decoder(
                #     cp_params,
                #     torch.full(
                #         (B, H),
                #         -1,
                #         dtype=torch.long,
                #         device=x.device,
                #     ),
                #     cp_decoder.T,
                #     margin_index=-1,
                #     use_scale_factors=True,
                # )

                # loss = (1 / H) * (  # avg across seq dimension
                #     -torch.log(p_tilde)  # (B,)
                #     + torch.log(z_tilde)  # (B,)
                #     # Contraction Stability Scale Factors
                #     - (gammas_p.log().sum(dim=-1))  # (B, H)
                #     + (gammas_z.log().sum(dim=-1))  # (B, H)
                # ).mean()  # avg across batch dimension

                # # TODO: plot as histogram
                # # DEBUGGING / ANALYSIS
                # # logging the following list: softmax(--r,h--|p_dists_tilde|--|decoder|--)  (R,H,V)
                # if self.debug:
                #     alphas = torch.softmax(
                #         torch.einsum("brhd,vd->brhv", cp_params, cp_decoder),
                #         dim=-1,
                #     )
                #     loss_dict["alphas"] = alphas.reshape(-1).detach().cpu()

            elif self.config.pos_func == "exp_lse":

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
                raise ValueError(f"Invalid pos_func: {self.config.pos_func}")

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
        B, D, H = x.shape[0], x.shape[1], self.config.horizon
        y_out = torch.empty(B, 0, dtype=torch.long, device=x.device)
        if self.config.pos_func in POS_FUNC_MAP:

            B, D, H, R = x.shape[0], x.shape[1], self.config.horizon, self.config.rank
            # y_out = torch.empty(B, H, dtype=torch.long, device=x.device)

            theta_cp = POS_FUNC_MAP[self.config.pos_func](
                torch.einsum("bi,rhid,vd->brhv", x, self.w_cp, self.decoder) + self.b_cp
            )

            # -1: marginalize
            y_out = torch.full((B, H), -1, dtype=torch.long, device=x.device)
            cp_mrgn = theta_cp.sum(dim=-1)  # (B, R, H)
            cp_mrgn_reduced = torch.ones(B, R, H + 1, device=x.device, dtype=x.dtype)

            # DP: cp_mrgn_reduced_h =  \prod_{i=h}^{H} \sum_v \theta_cp_{iv}
            res = torch.ones(B, R, device=x.device, dtype=x.dtype)
            for h in range(H - 1, -1, -1):
                res = res * cp_mrgn[:, :, h]  # (B, R)
                res = res / torch.max(res, dim=-1, keepdim=True)[0]
                cp_mrgn_reduced[:, :, h] = res

            cp_slct = torch.empty(B, R, 0, device=x.device, dtype=x.dtype)

            for h in range(H):
                y_slct_h = (
                    y_out[:, min(0, h - 1) : min(0, h - 1) + 1]
                    .reshape(B, 1, min(max(0, h), 1))
                    .expand(B, R, -1)
                )  # (B, R, 1/``)
                cp_slct_h = theta_cp[:, :, h].gather(  # (B, R, 1)
                    -1,
                    y_slct_h,
                )  # (B, R, 1/0)
                cp_slct = torch.cat([cp_slct, cp_slct_h], dim=-1)  # (B, R, 2)
                cp_slct = cp_slct.prod(dim=-1, keepdim=True)  # (B, R, 1)
                cp_slct = (
                    cp_slct / torch.max(cp_slct, dim=1, keepdim=True)[0]
                )  # (B, R, 1)

                cp_free = theta_cp[:, :, h]  # (B, R, V)
                p_tilde = (
                    cp_slct
                    * cp_mrgn_reduced[:, :, h + 1 : h + 2]
                    * cp_free
                    # (B, R, V) -> (B, V)
                ).sum(dim=1)
                probs = p_tilde / p_tilde.sum(-1, keepdim=True)
                dist = torch.distributions.Categorical(probs=probs)
                yi = dist.sample()  # (B,1)
                y_out[:, h] = yi

            return y_out

        elif self.config.pos_func == "exp_lse":
            cp_params_tilde = torch.einsum("bi,rhio->brho", x, self.w_cp) + self.b_cp
            for h in range(H):
                y_mrgn = torch.full(
                    (B, H - h),
                    -1,
                    dtype=torch.long,
                    device=x.device,
                )
                log_p = batch_cp_reduce_decoder_einlse(
                    cp_params_tilde,
                    torch.cat([y_out, y_mrgn], dim=-1),
                    self.decoder.T,
                    except_index=h,
                    backend="opt_einsum",
                )
                dist = torch.distributions.Categorical(logits=log_p)
                yi = dist.sample()  # (B,)
                y_out[:, h] = yi

        else:
            raise ValueError(f"Invalid pos_func: {self.config.pos_func}")


def run_test():
    B, H, D, V = 4, 28 * 28, 9, 2
    mt_head = CPDME(
        AbstractDisributionHeadConfig(
            d_model=D,
            d_output=V,
            horizon=H,
            rank=8,
            pos_func="exp_lse",
        ),
    )
    x = torch.randn(B, D)
    y = torch.randint(0, V, (B, H))
    # # set some 50% of y to ignore_index
    # for i, j in itertools.product(range(B), range(H)):
    #     if random.random() < 0.5:
    #         y[i, j] = -100

    # out = mt_head(x, y)
    # print(f"loss: {out.loss}")

    out = mt_head.generate(x)
    print(f"generated: {out}")


if __name__ == "__main__":
    run_test()
