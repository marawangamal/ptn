import torch
import torch.nn as nn

from mtp.mheads._abc import (
    AbstractDisributionHead,
    AbstractDisributionHeadConfig,
    AbstractDisributionHeadOutput,
)


def cp_select(cp_params: torch.Tensor, cp_decoder: torch.Tensor, index: torch.Tensor):
    """CP Select/Margin operator.

    Args:
        cp_params (torch.Tensor): CP Parameters. Shape: (R, H, Dh)
        cp_decoder (torch.Tensor): CP Decoder. Shape: (V, Dh)
        index (torch.Tensor): Selection. Shape: (H,)


    Returns:
        tuple:
            - logits: (B, F, V). F is the number of free legs.
            - scale factors: (B, H).
    """
    R, H, _ = cp_params.shape
    res = torch.ones(R, device=cp_params.device, dtype=cp_params.dtype)
    for h in range(H):
        decoder_yh = cp_decoder[index[h]]  # (Dh,)
        res = res * (cp_params[:, h] @ decoder_yh)
    return res.sum(dim=-1)


batch_cp_select = torch.vmap(cp_select, in_dims=(0, 0, 0))


class CP(AbstractDisributionHead):
    def __init__(self, config: AbstractDisributionHeadConfig):
        """Simple multi-head distribution with independent linear heads for each position."""
        super().__init__(config)

        # Separate linear heads for each position
        self.alpha = torch.nn.Parameter(torch.randn(config.rank))
        self.w_theta = torch.nn.Parameter(torch.randn(config.rank, config.horizon))
        self.decoder = nn.Linear(config.d_model, config.d_output)

    # def set_output_embeddings(self, embeddings: torch.Tensor):
    #     V, D = embeddings.shape
    #     assert embeddings.shape == (
    #         self.config.d_output,
    #         self.config.d_model,
    #     ), "embeddings shape must be (V, D)"
    #     u, s, vt = torch.svd(embeddings)  # (V, R), (R,), (R, D)
    #     self.decoder.weight = u[:, :D]
    #     self.heads[0].weight = s[:D] * vt[:D]  # (D, D)

    # def get_output_embeddings(self):
    #     return torch.einsum("vo,oi->vi", self.decoder.weight, self.heads[0].weight)

    def forward(self, x, y=None):
        # if y is none (eval), only compute logits for the first head
        H_ = 1 if y is None else self.config.horizon
        logits = torch.stack(
            [self.decoder(self.heads[h](x)) for h in range(H_)], dim=1
        )  # (B, H_, V)
        loss = None
        if y is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, self.config.d_output), y.reshape(-1)
            )
        return AbstractDisributionHeadOutput(logits=logits[:, 0], loss=loss)


def test_cp_select_margin():
    R, H, Dh = 2, 5, 3
    V = 4
    B = 2
    cp_params = torch.randn(R, H, Dh)
    cp_decoder = torch.randn(V, Dh)
    ops = torch.randint(0, V, (H,))
    out = cp_select(cp_params, cp_decoder, ops)
    print(out)
    return out is not None


def test_batch_cp_select_margin():
    B, R, H, Dh, V = 2, 5, 3, 4, 4
    cp_params = torch.randn(B, R, H, Dh)
    cp_decoder = torch.randn(B, V, Dh)
    ops = torch.randint(0, V, (B, H))
    out = batch_cp_select(cp_params, cp_decoder, ops)
    print(out)
    return out.shape == (B,)


def run_test():
    tests = [
        {
            "name": "cp_select_margin",
            "fn": test_cp_select_margin,
        },
        {
            "name": "batch_cp_select_margin",
            "fn": test_batch_cp_select_margin,
        },
    ]

    for test in tests:
        succ = "Pass" if test["fn"]() else "Fail"
        print(f"[{test['name']}]: {succ}")


if __name__ == "__main__":
    run_test()
