import unittest
import torch
from tqdm import tqdm

from ptn.dists.tensorops.mps import (
    mps_norm_batch,
    born_select_margin_batch,
)


class TestTensorOpsMpsSigmaLsf(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_select_margin_mps_tensor_batched_validity(self):
        for _ in tqdm(range(10_000)):
            B, R, H, Do = 1, 8, 2, 128
            a, b, g = (
                torch.randn(B, R).abs(),
                torch.randn(B, R).abs(),
                torch.randn(B, H, R, Do, R).abs(),
            )
            y = torch.randint(0, Do, (B, H))
            y_ = torch.nn.functional.one_hot(y, num_classes=Do).float()
            p_tilde, _ = born_select_margin_batch(g, a, b, y_, use_scale_factors=False)
            z_tilde, _ = mps_norm_batch(g, a, b, use_scale_factors=False)
            loss = (z_tilde - p_tilde).min()

            # loss is non-negative
            self.assertTrue(loss >= 0)

        for _ in tqdm(range(10_000)):
            B, R, H, Do = 1, 8, 2, 128
            a, b, g = (
                torch.randn(B, R).abs(),
                torch.randn(B, R).abs(),
                torch.randn(B, H, R, Do, R).abs(),
            )
            y = torch.randint(0, Do, (B, H))
            y_ = torch.nn.functional.one_hot(y, num_classes=Do).float()
            p_tilde, gammas_p = born_select_margin_batch(
                g, a, b, y_, use_scale_factors=False
            )
            z_tilde, gammas_z = mps_norm_batch(g, a, b, use_scale_factors=False)

            loss = (1 / H) * (  # avg across seq dimension
                -2 * torch.log(p_tilde.abs())  # (B,)
                + torch.log(z_tilde)  # (B,)
                # Contraction Stability Scale Factors
                - (2 * torch.log(gammas_p.abs()).sum(dim=-1))  # (B, H)
                + (gammas_z.log().sum(dim=-1))  # (B, H)
            ).mean()  # avg across batch dimension

            # loss is non-negative
            if loss < 0:
                print(f"Loss is negative: {loss}")
                print(f"p_tilde: {p_tilde}")
                print(f"z_tilde: {z_tilde}")
                print(f"gammas_p: {gammas_p}")
                print(f"gammas_z: {gammas_z}")
                raise ValueError("Loss is negative")
            self.assertTrue(loss >= 0)


if __name__ == "__main__":
    unittest.main()
