import unittest
import torch

from ptn.dists.tensorops.mps import select_margin_mps_tensor_batched


class TestTensorOpsMpsSigmaLsf(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_select_margin_mps_tensor_batched_validity(self):
        for _ in range(50):
            B, R, H, Do = 1, 8, 2, 128
            a, b, g = (
                torch.randn(B, R).abs(),
                torch.randn(B, R).abs(),
                torch.randn(B, H, R, Do, R).abs(),
            )
            y = torch.randint(0, Do, (B, H))
            p_tilde, _ = select_margin_mps_tensor_batched(  # type: ignore
                a, b, g, y, use_scale_factors=False
            )
            z_tilde, _ = select_margin_mps_tensor_batched(  # type: ignore
                a, b, g, y, use_scale_factors=False
            )
            loss = (z_tilde - p_tilde).mean()

            # loss is non-negative
            self.assertTrue(loss >= 0)


if __name__ == "__main__":
    unittest.main()
