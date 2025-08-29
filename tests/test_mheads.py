import unittest
import torch

from mtp.mheads import MHEADS
from mtp.mheads._abc import AbstractDisributionHeadConfig


class TestMHeads(unittest.TestCase):
    def test_moe(self):
        B, T, R, H, D, V = 1, 16, 4, 8, 512, 30000

        for model_name in ["moe", "moe_proj"]:
            model = MHEADS[model_name](
                AbstractDisributionHeadConfig(horizon=H, rank=R, d_model=D, d_output=V)
            )
            x = torch.randn(B, T, D)
            y = torch.randint(0, V, (B, T))
            output_seq = model.forward_seq(x, y)
            self.assertEqual(
                output_seq.logits.shape,
                (B, T, min(H, T), V),
                "logits should be (B, H, V)",
            )

            # output forward
            x = torch.randn(B, D)
            y = torch.randint(0, V, (B, H))
            output_forward = model(x, y)
            self.assertEqual(
                output_forward.logits.shape,
                (B, H, V),
                "logits should be (B, H, V)",
            )

    def test_stp(self):
        B, T, R, H, D, V = 1, 16, 4, 1, 512, 30000

        for model_name in ["stp"]:
            model = MHEADS[model_name](
                AbstractDisributionHeadConfig(horizon=H, rank=R, d_model=D, d_output=V)
            )
            x = torch.randn(B, T, D)
            y = torch.randint(0, V, (B, T))
            output_seq = model.forward_seq(x, y)
            self.assertEqual(
                output_seq.logits.shape,
                (B, T, min(H, T), V),
                "logits should be (B, H, V)",
            )

            # output forward
            x = torch.randn(B, D)
            y = torch.randint(0, V, (B, H))
            output_forward = model(x, y)
            self.assertEqual(
                output_forward.logits.shape,
                (B, H, V),
                "logits should be (B, H, V)",
            )


if __name__ == "__main__":
    unittest.main()
