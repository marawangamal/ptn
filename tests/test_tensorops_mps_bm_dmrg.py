import unittest
import torch

from ptn.dists._abc import AbstractDisributionHeadConfig
from ptn.dists.mps_bm_dmrg import MPS_BM_DMRG


class TestTensorOpsMpsBmDmrg(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_select_margin_mps_tensor_batched_validity(self):
        B, R, H, Do = 1, 2, 28 * 28, 2
        model = MPS_BM_DMRG(
            AbstractDisributionHeadConfig(
                d_model=1,
                d_output=Do,
                horizon=H,
                rank=R,
            )
        )
        for _ in range(50):
            # Train example
            x, y = torch.randn(B, 1), torch.randint(0, Do, (B, H))
            losses = model.train_example(x, y, n_sweeps=1, max_bond_dim=32)
            print(f"loss: {torch.stack(losses).mean():.2f}")


if __name__ == "__main__":
    unittest.main()
