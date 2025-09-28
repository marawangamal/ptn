import unittest
import torch

from ptn.dists._abc import AbstractDisributionHeadConfig
from ptn.dists.mps_bm_dmrg import (
    MPS_BM_DMRG,
    compute_lr_marginalization_cache,
    compute_lr_selection_cache,
)


class TestTensorOpsMpsBmDmrg(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    #
    # #########################################################
    #                Test MPS_BM_DMRG class                  #
    # #########################################################
    #

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
        for _ in range(5):
            # Train example
            x, y = torch.randn(B, 1), torch.randint(0, Do, (B, H))
            losses = model.train_example(x, y, n_sweeps=1, max_bond_dim=32)
            print(f"loss: {torch.stack(losses).mean():.2f}")

    #
    # #########################################################
    #                Test Cache Functions                    #
    # #########################################################
    #

    def test_margin_cache_validity(self):
        ml, mr = compute_lr_marginalization_cache(g, pad=False)

    def test_selection_cache_validity(self):
        pass


if __name__ == "__main__":
    unittest.main()
