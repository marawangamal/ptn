# test_einlogsumexp_unittest.py
import unittest
import math
import torch

from mtp.mheads._tensorops import (
    cp_reduce_decoder_einlse_margin_only,
    cp_reduce_decoder,
    cp_reduce_decoder_einlse,
    cp_reduce,
    cp_reduce_decoder_einlse_select_only,
    mps_reduce_decoder_einlse_margin_only,
    mps_reduce_decoder_einlse_select_only,
)

# If einlogsumexp is in another module, import it:
# from your_module import einlogsumexp


class TestEinLogSumExp(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_mps_validity(self):
        for _ in range(50):
            R, H, Do, V = 8, 2, 128, 30_000
            mps_params_tilde = torch.randn(H, R, Do, R)
            decoder = torch.randn(Do, V)
            ops = torch.randint(0, V, (H,))
            log_p_tilde = mps_reduce_decoder_einlse_select_only(
                mps_params_tilde, ops, decoder
            )
            log_z = mps_reduce_decoder_einlse_margin_only(
                mps_params_tilde,
                decoder,
                apply_logsumexp=True,
            )
            loss = (log_z - log_p_tilde).mean()

            # loss is non-negative
            self.assertTrue(loss >= 0)


if __name__ == "__main__":
    unittest.main()
