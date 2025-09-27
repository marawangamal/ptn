# test_einlogsumexp_unittest.py
import unittest
import math
import torch

from ptn.mheads._tensorops import (
    mps_reduce_decoder_einlse_margin_only,
    mps_reduce_decoder_einlse_select_only,
    mps_reduce_decoder_margin_only,
    mps_reduce_decoder_select_only,
)

# If einlogsumexp is in another module, import it:
# from your_module import einlogsumexp


class TestEinLogSumExp(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_mps_validity_einlse(self):
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

    def test_mps_select_only(self):
        R, H, Do, V = 2, 4, 8, 10
        mps_params_tilde = torch.randn(H, R, Do, R).abs()
        decoder = torch.randn(Do, V).abs()
        ops = torch.randint(0, V, (H,))
        res_1 = mps_reduce_decoder_einlse_select_only(
            mps_params_tilde, ops, decoder, apply_logsumexp=False
        )
        res_2 = mps_reduce_decoder_select_only(
            mps_params_tilde, ops, decoder, apply_scale_factors=False
        )
        self.assertTrue(torch.allclose(res_1, res_2))

    def test_mps_margin_only(self):
        R, H, Do, V = 2, 4, 8, 10
        mps_params_tilde = torch.randn(H, R, Do, R).abs()
        decoder = torch.randn(Do, V).abs()
        res_1 = mps_reduce_decoder_einlse_margin_only(
            mps_params_tilde, decoder, apply_logsumexp=False
        )
        res_2 = mps_reduce_decoder_margin_only(
            mps_params_tilde, decoder, apply_scale_factors=False
        )
        self.assertTrue(torch.allclose(res_1, res_2))

    def test_mps_validity(self):
        for _ in range(50):
            R, H, Do, V = 8, 2, 128, 30_000
            mps_params_tilde = torch.randn(H, R, Do, R).abs()
            decoder = torch.randn(Do, V).abs()
            ops = torch.randint(0, V, (H,))
            log_p_tilde = mps_reduce_decoder_select_only(mps_params_tilde, ops, decoder)
            log_z = mps_reduce_decoder_margin_only(mps_params_tilde, decoder)
            loss = (log_z - log_p_tilde).mean()

            # loss is non-negative
            self.assertTrue(loss >= 0)


if __name__ == "__main__":
    unittest.main()
