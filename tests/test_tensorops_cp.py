# test_einlogsumexp_unittest.py
import unittest
import math
import torch

from ptn.mheads._tensorops import (
    cp_reduce_decoder_einlse_margin_only,
    cp_reduce_decoder,
    cp_reduce_decoder_einlse,
    cp_reduce,
    cp_reduce_decoder_einlse_select_only,
    cp_reduce_decoder_margin_only,
)

# If einlogsumexp is in another module, import it:
# from your_module import einlogsumexp


class TestEinLogSumExp(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_reduce_equals_reduce_decoder(self):
        R, H, Di, Do, V = 8, 4, 386, 128, 30_000

        cp_params_tilde = torch.randn(R, H, Di)
        decoder = torch.randn(Di, V)
        ops = torch.randint(0, V, (H,))
        res_v1, _ = cp_reduce(
            torch.einsum("rhd,dv->rhv", cp_params_tilde, decoder),
            ops,
            use_scale_factors=False,
        )
        res_v2, _ = cp_reduce_decoder(
            cp_params_tilde,
            ops,
            decoder,
            use_scale_factors=False,
        )
        self.assertTrue(torch.allclose(res_v1, res_v2, atol=1))

    def test_cp_reduce(self):
        R, H, Di, Do, V = 8, 4, 386, 128, 2
        cp_params_tilde = torch.randn(R, H, Di).abs()
        ops = torch.randint(0, V, (H,))
        res_1, _ = cp_reduce(cp_params_tilde, ops, except_index=0)
        self.assertTrue(torch.isfinite(res_1).all())

    def test_reduce_equals_reduce_decoder_einlse(self):
        R, H, Di, Do, V = 8, 4, 386, 128, 30_000

        cp_params_tilde = torch.randn(R, H, Di)
        decoder = torch.randn(Di, V)
        ops = torch.randint(0, V, (H,))
        res_v1, _ = cp_reduce(
            torch.einsum("rhd,dv->rhv", cp_params_tilde, decoder),
            ops,
            use_scale_factors=False,
        )
        res_v2 = cp_reduce_decoder_einlse(
            cp_params_tilde,
            ops,
            decoder,
            apply_logsumexp=False,
        )
        self.assertTrue(torch.allclose(res_v1, res_v2, atol=1))

    def test_reduce_equals_value(self):
        R, H, Di, Do, V = 8, 4, 386, 128, 30_000

        cp_params_tilde = torch.eye(R, R).unsqueeze(1).repeat(1, H, 1)
        decoder = torch.eye(R, R)
        ops = torch.zeros((H,), dtype=torch.long)
        res_v1, _ = cp_reduce(
            torch.einsum("rhd,dv->rhv", cp_params_tilde, decoder),
            ops,
            use_scale_factors=False,
        )
        res_v2 = cp_reduce_decoder_einlse(
            cp_params_tilde,
            ops,
            decoder,
            apply_logsumexp=False,
        )
        self.assertTrue(torch.allclose(res_v1, res_v2, atol=1))

    def test_reduce_decoder_equals_reduce_decoder_einlse(self):
        R, H, Di, Do, V = 8, 4, 386, 128, 30_000

        cp_params_tilde = torch.randn(R, H, Di)
        decoder = torch.randn(Di, V)
        ops = torch.randint(0, V, (H,))
        res_v1, _ = cp_reduce_decoder(
            # torch.einsum("rhd,dv->rhv", cp_params_tilde, decoder),
            cp_params_tilde,
            ops,
            decoder,
            use_scale_factors=False,
        )
        res_v2 = cp_reduce_decoder_einlse(
            cp_params_tilde,
            ops,
            decoder,
            apply_logsumexp=False,
        )
        self.assertTrue(torch.allclose(res_v1, res_v2, atol=1))

    def test_reduce_decoder_equals_reduce_decoder_einlse_with_lse(self):
        R, H, Di, Do, V = 8, 4, 386, 128, 30_000

        cp_params_tilde = torch.randn(R, H, Di)
        decoder = torch.randn(Di, V)
        # ops = torch.randint(0, V, (H,))
        ops = torch.full((H,), -1, dtype=torch.long)
        res_v1, _ = cp_reduce_decoder(
            cp_params_tilde.exp(),
            ops,
            decoder.exp(),
            use_scale_factors=False,
        )
        res_v2 = cp_reduce_decoder_einlse_margin_only(
            cp_params_tilde,
            decoder,
            apply_logsumexp=True,
        )
        self.assertTrue(torch.allclose(torch.log(res_v1), res_v2, atol=1))

    def test_cp_marginalization(self):
        R, H, Di, Do, V = 8, 4, 386, 128, 30_000
        cp_params_tilde = torch.randn(R, H, Di)
        decoder = torch.randn(Di, V)
        res_1 = cp_reduce_decoder_einlse_margin_only(
            cp_params_tilde.exp(),
            decoder.exp(),
            apply_logsumexp=False,
        )
        res_2 = cp_reduce_decoder_einlse_margin_only(
            cp_params_tilde,
            decoder,
            apply_logsumexp=True,
        )
        self.assertTrue(torch.allclose(torch.log(res_1), res_2, atol=1))

    def test_cp_validity(self):
        for _ in range(50):
            R, H, Di, Do, V = 8, 4, 386, 128, 30_000
            cp_params_tilde = torch.randn(R, H, Di)
            decoder = torch.randn(Di, V)
            ops = torch.randint(0, V, (H,))
            log_p_tilde = cp_reduce_decoder_einlse(cp_params_tilde, ops, decoder)
            log_z = cp_reduce_decoder_einlse_margin_only(
                cp_params_tilde,
                decoder,
                apply_logsumexp=True,
            )
            loss = (log_z - log_p_tilde).mean()

            # loss is non-negative
            self.assertTrue(loss >= 0)

    def test_cp_reduce_decoder_einsle_marginal(self):
        R, H, Di, Do, V = 8, 4, 386, 128, 30_000
        cp_params_tilde = torch.randn(R, H, Di)
        decoder = torch.randn(Di, V)
        ops = torch.full((H,), -1, dtype=torch.long)
        res_1 = cp_reduce_decoder_einlse(cp_params_tilde, ops, decoder)
        res_2 = cp_reduce_decoder_einlse_margin_only(
            cp_params_tilde,
            decoder,
            apply_logsumexp=True,
        )
        self.assertTrue(torch.allclose(res_1, res_2, atol=1))

    def test_cp_reduce_decoder_einsle_select_only(self):
        R, H, Di, Do, V = 8, 4, 386, 128, 30_000
        cp_params_tilde = torch.randn(R, H, Di)
        decoder = torch.randn(Di, V)
        ops = torch.randint(0, V, (H,))
        res_1 = cp_reduce_decoder_einlse_select_only(cp_params_tilde, ops, decoder)
        res_2 = cp_reduce_decoder_einlse(cp_params_tilde, ops, decoder)
        self.assertTrue(torch.allclose(res_1, res_2, atol=1))

    # def test_cp_reduce_decoder_margin_only(self):
    #     R, H, Di, Do, V = 8, 4, 10, 10, 30
    #     cp_params_tilde = torch.randn(R, H, Di).abs()
    #     decoder = torch.randn(Di, V).abs()
    #     ops = torch.full((H,), -1, dtype=torch.long)
    #     res_1, _ = cp_reduce_decoder(
    #         cp_params_tilde,
    #         ops,
    #         decoder,
    #         use_scale_factors=False,
    #     )
    #     res_2 = cp_reduce_decoder_margin_only(
    #         cp_params_tilde,
    #         decoder,
    #         apply_scale_factors=False,
    #     )
    #     self.assertTrue(torch.allclose(res_1.log(), res_2, atol=1))


if __name__ == "__main__":
    unittest.main()
