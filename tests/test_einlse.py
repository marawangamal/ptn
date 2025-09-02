# test_einlogsumexp_unittest.py
import unittest
import math
import torch

from mtp.mheads.einlse import einlogsumexp

# If einlogsumexp is in another module, import it:
# from your_module import einlogsumexp


def ref_log_einsum(subscripts, *ops_log):
    """Reference via prob-space: einsum(exp(...)) then log."""
    ops = [x.exp() for x in ops_log]
    out = torch.einsum(subscripts, *ops)
    return out.log()


# Add below your imports / helpers
def ref_log_einsum_list(*args):
    """Reference for list/ellipsis einsum: einsum(exp(...)) then log."""
    args_exp = [(a.exp() if isinstance(a, torch.Tensor) else a) for a in args]
    out = torch.einsum(*args_exp)
    return out.log()


class TestEinLogSumExp(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_log_matmul_matches_probspace(self):
        I, K, J = 3, 4, 5
        A = torch.randn(I, K) * 0.2
        B = torch.randn(K, J) * 0.2
        out = einlogsumexp("ik,kj->ij", A, B)
        ref = ref_log_einsum("ik,kj->ij", A, B)
        self.assertTrue(torch.allclose(out, ref, atol=1e-6, rtol=1e-6))

    def test_batched_log_matmul(self):
        Bsz, I, K, J = 2, 3, 4, 5
        A = torch.randn(Bsz, I, K) * 0.1
        Bm = torch.randn(Bsz, K, J) * 0.1
        out = einlogsumexp("bik,bkj->bij", A, Bm)
        ref = ref_log_einsum("bik,bkj->bij", A, Bm)
        self.assertTrue(torch.allclose(out, ref, atol=1e-6, rtol=1e-6))

    def test_permutation_of_output_labels(self):
        I, K, J = 2, 3, 4
        A = torch.randn(I, K) * 0.2
        B = torch.randn(K, J) * 0.2
        out = einlogsumexp("ik,kj->ji", A, B)
        ref = ref_log_einsum("ik,kj->ji", A, B)
        self.assertTrue(torch.allclose(out, ref, atol=1e-6, rtol=1e-6))

    def test_scalar_result_all_contracted(self):
        I, J = 3, 4
        A = torch.randn(I, J) * 0.2
        B = torch.randn(I, J) * 0.2
        out = einlogsumexp("ij,ij->", A, B)
        ref = ref_log_einsum("ij,ij->", A, B)
        self.assertEqual(out.shape, torch.Size([]))
        self.assertTrue(torch.allclose(out, ref, atol=1e-6, rtol=1e-6))

    # def test_broadcasting_missing_labels(self):
    #     I, K, J, Bsz = 3, 4, 5, 7
    #     A = torch.randn(I, K) * 0.2
    #     Bm = torch.randn(Bsz, K, J) * 0.2
    #     out = einlogsumexp("ik,bkj->bij", A, Bm)
    #     ref = ref_log_einsum("ik,bkj->bij", A, Bm)
    #     self.assertTrue(torch.allclose(out, ref, atol=1e-6, rtol=1e-6))

    def test_numerical_stability_large(self):
        I, K, J = 3, 4, 5
        A = 1000.0 + torch.randn(I, K)
        B = 1000.0 + torch.randn(K, J)
        out = einlogsumexp("ik,kj->ij", A, B)
        self.assertTrue(torch.isfinite(out).all())
        approx = 2000.0 + math.log(K)  # rough magnitude check
        self.assertTrue((out > approx - 10.0).all())

    def test_mixture_marginalization(self):
        R, H = 5, 11
        w = torch.randn(R)
        cond = torch.randn(R, H)
        out = einlogsumexp("r,rh->h", w, cond)
        ref = ref_log_einsum("r,rh->h", w, cond)
        self.assertTrue(torch.allclose(out, ref, atol=1e-6, rtol=1e-6))

    # def test_autograd_backward(self):
    #     I, K, J = 2, 3, 2
    #     A = torch.randn(I, K, dtype=torch.double, requires_grad=True) * 0.05
    #     B = torch.randn(K, J, dtype=torch.double, requires_grad=True) * 0.05
    #     out = einlogsumexp("ik,kj->ij", A, B).sum()
    #     out.backward()
    #     self.assertIsNotNone(A.grad)
    #     self.assertIsNotNone(B.grad)
    #     self.assertTrue(torch.isfinite(A.grad).all())
    #     self.assertTrue(torch.isfinite(B.grad).all())

    # def test_mismatched_operands_raises(self):
    #     X = torch.randn(2, 3)
    #     with self.assertRaises(AssertionError):
    #         einlogsumexp("ij,kl->ik", X)  # expects 2 operands, got 1

    # def test_invalid_rhs_labels_raises(self):
    #     X = torch.randn(2, 3)
    #     with self.assertRaises(ValueError):
    #         einlogsumexp("ij->ik", X)  # 'k' never appears on LHS

    def test_list_basic_batched_matmul(self):
        # A: (B, I, K), B: (B, K, J) -> (B, I, J)
        Bsz, I, K, J = 3, 4, 5, 6
        A = torch.randn(Bsz, I, K) * 0.1
        Bm = torch.randn(Bsz, K, J) * 0.1

        out = einlogsumexp(A, [..., 0, 1], Bm, [..., 1, 2], [..., 0, 2])
        ref = ref_log_einsum_list(A, [..., 0, 1], Bm, [..., 1, 2], [..., 0, 2])

        self.assertEqual(out.shape, (Bsz, I, J))
        self.assertTrue(torch.allclose(out, ref, atol=1e-6, rtol=1e-6))

    def test_list_broadcast_batch(self):
        # A: (B1, B2, I, K), B: (1,  B2, K, J) -> (B1, B2, I, J)
        B1, B2, I, K, J = 2, 3, 4, 5, 6
        A = torch.randn(B1, B2, I, K) * 0.1
        Bm = torch.randn(1, B2, K, J) * 0.1  # broadcast over B1

        out = einlogsumexp(A, [..., 0, 1], Bm, [..., 1, 2], [..., 0, 2])
        ref = ref_log_einsum_list(A, [..., 0, 1], Bm, [..., 1, 2], [..., 0, 2])

        self.assertEqual(out.shape, (B1, B2, I, J))
        self.assertTrue(torch.allclose(out, ref, atol=1e-6, rtol=1e-6))

    def test_list_different_ellipsis_widths(self):
        # Different numbers of leading (ellipsis) dims per operand
        # A: (E1, E2, I, K), B: (E2, K, J) -> broadcast ellipsis to (E1, E2)
        E1, E2, I, K, J = 2, 3, 4, 5, 6
        A = torch.randn(E1, E2, I, K) * 0.1
        Bm = torch.randn(E2, K, J) * 0.1  # missing E1 -> broadcast

        out = einlogsumexp(A, [..., 0, 1], Bm, [..., 1, 2], [..., 0, 2])
        ref = ref_log_einsum_list(A, [..., 0, 1], Bm, [..., 1, 2], [..., 0, 2])

        self.assertEqual(out.shape, (E1, E2, I, J))
        self.assertTrue(torch.allclose(out, ref, atol=1e-6, rtol=1e-6))

    def test_list_full_contraction_scalar(self):
        # A: (I,), B: (I,) -> scalar via []
        I = 7
        a = torch.randn(I) * 0.1
        b = torch.randn(I) * 0.1

        out = einlogsumexp(a, [0], b, [0], [])
        ref = ref_log_einsum_list(a, [0], b, [0], [])

        self.assertEqual(out.shape, torch.Size([]))
        self.assertTrue(torch.allclose(out, ref, atol=1e-6, rtol=1e-6))

    def test_list_permute_output_order(self):
        # A: (B, I, K), B: (B, K, J) -> output as (B, J, I) using [..., 2, 0]
        Bsz, I, K, J = 2, 3, 4, 5
        A = torch.randn(Bsz, I, K) * 0.1
        Bm = torch.randn(Bsz, K, J) * 0.1

        out = einlogsumexp(A, [..., 0, 1], Bm, [..., 1, 2], [..., 2, 0])
        ref = ref_log_einsum_list(A, [..., 0, 1], Bm, [..., 1, 2], [..., 2, 0])

        self.assertEqual(out.shape, (Bsz, J, I))
        self.assertTrue(torch.allclose(out, ref, atol=1e-6, rtol=1e-6))

    def test_list_three_operands_chain(self):
        # (..., i,k) * (..., k,j) * (..., j,n) -> (..., i,n)
        B1, B2, I, K, J, N = 2, 1, 3, 4, 5, 2
        A = torch.randn(B1, B2, I, K) * 0.1
        Bm = torch.randn(1, B2, K, J) * 0.1
        C = torch.randn(B1, 1, J, N) * 0.1

        out = einlogsumexp(A, [..., 0, 1], Bm, [..., 1, 2], C, [..., 2, 3], [..., 0, 3])
        ref = ref_log_einsum_list(
            A, [..., 0, 1], Bm, [..., 1, 2], C, [..., 2, 3], [..., 0, 3]
        )

        self.assertEqual(out.shape, (B1, B2, I, N))
        self.assertTrue(torch.allclose(out, ref, atol=1e-6, rtol=1e-6))


def showcase():
    B, D, V = 10, 100, 1000
    a, b = torch.rand(B, D).square() * 1000, torch.randn(D, V).square() * 1000
    torch.einsum("bd,dv->db", a.exp(), b.exp()).isfinite().all()
    # >>> False
    einlogsumexp("bd,dv->db", a, b).isfinite().all()
    # >>> True


if __name__ == "__main__":
    unittest.main()
