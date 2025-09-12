# test_einlogsumexp_unittest.py
import unittest
import math
import torch

from mtp.mheads.mps_born_unconditional import canonicalize

# If einlogsumexp is in another module, import it:
# from your_module import einlogsumexp


def flatten(lst):
    return [item for sublist in lst for item in sublist]


class TestEinLogSumExp(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_born_mps_cano_leaves_map_unchanged(self):
        R, H, D = 8, 5, 2
        g = torch.randn(H, R, D, R)  # cores
        a = torch.eye(R)
        b = torch.eye(R)
        gc = canonicalize(g.clone(), [2])
        res_1 = torch.einsum(
            g[0],
            [0, 1, 2],
            g[1],
            [2, 3, 4],
            g[2],
            [4, 5, 6],
            g[3],
            [6, 7, 8],
            g[4],
            [8, 9, 10],
            [0, 10],
        )
        res_2 = torch.einsum(
            gc[0],
            [0, 1, 2],
            gc[1],
            [2, 3, 4],
            gc[2],
            [4, 5, 6],
            gc[3],
            [6, 7, 8],
            gc[4],
            [8, 9, 10],
            [0, 10],
        )
        self.assertTrue(torch.allclose(res_1, res_2, atol=1e-4), "Contraction mismatch")
        self.assertTrue(not torch.allclose(g, gc))

    def test_born_mps_cano_maringals(self):
        # test exact marginal and cano marginal match
        pass


if __name__ == "__main__":
    unittest.main()
