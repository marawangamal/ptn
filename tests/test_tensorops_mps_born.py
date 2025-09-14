# test_einlogsumexp_unittest.py
import unittest
import math
import torch

from ctn.mheads.tensorops.mps_born import born_mps_marginalize, born_mps_select

# If einlogsumexp is in another module, import it:
# from your_module import einlogsumexp


class TestEinLogSumExp(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_born_mps_marginalize(self):
        for _ in range(50):
            R, H, D = 8, 2, 128
            g = torch.randn(H, R, D, R)  # cores
            a = torch.randn(R)
            b = torch.randn(R)
            z = born_mps_marginalize(g, a, b)
            self.assertTrue(z >= 0)  # z is non-negative

    def test_born_mps_select(self):
        for _ in range(50):
            R, H, D = 8, 2, 128
            g = torch.randn(H, R, D, R)  # cores
            a = torch.randn(R)
            b = torch.randn(R)
            y = torch.randint(0, R, (H,))
            py = born_mps_select(g, a, b, y)
            self.assertTrue(py >= 0)  # py is non-negative

    def test_born_mps(self):
        for _ in range(50):
            R, H, D = 8, 2, 128
            g = torch.randn(H, R, D, R)  # cores
            a = torch.randn(R)
            b = torch.randn(R)
            y = torch.randint(0, R, (H,))
            py = born_mps_select(g, a, b, y)
            z = born_mps_marginalize(g, a, b)
            self.assertTrue(z >= py)  # valid distribution


if __name__ == "__main__":
    unittest.main()
