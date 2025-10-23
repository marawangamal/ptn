# decns_unittest.py
import unittest
import torch

# ------------ Implementation ------------


def dec2ns(x: torch.Tensor, base: int, bits: int) -> torch.Tensor:
    if base < 2:
        raise ValueError("base must be >= 2")
    if bits < 1:
        raise ValueError("bits must be >= 1")
    xl = x.to(torch.long)
    if (xl < 0).any():
        raise ValueError("non-negative integers only")
    tmp = xl.clone()
    digs = []
    for _ in range(bits):
        digs.append(tmp.remainder(base))
        tmp = torch.div(tmp, base, rounding_mode="floor")
    return torch.stack(digs[::-1], dim=-1).to(x.dtype)


def ns2dec(d: torch.Tensor, base: int, bits: int) -> torch.Tensor:
    if base < 2:
        raise ValueError("base must be >= 2")
    if d.size(-1) != bits:
        raise ValueError(f"last dim must be {bits}")
    dl = d.to(torch.long)
    if torch.any((dl < 0) | (dl >= base)):
        raise ValueError("digit out of range")
    exps = torch.arange(bits - 1, -1, -1, device=d.device, dtype=torch.long)
    weights = torch.tensor(base, device=d.device, dtype=torch.long) ** exps
    return torch.sum(dl * weights, dim=-1).to(d.dtype)


# ------------ Tests (builtin unittest) ------------


class TestDecNs(unittest.TestCase):
    def test_roundtrip_random(self):
        cases = [(2, 4), (3, 5), (4, 6), (7, 4), (10, 3), (16, 4)]
        for base, bits in cases:
            with self.subTest(base=base, bits=bits):
                torch.manual_seed(0)
                n = 100
                x = torch.randint(0, base**bits, (n,))
                d = dec2ns(x, base, bits)
                xb = ns2dec(d, base, bits)
                self.assertTrue(torch.equal(xb.to(torch.long), x.to(torch.long)))

    def test_overflow_truncates_msd(self):
        for base, bits in [(2, 5), (3, 4), (10, 2), (16, 3)]:
            with self.subTest(base=base, bits=bits):
                torch.manual_seed(1)
                x = torch.randint(0, 10 * base**bits, (64,))
                d = dec2ns(x, base, bits)
                xb = ns2dec(d, base, bits)
                self.assertTrue(
                    torch.equal(xb.to(torch.long), (x % (base**bits)).to(torch.long))
                )

    def test_shape_and_dtype(self):
        x = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.float32)
        d = dec2ns(x, base=3, bits=4)
        self.assertEqual(d.shape, (2, 3, 4))
        self.assertEqual(d.dtype, torch.float32)
        xb = ns2dec(d, base=3, bits=4)
        self.assertEqual(xb.shape, (2, 3))
        self.assertEqual(xb.dtype, torch.float32)

    def test_zero_and_max_edge(self):
        base, bits = 7, 5
        maxv = base**bits - 1
        x = torch.tensor([0, maxv])
        d = dec2ns(x, base, bits)
        xb = ns2dec(d, base, bits)
        self.assertTrue(torch.equal(xb.to(torch.long), x.to(torch.long)))

    def test_invalid_digits_raises(self):
        base, bits = 5, 3
        ok = torch.tensor([[0, 1, 4]])
        _ = ns2dec(ok, base, bits)  # no raise
        with self.assertRaises(ValueError):
            ns2dec(torch.tensor([[-1, 0, 0]]), base, bits)
        with self.assertRaises(ValueError):
            ns2dec(torch.tensor([[0, 5, 0]]), base, bits)

    def test_param_validation(self):
        x = torch.tensor([1, 2, 3])
        with self.assertRaises(ValueError):
            dec2ns(x, base=1, bits=3)
        with self.assertRaises(ValueError):
            dec2ns(x, base=2, bits=0)
        with self.assertRaises(ValueError):
            dec2ns(torch.tensor([-1, 0]), base=2, bits=3)
        with self.assertRaises(ValueError):
            ns2dec(torch.zeros(2, 4), base=2, bits=3)

    def test_float_inputs_supported(self):
        x = torch.tensor([0.0, 5.0, 10.0], dtype=torch.float64)
        d = dec2ns(x, base=2, bits=4)
        self.assertEqual(d.dtype, torch.float64)
        xb = ns2dec(d, base=2, bits=4)
        self.assertEqual(xb.dtype, torch.float64)
        self.assertTrue(torch.equal(xb.to(torch.long), (x.to(torch.long) % 16)))

    def test_batch_shapes(self):
        x = torch.arange(8).reshape(2, 4)
        d = dec2ns(x, base=2, bits=3)
        self.assertEqual(d.shape, (2, 4, 3))
        xb = ns2dec(d, base=2, bits=3)
        self.assertTrue(torch.equal(xb.to(torch.long), (x % 8).to(torch.long)))


# ------------ Main ------------


def main():
    print("Running dec2ns/ns2dec tests (unittest)...")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestDecNs)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        print("✅ All tests passed.")
    else:
        print("❌ Some tests failed.")


if __name__ == "__main__":
    main()
