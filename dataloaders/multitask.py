import numpy as np


class SimpleTiedHeadsGen:
    """
    Simplest tied multi-task generator:
      X ~ N(0, I_d)          (continuous)
      g ~ N(0, I_r)          (shared across heads per sample)
      logits_h = W_h X + U_h g + b_h
      Y_h ~ Categorical(softmax(logits_h))

    H heads; each head h has |V_h| classes (large).
    Dependence among Y_h|X arises from the shared latent g.
    """

    def __init__(
        self,
        d=64,
        V=(1000, 800, 600, 1200, 500),
        r=3,
        seed=0,
        w_scale=None,
        u_scale=1.0,
        bias_zipf=0.0,
        sample_mode="argmax",
    ):
        """
        d:      input dim
        V:      tuple of class counts per head (len(V) = H)
        r:      shared latent dim (set r>=1 to induce dependence)
        w_scale:std for W_h rows; default 1/sqrt(d)
        u_scale:std for U_h rows; larger -> stronger head coupling via g
        bias_zipf: >=0; adds skew to class biases (0=no skew)
        sample_mode: "argmax" (deterministic) or "sample" (stochastic)
        """
        self.rng = np.random.default_rng(seed)
        self.d = d
        self.V = list(V)
        self.H = len(V)
        self.r = r
        self.sample_mode = sample_mode

        if w_scale is None:
            w_scale = 1.0 / np.sqrt(d)

        # Head parameters
        self.W = [
            self.rng.normal(0, w_scale, size=(Vh, d)).astype(np.float32)
            for Vh in self.V
        ]
        self.U = [
            self.rng.normal(0, u_scale / np.sqrt(max(1, r)), size=(Vh, r)).astype(
                np.float32
            )
            for Vh in self.V
        ]
        self.b = [np.zeros(Vh, dtype=np.float32) for Vh in self.V]

        # Optional class-frequency skew via biases (Zipf-like)
        if bias_zipf > 0:
            for h, Vh in enumerate(self.V):
                ranks = np.arange(1, Vh + 1)
                bias = -bias_zipf * np.log(ranks)
                bias -= bias.mean()
                self.b[h] = bias.astype(np.float32)

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        ez = np.exp(z, dtype=np.float64)
        return (ez / ez.sum(axis=1, keepdims=True)).astype(np.float32)

    def generate(self, n=10000):
        """
        Returns:
          X: (n, d) float32
          Ys: list of length H; each is int64 array of shape (n,)
        """
        X = self.rng.normal(0, 1, size=(n, self.d)).astype(np.float32)
        g = (
            self.rng.normal(0, 1, size=(n, self.r)).astype(np.float32)
            if self.r > 0
            else None
        )

        Ys = []
        for h in range(self.H):
            logits = X @ self.W[h].T  # (n, Vh)
            if g is not None:
                logits += g @ self.U[h].T
            logits += self.b[h][None, :]

            if self.sample_mode == "sample":
                probs = self._softmax(logits)
                # sample per row
                cum = np.cumsum(probs, axis=1)
                u = self.rng.random(size=(n, 1))
                y = (cum < u).sum(axis=1).astype(np.int64)
            else:  # argmax
                y = logits.argmax(axis=1).astype(np.int64)

            Ys.append(y)

        return X, Ys


if __name__ == "__main__":
    gen = SimpleTiedHeadsGen(
        d=64,
        V=(2000, 1500, 3000, 1200, 800),  # large |V_h|
        r=3,  # shared latent -> dependence among heads
        u_scale=1.0,  # increase to strengthen coupling
        bias_zipf=0.3,  # optional class imbalance
        seed=123,
        sample_mode="sample",  # or "argmax"
    )
    X, Ys = gen.generate(n=5000)
    print("X:", X.shape, "  Y shapes:", [y.shape for y in Ys])
