import torch
from ctn.mheads import MHEADS, AbstractDisributionHeadConfig


class TrueModel:
    # P(y1, .., yH | x) (V^H x D)
    def __init__(self, d_model, d_vocab, rank=1):
        super().__init__()
        self.d_model = d_model
        self.d_vocab = d_vocab
        self.a = torch.randn(d_vocab, d_model, rank)
        self.rank = rank

        if rank > 1:
            raise NotImplementedError("Rank > 1 not implemented")

    def get_xy_samples(self, n_samples=1, horizon=1):
        """Sample x, y from p(x) and p(x,y1, y2, ..., yH | x)

        Args:
            n_samples (int, optional): Number of samples to draw from p(x). Defaults to 1.
            horizon (int, optional): Number of samples to draw from p(y1, y2, ..., yH | x). Defaults to 1.

        Returns:
            x: (n_samples, d_model)
            y: (n_samples, horizon)
        """
        x = torch.randn(n_samples, self.d_model)
        py_h_bar_x = torch.softmax(
            torch.einsum("bd, vdr->bvr", x, self.a).squeeze(-1), dim=-1
        )
        y = torch.cat(
            [torch.multinomial(py_h_bar_x, num_samples=1) for _ in range(horizon)],
            dim=-1,
        )
        return x, y


def train_model(
    model, gt_model, n_iterations=100, lr=1e-3, batch_size=128, device="cuda"
):

    results = {
        "train/loss": [],
        "eval/loss": [],
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for i in range(n_iterations):
        optimizer.zero_grad()
        x, y = gt_model.get_xy_samples(
            batch_size, model.config.horizon
        )  # (B, D) (B, H)
        x = x.to(device)
        y = y.to(device)

        output = model(x, y)
        loss = output.loss  # (B,)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            with torch.no_grad():
                x, y = gt_model.get_xy_samples(
                    batch_size, model.config.horizon
                )  # (B, D) (B, H)
                x = x.to(device)
                y = y.to(device)
                output = model(x, y)
                loss = output.loss  # (B,)
                results["eval/loss"].append(loss.mean().item())


# HPs
rank = 1
horizon = 4
d_model = 128
d_vocab = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
gt_model = TrueModel(d_model, d_vocab, rank=rank)

models = [
    {
        "mt_head": "moe",
        "model": {"rank": rank, "horizon": horizon, "d_output": d_vocab},
    },
    # {
    #     "mt_head": "moe_proj",
    #     "model": {
    #         "rank": rank,
    #         "horizon": horizon,
    #         "d_output": d_vocab
    #     },
    # },
    # {
    #     "mt_head": "cp",
    #     "model": {
    #         "rank": rank,
    #         "horizon": horizon,
    #         "d_output": d_vocab
    #     },
    # }
]

# Train models
for model in models:
    model = MHEADS[model["mt_head"]](
        config=AbstractDisributionHeadConfig(
            d_model=d_model, d_output=d_vocab, horizon=horizon, rank=rank
        )
    )
    results = train_model(model, gt_model, device=device)
