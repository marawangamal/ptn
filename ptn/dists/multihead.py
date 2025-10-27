from typing import Callable, Optional
import torch
import torch.nn as nn

from ptn.dists._abc import (
    AbstractDisributionHead,
    AbstractDisributionHeadConfig,
    AbstractDisributionHeadOutput,
)


class MultiHeadDist(AbstractDisributionHead):
    """Simple multi-head distribution with independent linear heads for each position."""

    def __init__(self, config: AbstractDisributionHeadConfig):
        super().__init__(config)
        # Separate linear heads for each position
        self.heads = nn.ModuleList(
            [nn.Linear(config.d_model, config.d_output) for _ in range(config.horizon)]
        )

    @classmethod
    def from_pretrained(
        cls, linear: torch.nn.Linear, config: AbstractDisributionHeadConfig
    ):
        raise NotImplementedError("from_pretrained not implemented")

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        """Forward pass.

        Args:
            x (torch.Tensor): Input features. Shape: (B, D)
            y (Optional[torch.Tensor], optional): Target tensor. Shape: (B, H). Defaults to None.

        Note:
            D = d_model (input features)
            H = horizon (number of positions/steps)
            V = d_output (number of output classes)

        Returns:
            AbstractDisributionHeadOutput: Output of the head. Shape: (B, H, V).
        """
        total_loss = torch.tensor(0.0, device=x.device)
        if y is not None:
            for h in range(self.config.horizon):
                logits = self.heads[h](x)
                loss = nn.CrossEntropyLoss(reduction="none")(logits, y[:, h])
                total_loss = total_loss + loss
            return AbstractDisributionHeadOutput(loss=total_loss, logits=logits)
        logits = logits = self.heads[0](x)
        return AbstractDisributionHeadOutput(logits=logits)

    def sample(
        self,
        x: torch.Tensor,
        sample_fn: Callable[[torch.Tensor], torch.Tensor],
        horizon: Optional[int] = None,
        return_logits: bool = False,
        **kwargs,
    ):
        """Sample from multi-head distribution."""
        H = horizon if horizon is not None else self.config.horizon
        B = x.size(0)
        device = x.device

        y_out = torch.empty(B, 0, device=device, dtype=torch.long)

        for h in range(H):
            logits = self.heads[h](x)
            probs = torch.softmax(logits, dim=-1)
            next_token = sample_fn(probs).unsqueeze(1)
            y_out = torch.cat([y_out, next_token], dim=1)
        return y_out

    def forward_backward(
        self, z: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer
    ):
        """Forward pass w/ memory efficient backward pass.

        Args:
            x (torch.Tensor): Input features. Shape: (B, D)
            y (Optional[torch.Tensor], optional): Target tensor. Shape: (B, H). Defaults to None.

        Note:
            D = d_model (input features)
            H = horizon (number of positions/steps)
            V = d_output (number of output classes)

        Returns:
            AbstractDisributionHeadOutput: Output of the head. Shape: (B, H, V).
        """
        total_loss = 0.0
        d = z.detach()
        d.requires_grad = True
        for h in range(self.config.horizon):
            logits = self.heads[h](d)
            loss = nn.CrossEntropyLoss(reduction="none")(logits, y[:, h])
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        z.backward(d.grad)
        return AbstractDisributionHeadOutput(
            loss=torch.tensor(total_loss, device=z.device),
            logits=logits,
        )


def test_forward():
    D, H, V, R = 8, 10, 32, 2
    model = MultiHeadDist(
        AbstractDisributionHeadConfig(
            d_model=D,
            d_output=V,
            horizon=H,
            rank=R,
        )
    )
    x = torch.randn(1, D)
    y = torch.randint(0, V, (1, H))
    output = model(x, y)
    print("[PASS] test_forward")


def test_forward_backward():
    D, H, V, R = 8, 10, 32, 2
    model = MultiHeadDist(
        AbstractDisributionHeadConfig(
            d_model=D,
            d_output=V,
            horizon=H,
            rank=R,
        )
    )
    x = torch.randn(1, D)
    y = torch.randint(0, V, (1, H))
    w_z = torch.nn.Linear(D, D)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output = model.forward_backward(w_z(x), y, optimizer=optimizer)
    print("[PASS] test_forward_backward")


def forward_backward_standard(model, x, y, optimizer):
    output = model.forward(x, y)
    loss = output.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return output


def forward_backward_memory_efficient(model, x, y, optimizer):
    output = model.forward_backward(x, y, optimizer=optimizer)
    # optimizer.step()
    # optimizer.zero_grad()
    return output


def test_forward_backward_memory_efficiency():
    """Test memory efficiency of forward_backward method.

    Example output (rtx8000):
        [forward_backward_standard] Memory (MB): 2977
        [forward_backward_memory_efficient] Memory (MB): 2028

    """
    device = torch.device("cuda")
    B, H, R, D, V = 1, 5, 2, 1024, 30_000
    model = MultiHeadDist(
        config=AbstractDisributionHeadConfig(rank=R, d_model=D, d_output=V, horizon=H)
    ).to(device)
    exps = [
        {
            "name": "forward_backward_standard",
            "fn": lambda **kwargs: forward_backward_standard(model, **kwargs),
        },
        {
            "name": "forward_backward_memory_efficient",
            "fn": lambda **kwargs: forward_backward_memory_efficient(model, **kwargs),
        },
    ]
    for exp in exps:

        encoder = torch.nn.Linear(D, D).to(device)
        x, y = encoder(torch.randn(B, D, device=device)), torch.randint(
            0, D, (B, H), device=device
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Reset peak memory stats after model creation but before forward pass
        torch.cuda.reset_peak_memory_stats()

        # ===== Memory measurement =========

        exp["fn"](x=x, y=y, optimizer=optimizer)

        # ===== Memory measurement end =====

        mem_after = torch.cuda.max_memory_allocated()
        print(f"[{exp['name']}] Memory (MB): {int(mem_after / (1024**2))}")
        exp["memory"] = mem_after

    mem_std = next(
        exp["memory"] for exp in exps if exp["name"] == "forward_backward_standard"
    )
    mem_fb = next(
        exp["memory"]
        for exp in exps
        if exp["name"] == "forward_backward_memory_efficient"
    )

    assert mem_fb < mem_std
    print("[PASS] test_forward_backward_memory_efficiency")


if __name__ == "__main__":
    test_forward()
    test_forward_backward()
    # test_forward_backward_memory_efficiency()  # requires cuda
