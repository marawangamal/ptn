import argparse
import torch
from ptn.dists.multihead import MultiHeadDist
from ptn.dists._abc import AbstractDisributionHeadConfig


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


def main():
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


if __name__ == "__main__":
    main()
