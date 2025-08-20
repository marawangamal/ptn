import random
import torch
from mtp.mheads._abc import AbstractDisributionHeadConfig
from mtp.mheads.cp import CP
from mtp.mheads.cp_cond import CPCond
from mtp.mheads.multihead import Multihead


def run_train(mt_head_name: str = "CPCond"):
    """Test if CP distribution can recover a target distribution on small scale."""
    import torch.optim as optim

    # set seed
    torch.manual_seed(42)
    random.seed(42)

    # Training parameters
    n_iters = 10_000
    B, R, H, D, V = 32, 2, 4, 128, 100

    mt_head = {
        "CP": CP,
        "CPCond": CPCond,
        "Multihead": Multihead,
    }[mt_head_name](
        AbstractDisributionHeadConfig(d_model=D, d_output=V, horizon=H, rank=R),
    )

    optimizer = optim.AdamW(mt_head.parameters(), lr=1e-4)
    for i in range(n_iters):
        x = torch.randn(B, D)
        # y = torch.randint(0, V, (B, H)) #
        y = torch.randint(0, 2, (B, H)) * (V - 1)
        out = mt_head(x, y)
        out.loss.backward()
        optimizer.step()

        if i % 100 == 0:
            log_dict = {
                "loss": out.loss.item(),
                "grad_norm": torch.nn.utils.clip_grad_norm_(
                    mt_head.parameters(), max_norm=float("inf")
                ).item(),
            }

            if out.loss_dict is not None:
                log_dict.update(out.loss_dict)

            log_dict = {
                k: f"{v:.2f}" if isinstance(v, float) else v
                for k, v in log_dict.items()
            }

            print(f"[{i}] " + " | ".join([f"{k}: {v}" for k, v in log_dict.items()]))

            if out.loss.isnan():
                print("Loss is NaN!")
                break

    print("Training test completed!")


# def run_train():
#     """Test if CP distribution can recover a target distribution on small scale."""
#     import torch.optim as optim

#     # Training parameters
#     n_iters = 10_000
#     B, R, H, D, V = 32, 1, 4, 128, 100

#     for mt_head_name in ["Multihead", "CP"]:
#         mt_head = {
#             "Multihead": Multihead,
#             "CP": CP,
#         }[mt_head_name](
#             AbstractDisributionHeadConfig(d_model=D, d_output=V, horizon=H, rank=R),
#         )
#         optimizer = optim.AdamW(mt_head.parameters(), lr=1e-3)
#         for i in range(n_iters):
#             x = torch.randn(B, D)
#             y = torch.randint(0, V, (B, H))
#             out = mt_head(x, y)
#             out.loss.backward()

#             # # clip gradients
#             # torch.nn.utils.clip_grad_norm_(mt_head.parameters(), max_norm=1.0)

#             optimizer.step()

#             if i % 100 == 0:
#                 print(f"[{mt_head_name}] Iteration {i}, Loss: {out.loss.item():.4f}")

#         print(f"Training test completed for {mt_head_name}!")


if __name__ == "__main__":
    run_train()
