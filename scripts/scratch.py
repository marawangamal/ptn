import torch
from ctn.mheads import MHEADS
from ctn.mheads._abc import AbstractDisributionHeadConfig
import pandas as pd

device = "cuda"
print(f"Using device: {device}")

B, H, R, Di = 1, 5, 2, 1  # Config used for Fig. 1
rows = []
for Do in [2, 64, 128, 256]:

    x = torch.randn(B, Di, device=device)
    y = torch.randint(0, Do, (B, H), device=device)

    # # MPS_BM:
    model = MHEADS["bm"](
        config=AbstractDisributionHeadConfig(rank=R, d_model=Di, d_output=Do, horizon=H)
    ).to(device)
    model.to(device)

    # MPS_sigma
    # model = MHEADS["mps"](
    #     config=AbstractDisributionHeadConfig(rank=R, d_model=Di, d_output=Do, horizon=H)
    # ).to(device)
    # model.to(device)

    torch.cuda.reset_peak_memory_stats()
    # MPS_BM
    model.train_example(x, y)
    # # MPS_sigma
    # model(x, y)

    mem_after = torch.cuda.max_memory_allocated()
    print(f"[Do={Do}] Mem: {(mem_after) / (1024**2):.4f} MB")
    rows.append(
        {
            "Do": Do,
            "Mem": mem_after / (1024**2),
        }
    )

df = pd.DataFrame(rows)
df.to_csv("results/memory_mps_bm.csv", index=False)
