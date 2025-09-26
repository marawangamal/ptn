import torch
from ctn.mheads import MHEADS
from ctn.mheads._abc import AbstractDisributionHeadConfig
import pandas as pd

device = "cuda"
print(f"Using device: {device}")

B, H, R, Di = 1, 5, 2, 1  # Config used for Fig. 1
rows = []
for Do in [64, 128, 256, 512, 1024]:

    torch.cuda.reset_peak_memory_stats()

    x = torch.randn(B, Di, device=device)
    y = torch.randint(0, Do, (B, H), device=device)

    # <<<<< SELECT MODEL HERE >>>>>
    # # MPS_sigma
    # model = MHEADS["mps"](
    #     config=AbstractDisributionHeadConfig(rank=R, d_model=Di, d_output=Do, horizon=H)
    # ).to(device)
    # model.to(device)  # 15.2329 MB
    # model(x, y)
    # ================================
    # # MPS_BMNC
    model = MHEADS["bmnc"](
        config=AbstractDisributionHeadConfig(rank=R, d_model=Di, d_output=Do, horizon=H)
    ).to(device)
    model.to(device)
    model(x, y)
    # # <<<<<<<<<<<<>>>>>>>>>>>>>>>>
    # # MPS_BM
    # model = MHEADS["bm"](
    #     config=AbstractDisributionHeadConfig(rank=R, d_model=Di, d_output=Do, horizon=H)
    # ).to(device)
    # model.to(device)
    # model.train_example(x, y)
    # # <<<<<<<<<<<<>>>>>>>>>>>>>>>>

    mem_after = torch.cuda.max_memory_allocated()
    print(f"[Do={Do}] Mem: {(mem_after) / (1024**2):.4f} MB")
    rows.append(
        {
            "Do": Do,
            "Mem": mem_after / (1024**2),
        }
    )

df = pd.DataFrame(rows)
df.to_csv("results/memory_mps_bmnc.csv", index=False)
