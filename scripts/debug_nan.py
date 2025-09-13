import torch
import torch.nn.functional as F
import os
from mtp.mheads import MHEADS
from mtp.mheads._abc import AbstractDisributionHeadConfig


def debug_nan():
    """Debug NaN issues by loading saved model and inputs."""
    debug_dir = "debug_nan"

    # Check if debug files exist
    if not os.path.exists(debug_dir):
        print(
            f"Debug directory {debug_dir} not found. Run training first to generate debug files."
        )
        return

    # Find the most recent debug files
    debug_files = [f for f in os.listdir(debug_dir) if f.startswith("model_nan_")]
    if not debug_files:
        print("No debug model files found.")
        return

    # Get the latest debug file
    latest_file = sorted(debug_files)[-1]
    batch_idx = latest_file.split("_")[-1].split(".")[0]

    print(f"Loading debug files for batch {batch_idx}...")

    # Load model state
    model_path = os.path.join(debug_dir, f"model_nan_{batch_idx}.pt")
    z_path = os.path.join(debug_dir, f"input_z_nan_{batch_idx}.pt")
    y_path = os.path.join(debug_dir, f"input_y_nan_{batch_idx}.pt")

    # Load inputs with CPU mapping
    z = torch.load(z_path, map_location=torch.device("cpu"))
    y = torch.load(y_path, map_location=torch.device("cpu"))
    print(f"Loaded inputs - z shape: {z.shape}, y shape: {y.shape}")

    # Create model with same config as training
    model = MHEADS["mps"](
        AbstractDisributionHeadConfig(
            horizon=(28 * 28),  # 28x28 for MNIST
            d_model=10,
            d_output=2,
            rank=8,
            pos_func="abs",
        )
    )

    # Load model state with CPU mapping
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    print("Model loaded successfully. Running forward pass...")
    # Run forward pass
    with torch.no_grad():
        output = model(z, y.reshape(z.shape[0], -1))
        if output.loss.isnan():
            raise ValueError("Loss is NaN!")

        # Check for NaN in model parameters
        nan_params = []
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                raise ValueError(f"NaN found in parameter: {name}")

    print("[PASS] Model is stable!")


if __name__ == "__main__":
    debug_nan()
