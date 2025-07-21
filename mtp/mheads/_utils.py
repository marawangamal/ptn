import torch


def get_windowed_input_ids(input_ids: torch.Tensor, horizon: int):
    # 1. Window the `input_ids` to get targets: (*, T) => (*, (T-H), H)
    #   each position should look H steps ahead
    batch_dims = input_ids.shape[:-1]
    input_ids = input_ids.reshape(-1, input_ids.shape[-1])  # Flatten batch dims
    input_ids_windowed = window_input_ids(input_ids, horizon=horizon)

    # 2. Make targets using windowed input_ids
    targets = input_ids_windowed[:, :-horizon]  # (*, T-H, H)
    targets = targets.reshape(*batch_dims, -1, horizon)  # (*,(T-H), H)
    return targets


def window_input_ids(input_ids: torch.Tensor, horizon: int, shift: int = 1):
    """Window the input_ids so that each position looks H steps ahead.

    Args:
        input_ids (torch.Tensor): The input tensor of shape (B, T).
        H (int): The number of steps ahead each position should look.

    Returns:
        torch.Tensor: The windowed tensor of shape (B, T, H).
    """
    B, T = input_ids.shape

    # Create the windowed input tensor
    input_ids_windowed = torch.stack(
        [torch.roll(input_ids, -i - shift, dims=1) for i in range(horizon)], dim=-1
    )

    # Mask out positions that roll beyond the sequence length
    for i in range(1, horizon):
        input_ids_windowed[:, -i - shift :, i] = (
            0  # Replace 0 with padding token if needed
        )

    # Correct the padding (zeroing) for positions that have rolled beyond the valid sequence length
    for i in range(horizon):
        # Calculate the index from which zeroing should start based on the shift
        zero_start = T - i - shift
        if zero_start < T:  # Only apply zeroing if we're within valid range
            input_ids_windowed[:, zero_start:, i] = 0

    return input_ids_windowed
