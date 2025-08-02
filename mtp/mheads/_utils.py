import torch


def get_windowed_input_ids(
    input_ids: torch.Tensor, horizon: int, offset: int = 0, ignore_index: int = -1
):
    # 1. Window the `input_ids` to get targets: (*, T) => (*, (T-H), H)
    #   each position should look H steps ahead
    batch_dims = input_ids.shape[:-1]
    input_ids = input_ids.reshape(-1, input_ids.shape[-1])  # Flatten batch dims
    input_ids_windowed = window_input_ids_v1(input_ids, horizon=horizon)

    # 2. Make targets using windowed input_ids
    targets = input_ids_windowed[:, :-horizon]  # (*, T-H, H)
    targets = targets.reshape(*batch_dims, -1, horizon)  # (*,(T-H), H)
    return targets


def window_input_ids(
    input_ids: torch.Tensor, horizon: int, shift: int = 1, ignore_index: int = -1
):
    """Window the input_ids so that each position looks H steps ahead.

    Args:
        input_ids (torch.Tensor): The input tensor of shape (B, T).
        H (int): The number of steps ahead each position should look.
        shift (int): The number of steps to shift the window.

    Returns:
        torch.Tensor: The windowed tensor of shape (B, T, H).

    Example:
        >>> input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        >>> window_input_ids(input_ids, horizon=2, shift=0)
        Input IDs:
        tensor([[0, 1, 2, 3, 4, 5]])
        Windowed Input IDs:
        tensor([[[ 0,  1],
                 [ 1,  2],
                 [ 2,  3],
                 [ 3,  4],
                 [ 4,  5],
                 [ 5, -1]]])

        >>> window_input_ids(input_ids, horizon=2, shift=2)
        Input IDs:
        tensor([[0, 1, 2, 3, 4, 5]])
        Windowed Input IDs:
        tensor([[[ 2,  3],
                 [ 3,  4],
                 [ 4,  5],
                 [ 5, -1],
                 [-1, -1],
                 [-1, -1]]])


    """

    B, T, H = input_ids.size(0), input_ids.size(1), horizon

    # (B, T) -> (B, T, H)
    input_ids_windowed = torch.stack(
        [torch.roll(input_ids, -i - shift, dims=1) for i in range(H)], dim=-1
    )
    # print(f"Input IDs: {input_ids}")
    # print(f"Input IDs (windowed): {input_ids_windowed}")
    input_ids_windowed[
        input_ids_windowed
        < torch.arange(T, device=input_ids.device).reshape(1, T, -1).repeat(1, 1, H)
    ] = ignore_index
    return input_ids_windowed


def window_input_ids_v1(input_ids: torch.Tensor, horizon: int, shift: int = 1):
    """Window the input_ids so that each position looks H steps ahead.

    Args:
        input_ids (torch.Tensor): The input tensor of shape (B, T).
        H (int): The number of steps ahead each position should look.

    Returns:
        torch.Tensor: The windowed tensor of shape (B, T, H).
    """
    B, T = input_ids.shape

    # Create the windowed input tensor
    # (B, T) -> (B, T, H)
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


if __name__ == "__main__":
    x = torch.arange(6).unsqueeze(0)
    cases = [
        {
            "kwargs": {
                "input_ids": torch.arange(6).unsqueeze(0),
                "horizon": 2,
                "offset": 0,
                "ignore_index": -1,
            },
            "y_true": torch.tensor(
                [
                    [1, 2],  # 1
                    [2, 3],  # 2
                    [3, 4],  # 3
                    [4, 5],  # 4
                ]
            ),
        },
        {
            "kwargs": {
                "input_ids": torch.arange(6).unsqueeze(0),
                "horizon": 2,
                "offset": 1,
                "ignore_index": -1,
            },
            "y_true": torch.tensor(
                [
                    [2, 3],  # 1
                    [3, 4],  # 2
                    [4, 5],  # 3
                    [5, -1],  # 4
                ]
            ),
        },
    ]

    for case in cases:
        y_pred = get_windowed_input_ids(**case["kwargs"])
        assert (
            y_pred.shape == case["y_true"].shape
        ), f"Case {case['kwargs']} failed. Shape mismatch: {y_pred.shape} != {case['y_true'].shape}"
        assert torch.all(
            y_pred == case["y_true"]
        ), f"Case {case['kwargs']} failed. Values mismatch: {y_pred} != {case['y_true']}"

    print("✅ All tests passed!")
