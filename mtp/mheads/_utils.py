import torch


def window_input_ids(
    input_ids: torch.Tensor,
    horizon: int = 1,
    shift: int = 1,
    ignore_index: int = -100,
):
    """Window the input_ids so that each position looks H steps ahead.

    Args:
        input_ids (torch.Tensor): The input tensor of shape (B, T).
        horizon (int): The number of steps ahead each position should look. Default is 1 (i.e. next-token prediction)
        shift (int): The number of steps to shift the window. Default is 1 (i.e. next-token prediction)
        ignore_index (int): Index for rolled-beyond positions. Default is -100 (i.e. ignore)

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
                 [ 5, -100]]])

        >>> window_input_ids(input_ids, horizon=2, shift=2)
        Input IDs:
        tensor([[0, 1, 2, 3, 4, 5]])
        Windowed Input IDs:
        tensor([[[ 2,  3],
                 [ 3,  4],
                 [ 4,  5],
                 [ 5, -1],
                 [-100, -100],
                 [-100, -100]]])


    """

    B, T, H = input_ids.size(0), input_ids.size(1), horizon

    # (B, T) -> (B, T, H)
    input_ids_windowed = torch.stack(
        [torch.roll(input_ids, -i - shift, dims=1) for i in range(H)], dim=-1
    )
    # Replace rolled-beyond positions with ignore_index
    # ...
    # print(f"Input IDs: {input_ids}")
    # print(f"Input IDs (windowed): {input_ids_windowed}")
    # input_ids_windowed[
    #     input_ids_windowed
    #     < torch.arange(T, device=input_ids.device).reshape(1, T, -1).repeat(1, 1, H)
    # ] = ignore_index

    # For each head i, mask out the last (shift + i) tokens that wrapped around
    for i in range(H):
        if shift + i > 0:
            input_ids_windowed[:, T - (shift + i) :, i] = ignore_index
    return input_ids_windowed


if __name__ == "__main__":
    x = torch.arange(6).unsqueeze(0)
    x_ignore = x.clone()
    x_ignore[x_ignore < 3] = -100
    cases = [
        {
            "kwargs": {
                "input_ids": x,
                "horizon": 2,
                "shift": 1,
                "ignore_index": -100,
            },
            "y_true": torch.tensor(
                [
                    [
                        [1, 2],  # 1
                        [2, 3],  # 2
                        [3, 4],  # 3
                        [4, 5],  # 4
                        [5, -100],  # 5
                        [-100, -100],  # 6
                    ]
                ]
            ),
        },
        {
            "kwargs": {
                "input_ids": x,
                "horizon": 2,
                "shift": 2,
                "ignore_index": -100,
            },
            "y_true": torch.tensor(
                [
                    [
                        [2, 3],  # 1
                        [3, 4],  # 2
                        [4, 5],  # 3
                        [5, -100],  # 4
                        [-100, -100],  # 5
                        [-100, -100],  # 6
                    ]
                ]
            ),
        },
        {
            "kwargs": {
                "input_ids": x_ignore,
                "horizon": 2,
                "shift": 2,
                "ignore_index": -100,
            },
            "y_true": torch.tensor(
                [
                    [
                        [-100, 3],  # 1
                        [3, 4],  # 2
                        [4, 5],  # 3
                        [5, -100],  # 4
                        [-100, -100],  # 5
                        [-100, -100],  # 6
                    ]
                ]
            ),
        },
        {
            "kwargs": {
                "input_ids": torch.tensor(
                    [-1 for _ in range(5)] + [i for i in range(5)]
                ).reshape(
                    1, -1
                ),  # (B, T)
                "horizon": 1,
                "shift": 0,
                "ignore_index": -1,
            },
            "y_true": torch.tensor(
                [-1 for _ in range(5)] + [i for i in range(5)]
            ).reshape(
                1, -1, 1
            ),  # (B, T, H)
        },
        {
            "kwargs": {
                "input_ids": torch.tensor(
                    [[0, 1, 2, 3], [10, 11, 12, 13]]
                ),  # (B=2, T=4)
                "horizon": 3,
                "shift": 1,
                "ignore_index": -100,
            },
            "y_true": torch.tensor(
                [
                    [
                        [1, 2, 3],
                        [2, 3, -100],
                        [3, -100, -100],
                        [-100, -100, -100],
                    ],
                    [
                        [11, 12, 13],
                        [12, 13, -100],
                        [13, -100, -100],
                        [-100, -100, -100],
                    ],
                ]
            ),
        },
    ]

    # , device='cuda:0')

    for case in cases:
        y_pred = window_input_ids(**case["kwargs"])
        assert (
            y_pred.shape == case["y_true"].shape
        ), f"Case {case['kwargs']} failed. Shape mismatch: {y_pred.shape} != {case['y_true'].shape}"
        assert torch.all(
            y_pred == case["y_true"]
        ), f"Case {case['kwargs']} failed. Values mismatch: {y_pred} != {case['y_true']}"

    print("✅ All tests passed!")
