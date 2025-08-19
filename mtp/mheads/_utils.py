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

    # NOTE: This next part was vibe-coded >>>
    # Replace rolled-beyond positions with ignore_index
    # For each head i, mask out the last (shift + i) tokens that wrapped around
    for i in range(H):
        if shift + i > 0:
            input_ids_windowed[:, T - (shift + i) :, i] = ignore_index
    return input_ids_windowed


def select_margin_cp_tensor_batched(
    cp_params: torch.Tensor, ops: torch.Tensor, use_scale_factors=True
):
    """Performs selection and marginalization operations on a CP tensor representation.

    Given a CP tensor T = ∑ᵢ aᵢ₁ ⊗ aᵢ₂ ⊗ ... ⊗ aᵢₜ where each aᵢⱼ ∈ ℝᵈ,
    applies a sequence of operations on each mode:
        - Selection: For op ∈ [0,V), selects op^th index of aᵢⱼ
        - Marginalization: For op = -2, sums all elements in aᵢⱼ
        - Free index: For op = -1, keeps aᵢⱼ unchanged

    Args:
        cp_params (torch.Tensor): CP tensor factors of shape (B, R, T, D) where:
            B: Batch size
            R: CP rank
            T: number of tensor modes/dimensions
            D: dimension of each mode
        ops (torch.Tensor): Operation codes of shape (T,) specifying:
            -2: marginalize mode (sum reduction)
            -1: keep mode as free index
            [0,V): select index v in mode

    Note:
        - The number of free indices in `ops` must be at most 1

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Result tensor of shape (n_free, D) where n_free is the number of free indices (-1 operations) in ops
            - Scale factors list of shape (T,)
    """

    # Validation:
    assert len(cp_params.shape) == 4, "CP params tensor must be $D (batched)"
    assert len(ops.shape) == 2, "Invalid ops tensor: must be 2D (batched)"
    assert (ops >= -2).all() and (
        ops < cp_params.size(3)
    ).all(), "Invalid ops tensor: must be in range [-2, vocab_size)"
    assert ops.size(0) == cp_params.size(0), "Batch size mismatch"

    batch_size, rank, horizon, vocab_size = cp_params.size()

    # Get breakpoints for 1st free leg and 1st margin leg
    bp_free, bp_margin = get_breakpoints(ops)  # (batch_size,), (batch_size,)

    res_left = torch.ones(
        batch_size, rank, device=cp_params.device, dtype=cp_params.dtype
    )
    res_right = torch.ones(
        batch_size, rank, device=cp_params.device, dtype=cp_params.dtype
    )
    if torch.any(bp_free != bp_margin):
        res_free = torch.ones(
            batch_size, rank, vocab_size, device=cp_params.device, dtype=cp_params.dtype
        )

    core_margins = cp_params.sum(dim=-1)  # (B, R, T)
    scale_factors = []

    for t in range(horizon):
        mask_select = t < bp_free
        mask_margin = t >= bp_margin
        mask_free = ~mask_select & ~mask_margin

        # Select
        if mask_select.any():
            update = torch.gather(
                cp_params[mask_select, :, t, :],  # (B', R, D)
                dim=-1,
                index=ops[mask_select, t]
                .reshape(-1, 1, 1)
                .expand(-1, rank, -1),  # (B', R, 1)
            ).squeeze(-1)
            sf = torch.ones(batch_size, device=cp_params.device, dtype=cp_params.dtype)

            # Post-contraction scaling
            res_left[mask_select] = res_left[mask_select] * update
            if use_scale_factors:
                # sf[mask_select] = torch.linalg.norm(
                #     res_left[mask_select], dim=-1
                # )  # (B',)
                sf[mask_select] = torch.max(res_left[mask_select], dim=-1)[0]  # (B',)
                res_left[mask_select] = res_left[mask_select] / sf[
                    mask_select
                ].unsqueeze(-1)
            scale_factors.append(sf)

        # Marginalize
        if mask_margin.any():
            update = core_margins[mask_margin, :, t]  # (B', R)
            sf = torch.ones(
                batch_size, device=cp_params.device, dtype=cp_params.dtype
            )  # (B,)

            # Post-contraction scaling
            res_right[mask_margin] = res_right[mask_margin] * update
            if use_scale_factors:
                # sf[mask_margin] = torch.linalg.norm(
                #     res_right[mask_margin], dim=-1
                # )  # (B',)
                sf[mask_margin] = torch.max(res_right[mask_margin], dim=-1)[0]  # (B',)
                res_right[mask_margin] = res_right[mask_margin] / sf[
                    mask_margin
                ].unsqueeze(-1)
            scale_factors.append(sf)

        # Free
        if mask_free.any():
            res_free[mask_free] = cp_params[mask_free, :, t, :]

    # Final result
    # if not use_scale_factors:
    #     scale_factors = []
    # Special case: pure select
    if torch.all(bp_free == horizon):
        return res_left.sum(dim=-1), scale_factors  # (B,)
    # Special case: pure marginalization
    elif torch.all(bp_margin == 0):
        return res_right.sum(dim=-1), scale_factors
    else:  # General case
        result = (
            res_left.unsqueeze(-1) * res_free * res_right.unsqueeze(-1)
        )  # (B, R, D)
        return result.sum(dim=1), scale_factors


def get_breakpoints(ops: torch.Tensor):
    """Get breakpoints for select, free, and marginalize operations.

    Args:
        ops (torch.Tensor): Operation codes of shape (B, T) specifying:
            -2: marginalize mode (sum reduction)
            -1: keep mode as free index
            [0,V): select index v in mode

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Breakpoints for select/free (h_slct) and free/marginalize (h_mrgn) operations

    """
    # For first non-select (first -1 or -2)
    non_select_mask = (ops < 0).int()  # Convert bool to int, shape: (B, T)
    has_non_select = non_select_mask.any(dim=1)
    h_free = non_select_mask.argmax(dim=1)  # shape: (B,)
    # For batches with all selects, set h_slct to T
    h_free = torch.where(
        has_non_select, h_free, torch.tensor(ops.size(1), device=ops.device)
    )

    # For first margin (first -2)
    is_margin_mask = (ops == -2).int()  # Convert bool to int
    has_margin = is_margin_mask.any(dim=1)
    h_mrgn = is_margin_mask.argmax(dim=1)
    h_mrgn = torch.where(
        has_margin, h_mrgn, torch.tensor(ops.size(1), device=ops.device)
    )
    return h_free.long(), h_mrgn.long()


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
