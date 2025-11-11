from typing import Optional
import torch


def diagnose(*args, **kwargs):
    pass


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


def select_margin_mps_tensor_batched(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    core: torch.Tensor,
    ops: torch.Tensor,
    use_scale_factors: bool = True,
    eps: float = 1e-12,
    norm: str = "l2",
    **kwargs,
):
    """Performs selection and marginalization operations on a MPS tensor representation.

    This function processes an MPS (Matrix Product State) tensor by applying
    selection or marginalization operations on each tensor in the chain according
    to the operation codes provided in `ops`.

    Args:
        alpha (torch.Tensor): Left boundary tensor of shape (B, R)
        beta (torch.Tensor): Right boundary tensor of shape (B, R)
        core (torch.Tensor): Core tensors. Shape (B, H, R, V?, R)
        emission (Optional[torch.Tensor]): Emission tensor.  Shape (B, H, V, R). Optional if V is not provided in core.
        ops (torch.Tensor): Operation codes of shape (B, H) specifying:
            -2: marginalize mode (sum over all indices)
            -1: keep mode as free index
            >=0: select specific index in mode
        use_scale_factors (bool): Whether to return scale factors. Default: True

    Notes:
        - The number of free indices (-1) in `ops` must be at most 1
        - Boundary tensors are cloned to avoid in-place modifications
        - B is batch size
        - H is the number of modes (cores)
        - R is the bond dimension
        - V is the vocabulary size
        - V? indicates that this dimension is missiong if emission is provided

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - result_tensor: Tensor of shape determined by free indices
            - scale_factors: Scale factors of shape (B,) if use_scale_factors=True, else None
    """

    # Validation:
    assert len(core.shape) == 5, "MPS params tensor must be 5D (batched)"
    assert len(ops.shape) == 2, "Invalid ops tensor: must be 2D (batched)"
    assert (ops >= -2).all() and (
        ops < core.size(3)
    ).all(), "Invalid ops tensor: must be in range [-2, vocab_size)"
    assert ops.size(0) == core.size(0), "Batch size mismatch"
    # TODO: add validation that ops must be in order select, free, marginalize

    batch_size, horizon, rank, vocab_size, rank = core.size()

    # Get breakpoints for 1st free leg and 1st margin leg
    bp_free, bp_margin = get_breakpoints(ops)  # (batch_size,), (batch_size,)

    # Detach the input tensors for intermediate calculations
    res_left = alpha.detach().clone()
    res_right = beta.detach().clone()
    res_free = (
        torch.eye(rank, rank, device=core.device)
        .reshape(1, rank, 1, rank)
        .repeat(batch_size, 1, vocab_size, 1)
    )

    core_margins = core.sum(dim=3)  # (B, H, R, V, R) => (B, H, R, R)
    diagnose(core_margins, "core_margins")
    scale_factors = []

    left_cache, right_cache = kwargs.get("left_cache", None), kwargs.get(
        "right_cache", None
    )
    if left_cache is None:
        left_cache = torch.ones(batch_size, horizon, rank, device=core.device) * -100
    if right_cache is None:
        right_cache = torch.ones(batch_size, horizon, rank, device=core.device) * -100

    norm_fn = {
        "l2": torch.linalg.norm,
        "linf": torch.amax,
    }[norm]

    for t in range(horizon):
        mask_select = t < bp_free  # (B,)
        mask_margin = t >= bp_margin  # (B,)
        mask_free = ~mask_select & ~mask_margin  # (B,)

        # Select
        if mask_select.any():
            if (left_cache[mask_select, t] != -100).all():  # cache hit
                res_left[mask_select] = left_cache[mask_select, t]
            else:
                core_select = torch.gather(
                    # (B, H, R, V, R) -> (B', R, V, R) -> (B', R, 1, R) -> (B', R, R)
                    core[mask_select, t],
                    dim=2,
                    index=ops[mask_select, t]  # (B',)
                    .reshape(-1, 1, 1, 1)
                    .expand(-1, rank, -1, rank),  # (B', R, 1, R)
                ).squeeze(2)
                #  (B', 1, R) @ (B', R, R) -> (B', 1, R) -> (B', R)
                update = (res_left[mask_select].unsqueeze(1) @ core_select).squeeze(1)
                sf = torch.ones(batch_size, device=core.device)  # (B,)
                if use_scale_factors:
                    sf[mask_select] = norm_fn(update, dim=-1)
                scale_factors.append(sf)
                res_left[mask_select] = update / sf[mask_select].unsqueeze(-1).clamp(
                    min=eps
                )
                left_cache[mask_select, t] = res_left[mask_select].clone()
                diagnose(core_margins, "res_left")

        # Free
        if mask_free.any():
            res_free[mask_free] = core[mask_free, t]  # (B', R, V, R)

    for t in range(horizon - 1, -1, -1):
        mask_margin = t >= bp_margin  # (B,)

        # Marginalize
        if mask_margin.any():
            if (right_cache[mask_margin, t] != -100).all():  # cache hit
                res_right[mask_margin] = right_cache[mask_margin, t]
            else:
                core_margin = core_margins[mask_margin, t]  # (B', R, R)
                # (B', R, R) @ (B', R, 1) -> (B', R, 1) -> (B', R)
                update = (core_margin @ res_right[mask_margin].unsqueeze(-1)).squeeze(
                    -1
                )
                sf = torch.ones(batch_size, device=core.device)  # (B,)
                if use_scale_factors:
                    sf[mask_margin] = norm_fn(update, dim=-1)
                scale_factors.append(sf)
                res_right[mask_margin] = update / sf[mask_margin].unsqueeze(-1).clamp(
                    min=eps
                )
                right_cache[mask_margin, t] = res_right[mask_margin].clone()
                diagnose(core_margins, "res_right")

    if not use_scale_factors:
        scale_factors = []

    # Special case: pure select
    if torch.all(bp_free == horizon):
        # (B, R) * (B, R)
        return (res_left * beta).sum(dim=-1), scale_factors
    # Special case: pure marginalization
    elif torch.all(bp_margin == 0):
        return (alpha * res_right).sum(dim=-1), scale_factors
    else:  # General case
        # NOTE: Backprop fails through this path
        result = torch.einsum("bi, bivj, bj -> bv", res_left, res_free, res_right)
        return result, scale_factors, left_cache, right_cache


def select_margin_hmm_tensor_batched(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    core: torch.Tensor,
    emission: torch.Tensor,
    ops: torch.Tensor,
    use_scale_factors: bool = True,
    eps: float = 1e-12,
    norm: str = "l2",
    **kwargs,
):
    """Performs selection and marginalization operations on a MPS tensor representation.

    This function processes an MPS (Matrix Product State) tensor by applying
    selection or marginalization operations on each tensor in the chain according
    to the operation codes provided in `ops`.

    Args:
        alpha (torch.Tensor): Left boundary tensor of shape (B, R)
        beta (torch.Tensor): Right boundary tensor of shape (B, R)
        core (torch.Tensor): Core tensors. Shape (B, H, R, R)
        emission (torch.Tensor): Emission tensor.  Shape (B, H, V, R).
        ops (torch.Tensor): Operation codes of shape (B, H) specifying:
            -2: marginalize mode (sum over all indices)
            -1: keep mode as free index
            >=0: select specific index in mode
        use_scale_factors (bool): Whether to return scale factors. Default: True

    Notes:
        - The number of free indices (-1) in `ops` must be at most 1
        - Boundary tensors are cloned to avoid in-place modifications
        - B is batch size
        - H is the number of modes (cores)
        - R is the bond dimension
        - V is the vocabulary size

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - result_tensor: Tensor of shape determined by free indices
            - scale_factors: Scale factors of shape (B,) if use_scale_factors=True, else None
    """

    # Validation:
    assert len(core.shape) == 4, "MPS params tensor must be 4D (batched)"
    assert len(ops.shape) == 2, "Invalid ops tensor: must be 2D (batched)"
    assert (ops >= -2).all() and (
        ops < emission.size(2)
    ).all(), "Invalid ops tensor: must be in range [-2, vocab_size)"
    assert ops.size(0) == core.size(0), "Batch size mismatch"
    # TODO: add validation that ops must be in order select, free, marginalize

    batch_size, horizon, rank, rank = core.size()
    _, _, vocab_size, _ = emission.size()

    # Get breakpoints for 1st free leg and 1st margin leg
    bp_free, bp_margin = get_breakpoints(ops)  # (batch_size,), (batch_size,)

    # Detach the input tensors for intermediate calculations
    res_left = alpha.detach().clone()
    res_right = beta.detach().clone()
    res_free = (
        torch.eye(rank, rank, device=core.device)
        .reshape(1, rank, 1, rank)
        .repeat(batch_size, 1, vocab_size, 1)
    )

    emission_margins = emission.sum(dim=2)  # (B, H, V, R) => (B, H, R)
    diagnose(emission_margins, "emission_margins")
    scale_factors = []

    left_cache, right_cache = kwargs.get("left_cache", None), kwargs.get(
        "right_cache", None
    )
    if left_cache is None:
        left_cache = torch.ones(batch_size, horizon, rank, device=core.device) * -100
    if right_cache is None:
        right_cache = torch.ones(batch_size, horizon, rank, device=core.device) * -100

    norm_fn = {
        "l2": torch.linalg.norm,
        "linf": torch.amax,
    }[norm]

    for t in range(horizon):
        mask_select = t < bp_free  # (B,)
        mask_margin = t >= bp_margin  # (B,)
        mask_free = ~mask_select & ~mask_margin  # (B,)

        # Select
        if mask_select.any():
            if (left_cache[mask_select, t] != -100).all():  # cache hit
                res_left[mask_select] = left_cache[mask_select, t]
            else:
                emission_select = torch.gather(
                    # (B, H, V, R) -> (B', V, R) -> (B', R, 1, R) -> (B', R, R)
                    emission[mask_select, t],
                    dim=1,
                    index=ops[mask_select, t]  # (B',)
                    .reshape(-1, 1, 1)
                    .expand(-1, -1, rank),  # (B', 1, R)
                ).squeeze(1)
                core_select = torch.einsum(
                    "bj,bij,kj->bik",
                    emission_select,
                    core[mask_select, t],
                    torch.eye(rank, rank, device=core.device),
                )
                #  (B', 1, R) @ (B', R, R) -> (B', 1, R) -> (B', R)
                update = (res_left[mask_select].unsqueeze(1) @ core_select).squeeze(1)
                sf = torch.ones(batch_size, device=core.device)  # (B,)
                if use_scale_factors:
                    sf[mask_select] = norm_fn(update, dim=-1)
                scale_factors.append(sf)
                res_left[mask_select] = update / sf[mask_select].unsqueeze(-1).clamp(
                    min=eps
                )
                left_cache[mask_select, t] = res_left[mask_select].clone()
                diagnose(emission_margins, "res_left")

        # Free
        if mask_free.any():
            # TODO: can avoid (B,R,V,R) mem bottleneck but not important since only used during inference
            core_free = torch.einsum(
                "bvj,bij,kj->bkvi",
                emission[mask_free, t],
                core[mask_free, t],
                torch.eye(rank, rank, device=core.device),
            )  # (B', R, V, R)
            res_free[mask_free] = core_free

    for t in range(horizon - 1, -1, -1):
        mask_margin = t >= bp_margin  # (B,)

        # Marginalize
        if mask_margin.any():
            if (right_cache[mask_margin, t] != -100).all():  # cache hit
                res_right[mask_margin] = right_cache[mask_margin, t]
            else:
                emission_margin = emission_margins[mask_margin, t]  # (B', R)
                # (B', R, R) @ (B', R, 1) -> (B', R, 1) -> (B', R)
                core_margin = torch.einsum(
                    "bj,bij,kj->bik",
                    emission_margin,
                    core[mask_margin, t],
                    torch.eye(rank, rank, device=core.device),
                )
                update = (core_margin @ res_right[mask_margin].unsqueeze(-1)).squeeze(
                    -1
                )
                sf = torch.ones(batch_size, device=core.device)  # (B,)
                if use_scale_factors:
                    sf[mask_margin] = norm_fn(update, dim=-1)
                scale_factors.append(sf)
                res_right[mask_margin] = update / sf[mask_margin].unsqueeze(-1).clamp(
                    min=eps
                )
                right_cache[mask_margin, t] = res_right[mask_margin].clone()
                diagnose(emission_margins, "res_right")

    if not use_scale_factors:
        scale_factors = []

    # Special case: pure select
    if torch.all(bp_free == horizon):
        # (B, R) * (B, R)
        return (res_left * beta).sum(dim=-1), scale_factors
    # Special case: pure marginalization
    elif torch.all(bp_margin == 0):
        return (alpha * res_right).sum(dim=-1), scale_factors
    else:  # General case
        # NOTE: Backprop fails through this path
        result = torch.einsum("bi, bivj, bj -> bv", res_left, res_free, res_right)
        return result, scale_factors, left_cache, right_cache


def mps_norm(
    g: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    use_scale_factors: bool = True,
    norm: str = "l2",
    **kwargs,
):
    """Marginalize a Born MPS tensor.

    Args:
        g (torch.Tensor): g tensor. Shape: (H, R, D, R)

    Returns:
        torch.Tensor: Marginalized tensor. Shape: (1,)
    """
    H, _, _, _ = g.shape
    scale_factors = []
    norm_fn = {
        "l2": torch.linalg.norm,
        "linf": torch.amax,
    }[norm]
    # L = torch.einsum("p,pdq,r,rds->qs", a, g[0], a, g[0])
    L = torch.einsum("p,r->pr", a, a)  # (B, R)
    for h in range(0, H):
        L = torch.einsum("pdq,pr,rds ->qs", g[h], L, g[h])
        if use_scale_factors:
            sf = norm_fn(L.abs())
            scale_factors.append(sf)
            L = L / sf
    L = torch.einsum("pq,p,q->", L, b, b)
    if not use_scale_factors:
        scale_factors = [torch.tensor(1.0)]
    return L, torch.stack(scale_factors)  # (1,), (H,)


def reduce_select(
    t: torch.Tensor,  # (B, H, R, D, R)
    y: torch.Tensor,
):
    pass


def born_select_margin(
    g: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    y: torch.Tensor,  # (H, D)
    use_scale_factors: bool = True,
    free_index: Optional[int] = None,
    norm: str = "l2",
    **kwargs,
):
    """Marginalize a Born MPS tensor.

    Args:
        g (torch.Tensor): Core tensors. Shape: (H, R, D, R)
        a (torch.Tensor): Left boundary tensor. Shape: (R,)
        b (torch.Tensor): Right boundary tensor. Shape: (R,)
        y (torch.Tensor): Target tensor. Shape: (H, D)
        use_scale_factors (bool): Whether to use scale factors.
        free_index (Optional[int]): Free index. Shape: (B,)
        norm (str): Norm to use for scale factors.

    Returns:
        torch.Tensor: Marginalized tensor. Shape: (1,)
    """
    assert y.ndim == 2, "Feature tensor `y` must be 2D (H, D)"
    assert y.size(0) == g.size(0), "Horizon mismatch"
    assert y.size(1) == g.size(2), "Physical dimension mismatch"
    assert a.size(0) == g.size(1) == b.size(0), "Bond dimension mismatch"
    H, _, _, _ = g.shape
    scale_factors = []

    def amax_fn(x: torch.Tensor):
        return torch.amax(x.abs().flatten())

    norm_fn = {
        "l2": torch.linalg.norm,
        "linf": amax_fn,
    }[norm]
    L = torch.einsum("p,r->pr", a, a)  # (B, R)
    for h in range(0, H):
        if free_index is not None and h == free_index:  # free index encountered
            L = torch.einsum("pdq,pr,rds->qds", g[h], L, g[h])
        elif free_index is not None and h < free_index:  # before free index
            L = torch.einsum("pdq,pr,rds,d->qs", g[h], L, g[h], y[h])
        elif free_index is not None and h > free_index:  # before free index
            L = torch.einsum("pdq,pvr,rds,d->qvs", g[h], L, g[h], y[h])
        else:  # no free index
            L = torch.einsum("pdq,pr,rds,d->qs", g[h], L, g[h], y[h])
        if use_scale_factors:
            sf = norm_fn(L)
            scale_factors.append(sf)
            L = L / sf
    if free_index is not None:
        L = torch.einsum("pvq,p,q->v", L, b, b)  # return dist
    else:
        L = torch.einsum("pq,p,q->", L, b, b)
    if not use_scale_factors:
        scale_factors = [torch.tensor(1.0)]
    return L, torch.stack(scale_factors)  # (1,), (H,)


mps_norm_batch = torch.vmap(mps_norm, in_dims=(0, 0, 0))
born_select_margin_batch = torch.vmap(born_select_margin, in_dims=(0, 0, 0, 0))


if __name__ == "__main__":
    # test select_margin_hmm_tensor_batched
    B, H, R, V = 10, 3, 4, 3
    cores = torch.randn(B, H, R, R)
    emissions = torch.randn(B, H, V, R)
    a = torch.randn(B, R)
    b = torch.randn(B, R)
    y = torch.randint(0, V, (B, H))
    print(len(select_margin_hmm_tensor_batched(a, b, cores, emissions, y)))
