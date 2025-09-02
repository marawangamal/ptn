import torch

from mtp.mheads.einlse import einlogsumexp


def select_margin_cp_tensor_batched(
    cp_params: torch.Tensor, ops: torch.Tensor, use_scale_factors=False
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


def cp_reduce(
    cp_params: torch.Tensor, ops: torch.Tensor, use_scale_factors=True, margin_index=-1
):
    """Reduce a CP tensor via select/marginalize operations.
    Args:
        cp_params (torch.Tensor): CP params. Shape: (R, H, V)
        ops (torch.Tensor): Ops \\in [0, V) + [margin_index]. Shape: (H,)
        use_scale_factors (bool): Whether to apply scale factors during reduction

    Returns:
        torch.Tensor: Reduced tensor result (scalar)
    """
    assert margin_index < 0, "margin_index must be negative"
    R, H, V = cp_params.shape

    # For each mode: marginalize (sum) if ops[h] == -1, else select ops[h]
    marginalize_mask = ops == margin_index  # (H,)
    marginalized = cp_params.sum(dim=-1)  # (R, H) - sum over V dimension

    # Gather selected indices (clamp to handle -1 values safely)
    selected_indices = ops.clamp(min=0).unsqueeze(0).unsqueeze(-1)  # (1, H, 1)
    selected = cp_params.gather(-1, selected_indices.expand(R, -1, -1)).squeeze(
        -1
    )  # (R, H)

    # Choose marginalized or selected values based on ops
    factors = torch.where(marginalize_mask, marginalized, selected)  # (R, H)

    scale_factors = []
    res = torch.ones(R, device=cp_params.device, dtype=cp_params.dtype)

    # BUG: do scale factors online
    # if use_scale_factors:
    #     # norm of each factor
    #     # scale_factors = torch.max(factors, dim=0)[0]
    #     # factors = factors / scale_factors

    # Do it during contraction
    for i in range(H):
        res = res * factors[:, i]  # (R,)
        if use_scale_factors:
            scale_factors.append(torch.max(res))
            res = res / scale_factors[-1]
    res = res.sum()
    scale_factors = torch.stack(scale_factors) if scale_factors else torch.empty(0)

    # Compute CP tensor value: product over modes, sum over components
    return res, scale_factors


# BUG: this version cannot do marginals properly, for marginalization see ``cp_normalize_decoder_einlse``
def cp_reduce_decoder_einlse(
    cp_params_tilde: torch.Tensor,
    ops: torch.Tensor,
    decoder_tilde: torch.Tensor,
    margin_index=-1,
    apply_logsumexp=True,
):
    """Reduce a CP tensor via select/marginalize operations.
    Args:
        cp_params_tilde (torch.Tensor): CP params logits. Shape: (R, H, D)
        decoder_tilde (torch.Tensor): Decoder logits. Shape: (D, V)
        ops (torch.Tensor): Ops \\in [0, V) + [margin_index]. Shape: (H,)
        use_scale_factors (bool): Whether to apply scale factors during reduction

    Note:
        Computes logP(y1, ..., yH | x) = log \\sum exp(a_h)exp(d_h(ops)) ... exp(a_H) exp(d_H(ops))

    Returns:
        torch.Tensor: Reduced tensor result (scalar)
    """
    H, V = cp_params_tilde.size(1), decoder_tilde.size(1)
    esum_recipe = []
    dvc = cp_params_tilde.device
    dtype = cp_params_tilde.dtype
    # BUG: not sure if it is okay to use the same ones tensor for all modes
    # ones = torch.ones(V, device=dvc, dtype=dtype)

    # ops: (H,)
    margin_mask = ops == margin_index  # (H,) bool
    one_hots = torch.nn.functional.one_hot(ops.clamp(min=0), num_classes=V).to(
        device=dvc, dtype=dtype
    )  # (H, V)
    ones = torch.ones((H, V), device=dvc, dtype=dtype)
    # For margin positions, replace one-hot with all-ones
    c = torch.where(
        margin_mask.unsqueeze(-1),  # (H, 1) -> broadcast
        # torch.ones((H, V), device=dvc, dtype=dtype),
        ones,  # (H, V)
        one_hots,  # (H, V)
    )  # (H, V)

    consts = []
    for h in range(H):
        a_h = cp_params_tilde[:, h, :]  # (R, D)
        d_h = decoder_tilde  # (D, V)
        c_h = c[h]  # (V,)
        s = h * 3  # num indexes used in each iteration
        if apply_logsumexp:
            m_a = a_h.max()
            m_d = d_h.max()
            esum_recipe.append((a_h - m_a).exp())  # (R, D)
            esum_recipe.append([0, s + 1])
            esum_recipe.append((d_h - m_d).exp())  # (D, V)
            esum_recipe.append([s + 1, s + 2])
            esum_recipe.append(c_h)  # (V,)
            esum_recipe.append([s + 2])
            consts.extend([m_a, m_d])
        else:
            esum_recipe.append(a_h)  # (R, D)
            esum_recipe.append([0, s + 1])
            esum_recipe.append(d_h)  # (D,)
            esum_recipe.append([s + 1, s + 2])
            esum_recipe.append(c_h)  # (V,)
            esum_recipe.append([s + 2])

    esum_recipe.append([])  # output is scalar
    if apply_logsumexp:
        return torch.log(torch.einsum(*esum_recipe)) + torch.stack(consts).sum()
    else:
        return torch.einsum(*esum_recipe)


def cp_normalize_decoder_einlse(
    cp_params_tilde: torch.Tensor,
    decoder_tilde: torch.Tensor,
    apply_logsumexp=True,
):
    """Reduce a CP tensor via select/marginalize operations.
    Args:
        cp_params_tilde (torch.Tensor): CP params logits. Shape: (R, H, D)
        decoder_tilde (torch.Tensor): Decoder logits. Shape: (D, V)
        ops (torch.Tensor): Ops \\in [0, V) + [margin_index]. Shape: (H,)
        use_scale_factors (bool): Whether to apply scale factors during reduction

    Note:
        Computes logP(y1, ..., yH | x) = log \\sum exp(a_h)exp(d_h(ops)) ... exp(a_H) exp(d_H(ops))

    Returns:
        torch.Tensor: Reduced tensor result (scalar)
    """
    H, V = cp_params_tilde.size(1), decoder_tilde.size(1)
    esum_recipe = []
    dvc = cp_params_tilde.device
    dtype = cp_params_tilde.dtype

    # BUG: not sure if it is okay to use the same ones tensor for all modes
    # ones = torch.ones(V, device=dvc, dtype=dtype)

    consts = []
    for h in range(H):
        a_h = cp_params_tilde[:, h, :]  # (R, D)
        d_h = decoder_tilde  # (D, V)
        s = h * 3  # num indexes used in each iteration
        ones = torch.ones(V, device=dvc, dtype=dtype)
        if apply_logsumexp:
            m_a = a_h.max()
            m_d = d_h.max()
            esum_recipe.append((a_h - m_a).exp())  # (R, D)
            esum_recipe.append([0, s + 1])
            esum_recipe.append((d_h - m_d).exp())  # (D, V)
            esum_recipe.append([s + 1, s + 2])
            esum_recipe.append(ones)  # (V,)
            esum_recipe.append([s + 2])
            consts.extend([m_a, m_d])
        else:
            esum_recipe.append(a_h)  # (R, D)
            esum_recipe.append([0, s + 1])
            esum_recipe.append(d_h)  # (D,)
            esum_recipe.append([s + 1, s + 2])
            esum_recipe.append(ones)  # (V,)
            esum_recipe.append([s + 2])

    esum_recipe.append([])  # output is scalar
    if apply_logsumexp:
        return torch.log(torch.einsum(*esum_recipe)) + torch.stack(consts).sum()
    else:
        return torch.einsum(*esum_recipe)


def cp_reduce_decoder(
    cp_params: torch.Tensor,
    ops: torch.Tensor,
    decoder: torch.Tensor,
    use_scale_factors=True,
    einlse=False,
    margin_index=-1,
):
    """Reduce a CP tensor via select/marginalize operations.
    Args:
        cp_params (torch.Tensor): CP params. Shape: (R, H, D)
        decoder (torch.Tensor): Decoder. Shape: (D, V)
        ops (torch.Tensor): Ops \\in [0, V) + [margin_index]. Shape: (H,)
        use_scale_factors (bool): Whether to apply scale factors during reduction

    Returns:
        torch.Tensor: Reduced tensor result (scalar)
    """
    assert margin_index < 0, "margin_index must be negative"
    R, H, D = cp_params.shape

    esum_fn = einlogsumexp if einlse else torch.einsum

    # For each mode: marginalize (sum) if ops[h] == -1, else select ops[h]
    marginalize_mask = ops == margin_index  # (H,)
    marginalized = esum_fn(
        "rhd,d->rh", cp_params, decoder.sum(dim=-1)
    )  # (R, H) - sum over V dimension

    # Gather selected indices (clamp to handle -1 values safely)
    selected_indices = ops.clamp(min=0).unsqueeze(0)  # (1, H)
    # (D, V) -> (D,H) then,  (D,H), (R, H, D) -> (R, H)
    selected = esum_fn(
        "dh, rhd->rh", decoder.gather(-1, selected_indices.expand(D, -1)), cp_params
    )

    # Choose marginalized or selected values based on ops
    factors = torch.where(marginalize_mask, marginalized, selected)  # (R, H)

    scale_factors = []
    res = torch.ones(R, device=cp_params.device, dtype=cp_params.dtype)

    # BUG: do scale factors online
    # if use_scale_factors:
    #     # norm of each factor
    #     # scale_factors = torch.max(factors, dim=0)[0]
    #     # factors = factors / scale_factors

    # Do it during contraction
    for i in range(H):
        res = res * factors[:, i]  # (R,)
        if use_scale_factors:
            scale_factors.append(torch.max(res))
            res = res / scale_factors[-1]
    res = res.sum()
    scale_factors = torch.stack(scale_factors) if scale_factors else torch.empty(0)

    # Compute CP tensor value: product over modes, sum over components
    return res, scale_factors


# *************************************************************************************************************************
#  VECTORIZED VERSIONS
# *************************************************************************************************************************

batch_cp_reduce = torch.vmap(cp_reduce, in_dims=(0, 0))
batch_cp_reduce_decoder = torch.vmap(cp_reduce_decoder, in_dims=(0, 0, None))
batch_cp_reduce_decoder_einlse = torch.vmap(
    cp_reduce_decoder_einlse, in_dims=(0, 0, None)
)
batch_cp_normalize_decoder_einlse = torch.vmap(
    cp_normalize_decoder_einlse, in_dims=(0, None)
)
