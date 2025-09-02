# einlogsumexp.py
import torch


def einlogsumexp(*args):
    r"""
    Log-domain einsum.

    String form:
        einlogsumexp("ik,kj->ij", A_log, B_log)

    List/ellipsis form (like torch.einsum):
        einlogsumexp(A_log, [..., 0, 1], B_log, [..., 1, 2], [..., 0, 2])

    Computes in log-space:
        Y[i_R] = log( sum_{i_C} exp( sum_l X^(l)[i_{t_l}] ) )
    where i_R are RHS/free labels (or the final index list in list-form),
    i_C are contracted labels (present on inputs but not on the RHS),
    and labels can be integers; `...` denotes a block of (broadcastable) axes.

    Equivalent (but numerically safer) to:
        torch.log(torch.einsum(..., *[x.exp() for x in inputs]))
    """
    if len(args) == 0:
        raise ValueError("einlogsumexp expects arguments.")

    # ------------------ Parse API variants ------------------
    if isinstance(args[0], str):
        # String form: subscripts, *tensors
        subscripts = args[0].replace(" ", "")
        tensors = list(args[1:])
        if "->" not in subscripts:
            raise ValueError("String subscripts must contain '->'.")
        lhs, rhs = subscripts.split("->")
        terms = lhs.split(",")
        assert len(terms) == len(tensors), "mismatched operands"
        op_labels = [list(term) for term in terms]
        rhs_labels = list(rhs)
        ell_labels = []  # no ellipsis in string mode
    else:
        # List/ellipsis form: T1, idx1, T2, idx2, ..., out_idx
        if len(args) < 3 or (len(args) % 2) != 1:
            raise ValueError("List form must be: T1, idx1, T2, idx2, ..., out_idx")

        *pairs, out_idx = args
        tensors = []
        raw_idx_lists = []

        for t, idx in zip(pairs[0::2], pairs[1::2]):
            if not isinstance(t, torch.Tensor):
                raise ValueError("Expected Tensor in operand position.")
            if not isinstance(idx, (list, tuple)):
                raise ValueError(
                    "Index spec must be list/tuple (may contain Ellipsis)."
                )
            tensors.append(t)
            raw_idx_lists.append(list(idx))

        # Infer per-operand ellipsis widths (how many unnamed axes)
        ell_widths = []
        for t, idx in zip(tensors, raw_idx_lists):
            named = sum(1 for x in idx if x is not Ellipsis)
            w = t.ndim - named
            if w < 0:
                raise ValueError("Index list longer than tensor rank.")
            ell_widths.append(w)
        E = max(ell_widths) if ell_widths else 0  # global ellipsis width

        # Normalize output index list: ensure it has Ellipsis if E>0
        if not isinstance(out_idx, (list, tuple)):
            raise ValueError("Final output index list required.")
        out_idx = list(out_idx)
        if E > 0 and all(x is not Ellipsis for x in out_idx):
            # mimic einsum behavior: if inputs have ellipsis, output must too
            out_idx = [Ellipsis] + out_idx

        # Build global ellipsis label tokens (shared across operands and output)
        ell_labels = [f"@e{k}" for k in range(E)]

        def canon_tok(x):
            # Canonicalize labels to strings (ints -> '#<int>')
            if x is Ellipsis:
                return None
            if isinstance(x, int):
                return f"#{x}"
            if isinstance(x, str):
                return x
            raise ValueError("Labels must be int or str (plus Ellipsis).")

        # Expand each operand's ellipsis to the LAST w tokens of ell_labels
        # (aligns trailing unnamed axes for broadcasting).
        op_labels = []
        for idx, w in zip(raw_idx_lists, ell_widths):
            expanded = []
            for x in idx:
                if x is Ellipsis:
                    expanded.extend(ell_labels[E - w : E])
                else:
                    expanded.append(canon_tok(x))
            if len(expanded) != tensors[len(op_labels)].ndim:
                raise ValueError(
                    "Index list after ellipsis expansion must match tensor rank."
                )
            op_labels.append(expanded)

        # Expand RHS ellipsis to ALL ell_labels (order preserved where placed)
        rhs_labels = []
        saw_ellipsis = False
        for x in out_idx:
            if x is Ellipsis:
                rhs_labels.extend(ell_labels)
                saw_ellipsis = True
            else:
                rhs_labels.append(canon_tok(x))
        # If no ellipsis on RHS (E==0), rhs_labels stays as given.

    # ------------------ Build global order & reshape ------------------
    # Ensure all ell labels are present first (so we can broadcast missing ones)
    labels_unique = list(ell_labels)
    # Then add labels in order of first occurrence across ops and RHS
    for labs in op_labels + [rhs_labels]:
        for L in labs:
            if L is None:
                continue
            if L not in labels_unique:
                labels_unique.append(L)

    shaped = []
    for labs, x in zip(op_labels, tensors):
        axis_map = {lbl: i for i, lbl in enumerate(labs)}
        new_shape = [
            x.shape[axis_map[L]] if L in axis_map else 1 for L in labels_unique
        ]
        y = x.reshape(new_shape)
        shaped.append(y)

    # ------------------ Log-space "multiply" and contract ------------------
    total = shaped[0]
    for y in shaped[1:]:
        total = total + y

    contracted_axes = [i for i, L in enumerate(labels_unique) if L not in rhs_labels]
    out = total if not contracted_axes else torch.logsumexp(total, dim=contracted_axes)

    # ------------------ Permute to RHS order ------------------
    remaining = [L for L in labels_unique if L in rhs_labels]
    if set(rhs_labels) != set(remaining):
        raise ValueError("RHS labels must be a subset of free labels.")
    if out.ndim == 0 or remaining == rhs_labels:
        return out
    perm = [remaining.index(L) for L in rhs_labels]
    return out.permute(perm)
