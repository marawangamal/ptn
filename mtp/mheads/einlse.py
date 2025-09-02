# einlogsumexp: einsum in log-space

import torch
import string


def einlogsumexp(subscripts, *ops):
    r"""
    Log-domain einsum.

    Given log-domain tensors :math:`X^{(1)},\dots,X^{(m)}` and an einsum
    pattern like ``'ik,kj->ij'``, this computes

    .. math::
        Y[\mathbf{i}_R]
        \;=\;
        \log \sum_{\mathbf{i}_C}
            \exp\!\Big(
              \sum_{\ell=1}^m X^{(\ell)}[\mathbf{i}_{t_\ell}]
            \Big),

    where
      * :math:`R` are the RHS (free) labels,
      * :math:`C` are the contracted labels (present on LHS but not RHS),
      * :math:`t_\ell` is the label set of operand :math:`X^{(\ell)}`.

    Equivalent to:

    .. code-block:: python

        torch.log(torch.einsum(subscripts, *[x.exp() for x in ops]))

    but implemented stably via broadcasting and ``torch.logsumexp``.

    Examples
    --------
    >>> A_log = torch.randn(3, 4)
    >>> B_log = torch.randn(4, 5)
    >>> C_log = einlogsumexp('ik,kj->ij', A_log, B_log)
    """

    # Example subscripts: 'ik,kj->ij'
    lhs, rhs = subscripts.replace(" ", "").split("->")
    terms = lhs.split(",")
    assert len(terms) == len(ops), "mismatched operands"

    # Map each label to a global axis order
    labels_all = []
    for t in terms:
        labels_all.extend(list(t))
    labels_unique = []
    for c in labels_all:
        if c not in labels_unique:
            labels_unique.append(c)

    # Bring every operand to the same global order via unsqueeze/expand
    shaped = []
    for term, x in zip(terms, ops):
        # build shape with singleton dims for all unique labels
        view = []
        for L in labels_unique:
            if L in term:
                view.append(slice(None))
            else:
                view.append(None)  # will unsqueeze here
        # unsqueeze into the global grid
        y = x
        # walk from left to right, insert singleton dims where label absent
        expand_shape = []
        xi = 0
        for L in labels_unique:
            if L in term:
                expand_shape.append(x.shape[xi])
                xi += 1
            else:
                expand_shape.append(1)
        y = y.reshape(
            [s for s in expand_shape if s != 1]
            if len(term) == len(expand_shape)
            else x.shape
        )  # safety
        # Now unsqueeze to global rank
        y = x
        xi = 0
        new_shape = []
        for L in labels_unique:
            if L in term:
                new_shape.append(x.shape[xi])
                xi += 1
            else:
                new_shape.append(1)
        y = y.reshape(new_shape)
        shaped.append(y)

    # log-space "multiply" = addition
    total = shaped[0]
    for y in shaped[1:]:
        total = total + y  # broadcast add

    # reduce (log-sum-exp) over contracted labels = those not in rhs
    contracted = [i for i, L in enumerate(labels_unique) if L not in rhs]
    if len(contracted) == 0:
        out = total  # no contraction
    else:
        out = torch.logsumexp(total, dim=contracted)

    # finally, permute to match rhs order
    if set(rhs) != set([L for i, L in enumerate(labels_unique) if i not in contracted]):
        # sanity check: rhs must be subset of labels_unique minus contracted
        raise ValueError("rhs labels must be a subset of free labels")
    # build permutation from current free-label order to rhs order
    free_labels = [L for i, L in enumerate(labels_unique) if i not in contracted]
    perm = [free_labels.index(L) for L in rhs]
    if out.ndim != len(free_labels):
        # If out has dropped dims in a different order, just return as is
        return out
    return out.permute(perm)
