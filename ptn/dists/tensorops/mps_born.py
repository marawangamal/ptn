from typing import List
import torch


def born_mps_marginalize(
    g: torch.Tensor, a: torch.Tensor, b: torch.Tensor, use_scale_factors: bool = True
):
    """Marginalize a Born MPS tensor.

    Args:
        g (torch.Tensor): g tensor. Shape: (H, R, D, R)

    Returns:
        torch.Tensor: Marginalized tensor. Shape: (1,)
    """
    H, _, _, _ = g.shape
    scale_factors = []
    L = torch.einsum("p,pdq,r,rds->qs", a, g[0], a, g[0])
    for h in range(1, H):
        L = torch.einsum("pdq,pr,rds ->qs", g[h], L, g[h])
        if use_scale_factors:
            sf = L.abs().max()
            scale_factors.append(sf)
            L = L / sf
    L = torch.einsum("pq,p,q->", L, b, b)
    if not use_scale_factors:
        scale_factors = [torch.tensor(1.0)]
    return L, torch.stack(scale_factors)  # (1,), (H,)


def born_mps_select(
    g: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    y: torch.Tensor,
    use_scale_factors: bool = False,
):
    """Select a Born MPS tensor.
    Args:
        g (torch.Tensor): g tensor. Shape: (H, R, D, R)
        a (torch.Tensor): a tensor. Shape: (R,)
        b (torch.Tensor): b tensor. Shape: (R,)
        y (torch.Tensor): y tensor. Shape: (H,)

    Returns:
        torch.Tensor: Selected tensor. Shape: (1,)
    """
    H, R, _, _ = g.shape
    y_slct = y.reshape(-1, 1, 1, 1).expand(-1, R, -1, R)
    g_slct = g.gather(-2, y_slct).squeeze(-2)  # (H, R, R)
    L = a
    scale_factors = [torch.tensor(1.0)]
    for h in range(H):
        L = torch.einsum("p,pq->q", L, g_slct[h])
        if use_scale_factors:
            sf = L.abs().max()
            scale_factors.append(sf)
            L /= sf
    L = torch.einsum("p,p->", L, b)
    return L.pow(2), torch.stack(scale_factors)


def born_mps_ortho_marginalize(g: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    """Marginalize an Orthogonal Born MPS tensor.

    Args:
        g (torch.Tensor): g tensor. Shape: (H, R, D, R)

    Returns:
        torch.Tensor: Marginalized tensor. Shape: (1,)
    """
    return torch.einsum(
        "r,q,ris,qis,sjp,sjt,p,t->", a, a, g[0], g[0], g[-1], g[-1], b, b
    )


def born_mps_canonical_marginalize(
    g: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    canonical_index: List[int],
    use_scale_factors: bool = False,
):
    """Marginalize a Canonical Born MPS tensor.

    Args:
        g (torch.Tensor): g tensor. Shape: (H, R, D, R)
        a (torch.Tensor): a tensor. Shape: (R,)
        b (torch.Tensor): b tensor. Shape: (R,)
        canonical_index (List[int]): Canonical index.
    """
    assert len(canonical_index) in [1, 2], "Canonical index must be 1 or 2"
    scale_factors = [torch.tensor(1.0)]
    if len(canonical_index) == 1:
        g0 = g[canonical_index[0]]
        if use_scale_factors:
            sf = g0.abs().max()
            g0 = g0 / sf
            scale_factors.append(sf)
        z_tilde = torch.einsum("idj,idj->", g0, g0)
    else:
        i0, i1 = canonical_index
        g0, g1 = g[i0], g[i1]
        if use_scale_factors:
            sf0 = g0.abs().max()
            sf1 = g1.abs().max()
            g0 = g0 / sf0
            g1 = g1 / sf1
            scale_factors.append(sf0)
            scale_factors.append(sf1)
        z_tilde = torch.einsum("idj,jvr,idq,qvr->", g0, g1, g0, g1)
    return z_tilde, torch.stack(scale_factors)


batch_born_mps_marginalize = torch.vmap(born_mps_marginalize, in_dims=(0, 0, 0))
batch_born_mps_select = torch.vmap(born_mps_select, in_dims=(0, 0, 0, 0))
batch_ortho_born_mps_marginalize = torch.vmap(
    born_mps_ortho_marginalize, in_dims=(0, 0, 0)
)
batch_born_mps_canonical_marginalize = torch.vmap(
    born_mps_canonical_marginalize, in_dims=(0, 0, 0, None)
)
