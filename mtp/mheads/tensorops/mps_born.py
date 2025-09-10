import torch


def born_mps_marginalize(g: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    """Marginalize a Born MPS tensor.

    Args:
        g (torch.Tensor): g tensor. Shape: (H, R, D, R)

    Returns:
        torch.Tensor: Marginalized tensor. Shape: (1,)
    """
    H, _, _, _ = g.shape
    L = torch.einsum("p,pdq,r,rds->qs", a, g[0], a, g[0])
    for h in range(1, H):
        L = torch.einsum("pdq,qr,rds ->ps", g[h], L, g[h])
    L = torch.einsum("pq,p,q->", L, b, b)
    return L


def born_mps_select(g: torch.Tensor, a: torch.Tensor, b: torch.Tensor, y: torch.Tensor):
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
    for h in range(H):
        L = torch.einsum("p,pq->q", L, g_slct[h])
    L = torch.einsum("p,p->", L, b)
    return L.pow(2)


batch_born_mps_marginalize = torch.vmap(born_mps_marginalize, in_dims=(0, 0, 0))
batch_born_mps_select = torch.vmap(born_mps_select, in_dims=(0, 0, 0, 0))
