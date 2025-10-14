from typing import List
import torch


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


batch_ortho_born_mps_marginalize = torch.vmap(
    born_mps_ortho_marginalize, in_dims=(0, 0, 0)
)
batch_born_mps_canonical_marginalize = torch.vmap(
    born_mps_canonical_marginalize, in_dims=(0, 0, 0, None)
)
