from ptn.dists.bm2s import BM
from ptn.dists.mps import MPS
from ptn.dists.moe import MoE
from ptn.dists.cp import CP
from ptn.dists.bmnc import BMNC
from ._abc import AbstractDisributionHeadConfig, AbstractDisributionHeadOutput

# !! IMPORTANT !! ::  Keep in sync with ptn.dists.types
dists = {
    "cp": CP,  # Canonical Polyadic
    "moe": MoE,  # Mixture of Experts
    "mps": MPS,  # Matrix Product State
    "bm": BM,  # Born Machine Canonical Form w/ DMRG)
    "bmnc": BMNC,  # Born Machine (Born Machine Non-Canonical Form w/ LogSF algo)
}
