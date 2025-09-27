from ptn.mheads.bm2s import BM
from ptn.mheads.mps import MPS
from ptn.mheads.moe import MoE
from ptn.mheads.cp import CP
from ptn.mheads.bmnc import BMNC
from ._abc import AbstractDisributionHeadConfig, AbstractDisributionHeadOutput

# !! IMPORTANT !! ::  Keep in sync with ptn.mheads.types
MHEADS = {
    "cp": CP,  # Canonical Polyadic
    "moe": MoE,  # Mixture of Experts
    "mps": MPS,  # Matrix Product State
    "bm": BM,  # Born Machine Canonical Form w/ DMRG)
    "bmnc": BMNC,  # Born Machine (Born Machine Non-Canonical Form w/ LogSF algo)
}
