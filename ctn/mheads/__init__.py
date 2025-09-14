from ctn.mheads.mps import MPS
from ctn.mheads.moe import MoE
from ctn.mheads.cp import CP
from ctn.mheads.born import BM
from ._abc import AbstractDisributionHeadConfig, AbstractDisributionHeadOutput

# !! IMPORTANT !! ::  Keep in sync with ctn.mheads.types
MHEADS = {
    "cp": CP,  # Canonical Polyadic
    "moe": MoE,  # Mixture of Experts
    "mps": MPS,  # Matrix Product State
    "bm": BM,  # Born Machine
}
