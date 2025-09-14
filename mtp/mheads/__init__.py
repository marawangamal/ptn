from mtp.mheads.mps import MPS
from mtp.mheads.moe import MoE
from mtp.mheads.cp import CP
from mtp.mheads.born import BM
from ._abc import AbstractDisributionHeadConfig, AbstractDisributionHeadOutput

# !! IMPORTANT !! ::  Keep in sync with mtp.mheads.types
MHEADS = {
    "cp": CP,  # Canonical Polyadic
    "moe": MoE,  # Mixture of Experts
    "mps": MPS,  # Matrix Product State
    "bm": BM,  # Born Machine
}
