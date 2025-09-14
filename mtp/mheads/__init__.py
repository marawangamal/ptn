from mtp.mheads.mps import MPS
from mtp.mheads.moe import MoE
from mtp.mheads.cp import CP
from ._abc import AbstractDisributionHeadConfig, AbstractDisributionHeadOutput

# !! IMPORTANT !! ::  Keep in sync with mtp.mheads.types
MHEADS = {
    "cp": CP,
    "moe": MoE,
    "mps": MPS,
    "born": BornMachineUnconditional,
}
