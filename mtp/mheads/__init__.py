from mtp.mheads.cp_proj import CPProjector
from mtp.mheads.moe import MoE
from mtp.mheads.moe_proj import MoEProjector
from mtp.mheads.multihead import Multihead
from mtp.mheads.stp import STP
from ._abc import AbstractDisributionHeadConfig, AbstractDisributionHeadOutput

# !! IMPORTANT !! ::  Keep in sync with mtp.mheads.types
MHEADS = {
    "stp": STP,
    "multihead": Multihead,
    "moe": MoE,
    "moe_proj": MoEProjector,
    "cp_proj": CPProjector,
}
