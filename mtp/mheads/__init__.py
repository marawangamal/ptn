from mtp.mheads.cp_cond import CPCond
from mtp.mheads.cp_condl import CPCondl
from mtp.mheads.eagle import EAGLE
from mtp.mheads.moe import MoE
from mtp.mheads.moe_proj import MoEProjector
from mtp.mheads.multihead import Multihead
from mtp.mheads.stp import STP
from mtp.mheads.cp import CP
from ._abc import AbstractDisributionHeadConfig, AbstractDisributionHeadOutput

# !! IMPORTANT !! ::  Keep in sync with mtp.mheads.types
MHEADS = {
    "stp": STP,
    "multihead": Multihead,
    "moe": MoE,
    "moe_proj": MoEProjector,
    "cp": CP,
    "cp_cond": CPCond,
    "cp_condl": CPCondl,
    "eagle": EAGLE,
}
