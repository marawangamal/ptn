from mtp.mheads.cp import CP
from mtp.mheads.cp_decoder import CPD
from mtp.mheads.moe_decoder import MoED
from mtp.mheads.mps import MPS
from mtp.mheads.mps_decoder import MPSD
from mtp.mheads.multihead import Multihead
from mtp.mheads.moe import MoE
from mtp.mheads.stp import STP
from mtp.mheads.umps import UMPS
from ._abc import AbstractDisributionHeadConfig, AbstractDisributionHeadOutput

# !! IMPORTANT !! ::  Keep in sync with mtp.mheads.types
MHEADS = {
    "stp": STP,
    "multihead": Multihead,
    "cp": CP,
    "cp_decoder": CPD,
    "moe": MoE,
    "moe_decoder": MoED,
    "mps": MPS,
    "mps_decoder": MPSD,
    "umps": UMPS,
}
