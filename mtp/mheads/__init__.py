from mtp.mheads.cp_cond import CPCond
from mtp.mheads.multihead import Multihead
from mtp.mheads.stp import STP
from mtp.mheads.cp import CP
from ._abc import AbstractDisributionHeadConfig, AbstractDisributionHeadOutput

# !! IMPORTANT !! ::  Keep in sync with mtp.mheads
MHEADS = {"stp": STP, "multihead": Multihead, "cp": CP, "cp_cond": CPCond}
