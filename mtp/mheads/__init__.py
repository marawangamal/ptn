from mtp.mheads.multihead import Multihead
from mtp.mheads.stp import STP

# NOTE: Should be in sync with mtp._types.ModelHeadType
MHEADS = {"stp": STP, "multihead": Multihead, "none": lambda *args, **kwargs: None}
