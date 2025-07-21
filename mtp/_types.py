from typing import Literal

PositivityFuncType = Literal[
    "relu", "leaky_relu", "sq", "abs", "exp", "safe_exp", "sigmoid", "none"
]

# NOTE: Should be in sync with mtp.mheads
ModelHeadType = Literal["stp"]
