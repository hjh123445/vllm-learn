"""Explicit model boundary for forward-only pipeline stages."""
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class IntermediateTensors:
    hidden_states: "Tensor"  # [scheduled_tokens, hidden_size]
    residual: "Tensor"       # same shape/dtype; delayed residual addition
