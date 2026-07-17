"""
Exotic Option Pricer — exotic_option_pricer.instruments

Exotic option instruments that define payoff structures for path-dependent
derivatives. Each instrument implements ExoticOption.payoff(), which plugs
directly into MonteCarloEngine.price().

Available instruments (Phase 3):
- AsianOption    : Arithmetic / geometric averaging, fixed / floating strike
- BarrierOption  : Single-barrier knock-in / knock-out (up/down, call/put)
- DigitalOption  : Cash-or-nothing / asset-or-nothing
- LookbackOption : Floating / fixed strike, min / max of path
"""

from .asian import AsianOption
from .barrier import BarrierOption
from .base import ExoticOption
from .digital import DigitalOption
from .lookback import LookbackOption

__all__ = [
    "AsianOption",
    "BarrierOption",
    "DigitalOption",
    "ExoticOption",
    "LookbackOption",
]
