"""
Exotic Option Pricer — src.engines

Numerical pricing engines for derivative valuation.

MonteCarloEngine is the primary engine, supporting:
- GBM simulation (exact, Euler-Maruyama, Milstein)
- European option pricing with variance reduction
- Generic payoff pricing for path-dependent exotics
"""

from .monte_carlo import MCResult, MonteCarloEngine

__all__ = ["MCResult", "MonteCarloEngine"]
