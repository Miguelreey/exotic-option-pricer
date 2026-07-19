"""
Exotic Option Pricer — Production-grade derivatives pricing library.

Modules
-------
models  : Analytical and numerical pricing models (BS, Heston, rBergomi)
engines : Numerical engines (Monte Carlo, PDE solvers)
utils   : Visualization and Greeks utilities
"""

__version__ = "2.0.0"

from .engines import MCResult, MonteCarloEngine
from .models import BlackScholesModel, PricingModel

__all__ = ["PricingModel", "BlackScholesModel", "MonteCarloEngine", "MCResult"]
