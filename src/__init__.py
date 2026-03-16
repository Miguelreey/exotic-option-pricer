"""
Exotic Option Pricer — Production-grade derivatives pricing library.

Modules
-------
models : Analytical and numerical pricing models (BS, Heston, rBergomi)
utils  : Visualization and Greeks utilities
"""

__version__ = '1.0.0'

from .models import PricingModel, BlackScholesModel

__all__ = ['PricingModel', 'BlackScholesModel']
