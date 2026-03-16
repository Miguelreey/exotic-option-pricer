"""
Exotic Option Pricer — src/models/base.py

Abstract base class defining the interface for all pricing models.
Ensures interchangeability between BlackScholesModel, HestonModel,
RoughBergomiModel, etc.

Design Principles
-----------------
- Common interface: all models accept (S, K, T, r, option_type, q)
- Model parameters (sigma, kappa, v0, etc.) are set at construction
- Market parameters (S, K, T, r, q) are passed to methods
- All methods support scalar and vectorized NumPy inputs

References
----------
.. [1] Glasserman (2003). Monte Carlo Methods in Financial Engineering.
"""

from abc import ABC, abstractmethod
from typing import Union, Dict
import numpy as np

Numeric = Union[float, np.ndarray]


class PricingModel(ABC):
    """
    Abstract base class for derivative pricing models.

    All concrete pricing models must implement this interface.
    This ensures that any model can be used interchangeably in:
    - Portfolio valuation engines
    - Calibration routines
    - Risk management systems
    - Visualization tools

    Subclasses
    ----------
    - BlackScholesModel : Analytical BS-Merton (1973)
    - HestonModel : Stochastic volatility via characteristic function (Phase 4)
    - RoughBergomiModel : Rough volatility via hybrid MC (Phase 5)
    """

    @abstractmethod
    def price(self, S: Numeric, K: Numeric, T: Numeric, r: Numeric,
              option_type: str = 'call', q: float = 0.0) -> Numeric:
        """
        European option price.

        Parameters
        ----------
        S : float or ndarray
            Spot price(s). Must be > 0.
        K : float or ndarray
            Strike price(s). Must be >= 0.
        T : float or ndarray
            Time to expiry in years. Must be > 0.
        r : float or ndarray
            Risk-free rate (annualized, continuous compounding).
        option_type : str, default 'call'
            'call' or 'put'.
        q : float, default 0.0
            Continuous dividend yield.

        Returns
        -------
        float or ndarray
            Option price(s).
        """
        ...

    @abstractmethod
    def delta(self, S: Numeric, K: Numeric, T: Numeric, r: Numeric,
              option_type: str = 'call', q: float = 0.0) -> Numeric:
        """First derivative with respect to spot: dV/dS."""
        ...

    @abstractmethod
    def gamma(self, S: Numeric, K: Numeric, T: Numeric, r: Numeric,
              option_type: str = 'call', q: float = 0.0) -> Numeric:
        """Second derivative with respect to spot: d²V/dS²."""
        ...

    @abstractmethod
    def vega(self, S: Numeric, K: Numeric, T: Numeric, r: Numeric,
             option_type: str = 'call', q: float = 0.0) -> Numeric:
        """Derivative with respect to volatility: dV/dσ."""
        ...

    @abstractmethod
    def theta(self, S: Numeric, K: Numeric, T: Numeric, r: Numeric,
              option_type: str = 'call', q: float = 0.0) -> Numeric:
        """Derivative with respect to calendar time: dV/dt (per year)."""
        ...

    @abstractmethod
    def rho(self, S: Numeric, K: Numeric, T: Numeric, r: Numeric,
            option_type: str = 'call', q: float = 0.0) -> Numeric:
        """Derivative with respect to risk-free rate: dV/dr."""
        ...

    @abstractmethod
    def greeks(self, S: Numeric, K: Numeric, T: Numeric, r: Numeric,
               option_type: str = 'call', q: float = 0.0) -> Dict[str, Numeric]:
        """
        Compute all available Greeks in a single pass.

        Returns
        -------
        dict of {str: float or ndarray}
            At minimum: price, delta, gamma, vega, theta, rho.
            Models may add additional Greeks.
        """
        ...
