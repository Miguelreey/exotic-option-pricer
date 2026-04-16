"""
Exotic Option Pricer — src/instruments/base.py

Abstract base class for exotic option instruments.

Unlike PricingModel (which represents a pricing MODEL — Black-Scholes, Heston,
etc.), ExoticOption represents a PAYOFF STRUCTURE. Exotic options define how
the payoff depends on the price path, and delegate the actual pricing to an
external engine (Monte Carlo, PDE, etc.).

Design Principles
-----------------
- ExoticOption.payoff() is the single integration point with MonteCarloEngine.
  It maps simulated paths to undiscounted payoffs, and the engine handles
  discounting, variance reduction, and confidence intervals.
- Subclasses may provide analytical prices where closed-form solutions exist
  (e.g., geometric Asian via Kemna-Vorst, digital via BS). These serve as:
  (a) cross-validation targets for MC
  (b) control variates for MC variance reduction
- Instrument parameters (strike, barrier, averaging type) are set at
  construction. Market parameters (S, T, r, sigma) are passed to analytical
  methods or handled by the engine.

References
----------
.. [1] Glasserman (2003). Monte Carlo Methods in Financial Engineering.
.. [2] Hull (2018). Options, Futures & Other Derivatives, 10th ed. Ch. 26.
"""

from abc import ABC, abstractmethod

import numpy as np


class ExoticOption(ABC):
    """
    Abstract base class for exotic option instruments.

    All exotic option classes must implement ``payoff()``, which maps
    simulated price paths to undiscounted payoff values. This method is
    designed to plug directly into ``MonteCarloEngine.price()``:

    >>> mc = MonteCarloEngine(n_paths=100_000, seed=42)
    >>> paths = mc.simulate_gbm(100, 1.0, 0.05, 0.20)
    >>> result = mc.price(exotic.payoff, paths, r=0.05, T=1.0)

    Subclasses
    ----------
    - AsianOption : Arithmetic/geometric averaging (Phase 3)
    - BarrierOption : Knock-in/knock-out with barrier level (Phase 3)
    - LookbackOption : Floating/fixed strike on path extremum (Phase 3)
    - DigitalOption : Cash-or-nothing / asset-or-nothing (Phase 3)
    """

    @abstractmethod
    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute undiscounted payoff from simulated price paths.

        Parameters
        ----------
        paths : np.ndarray, shape (n_paths, n_steps + 1)
            Simulated price paths from an SDE solver.
            ``paths[:, 0]`` is the initial price S_0 (same for all paths).
            ``paths[:, -1]`` is the terminal price S_T.
            Intermediate columns are prices at monitoring times.

        Returns
        -------
        np.ndarray, shape (n_paths,)
            Undiscounted payoff for each path. The pricing engine applies
            the discount factor e^{-rT} separately.

        Notes
        -----
        The payoff function must be vectorized over paths (first axis).
        It must NOT apply discounting — that is the engine's responsibility.
        """
        ...
