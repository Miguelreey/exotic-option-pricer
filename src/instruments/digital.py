"""
Exotic Option Pricer — src/instruments/digital.py

Digital (binary) options: cash-or-nothing and asset-or-nothing.

Digital options are the fundamental building blocks of all European-style
derivatives. A vanilla call decomposes exactly as:

    C_vanilla = V_asset_call - K * V_cash_call

which IS the Black-Scholes formula rewritten in terms of its two components.
Understanding this decomposition is essential for structured products pricing,
where autocallables, reverse convertibles, and bonus certificates embed
digital payoffs as coupons conditional on barrier events.

Payoff Definitions
------------------
Cash-or-nothing call : pays Q if S_T > K, else 0
Cash-or-nothing put  : pays Q if S_T < K, else 0
Asset-or-nothing call: pays S_T if S_T > K, else 0
Asset-or-nothing put : pays S_T if S_T < K, else 0

Analytical Prices (under GBM / Black-Scholes)
----------------------------------------------
Cash-or-nothing call : e^{-rT} Q N(d_2)
Cash-or-nothing put  : e^{-rT} Q N(-d_2)
Asset-or-nothing call: S e^{-qT} N(d_1)
Asset-or-nothing put : S e^{-qT} N(-d_1)

where d_1, d_2 are the standard Black-Scholes parameters.

Key Identities
--------------
1. Complementarity: V_cash_call + V_cash_put = e^{-rT} Q
   (the events S_T > K and S_T < K are complementary under continuous measure)

2. Complementarity: V_asset_call + V_asset_put = S e^{-qT}
   (total asset payout equals the forward)

3. Vanilla decomposition: C = V_asset_call - K * V_cash_call
   (this is literally the BS formula split into its N(d_1) and N(d_2) terms)

Numerical Considerations
-------------------------
Digital payoffs are discontinuous (indicator function). This causes:
- Slow MC convergence: Var = p(1-p)Q^2, maximized at ATM where p ~ 0.5
- Pathwise Greeks do not exist (derivative of indicator is Dirac delta)
- Must use bump-and-revalue or likelihood ratio for Greeks

References
----------
.. [1] Hull (2018). Options, Futures & Other Derivatives, 10th ed. Ch. 26.
.. [2] Haug (2007). The Complete Guide to Option Pricing Formulas, 2nd ed.
.. [3] Reiner & Rubinstein (1991). "Unscrambling the Binary Code." Risk 4(9).
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from .base import ExoticOption


def _compute_d1_d2(
    S: float, K: float, T: float, r: float, sigma: float, q: float,
) -> tuple[float, float]:
    """
    Standard Black-Scholes d_1 and d_2.

    d_1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T))
    d_2 = d_1 - sigma sqrt(T)

    Parameters
    ----------
    S : float
        Spot price. Must be > 0.
    K : float
        Strike price. Must be > 0.
    T : float
        Time to expiry in years. Must be > 0.
    r : float
        Risk-free rate.
    sigma : float
        Annualized volatility. Must be > 0.
    q : float
        Continuous dividend yield.

    Returns
    -------
    d1 : float
    d2 : float
    """
    sqrt_T = np.sqrt(T)
    sigma_sqrt_T = sigma * sqrt_T
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T
    return float(d1), float(d2)


class DigitalOption(ExoticOption):
    """
    Digital (binary) option instrument.

    Digital options pay a fixed amount (cash-or-nothing) or the asset value
    (asset-or-nothing) contingent on the terminal price exceeding or falling
    below the strike.

    Parameters
    ----------
    K : float
        Strike price. Must be > 0.
    option_type : str, default 'call'
        'call' (pays if S_T > K) or 'put' (pays if S_T < K).
    payout_type : str, default 'cash'
        'cash' (pays fixed amount Q) or 'asset' (pays S_T).
    cash_amount : float, default 1.0
        Fixed payout amount Q for cash-or-nothing. Ignored for asset-or-nothing.
        Must be >= 0.

    Examples
    --------
    >>> from src.instruments.digital import DigitalOption
    >>> from src.engines.monte_carlo import MonteCarloEngine
    >>> import numpy as np
    >>> dig = DigitalOption(K=100, option_type='call', payout_type='cash')
    >>> mc = MonteCarloEngine(n_paths=500_000, seed=42)
    >>> paths = mc.simulate_gbm(100, 1.0, 0.05, 0.20, n_steps=1)
    >>> result = mc.price(dig.payoff, paths, r=0.05, T=1.0)
    >>> analytical = DigitalOption.analytical_price(100, 100, 1.0, 0.05, 0.20)
    >>> abs(result.price - analytical) < 0.01
    True
    """

    def __init__(
        self,
        K: float,
        option_type: str = "call",
        payout_type: str = "cash",
        cash_amount: float = 1.0,
    ) -> None:
        if K <= 0:
            raise ValueError(f"K must be > 0, got {K}")

        opt = option_type.strip().lower()
        if opt == "c":
            opt = "call"
        elif opt == "p":
            opt = "put"
        if opt not in ("call", "put"):
            raise ValueError(
                f"option_type must be 'call' or 'put', got '{option_type}'"
            )

        pay = payout_type.strip().lower()
        if pay not in ("cash", "asset"):
            raise ValueError(
                f"payout_type must be 'cash' or 'asset', got '{payout_type}'"
            )

        if cash_amount < 0:
            raise ValueError(f"cash_amount must be >= 0, got {cash_amount}")

        self.K = float(K)
        self._option_type = opt
        self._payout_type = pay
        self.cash_amount = cash_amount

    @property
    def option_type(self) -> str:
        """'call' or 'put'."""
        return self._option_type

    @property
    def payout_type(self) -> str:
        """'cash' or 'asset'."""
        return self._payout_type

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute undiscounted digital payoff from simulated paths.

        For digital options, only the terminal value S_T = paths[:, -1]
        matters. The payoff is:

        - Cash-or-nothing call: Q * 1_{S_T > K}
        - Cash-or-nothing put:  Q * 1_{S_T < K}
        - Asset-or-nothing call: S_T * 1_{S_T > K}
        - Asset-or-nothing put:  S_T * 1_{S_T < K}

        Parameters
        ----------
        paths : np.ndarray, shape (n_paths, n_steps + 1)
            Simulated price paths.

        Returns
        -------
        np.ndarray, shape (n_paths,)
            Undiscounted payoffs.

        Notes
        -----
        The strict inequality (S_T > K, not >=) is the market convention
        for cash-or-nothing options. Under a continuous distribution,
        P(S_T = K) = 0, so the choice is immaterial for pricing but
        matters for consistency in tests with deterministic paths.
        """
        S_T = paths[:, -1]

        if self._option_type == "call":
            indicator = (S_T > self.K).astype(np.float64)
        else:
            indicator = (S_T < self.K).astype(np.float64)

        if self._payout_type == "cash":
            return self.cash_amount * indicator
        else:
            return S_T * indicator

    @staticmethod
    def analytical_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        payout_type: str = "cash",
        cash_amount: float = 1.0,
        q: float = 0.0,
    ) -> float:
        """
        Black-Scholes analytical price for digital options.

        Parameters
        ----------
        S : float
            Spot price. Must be > 0.
        K : float
            Strike price. Must be > 0.
        T : float
            Time to expiry in years. Must be > 0.
        r : float
            Risk-free rate (annualized, continuous compounding).
        sigma : float
            Annualized volatility. Must be > 0.
        option_type : str, default 'call'
            'call' or 'put'.
        payout_type : str, default 'cash'
            'cash' (pays Q) or 'asset' (pays S_T).
        cash_amount : float, default 1.0
            Fixed payout Q for cash-or-nothing.
        q : float, default 0.0
            Continuous dividend yield.

        Returns
        -------
        float
            Analytical digital option price.

        Notes
        -----
        Cash-or-nothing call  = e^{-rT} Q N(d_2)
        Cash-or-nothing put   = e^{-rT} Q N(-d_2)
        Asset-or-nothing call = S e^{-qT} N(d_1)
        Asset-or-nothing put  = S e^{-qT} N(-d_1)

        These follow directly from the Black-Scholes formula. The vanilla
        call price C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2) is literally
        the asset-or-nothing call minus K times the cash-or-nothing call.

        References
        ----------
        .. [1] Hull (2018). Ch. 26.
        .. [2] Haug (2007). Ch. 10.
        """
        if S <= 0:
            raise ValueError(f"S must be > 0, got {S}")
        if K <= 0:
            raise ValueError(f"K must be > 0, got {K}")
        if T <= 0:
            raise ValueError(f"T must be > 0, got {T}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")

        opt = option_type.strip().lower()
        if opt == "c":
            opt = "call"
        elif opt == "p":
            opt = "put"
        if opt not in ("call", "put"):
            raise ValueError(
                f"option_type must be 'call' or 'put', got '{option_type}'"
            )

        pay = payout_type.strip().lower()
        if pay not in ("cash", "asset"):
            raise ValueError(
                f"payout_type must be 'cash' or 'asset', got '{payout_type}'"
            )

        d1, d2 = _compute_d1_d2(S, K, T, r, sigma, q)

        if pay == "cash":
            if opt == "call":
                return float(np.exp(-r * T) * cash_amount * ndtr(d2))
            else:
                return float(np.exp(-r * T) * cash_amount * ndtr(-d2))
        else:
            if opt == "call":
                return float(S * np.exp(-q * T) * ndtr(d1))
            else:
                return float(S * np.exp(-q * T) * ndtr(-d1))

    def __repr__(self) -> str:
        if self._payout_type == "cash":
            return (
                f"DigitalOption(K={self.K}, type='{self._option_type}', "
                f"payout='cash', Q={self.cash_amount})"
            )
        return (
            f"DigitalOption(K={self.K}, type='{self._option_type}', "
            f"payout='asset')"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DigitalOption):
            return NotImplemented
        return (
            self.K == other.K
            and self._option_type == other._option_type
            and self._payout_type == other._payout_type
            and self.cash_amount == other.cash_amount
        )

    def __hash__(self) -> int:
        return hash((self.K, self._option_type, self._payout_type, self.cash_amount))
