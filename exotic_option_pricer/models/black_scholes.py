"""
Exotic Option Pricer — exotic_option_pricer/models/black_scholes.py

Production-grade Black-Scholes-Merton (1973) analytical pricing engine.

Implements:
- European vanilla pricing with continuous dividend yield
- First-order Greeks: delta, gamma, vega, theta, rho
- Second-order Greeks: vanna, volga, charm, speed, zomma, color
- Dual Greeks: dual_delta, dual_gamma (strike sensitivities)
- Implied volatility solver (Halley's method with cubic convergence)
- Batch Greeks computation in single pass
- Utility: probability_itm, elasticity

All methods support scalar and vectorized NumPy inputs.

References
----------
.. [1] Black & Scholes (1973). JPE 81(3), 637-654.
.. [2] Merton (1973). Bell J. Econ. 4(1), 141-183.
.. [3] Hull (2018). Options, Futures & Other Derivatives, 10th ed.
.. [4] Haug (2007). The Complete Guide to Option Pricing Formulas, 2nd ed.
.. [5] Jaeckel (2017). "Let's Be Rational." Wilmott.
.. [6] Brenner & Subrahmanyam (1988). FAJ 5, 80-83.
"""

import warnings
from typing import Dict

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from .base import Numeric, PricingModel

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _norm_pdf(x: Numeric) -> Numeric:
    """Standard normal PDF. ~5x faster than scipy.stats.norm.pdf for arrays."""
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)  # type: ignore[no-any-return]


class BlackScholesModel(PricingModel):
    """
    Black-Scholes-Merton analytical engine for European options.

    Parameters
    ----------
    sigma : float
        Annualized volatility as decimal (0.20 = 20%). Must be > 0.

    Examples
    --------
    >>> bs = BlackScholesModel(sigma=0.20)
    >>> round(bs.price(100, 100, 1.0, 0.05, 'call'), 4)
    10.4506
    >>> round(bs.price(100, 100, 1.0, 0.05, 'call', q=0.03), 2)
    8.66
    """

    def __init__(self, sigma: float) -> None:
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(
                f"Volatility must be finite and strictly positive, got sigma={sigma}. "
                f"Use decimal: 0.20 for 20%."
            )
        if sigma > 5.0:
            warnings.warn(
                f"sigma={sigma} is unusually large. Did you mean {sigma / 100:.4f}?",
                UserWarning,
                stacklevel=2,
            )
        self.sigma: float = sigma

    # Input validation (_validate_inputs / _validate_option_type) is
    # inherited from the PricingModel ABC.

    # ──────────────────────────────────────────────
    # Core: d1, d2
    # ──────────────────────────────────────────────

    def _compute_d1_d2(
        self, S: Numeric, K: Numeric, T: Numeric, r: Numeric, q: float = 0.0
    ) -> tuple:
        """
        d1 = [ln(S/K) + (r - q + sigma^2/2)T] / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)

        Handles K=0 by setting d1=d2=50 (N(50)~1).
        """
        S = np.asarray(S, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64)

        sigma = self.sigma
        sqrt_T = np.sqrt(T)
        sigma_sqrt_T = sigma * sqrt_T

        safe_K = np.where(K > 0, K, 1.0)
        log_m = np.log(S / safe_K)

        d1 = (log_m + (r - q + 0.5 * sigma**2) * T) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T

        d1 = np.where(K > 0, d1, 50.0)
        d2 = np.where(K > 0, d2, 50.0)
        return d1, d2

    # ──────────────────────────────────────────────
    # Price
    # ──────────────────────────────────────────────

    def price(
        self,
        S: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        option_type: str = "call",
        q: float = 0.0,
    ) -> Numeric:
        """
        European option price.

        C = S*exp(-qT)*N(d1) - K*exp(-rT)*N(d2)
        P = K*exp(-rT)*N(-d2) - S*exp(-qT)*N(-d1)

        Parameters
        ----------
        S, K, T, r : float or ndarray
        option_type : 'call' or 'put'
        q : float, default 0.0
            Continuous dividend yield.
        """
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)

        S = np.asarray(S, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64)

        d1, d2 = self._compute_d1_d2(S, K, T, r, q)
        disc_r = np.exp(-r * T)
        disc_q = np.exp(-q * T)

        if opt == "call":
            p = S * disc_q * ndtr(d1) - K * disc_r * ndtr(d2)
        else:
            p = K * disc_r * ndtr(-d2) - S * disc_q * ndtr(-d1)

        p = np.maximum(p, 0.0)
        return float(p) if p.ndim == 0 else p

    # ──────────────────────────────────────────────
    # First-order Greeks
    # ──────────────────────────────────────────────

    def delta(
        self,
        S: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        option_type: str = "call",
        q: float = 0.0,
    ) -> Numeric:
        """Delta = dV/dS. Call: exp(-qT)*N(d1). Put: -exp(-qT)*N(-d1)."""
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)
        d1, _ = self._compute_d1_d2(S, K, T, r, q)
        disc_q = np.exp(-q * T)
        result = disc_q * ndtr(d1) if opt == "call" else -disc_q * ndtr(-d1)
        return float(result) if np.ndim(result) == 0 else result

    def gamma(
        self,
        S: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        option_type: str = "call",
        q: float = 0.0,
    ) -> Numeric:
        """Gamma = d²V/dS². Same for call/put. = exp(-qT)*phi(d1)/(S*sigma*sqrt(T))."""
        self._validate_inputs(S, K, T, r)
        self._validate_option_type(option_type)
        S = np.asarray(S, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        d1, _ = self._compute_d1_d2(S, K, T, r, q)
        result = np.exp(-q * T) * _norm_pdf(d1) / (S * self.sigma * np.sqrt(T))
        return float(result) if np.ndim(result) == 0 else result

    def vega(
        self,
        S: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        option_type: str = "call",
        q: float = 0.0,
    ) -> Numeric:
        """Vega = dV/dsigma. Same for call/put. = S*exp(-qT)*phi(d1)*sqrt(T)."""
        self._validate_inputs(S, K, T, r)
        self._validate_option_type(option_type)
        S = np.asarray(S, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        d1, _ = self._compute_d1_d2(S, K, T, r, q)
        result = S * np.exp(-q * T) * _norm_pdf(d1) * np.sqrt(T)
        return float(result) if np.ndim(result) == 0 else result

    def theta(
        self,
        S: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        option_type: str = "call",
        q: float = 0.0,
    ) -> Numeric:
        """
        Theta = dV/dt (per year). Typically negative for long positions.

        Call: -S*exp(-qT)*phi(d1)*sigma/(2*sqrt(T)) + q*S*exp(-qT)*N(d1)
              - r*K*exp(-rT)*N(d2)
        Put:  -S*exp(-qT)*phi(d1)*sigma/(2*sqrt(T)) - q*S*exp(-qT)*N(-d1)
              + r*K*exp(-rT)*N(-d2)
        """
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)
        S = np.asarray(S, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64)

        d1, d2 = self._compute_d1_d2(S, K, T, r, q)
        disc_r = np.exp(-r * T)
        disc_q = np.exp(-q * T)
        phi_d1 = _norm_pdf(d1)
        sqrt_T = np.sqrt(T)

        time_decay = -S * disc_q * phi_d1 * self.sigma / (2.0 * sqrt_T)
        if opt == "call":
            result = time_decay + q * S * disc_q * ndtr(d1) - r * K * disc_r * ndtr(d2)
        else:
            result = time_decay - q * S * disc_q * ndtr(-d1) + r * K * disc_r * ndtr(-d2)
        return float(result) if np.ndim(result) == 0 else result

    def rho(
        self,
        S: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        option_type: str = "call",
        q: float = 0.0,
    ) -> Numeric:
        """Rho = dV/dr. Call: K*T*exp(-rT)*N(d2). Put: -K*T*exp(-rT)*N(-d2)."""
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)
        K = np.asarray(K, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64)
        _, d2 = self._compute_d1_d2(S, K, T, r, q)
        disc_r = np.exp(-r * T)
        if opt == "call":
            result = K * T * disc_r * ndtr(d2)
        else:
            result = -K * T * disc_r * ndtr(-d2)
        return float(result) if np.ndim(result) == 0 else result

    # ──────────────────────────────────────────────
    # Second-order Greeks
    # ──────────────────────────────────────────────

    def vanna(
        self,
        S: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        option_type: str = "call",
        q: float = 0.0,
    ) -> Numeric:
        """
        Vanna = d²V/(dS dsigma) = dDelta/dsigma = dVega/dS.
        = -exp(-qT) * phi(d1) * d2 / sigma. Same for call/put.
        """
        self._validate_inputs(S, K, T, r)
        self._validate_option_type(option_type)
        d1, d2 = self._compute_d1_d2(S, K, T, r, q)
        result = -np.exp(-q * T) * _norm_pdf(d1) * d2 / self.sigma
        return float(result) if np.ndim(result) == 0 else result

    def volga(
        self,
        S: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        option_type: str = "call",
        q: float = 0.0,
    ) -> Numeric:
        """
        Volga (Vomma) = d²V/dsigma² = dVega/dsigma.
        = vega * d1 * d2 / sigma. Same for call/put.
        """
        self._validate_inputs(S, K, T, r)
        self._validate_option_type(option_type)
        S = np.asarray(S, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        d1, d2 = self._compute_d1_d2(S, K, T, r, q)
        v = S * np.exp(-q * T) * _norm_pdf(d1) * np.sqrt(T)
        result = v * d1 * d2 / self.sigma
        return float(result) if np.ndim(result) == 0 else result

    def charm(
        self,
        S: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        option_type: str = "call",
        q: float = 0.0,
    ) -> Numeric:
        """
        Charm (delta decay) = -dDelta/dT = dDelta/dt.

        Call: q*exp(-qT)*N(d1) - exp(-qT)*phi(d1)*[2(r-q)T - d2*sigma*sqrt(T)]/(2*sigma*T*sqrt(T))
        Put:  -q*exp(-qT)*N(-d1) - exp(-qT)*phi(d1)*[2(r-q)T - d2*sigma*sqrt(T)]/(2*sigma*T*sqrt(T))
        """
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)
        T = np.asarray(T, dtype=np.float64)
        d1, d2 = self._compute_d1_d2(S, K, T, r, q)
        disc_q = np.exp(-q * T)
        phi_d1 = _norm_pdf(d1)
        sigma = self.sigma
        sqrt_T = np.sqrt(T)

        common = (
            -disc_q * phi_d1 * (2 * (r - q) * T - d2 * sigma * sqrt_T) / (2 * sigma * T * sqrt_T)
        )
        if opt == "call":
            result = q * disc_q * ndtr(d1) + common
        else:
            result = -q * disc_q * ndtr(-d1) + common
        return float(result) if np.ndim(result) == 0 else result

    def speed(
        self,
        S: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        option_type: str = "call",
        q: float = 0.0,
    ) -> Numeric:
        """
        Speed = dGamma/dS = d³V/dS³.
        = -Gamma/S * [d1/(sigma*sqrt(T)) + 1]. Same for call/put.
        """
        self._validate_inputs(S, K, T, r)
        self._validate_option_type(option_type)
        S = np.asarray(S, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        d1, _ = self._compute_d1_d2(S, K, T, r, q)
        g = np.exp(-q * T) * _norm_pdf(d1) / (S * self.sigma * np.sqrt(T))
        result = -g / S * (d1 / (self.sigma * np.sqrt(T)) + 1.0)
        return float(result) if np.ndim(result) == 0 else result

    def zomma(
        self,
        S: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        option_type: str = "call",
        q: float = 0.0,
    ) -> Numeric:
        """
        Zomma = dGamma/dsigma. = Gamma * (d1*d2 - 1) / sigma. Same for call/put.
        """
        self._validate_inputs(S, K, T, r)
        self._validate_option_type(option_type)
        S = np.asarray(S, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        d1, d2 = self._compute_d1_d2(S, K, T, r, q)
        g = np.exp(-q * T) * _norm_pdf(d1) / (S * self.sigma * np.sqrt(T))
        result = g * (d1 * d2 - 1.0) / self.sigma
        return float(result) if np.ndim(result) == 0 else result

    def color(
        self,
        S: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        option_type: str = "call",
        q: float = 0.0,
    ) -> Numeric:
        """
        Color = dGamma/dt = -dGamma/dT.
        = -Gamma/(2T) * [2qT + 1 + d1*(2(r-q)T - d2*sigma*sqrt(T))/(sigma*sqrt(T))].
        Same for call/put.
        """
        self._validate_inputs(S, K, T, r)
        self._validate_option_type(option_type)
        S = np.asarray(S, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        d1, d2 = self._compute_d1_d2(S, K, T, r, q)
        sigma = self.sigma
        sqrt_T = np.sqrt(T)
        g = np.exp(-q * T) * _norm_pdf(d1) / (S * sigma * sqrt_T)
        bracket = 2 * q * T + 1.0 + d1 * (2 * (r - q) * T - d2 * sigma * sqrt_T) / (sigma * sqrt_T)
        result = -g / (2.0 * T) * bracket
        return float(result) if np.ndim(result) == 0 else result

    # ──────────────────────────────────────────────
    # Dual Greeks (strike sensitivities)
    # ──────────────────────────────────────────────

    def dual_delta(
        self,
        S: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        option_type: str = "call",
        q: float = 0.0,
    ) -> Numeric:
        """
        Dual delta = dV/dK.
        Call: -exp(-rT)*N(d2).  Put: exp(-rT)*N(-d2).
        """
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)
        T = np.asarray(T, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64)
        _, d2 = self._compute_d1_d2(S, K, T, r, q)
        disc_r = np.exp(-r * T)
        result = -disc_r * ndtr(d2) if opt == "call" else disc_r * ndtr(-d2)
        return float(result) if np.ndim(result) == 0 else result

    def dual_gamma(
        self,
        S: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        option_type: str = "call",
        q: float = 0.0,
    ) -> Numeric:
        """
        Dual gamma = d²V/dK².  = exp(-rT)*phi(d2)/(K*sigma*sqrt(T)).
        Same for call/put. Proportional to risk-neutral density (Breeden-Litzenberger).
        """
        self._validate_inputs(S, K, T, r)
        self._validate_option_type(option_type)
        K = np.asarray(K, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64)
        _, d2 = self._compute_d1_d2(S, K, T, r, q)
        result = np.exp(-r * T) * _norm_pdf(d2) / (K * self.sigma * np.sqrt(T))
        return float(result) if np.ndim(result) == 0 else result

    # ──────────────────────────────────────────────
    # Utility methods
    # ──────────────────────────────────────────────

    def probability_itm(
        self,
        S: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        option_type: str = "call",
        q: float = 0.0,
    ) -> Numeric:
        """Risk-neutral probability of finishing ITM. Call: N(d2). Put: N(-d2)."""
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)
        _, d2 = self._compute_d1_d2(S, K, T, r, q)
        result = ndtr(d2) if opt == "call" else ndtr(-d2)
        return float(result) if np.ndim(result) == 0 else result

    def elasticity(
        self,
        S: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        option_type: str = "call",
        q: float = 0.0,
    ) -> Numeric:
        """
        Lambda (elasticity/leverage) = Delta * S / V.
        Measures percentage change in option per 1% change in underlying.
        """
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)
        d = self.delta(S, K, T, r, opt, q)
        v = self.price(S, K, T, r, opt, q)
        S = np.asarray(S, dtype=np.float64)
        safe_v = np.where(np.abs(v) > 1e-15, v, 1.0)
        result = np.where(np.abs(v) > 1e-15, d * S / safe_v, np.nan)
        return float(result) if np.ndim(result) == 0 else result

    # ──────────────────────────────────────────────
    # Implied volatility
    # ──────────────────────────────────────────────

    @staticmethod
    def implied_vol(
        price_market: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = "call",
        q: float = 0.0,
        tol: float = 1e-12,
        max_iter: int = 20,
    ) -> float:
        """
        Compute implied volatility via Halley's method (cubic convergence).

        Inspired by Jaeckel (2017) "Let's Be Rational":
        1. Regime-dependent initial guess (Brenner-Subrahmanyam near ATM,
           asymptotic far from ATM)
        2. Halley iterations using vega and volga (3rd order convergence,
           typically converges in 2-4 iterations vs 50+ for Newton-Raphson)
        3. Inline d1/d2 computation (zero object creation overhead)
        4. Brent fallback for pathological cases

        Parameters
        ----------
        price_market : float
            Observed option price.
        S, K, T, r : float
            Contract parameters.
        option_type : str
            'call' or 'put'.
        q : float
            Continuous dividend yield.
        tol : float
            Price tolerance for convergence.
        max_iter : int
            Max Halley iterations (default 20; typically needs 2-4).

        Returns
        -------
        float
            Implied volatility (annualized decimal).

        Raises
        ------
        ValueError
            If price violates no-arbitrage bounds.

        References
        ----------
        .. [1] Jaeckel (2017). "Let's Be Rational." Wilmott.
        .. [2] Brenner & Subrahmanyam (1988). FAJ 5, 80-83.
        """
        opt = BlackScholesModel._validate_option_type(option_type)

        # Precompute constants (used throughout)
        disc_r = np.exp(-r * T)
        disc_q = np.exp(-q * T)
        sqrt_T = np.sqrt(T)

        # No-arbitrage bounds
        if opt == "call":
            lb = max(S * disc_q - K * disc_r, 0.0)
            ub = S * disc_q
        else:
            lb = max(K * disc_r - S * disc_q, 0.0)
            ub = K * disc_r

        if price_market < lb - 1e-10:
            raise ValueError(f"Price {price_market:.6f} below no-arbitrage bound {lb:.6f}")
        if price_market > ub + 1e-10:
            raise ValueError(f"Price {price_market:.6f} above no-arbitrage bound {ub:.6f}")

        log_SK = np.log(S / K) if K > 0 else 50.0

        # --- Regime-dependent initial guess ---
        # Log-forward-moneyness: x = ln(F/K) where F = S*exp((r-q)*T)
        x = log_SK + (r - q) * T

        if abs(x) < 0.5:
            # Near ATM-forward: Brenner-Subrahmanyam (1988)
            sigma = np.sqrt(2.0 * np.pi / T) * price_market / (S * disc_q)
        else:
            # Far from ATM: asymptotic σ√T ≈ √(2|x|) (Bachelier limit)
            sigma = np.sqrt(2.0 * abs(x)) / sqrt_T

        sigma = np.clip(sigma, 0.005, 5.0)

        # --- Halley iterations (cubic convergence) ---
        # All computation is inline — no BlackScholesModel objects created.
        for _ in range(max_iter):
            sigma_sqrt_T = sigma * sqrt_T
            d1 = (log_SK + (r - q + 0.5 * sigma**2) * T) / sigma_sqrt_T
            d2 = d1 - sigma_sqrt_T
            phi_d1 = _norm_pdf(d1)

            if opt == "call":
                p = max(S * disc_q * ndtr(d1) - K * disc_r * ndtr(d2), 0.0)
            else:
                p = max(K * disc_r * ndtr(-d2) - S * disc_q * ndtr(-d1), 0.0)

            diff = p - price_market
            if abs(diff) < tol:
                return float(sigma)

            vega = S * disc_q * phi_d1 * sqrt_T
            if abs(vega) < 1e-300:
                break

            # Halley step: h / (1 - h*d1*d2/(2*sigma))
            # where h = diff/vega is the Newton step
            # Cubic convergence: O(eps^3) per iteration
            newton_step = diff / vega
            halley_denom = 1.0 - 0.5 * newton_step * d1 * d2 / sigma

            if abs(halley_denom) > 0.1:
                sigma -= newton_step / halley_denom
            else:
                # Damped Newton fallback when Halley correction diverges
                sigma -= 0.5 * newton_step

            sigma = np.clip(sigma, 1e-6, 10.0)

        # Brent fallback (inline — no object creation)
        def obj(sig):
            ssqt = sig * sqrt_T
            dd1 = (log_SK + (r - q + 0.5 * sig**2) * T) / ssqt
            dd2 = dd1 - ssqt
            if opt == "call":
                return max(S * disc_q * ndtr(dd1) - K * disc_r * ndtr(dd2), 0.0) - price_market
            else:
                return max(K * disc_r * ndtr(-dd2) - S * disc_q * ndtr(-dd1), 0.0) - price_market

        try:
            return float(brentq(obj, 1e-6, 10.0, xtol=tol, maxiter=500))
        except ValueError:
            return float("nan")

    @staticmethod
    def implied_vol_batch(
        prices: np.ndarray,
        S: float,
        strikes: np.ndarray,
        T: Numeric,
        r: float,
        option_type: str = "call",
        q: float = 0.0,
        tol: float = 1e-12,
        max_iter: int = 20,
    ) -> np.ndarray:
        """
        Vectorized implied volatilities for many (strike, maturity) points.

        Follows the exact Halley trajectory of ``implied_vol`` element by
        element: same regime-dependent initial guess, same update rule with
        each element FROZEN the moment it converges (so extra iterations
        for slow elements cannot perturb already-converged ones), and the
        same Brent fallback — delegated to the scalar solver for the rare
        elements that do not converge in ``max_iter`` Halley steps. Output
        therefore matches a scalar loop to within 1-2 ulp (NumPy's SIMD
        array kernels for exp/log/ndtr may round the last bit differently
        from the scalar path), at a fraction of the cost.
        ``T`` may be an array so an ENTIRE option chain inverts in one
        call — this is the hot path of the Heston calibration objective,
        and per-expiry sub-batches of ~40 options would be dominated by
        NumPy small-array overhead rather than arithmetic.

        The error contract differs from the scalar solver: prices outside
        the no-arbitrage band (or non-finite) yield NaN instead of raising,
        because a batch caller needs per-element failure, not
        all-or-nothing.

        Parameters
        ----------
        prices : np.ndarray
            Observed option prices, same shape as ``strikes``.
        S : float
            Spot price.
        strikes : np.ndarray
            Strikes.
        T : float or np.ndarray
            Maturity (years), scalar or per-option array.
        r : float
            Risk-free rate, common to the batch.
        option_type : str, default 'call'
            'call' or 'put'.
        q : float, default 0.0
            Continuous dividend yield.
        tol : float, default 1e-12
            Price tolerance for convergence.
        max_iter : int, default 20
            Max Halley iterations before the Brent fallback.

        Returns
        -------
        np.ndarray
            Implied vols, same shape as the inputs; NaN where the price is
            not invertible.
        """
        opt = BlackScholesModel._validate_option_type(option_type)
        prices_a = np.asarray(prices, dtype=np.float64)
        strikes_a = np.asarray(strikes, dtype=np.float64)
        if prices_a.shape != strikes_a.shape:
            raise ValueError(
                f"prices and strikes must have the same shape, "
                f"got {prices_a.shape} and {strikes_a.shape}"
            )
        shape = prices_a.shape
        p = prices_a.ravel()
        K = strikes_a.ravel()
        T_a = np.broadcast_to(np.asarray(T, dtype=np.float64), shape).ravel()

        disc_r = np.exp(-r * T_a)
        disc_q = np.exp(-q * T_a)
        sqrt_T = np.sqrt(T_a)

        S_disc_q = S * disc_q
        K_disc_r = K * disc_r
        if opt == "call":
            lb = np.maximum(S_disc_q - K_disc_r, 0.0)
            ub = S_disc_q
        else:
            lb = np.maximum(K_disc_r - S_disc_q, 0.0)
            ub = K_disc_r
        valid = (p >= lb - 1e-10) & (p <= ub + 1e-10) & np.isfinite(p)

        log_SK = np.where(K > 0.0, np.log(S / np.where(K > 0.0, K, 1.0)), 50.0)
        x = log_SK + (r - q) * T_a

        # Regime-dependent initial guess: Brenner-Subrahmanyam near
        # ATM-forward, asymptotic sigma*sqrt(T) ~ sqrt(2|x|) far from it.
        # Invalid slots get a dummy 0.2 so the vector math stays finite.
        sigma = np.where(
            np.abs(x) < 0.5,
            np.sqrt(2.0 * np.pi / T_a) * p / S_disc_q,
            np.sqrt(2.0 * np.abs(x)) / sqrt_T,
        )
        sigma = np.clip(np.where(valid, sigma, 0.2), 0.005, 5.0)

        converged = np.zeros(p.shape, dtype=bool)
        needs_fallback = np.zeros(p.shape, dtype=bool)
        for _ in range(max_iter):
            active = valid & ~converged & ~needs_fallback
            if not np.any(active):
                break
            sigma_sqrt_T = sigma * sqrt_T
            # Expression grouped exactly as in the scalar solver — any
            # floating-point regrouping would break the bit-identical
            # trajectory contract.
            d1 = (log_SK + (r - q + 0.5 * sigma**2) * T_a) / sigma_sqrt_T
            d2 = d1 - sigma_sqrt_T
            phi_d1 = _norm_pdf(d1)

            if opt == "call":
                model_p = np.maximum(S_disc_q * ndtr(d1) - K_disc_r * ndtr(d2), 0.0)
            else:
                model_p = np.maximum(K_disc_r * ndtr(-d2) - S_disc_q * ndtr(-d1), 0.0)

            diff = model_p - p
            newly = active & (np.abs(diff) < tol)
            converged |= newly
            active &= ~newly

            vega = S_disc_q * phi_d1 * sqrt_T
            dead = active & (np.abs(vega) < 1e-300)
            needs_fallback |= dead
            active &= ~dead

            # Halley step, damped-Newton fallback when the correction
            # diverges — identical branch logic to the scalar solver.
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                newton = diff / vega
                halley_denom = 1.0 - 0.5 * newton * d1 * d2 / sigma
                step = np.where(np.abs(halley_denom) > 0.1, newton / halley_denom, 0.5 * newton)
            sigma = np.where(active, np.clip(sigma - step, 1e-6, 10.0), sigma)
        needs_fallback |= valid & ~converged

        out = np.where(valid & converged, sigma, np.nan)
        for i in np.flatnonzero(needs_fallback):
            try:
                out[i] = BlackScholesModel.implied_vol(
                    float(p[i]),
                    S,
                    float(K[i]),
                    float(T_a[i]),
                    r,
                    opt,
                    q=q,
                    tol=tol,
                    max_iter=max_iter,
                )
            except ValueError:
                out[i] = np.nan
        return out.reshape(shape)

    # ──────────────────────────────────────────────
    # Batch Greeks (single-pass, avoids redundant d1/d2)
    # ──────────────────────────────────────────────

    def greeks(
        self,
        S: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        option_type: str = "call",
        q: float = 0.0,
    ) -> Dict[str, Numeric]:
        """
        Compute all Greeks in a single pass.

        Returns dict with keys: price, delta, gamma, vega, theta, rho,
        vanna, volga, charm, speed, zomma, color, dual_delta, dual_gamma,
        probability_itm, elasticity.
        """
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)

        S = np.asarray(S, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64)
        sigma = self.sigma

        d1, d2 = self._compute_d1_d2(S, K, T, r, q)
        sqrt_T = np.sqrt(T)
        sigma_sqrt_T = sigma * sqrt_T

        disc_r = np.exp(-r * T)
        disc_q = np.exp(-q * T)
        phi_d1 = _norm_pdf(d1)
        phi_d2 = _norm_pdf(d2)
        N_d1 = ndtr(d1)
        N_d2 = ndtr(d2)
        Nn_d1 = ndtr(-d1)
        Nn_d2 = ndtr(-d2)

        # ----- Price -----
        if opt == "call":
            price = np.maximum(S * disc_q * N_d1 - K * disc_r * N_d2, 0.0)
            delta_val = disc_q * N_d1
            theta_val = (
                -S * disc_q * phi_d1 * sigma / (2 * sqrt_T)
                + q * S * disc_q * N_d1
                - r * K * disc_r * N_d2
            )
            rho_val = K * T * disc_r * N_d2
            dd_val = -disc_r * N_d2
            charm_n = q * disc_q * N_d1
            prob_itm = N_d2
        else:
            price = np.maximum(K * disc_r * Nn_d2 - S * disc_q * Nn_d1, 0.0)
            delta_val = -disc_q * Nn_d1
            theta_val = (
                -S * disc_q * phi_d1 * sigma / (2 * sqrt_T)
                - q * S * disc_q * Nn_d1
                + r * K * disc_r * Nn_d2
            )
            rho_val = -K * T * disc_r * Nn_d2
            dd_val = disc_r * Nn_d2
            charm_n = -q * disc_q * Nn_d1
            prob_itm = Nn_d2

        # ----- Greeks same for call/put -----
        gamma_val = disc_q * phi_d1 / (S * sigma * sqrt_T)
        vega_val = S * disc_q * phi_d1 * sqrt_T
        vanna_val = -disc_q * phi_d1 * d2 / sigma
        volga_val = vega_val * d1 * d2 / sigma
        speed_val = -gamma_val / S * (d1 / sigma_sqrt_T + 1.0)
        zomma_val = gamma_val * (d1 * d2 - 1.0) / sigma
        charm_common = (
            -disc_q * phi_d1 * (2 * (r - q) * T - d2 * sigma * sqrt_T) / (2 * sigma * T * sqrt_T)
        )
        charm_val = charm_n + charm_common
        color_bracket = (
            2 * q * T + 1.0 + d1 * (2 * (r - q) * T - d2 * sigma * sqrt_T) / (sigma * sqrt_T)
        )
        color_val = -gamma_val / (2.0 * T) * color_bracket
        dg_val = disc_r * phi_d2 / (K * sigma * sqrt_T)
        safe_price = np.where(np.abs(price) > 1e-15, price, 1.0)
        elast_val = np.where(np.abs(price) > 1e-15, delta_val * S / safe_price, np.nan)

        res = {
            "price": price,
            "delta": delta_val,
            "gamma": gamma_val,
            "vega": vega_val,
            "theta": theta_val,
            "rho": rho_val,
            "vanna": vanna_val,
            "volga": volga_val,
            "charm": charm_val,
            "speed": speed_val,
            "zomma": zomma_val,
            "color": color_val,
            "dual_delta": dd_val,
            "dual_gamma": dg_val,
            "probability_itm": prob_itm,
            "elasticity": elast_val,
        }
        return {k: float(v) if np.ndim(v) == 0 else v for k, v in res.items()}

    def __repr__(self) -> str:
        return f"BlackScholesModel(sigma={self.sigma})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BlackScholesModel) and self.sigma == other.sigma

    def __hash__(self) -> int:
        return hash(self.sigma)
