"""
Exotic Option Pricer — src/instruments/asian.py

Asian (average-price) options: arithmetic and geometric averaging,
fixed and floating strike variants.

Asian options are the most liquid path-dependent derivatives traded
globally. They are popular in commodity markets (protection against
fixing-date manipulation), FX corporate hedging (coverage over an
average rather than a single fixing), and equity structured products
(averaging reduces gamma risk near expiry). The averaging makes them
cheaper than vanilla options — a feature exploited for corporate sales
and for reducing hedging costs.

Mathematical Background
-----------------------
For a set of monitoring times t_1 < t_2 < ... < t_n = T, define

    arithmetic average:  A(S) = (1/n) * sum_{i=1..n} S_{t_i}
    geometric average:   G(S) = (prod_{i=1..n} S_{t_i})^{1/n}
                              = exp( (1/n) * sum_{i=1..n} ln S_{t_i} )

Payoffs (under the four common conventions):

    fixed-strike call:     max( A(S) - K, 0 )           or max( G(S) - K, 0 )
    fixed-strike put:      max( K - A(S), 0 )           or max( K - G(S), 0 )
    floating-strike call:  max( S_T - A(S), 0 )         or max( S_T - G(S), 0 )
    floating-strike put:   max( A(S) - S_T, 0 )         or max( G(S) - S_T, 0 )

Under GBM, G(S) is lognormal (since log-GBM increments are Gaussian and
a linear combination of Gaussians is Gaussian). This gives a closed-form
Black-Scholes-like formula for the geometric Asian (Kemna-Vorst 1990).
The arithmetic Asian has no closed form because the sum of lognormals is
not lognormal — it must be priced by Monte Carlo or PDE methods.

Kemna-Vorst Geometric Asian Formula (discrete monitoring)
---------------------------------------------------------
With n equally spaced observations t_i = iT/n (i = 1..n), under the
risk-neutral measure with rate r and continuous dividend yield q:

    sigma_hat^2 = sigma^2 * (n+1)*(2*n+1) / (6*n^2)
    r_hat       = 0.5 * sigma_hat^2 + (r - q - 0.5*sigma^2) * (n+1)/(2*n)

    d1 = ( ln(S/K) + (r_hat + sigma_hat^2/2) * T ) / ( sigma_hat * sqrt(T) )
    d2 = d1 - sigma_hat * sqrt(T)

    C_geo = exp( (r_hat - r) * T ) * ( S*N(d1) - K*exp(-r_hat*T)*N(d2) )
    P_geo = exp( (r_hat - r) * T ) * ( K*exp(-r_hat*T)*N(-d2) - S*N(-d1) )

The adjustment exp((r_hat - r)T) converts the Black-Scholes price in the
"effective" rate-r_hat economy back to discounting at the true rate r.
The formula recovers the continuous-monitoring limit as n -> infinity:

    sigma_hat^2 -> sigma^2 / 3       (so sigma_hat -> sigma / sqrt(3))
    r_hat       -> (r - q)/2 - sigma^2/12

Control Variate Technique (Kemna-Vorst 1990, Glasserman 2003, Sec. 4.1)
-----------------------------------------------------------------------
The arithmetic Asian has no closed form but is almost perfectly
correlated with the geometric Asian (correlation > 0.99 for typical
parameters). Using the geometric Asian as a control variate yields
variance reduction ratios > 95%:

    V_arith_cv = V_arith_MC - beta * (V_geo_MC - V_geo_analytical)

where beta is estimated from the sample via the OLS formula
beta* = Cov(arith, geo) / Var(geo) and V_geo_analytical is the
Kemna-Vorst closed form above. This is the single most important
variance-reduction trick for Asian options.

Design Decisions
----------------
- ``d1``/``d2`` are computed inline rather than imported from
  ``BlackScholesModel``. This mirrors ``DigitalOption`` and avoids
  coupling ``instruments/`` to ``models/``.
- The geometric average is computed via ``exp(mean(log(S)))`` rather
  than ``prod(S)**(1/n)`` to avoid overflow for n large.
- ``paths[:, 0]`` (the initial spot S_0) is NOT included in the average.
  The standard market convention is to average over the fixing dates
  t_1, ..., t_n only. Including S_0 would bias the average and contradict
  the Kemna-Vorst formula, which assumes n fixings strictly after t=0.

References
----------
.. [1] Kemna, A. G. Z. & Vorst, A. C. F. (1990). "A Pricing Method for
   Options Based on Average Asset Values." Journal of Banking and Finance
   14, pp. 113-129.
.. [2] Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering.
   Springer. Chapter 4 (control variates), Sec. 7.2 (Asian pricing).
.. [3] Hull, J. (2018). Options, Futures & Other Derivatives, 10th ed.
   Chapter 26 (exotic options).
.. [4] Haug, E. G. (2007). The Complete Guide to Option Pricing Formulas,
   2nd ed. McGraw-Hill, Chapter 4 (Asian options).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.special import ndtr

from .base import ExoticOption


class AsianOption(ExoticOption):
    """
    Asian (average-price) option instrument.

    Asian options derive their payoff from an average of the underlying
    price over a set of monitoring times. Four variants are supported,
    combining {arithmetic, geometric} averaging with {fixed, floating}
    strike:

    - Fixed-strike arithmetic call: ``max(mean(S) - K, 0)``
    - Fixed-strike geometric call:  ``max(geomean(S) - K, 0)``
    - Floating-strike arithmetic call: ``max(S_T - mean(S), 0)``
    - Floating-strike geometric call:  ``max(S_T - geomean(S), 0)``
    (plus the corresponding puts with reversed max arguments)

    Only fixed-strike geometric Asians have a closed-form price under GBM
    (Kemna-Vorst 1990), exposed as :meth:`geometric_price`. Arithmetic
    Asians are priced by Monte Carlo and can leverage the geometric
    Asian as a control variate via :meth:`geometric_control_fn`.

    Parameters
    ----------
    K : float or None, default None
        Strike price. Required when ``strike_type='fixed'`` (must be > 0).
        Ignored when ``strike_type='floating'`` and stored as ``None``.
    option_type : {'call', 'put', 'c', 'p'}, default 'call'
        Payoff direction. Case-insensitive; single-letter forms accepted.
    avg_type : {'arithmetic', 'geometric'}, default 'arithmetic'
        Type of average used in the payoff.
    strike_type : {'fixed', 'floating'}, default 'fixed'
        Whether the strike is a pre-agreed constant (fixed) or the
        terminal price ``S_T`` (floating).

    Notes
    -----
    The payoff method averages over ``paths[:, 1:]``, excluding the
    initial spot ``S_0`` in ``paths[:, 0]``. This matches the convention
    used by Kemna-Vorst and the market standard for fixing-date Asians.
    The number of fixings is therefore ``n_steps`` (one fewer than the
    number of columns in the paths array).

    Examples
    --------
    >>> from src.instruments.asian import AsianOption
    >>> from src.engines.monte_carlo import MonteCarloEngine
    >>> asian = AsianOption(K=100, option_type='call', avg_type='arithmetic')
    >>> mc = MonteCarloEngine(n_paths=200_000, seed=42)
    >>> paths = mc.simulate_gbm(100, 1.0, 0.05, 0.20, n_steps=252)
    >>> cv_fn = asian.geometric_control_fn(100, 1.0, 0.05, 0.20, n_obs=252)
    >>> result = mc.price(asian.payoff, paths, 0.05, 1.0, control_fn=cv_fn)
    """

    def __init__(
        self,
        K: float | None = None,
        option_type: str = "call",
        avg_type: str = "arithmetic",
        strike_type: str = "fixed",
    ) -> None:
        opt = option_type.strip().lower()
        if opt == "c":
            opt = "call"
        elif opt == "p":
            opt = "put"
        if opt not in ("call", "put"):
            raise ValueError(
                f"option_type must be 'call' or 'put', got '{option_type}'"
            )

        avg = avg_type.strip().lower()
        if avg not in ("arithmetic", "geometric"):
            raise ValueError(
                f"avg_type must be 'arithmetic' or 'geometric', got '{avg_type}'"
            )

        stk = strike_type.strip().lower()
        if stk not in ("fixed", "floating"):
            raise ValueError(
                f"strike_type must be 'fixed' or 'floating', got '{strike_type}'"
            )

        if stk == "fixed":
            if K is None:
                raise ValueError("K is required when strike_type='fixed'")
            if K <= 0:
                raise ValueError(f"K must be > 0, got {K}")
            self.K: float | None = float(K)
        else:
            self.K = None

        self._option_type = opt
        self._avg_type = avg
        self._strike_type = stk

    @property
    def option_type(self) -> str:
        """'call' or 'put'."""
        return self._option_type

    @property
    def avg_type(self) -> str:
        """'arithmetic' or 'geometric'."""
        return self._avg_type

    @property
    def strike_type(self) -> str:
        """'fixed' or 'floating'."""
        return self._strike_type

    def _compute_average(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute the running average over the fixing dates ``paths[:, 1:]``.

        Returns an array of shape ``(n_paths,)``. The initial spot
        ``paths[:, 0] = S_0`` is deliberately excluded, matching the
        standard market convention for Asian fixing schedules.
        """
        fixings = paths[:, 1:]
        if self._avg_type == "arithmetic":
            return np.asarray(np.mean(fixings, axis=1))
        return np.asarray(np.exp(np.mean(np.log(fixings), axis=1)))

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute undiscounted Asian payoff from simulated paths.

        Parameters
        ----------
        paths : np.ndarray, shape (n_paths, n_steps + 1)
            Simulated price paths. Column 0 is the initial spot ``S_0``
            (same for all paths) and is excluded from the average.

        Returns
        -------
        np.ndarray, shape (n_paths,)
            Undiscounted payoffs. For fixed strike, ``max(avg - K, 0)``
            (call) or ``max(K - avg, 0)`` (put). For floating strike,
            ``max(S_T - avg, 0)`` (call) or ``max(avg - S_T, 0)`` (put).
        """
        avg = self._compute_average(paths)

        if self._strike_type == "fixed":
            assert self.K is not None  # invariant from __init__
            if self._option_type == "call":
                return np.asarray(np.maximum(avg - self.K, 0.0))
            return np.asarray(np.maximum(self.K - avg, 0.0))

        S_T = paths[:, -1]
        if self._option_type == "call":
            return np.asarray(np.maximum(S_T - avg, 0.0))
        return np.asarray(np.maximum(avg - S_T, 0.0))

    @staticmethod
    def geometric_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n_obs: int,
        option_type: str = "call",
        q: float = 0.0,
    ) -> float:
        """
        Kemna-Vorst (1990) analytical price for a fixed-strike geometric Asian.

        Uses the fact that under GBM the geometric average is lognormal,
        so the pricing integral has the same form as Black-Scholes with
        adjusted volatility and drift:

            sigma_hat^2 = sigma^2 * (n+1)*(2*n+1) / (6*n^2)
            r_hat       = 0.5*sigma_hat^2 + (r - q - 0.5*sigma^2) * (n+1)/(2*n)

            d1 = ( ln(S/K) + (r_hat + sigma_hat^2/2) T ) / ( sigma_hat sqrt(T) )
            d2 = d1 - sigma_hat sqrt(T)

            C_geo = exp((r_hat - r) T) * ( S*N(d1) - K*exp(-r_hat T)*N(d2) )
            P_geo = exp((r_hat - r) T) * ( K*exp(-r_hat T)*N(-d2) - S*N(-d1) )

        The factor ``exp((r_hat - r)T)`` converts the Black-Scholes price
        in the effective rate-``r_hat`` economy back to discounting at the
        actual risk-free rate ``r``.

        Parameters
        ----------
        S : float
            Initial spot price. Must be > 0.
        K : float
            Strike. Must be > 0.
        T : float
            Time to expiry in years. Must be > 0.
        r : float
            Risk-free rate (annualized, continuous compounding).
        sigma : float
            Annualized volatility. Must be > 0.
        n_obs : int
            Number of equally spaced fixing dates (n_obs >= 1).
        option_type : {'call', 'put', 'c', 'p'}, default 'call'
            Payoff direction.
        q : float, default 0.0
            Continuous dividend yield.

        Returns
        -------
        float
            Fair value of the fixed-strike geometric Asian option.

        Notes
        -----
        As ``n_obs`` grows, the formula converges to the continuous-time
        limit with ``sigma_hat = sigma/sqrt(3)`` and
        ``r_hat = (r-q)/2 - sigma^2/12``. The discrete case is always
        the appropriate one for market-standard fixing schedules; the
        continuous limit is only relevant for theoretical sanity checks.

        References
        ----------
        .. [1] Kemna & Vorst (1990). J. Banking & Finance 14, 113-129.
        """
        if S <= 0:
            raise ValueError(f"S must be > 0, got {S}")
        if K <= 0:
            raise ValueError(f"K must be > 0, got {K}")
        if T <= 0:
            raise ValueError(f"T must be > 0, got {T}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if n_obs < 1:
            raise ValueError(f"n_obs must be >= 1, got {n_obs}")

        opt = option_type.strip().lower()
        if opt == "c":
            opt = "call"
        elif opt == "p":
            opt = "put"
        if opt not in ("call", "put"):
            raise ValueError(
                f"option_type must be 'call' or 'put', got '{option_type}'"
            )

        n = float(n_obs)
        sigma_hat_sq = sigma * sigma * (n + 1.0) * (2.0 * n + 1.0) / (6.0 * n * n)
        sigma_hat = np.sqrt(sigma_hat_sq)
        r_hat = 0.5 * sigma_hat_sq + (r - q - 0.5 * sigma * sigma) * (n + 1.0) / (2.0 * n)

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r_hat + 0.5 * sigma_hat_sq) * T) / (sigma_hat * sqrt_T)
        d2 = d1 - sigma_hat * sqrt_T
        adj = np.exp((r_hat - r) * T)

        if opt == "call":
            return float(
                adj * (S * ndtr(d1) - K * np.exp(-r_hat * T) * ndtr(d2))
            )
        return float(
            adj * (K * np.exp(-r_hat * T) * ndtr(-d2) - S * ndtr(-d1))
        )

    def geometric_control_fn(
        self,
        S0: float,
        T: float,
        r: float,
        sigma: float,
        n_obs: int,
        q: float = 0.0,
    ) -> Callable[[np.ndarray], tuple[np.ndarray, float]]:
        """
        Build a control-variate function using the geometric Asian.

        Only applicable to arithmetic fixed-strike Asians: for these,
        the arithmetic and geometric average are almost perfectly
        correlated (correlation > 0.99 for typical equity parameters),
        and the geometric Asian has a closed-form price via
        :meth:`geometric_price`. Plugging the returned ``control_fn``
        into ``MonteCarloEngine.price(..., control_fn=...)`` yields a
        variance reduction ratio typically above 95%.

        The returned function matches the engine's control-variate
        contract: given a paths array, it returns

            (discount * geo_payoffs, analytical_price)

        where both quantities are expressed on the same (discounted)
        scale as the discounted arithmetic payoffs computed inside
        ``price()``. The engine then applies the OLS-optimal
        ``beta*`` estimator to form the adjusted estimator.

        Parameters
        ----------
        S0 : float
            Initial spot used when the engine simulated the paths.
        T : float
            Time to expiry.
        r : float
            Risk-free rate.
        sigma : float
            Volatility used when simulating the paths.
        n_obs : int
            Number of fixing dates. Must match ``paths.shape[1] - 1``
            when the control function is later invoked.
        q : float, default 0.0
            Continuous dividend yield.

        Returns
        -------
        Callable[[np.ndarray], tuple[np.ndarray, float]]
            Control-variate function compatible with
            ``MonteCarloEngine.price``.

        Raises
        ------
        ValueError
            If the instrument is not an arithmetic fixed-strike Asian.

        Notes
        -----
        This method captures the analytical geometric price at call
        time — do not mutate ``S0``, ``sigma``, ``r``, ``q`` between
        building the control function and running Monte Carlo.
        """
        if self._strike_type != "fixed" or self._avg_type != "arithmetic":
            raise ValueError(
                "geometric_control_fn only applies to arithmetic fixed-strike "
                f"Asians, got avg_type='{self._avg_type}' "
                f"strike_type='{self._strike_type}'"
            )

        assert self.K is not None
        K = self.K
        opt = self._option_type
        analytical = AsianOption.geometric_price(
            S0, K, T, r, sigma, n_obs, opt, q,
        )
        discount = float(np.exp(-r * T))

        def control_fn(paths: np.ndarray) -> tuple[np.ndarray, float]:
            geo_avg = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
            if opt == "call":
                geo_payoffs = np.maximum(geo_avg - K, 0.0)
            else:
                geo_payoffs = np.maximum(K - geo_avg, 0.0)
            return discount * geo_payoffs, analytical

        return control_fn

    def __repr__(self) -> str:
        if self._strike_type == "fixed":
            return (
                f"AsianOption(K={self.K}, type='{self._option_type}', "
                f"avg='{self._avg_type}', strike='fixed')"
            )
        return (
            f"AsianOption(type='{self._option_type}', "
            f"avg='{self._avg_type}', strike='floating')"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AsianOption):
            return NotImplemented
        return (
            self.K == other.K
            and self._option_type == other._option_type
            and self._avg_type == other._avg_type
            and self._strike_type == other._strike_type
        )

    def __hash__(self) -> int:
        return hash(
            (self.K, self._option_type, self._avg_type, self._strike_type)
        )
