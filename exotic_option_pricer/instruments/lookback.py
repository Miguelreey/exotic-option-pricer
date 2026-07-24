"""
Exotic Option Pricer — exotic_option_pricer/instruments/lookback.py

Lookback options: path-dependent payoffs based on the running maximum
or minimum of the underlying. Four flavors are supported:

- Floating strike call  : payoff = S_T - min_{0<=t<=T} S_t
- Floating strike put   : payoff = max_{0<=t<=T} S_t - S_T
- Fixed strike call     : payoff = max(max_{0<=t<=T} S_t - K, 0)
- Fixed strike put      : payoff = max(K - min_{0<=t<=T} S_t, 0)

Floating-strike payoffs are **always non-negative** (the holder always
exercises): the terminal price is always at least the running min and
at most the running max. There is no max(., 0) wrapper.

Industry Usage
--------------
Lookback options remove timing risk — the holder always gets the best
price in the observation window. They are the most expensive of the
standard exotics and appear in:

- Structured products with "best-of" legs (rainbow lookbacks, cliquets).
- Hedging strategies that guarantee entry or exit at the extremum.
- Equity notes and autocallables with performance-based coupons.

Analytical Formulas — Conze-Viswanathan (1991)
----------------------------------------------
Under GBM with rate ``r``, dividend yield ``q`` and volatility ``sigma``,
the joint distribution of ``(S_T, max_t S_t)`` / ``(S_T, min_t S_t)``
admits a closed form via the reflection principle applied to Brownian
motion with drift.

Let ``b = r - q``, ``sqrtT = sigma * sqrt(T)`` and define

    a1 = [ln(S / S_min) + (b + sigma^2 / 2) * T] / sqrtT
    a2 = a1 - sqrtT

Floating-strike call — Goldman, Sosin & Gatto (1979):

    C_float = S * e^{-qT} * N(a1) - S_min * e^{-rT} * N(a2)
            + (S * sigma^2 / (2 * b)) * [
                  e^{-rT} * (S / S_min)^{-2b/sigma^2}
                            * N(-a1 + 2*b*sqrt(T)/sigma)
                - e^{-qT} * N(-a1)
              ]

Floating-strike put — symmetric reflection (same 1979 paper):

    P_float = -S * e^{-qT} * N(-b1) + S_max * e^{-rT} * N(-b2)
            + (S * sigma^2 / (2 * b)) * [
                - e^{-rT} * (S / S_max)^{-2b/sigma^2}
                            * N(b1 - 2*b*sqrt(T)/sigma)
                + e^{-qT} * N(b1)
              ]

with ``b1``, ``b2`` defined by replacing ``S_min`` with ``S_max``
in ``a1``, ``a2``.

**Singularity at r = q.** Both formulas contain ``sigma^2 / (2*b)``,
which diverges as ``b -> 0``. The bracket simultaneously vanishes, so
the overall expression admits a finite limit obtained by L'Hopital's
rule. For ``|r - q| < EPS`` we switch to the explicit limit form
derived term-by-term in :func:`_floating_call_zero_drift` and
:func:`_floating_put_zero_drift`. The threshold ``EPS = 1e-10`` is
tight enough to avoid numerical cancellation in the main formula.

Fixed-strike Lookback — Conze-Viswanathan (1991)
-------------------------------------------------
Payoffs: ``max(S_max - K, 0)`` (call) and ``max(K - S_min, 0)`` (put).

The fixed-strike formula is **structurally different** from the
floating-strike one — a common mistake is to assume that
``C_fix(K) = C_float`` evaluated at ``S_min := K``. That identity is
false: algebraic comparison of the two formulas yields

    C_float(S, S_min=K) - C_fix(S, M=K)
        = (S * sigma^2 / (2*b)) * [(S/K)^{-2b/sigma^2} - e^{bT}]

which vanishes only at the boundary. The correct fixed-strike call is
(Haug 2007, Table 4-9):

    C_fix = S * e^{-qT} * N(e1) - K * e^{-rT} * N(e2)
          + (S * sigma^2 / (2*b)) * e^{-rT} * [
                - (S / K)^{-2b/sigma^2} * N(e1 - 2*b*sqrt(T)/sigma)
                + e^{b*T} * N(e1)
            ]

with ``e1 = [ln(S/K) + (b + sigma^2/2) * T] / (sigma * sqrt(T))`` and
``e2 = e1 - sigma * sqrt(T)`` (replace ``K`` by ``S_max`` when already
in the money — see below). The put formula is symmetric.

When the current extremum already satisfies the strike, the fixed
lookback separates the deterministic intrinsic from the stochastic
part by evaluating the fixed-strike formula at ``M = S_max`` (call)
or ``M = S_min`` (put):

    C_fix(K) = (S_max - K) * e^{-rT} + C_fix_helper(S, M=S_max)
        if S_max >= K
    C_fix(K) = C_fix_helper(S, M=K)
        if S_max <  K

(and analogously for the put). Both branches route through
:func:`_fixed_call` / :func:`_fixed_put`, which expose an ``M``
parameter precisely to accommodate this.

Discrete-Monitoring Bias
------------------------
Discrete monitoring systematically underestimates the true range of the
path, biasing the running min upward and the running max downward. The
correction is the same Broadie-Glasserman-Kou constant used for
barriers:

    M_eff = M_disc * exp(+beta * sigma * sqrt(dt))
    m_eff = m_disc * exp(-beta * sigma * sqrt(dt))

with ``beta ~= 0.5826``. Applied to the discrete MC extremum **before**
computing the payoff, this removes the O(1/sqrt(n_steps)) bias and
recovers the continuous-monitoring analytical price. The adjustment is
exposed as :meth:`LookbackOption.continuity_correction` (the lookback
counterpart of ``BarrierOption.continuity_correction``); it operates on
the extremum itself because, unlike a barrier level, the corrected
quantity here is a per-path statistic rather than a contract parameter.

Design Decisions
----------------
- The :meth:`payoff` method includes ``S_0 = paths[:, 0]`` in the
  running extremum — the standard contract convention. At inception,
  with ``S_min = S_max = S_0``, the floating payoff at the first
  monitoring date is exactly 0.
- The class supports **seasoned contracts** via the ``S_min`` and
  ``S_max`` parameters in :meth:`analytical_price`. During the option's
  life, the running extremum is a state variable that moves
  monotonically towards the true extremum. Callers re-price at each
  time point by passing the current ``S_min`` / ``S_max``.
- For fixed-strike lookbacks, the ``K`` argument is required only for
  that flavor. The constructor validates this at construction time, so
  runtime errors cannot arise from misconfiguration.

References
----------
.. [1] Goldman, M. B., Sosin, H. B. & Gatto, M. A. (1979). "Path
   Dependent Options: Buy at the Low, Sell at the High."
   Journal of Finance 34(5), 1111-1127.
.. [2] Conze, A. & Viswanathan (1991). "Path Dependent Options: The
   Case of Lookback Options." Journal of Finance 46(5), 1893-1907.
.. [3] Haug, E. G. (2007). The Complete Guide to Option Pricing
   Formulas, 2nd ed., Chapter 4.
.. [4] Hull, J. (2018). Options, Futures & Other Derivatives, 10th ed.,
   Chapter 26.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from .base import ExoticOption

# Threshold below which the "r = q" limit form is used to avoid
# catastrophic cancellation in the (sigma^2) / (2 * (r - q)) coefficient.
_DRIFT_EPS: float = 1e-10

# Broadie-Glasserman-Kou (1997) discrete-monitoring constant
# beta = -zeta(1/2) / sqrt(2*pi). Same constant as barrier options.
_BGK_BETA: float = 0.5826


def _norm_pdf(x: float) -> float:
    """Standard normal density, used in the zero-drift limit formulas."""
    return float(np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi))


def _floating_call(
    S: float,
    S_min: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
) -> float:
    """
    Goldman-Sosin-Gatto (1979) floating-strike lookback call.

    Evaluated with ``S`` (current spot) and ``S_min``. For genuinely
    floating-strike contracts ``S_min <= S`` holds by construction, but
    this helper is also used by the Conze-Viswanathan fixed-strike
    decomposition where ``S_min`` is replaced by the strike ``K``
    (possibly ``> S`` in the OTM case). The algebraic formula is valid
    for any positive ``S_min``, so validation is left to the caller.

    Falls back to the zero-drift limit when ``|r - q| < _DRIFT_EPS``.
    """
    b = r - q
    sqrt_T = sigma * np.sqrt(T)
    a1 = (np.log(S / S_min) + (b + 0.5 * sigma * sigma) * T) / sqrt_T
    a2 = a1 - sqrt_T

    if abs(b) < _DRIFT_EPS:
        return _floating_call_zero_drift(S, S_min, T, r, sigma, a1, a2)

    two_b_over_sig2 = 2.0 * b / (sigma * sigma)
    shift = 2.0 * b * np.sqrt(T) / sigma
    bracket = np.exp(-r * T) * (S / S_min) ** (-two_b_over_sig2) * ndtr(-a1 + shift) - np.exp(
        -q * T
    ) * ndtr(-a1)
    value = (
        S * np.exp(-q * T) * ndtr(a1)
        - S_min * np.exp(-r * T) * ndtr(a2)
        + S * (sigma * sigma / (2.0 * b)) * bracket
    )
    return float(value)


def _floating_call_zero_drift(
    S: float,
    S_min: float,
    T: float,
    r: float,
    sigma: float,
    a1: float,
    a2: float,
) -> float:
    """
    L'Hopital limit of the floating-strike lookback call as ``r -> q``.

    Derivation (valid for any ``S_min <= S``)::

        lim_{b->0} (sigma^2 / (2b)) * [
            e^{-rT} * (S/S_min)^{-2b/sigma^2} * N(-a1(b) + 2b*sqrtT/sigma)
          - e^{-qT} * N(-a1(b))
        ]
        = e^{-rT} * [
              sigma * sqrt(T) * n(a1_0)
            - ln(S / S_min) * N(-a1_0)
            - (sigma^2 * T / 2) * N(-a1_0)
          ]

    so::

        C_float(r=q) = e^{-rT} * [
              S * N(a1_0) - S_min * N(a2_0)
            + S * sigma * sqrt(T) * n(a1_0)
            - S * ln(S / S_min) * N(-a1_0)
            - S * (sigma^2 * T / 2) * N(-a1_0)
          ]

    The two terms involving ``S_min`` collapse to ``S`` at inception
    (``S_min = S``) and the formula reduces to the simpler expression
    reported in Haug (2007, eq. 4-58).
    """
    disc = np.exp(-r * T)
    ln_ratio = np.log(S / S_min)
    limit_term = S * (
        sigma * np.sqrt(T) * _norm_pdf(a1)
        - ln_ratio * ndtr(-a1)
        - 0.5 * sigma * sigma * T * ndtr(-a1)
    )
    value = disc * (S * ndtr(a1) - S_min * ndtr(a2) + limit_term)
    return float(value)


def _floating_put(
    S: float,
    S_max: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
) -> float:
    """
    Goldman-Sosin-Gatto (1979) floating-strike lookback put.

    Mirror of :func:`_floating_call`, parameterised by the running
    maximum ``S_max``. Like its call counterpart, no ``S_max >= S``
    check is performed because the Conze-Viswanathan fixed-strike
    decomposition calls this helper with ``S_max`` replaced by the
    strike ``K`` (possibly ``< S`` in the OTM put case). User-facing
    validation lives in :meth:`LookbackOption.analytical_price`.
    """
    b = r - q
    sqrt_T = sigma * np.sqrt(T)
    b1 = (np.log(S / S_max) + (b + 0.5 * sigma * sigma) * T) / sqrt_T
    b2 = b1 - sqrt_T

    if abs(b) < _DRIFT_EPS:
        return _floating_put_zero_drift(S, S_max, T, r, sigma, b1, b2)

    two_b_over_sig2 = 2.0 * b / (sigma * sigma)
    shift = 2.0 * b * np.sqrt(T) / sigma
    bracket = -np.exp(-r * T) * (S / S_max) ** (-two_b_over_sig2) * ndtr(b1 - shift) + np.exp(
        -q * T
    ) * ndtr(b1)
    value = (
        -S * np.exp(-q * T) * ndtr(-b1)
        + S_max * np.exp(-r * T) * ndtr(-b2)
        + S * (sigma * sigma / (2.0 * b)) * bracket
    )
    return float(value)


def _floating_put_zero_drift(
    S: float,
    S_max: float,
    T: float,
    r: float,
    sigma: float,
    b1: float,
    b2: float,
) -> float:
    """
    L'Hopital limit of the floating-strike lookback put as ``r -> q``.

    Symmetric to :func:`_floating_call_zero_drift`::

        lim_{b->0} (sigma^2 / (2b)) * [
          - e^{-rT} * (S/S_max)^{-2b/sigma^2} * N(b1(b) - 2b*sqrtT/sigma)
          + e^{-qT} * N(b1(b))
        ]
        = e^{-rT} * [
              sigma * sqrt(T) * n(b1_0)
            + ln(S / S_max) * N(b1_0)
            + (sigma^2 * T / 2) * N(b1_0)
          ]

    Note that ``ln(S/S_max) <= 0`` because ``S_max >= S``, so the
    middle term subtracts from the limit. The sign pattern differs
    from the call because the put formula starts with a minus sign
    on the ``N(-b1)`` term.
    """
    disc = np.exp(-r * T)
    ln_ratio = np.log(S / S_max)  # <= 0
    limit_term = S * (
        sigma * np.sqrt(T) * _norm_pdf(b1)
        + ln_ratio * ndtr(b1)
        + 0.5 * sigma * sigma * T * ndtr(b1)
    )
    value = disc * (-S * ndtr(-b1) + S_max * ndtr(-b2) + limit_term)
    return float(value)


def _fixed_call(
    S: float,
    M: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
) -> float:
    """
    Conze-Viswanathan (1991) fixed-strike lookback call formula.

    This is structurally distinct from the floating-strike formula:
    the bracketed term inside the ``(sigma^2 / (2*b))`` coefficient
    has opposite signs on both halves. See Haug (2007, Table 4-9)
    for the canonical reference.

    Parameters
    ----------
    S : float
        Current spot.
    M : float
        Reference level used inside the formula. For the OTM branch
        (``K > S_max``) pass ``M = K``. For the ITM branch
        (``K <= S_max``) pass ``M = S_max`` — the caller adds the
        discounted intrinsic ``(S_max - K) * e^{-rT}`` on top.

    Notes
    -----
    The formula::

        C = S*e^{-qT}*N(e1) - M*e^{-rT}*N(e2)
          + S*e^{-rT}*(sigma^2 / (2*b))
              * [-(S/M)^{-2b/sigma^2} * N(e1 - 2*b*sqrtT/sigma)
                 + e^{bT} * N(e1)]

    with ``e1 = [ln(S/M) + (b + sigma^2/2) * T] / (sigma * sqrtT)``.
    """
    b = r - q
    sqrt_T = sigma * np.sqrt(T)
    e1 = (np.log(S / M) + (b + 0.5 * sigma * sigma) * T) / sqrt_T
    e2 = e1 - sqrt_T

    if abs(b) < _DRIFT_EPS:
        return _fixed_call_zero_drift(S, M, T, r, sigma, e1, e2)

    two_b_over_sig2 = 2.0 * b / (sigma * sigma)
    shift = 2.0 * b * np.sqrt(T) / sigma
    bracket = -((S / M) ** (-two_b_over_sig2)) * ndtr(e1 - shift) + np.exp(b * T) * ndtr(e1)
    value = (
        S * np.exp(-q * T) * ndtr(e1)
        - M * np.exp(-r * T) * ndtr(e2)
        + S * np.exp(-r * T) * (sigma * sigma / (2.0 * b)) * bracket
    )
    return float(value)


def _fixed_call_zero_drift(
    S: float,
    M: float,
    T: float,
    r: float,
    sigma: float,
    e1: float,
    e2: float,
) -> float:
    """
    L'Hopital limit of the fixed-strike lookback call as ``r -> q``.

    Derivation::

        F(b) = -(S/M)^{-2b/sigma^2} * N(e1(b) - 2b*sqrtT/sigma)
             + e^{bT} * N(e1(b))

    with ``F(0) = 0``. Differentiating term-by-term at ``b = 0`` and
    multiplying by ``sigma^2 / 2`` gives::

        lim_{b->0} (sigma^2 / (2*b)) * F(b)
            = ln(S/M) * N(e1_0)
            + sigma * sqrt(T) * n(e1_0)
            + (sigma^2 * T / 2) * N(e1_0)

    so that::

        C_fix(r=q) = e^{-rT} * [
              S * N(e1_0) - M * N(e2_0)
            + S * ln(S/M) * N(e1_0)
            + S * sigma * sqrt(T) * n(e1_0)
            + S * (sigma^2 * T / 2) * N(e1_0)
          ]
    """
    disc = np.exp(-r * T)
    ln_ratio = np.log(S / M)
    limit_term = S * (
        ln_ratio * ndtr(e1)
        + sigma * np.sqrt(T) * _norm_pdf(e1)
        + 0.5 * sigma * sigma * T * ndtr(e1)
    )
    value = disc * (S * ndtr(e1) - M * ndtr(e2) + limit_term)
    return float(value)


def _fixed_put(
    S: float,
    M: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
) -> float:
    """
    Conze-Viswanathan (1991) fixed-strike lookback put formula.

    Mirror of :func:`_fixed_call`. The reference level ``M`` is ``K``
    in the OTM branch (``K < S_min``) and ``S_min`` in the ITM branch
    (``K >= S_min``), with the intrinsic added separately.

    Canonical form::

        P = -S*e^{-qT}*N(-e1) + M*e^{-rT}*N(-e2)
          + S*e^{-rT}*(sigma^2 / (2*b))
              * [(S/M)^{-2b/sigma^2} * N(-e1 + 2*b*sqrtT/sigma)
                 - e^{bT} * N(-e1)]
    """
    b = r - q
    sqrt_T = sigma * np.sqrt(T)
    e1 = (np.log(S / M) + (b + 0.5 * sigma * sigma) * T) / sqrt_T
    e2 = e1 - sqrt_T

    if abs(b) < _DRIFT_EPS:
        return _fixed_put_zero_drift(S, M, T, r, sigma, e1, e2)

    two_b_over_sig2 = 2.0 * b / (sigma * sigma)
    shift = 2.0 * b * np.sqrt(T) / sigma
    bracket = ((S / M) ** (-two_b_over_sig2)) * ndtr(-e1 + shift) - np.exp(b * T) * ndtr(-e1)
    value = (
        -S * np.exp(-q * T) * ndtr(-e1)
        + M * np.exp(-r * T) * ndtr(-e2)
        + S * np.exp(-r * T) * (sigma * sigma / (2.0 * b)) * bracket
    )
    return float(value)


def _fixed_put_zero_drift(
    S: float,
    M: float,
    T: float,
    r: float,
    sigma: float,
    e1: float,
    e2: float,
) -> float:
    """
    L'Hopital limit of the fixed-strike lookback put as ``r -> q``.

    By analogous derivation to :func:`_fixed_call_zero_drift`, the
    limit of the bracket is::

        -ln(S/M) * N(-e1_0) + sigma * sqrt(T) * n(e1_0)
        - (sigma^2 * T / 2) * N(-e1_0)

    yielding::

        P_fix(r=q) = e^{-rT} * [
              -S * N(-e1_0) + M * N(-e2_0)
            - S * ln(S/M) * N(-e1_0)
            + S * sigma * sqrt(T) * n(e1_0)
            - S * (sigma^2 * T / 2) * N(-e1_0)
          ]
    """
    disc = np.exp(-r * T)
    ln_ratio = np.log(S / M)
    limit_term = S * (
        -ln_ratio * ndtr(-e1)
        + sigma * np.sqrt(T) * _norm_pdf(e1)
        - 0.5 * sigma * sigma * T * ndtr(-e1)
    )
    value = disc * (-S * ndtr(-e1) + M * ndtr(-e2) + limit_term)
    return float(value)


class LookbackOption(ExoticOption):
    """
    Lookback option: floating or fixed strike, call or put.

    Parameters
    ----------
    option_type : {'call', 'put', 'c', 'p'}, default 'call'
        Payoff direction.
    strike_type : {'floating', 'fixed'}, default 'floating'
        - ``'floating'``: the strike is set to the path extremum (min
          for the call, max for the put). The payoff is always
          non-negative.
        - ``'fixed'``: the strike is the user-specified ``K`` and the
          payoff is ``max(S_max - K, 0)`` / ``max(K - S_min, 0)``.
    K : float or None, default None
        Strike for the fixed-strike flavour. Required iff
        ``strike_type='fixed'``. Ignored for floating strike.

    Examples
    --------
    Floating-strike call (always non-negative):

    >>> from exotic_option_pricer.instruments.lookback import LookbackOption
    >>> from exotic_option_pricer.engines.monte_carlo import MonteCarloEngine
    >>> lb = LookbackOption(option_type='call', strike_type='floating')
    >>> mc = MonteCarloEngine(n_paths=200_000, seed=42)
    >>> paths = mc.simulate_gbm(100, 1.0, 0.05, 0.20, n_steps=252)
    >>> result = mc.price(lb.payoff, paths, 0.05, 1.0)

    Fixed-strike put:

    >>> lb = LookbackOption(option_type='put', strike_type='fixed', K=105)
    >>> result = mc.price(lb.payoff, paths, 0.05, 1.0)
    """

    def __init__(
        self,
        option_type: str = "call",
        strike_type: str = "floating",
        K: float | None = None,
    ) -> None:
        opt = option_type.strip().lower()
        if opt == "c":
            opt = "call"
        elif opt == "p":
            opt = "put"
        if opt not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

        st = strike_type.strip().lower()
        if st not in ("floating", "fixed"):
            raise ValueError(f"strike_type must be 'floating' or 'fixed', got '{strike_type}'")

        if st == "fixed":
            if K is None:
                raise ValueError("K is required when strike_type='fixed'")
            if K <= 0:
                raise ValueError(f"K must be > 0, got {K}")

        self._option_type = opt
        self._strike_type = st
        self.K = float(K) if K is not None else None

    @property
    def option_type(self) -> str:
        return self._option_type

    @property
    def strike_type(self) -> str:
        return self._strike_type

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute the lookback payoff from simulated paths.

        Parameters
        ----------
        paths : np.ndarray, shape (n_paths, n_steps + 1)
            Simulated price paths. The running extremum is computed over
            the full path including the initial spot ``paths[:, 0]``.

        Returns
        -------
        np.ndarray, shape (n_paths,)
            Undiscounted payoffs (non-negative).

        Notes
        -----
        Floating-strike payoffs are always ``>= 0`` because the
        terminal value is bounded by the path extremum. No
        ``max(., 0)`` wrapper is needed, and adding one would hide
        bugs that corrupt the running extremum.
        """
        S_T = paths[:, -1]

        if self._strike_type == "floating":
            if self._option_type == "call":
                return np.asarray(S_T - np.min(paths, axis=1))
            return np.asarray(np.max(paths, axis=1) - S_T)

        # fixed strike
        assert self.K is not None  # narrowed by constructor
        if self._option_type == "call":
            return np.asarray(np.maximum(np.max(paths, axis=1) - self.K, 0.0))
        return np.asarray(np.maximum(self.K - np.min(paths, axis=1), 0.0))

    @staticmethod
    def analytical_price(
        S: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        strike_type: str = "floating",
        K: float | None = None,
        q: float = 0.0,
        S_min: float | None = None,
        S_max: float | None = None,
    ) -> float:
        """
        Closed-form lookback price (continuous monitoring).

        Parameters
        ----------
        S : float
            Current spot.
        T : float
            Time remaining to expiry.
        r, sigma : float
            Risk-free rate and volatility.
        option_type : {'call', 'put', 'c', 'p'}
        strike_type : {'floating', 'fixed'}
        K : float or None
            Strike for fixed-strike. Ignored for floating.
        q : float, default 0.0
            Continuous dividend yield.
        S_min, S_max : float or None, default None
            Running extremum of the path observed so far. Defaults to
            ``S`` for a fresh contract. During the life of a floating-
            strike lookback, ``S_min`` is a state variable that drifts
            downward towards the true minimum (and ``S_max`` upward);
            re-pricing at a later time requires the current value.

        Returns
        -------
        float
            Fair value under continuous monitoring.

        Notes
        -----
        The fixed-strike price uses the dedicated Conze-Viswanathan
        (1991) formula in :func:`_fixed_call` / :func:`_fixed_put`,
        which is **structurally distinct** from the floating-strike
        formula (the bracketed correction term has different signs).
        When the current extremum already satisfies the strike
        (``S_max >= K`` for the call, ``S_min <= K`` for the put) the
        in-the-money intrinsic is split off::

            C_fix(K) = (S_max - K) * e^{-rT} + C_fix_helper(S, M=S_max)
                if S_max >= K,
            C_fix(K) = C_fix_helper(S, M=K)
                if S_max <  K,

        and analogously for the put. Splitting off the intrinsic keeps
        ``M`` bounded when the option is deep in the money and avoids
        the ``(S_max - K)`` term drowning the stochastic component at
        evaluation time. The ``r = q`` limit is handled inside the
        helpers via L'Hopital's rule.
        """
        if S <= 0:
            raise ValueError(f"S must be > 0, got {S}")
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
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

        st = strike_type.strip().lower()
        if st not in ("floating", "fixed"):
            raise ValueError(f"strike_type must be 'floating' or 'fixed', got '{strike_type}'")

        if S_min is None:
            S_min = S
        if S_max is None:
            S_max = S

        if S_min <= 0 or S_max <= 0:
            raise ValueError(f"S_min and S_max must be > 0, got S_min={S_min}, S_max={S_max}")

        if st == "floating":
            if opt == "call":
                if S_min > S:
                    raise ValueError(
                        f"S_min must be <= S (running min cannot exceed "
                        f"current spot), got S={S}, S_min={S_min}"
                    )
                return _floating_call(S, S_min, T, r, sigma, q)
            if S_max < S:
                raise ValueError(
                    f"S_max must be >= S (running max cannot be below "
                    f"current spot), got S={S}, S_max={S_max}"
                )
            return _floating_put(S, S_max, T, r, sigma, q)

        # Fixed strike
        if K is None:
            raise ValueError("K is required when strike_type='fixed'")
        if K <= 0:
            raise ValueError(f"K must be > 0, got {K}")

        if opt == "call":
            if S_max >= K:
                intrinsic = np.exp(-r * T) * (S_max - K)
                return float(intrinsic + _fixed_call(S, S_max, T, r, sigma, q))
            return _fixed_call(S, K, T, r, sigma, q)

        # Fixed strike put
        if S_min <= K:
            intrinsic = np.exp(-r * T) * (K - S_min)
            return float(intrinsic + _fixed_put(S, S_min, T, r, sigma, q))
        return _fixed_put(S, K, T, r, sigma, q)

    @staticmethod
    def continuity_correction(
        extremum: float | np.ndarray,
        sigma: float,
        dt: float,
        kind: str,
    ) -> float | np.ndarray:
        """
        Broadie-Glasserman-Kou (1997) adjusted extremum for discrete
        monitoring.

        Discrete monitoring underestimates the true range of the path:
        the observed running max is biased low and the running min biased
        high, by O(1/sqrt(n_steps)). The continuous-monitoring analytical
        price is recovered from discrete MC paths by shifting the observed
        extremum outward before computing the payoff:

            M_eff = M_disc * exp(+beta * sigma * sqrt(dt))    (kind='max')
            m_eff = m_disc * exp(-beta * sigma * sqrt(dt))    (kind='min')

        with ``beta ~= 0.5826``. This is the lookback counterpart of
        ``BarrierOption.continuity_correction``; it acts on the extremum
        itself (a per-path statistic) rather than on a contract level,
        hence the array support.

        Parameters
        ----------
        extremum : float or np.ndarray
            Discretely observed running extremum (e.g.
            ``np.max(paths, axis=1)``). Must be > 0 elementwise.
        sigma : float
            Annualized volatility. Must be > 0.
        dt : float
            Monitoring interval ``T / n_steps`` (in years). Must be > 0.
        kind : {'max', 'min'}
            Which extremum is being corrected: the running max is pushed
            up (+), the running min down (-).

        Returns
        -------
        float or np.ndarray
            Adjusted extremum, same shape as the input.

        Examples
        --------
        >>> LookbackOption.continuity_correction(120.0, 0.20, 1/252, 'max')
        120.883...
        >>> LookbackOption.continuity_correction(80.0, 0.20, 1/252, 'min')
        79.415...

        References
        ----------
        .. [1] Broadie, Glasserman & Kou (1997). "A Continuity Correction
           for Discrete Barrier Options." Math. Finance 7(4), 325-349.
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")

        k = kind.strip().lower()
        if k not in ("max", "min"):
            raise ValueError(f"kind must be 'max' or 'min', got '{kind}'")

        ext = np.asarray(extremum, dtype=np.float64)
        if np.any(ext <= 0):
            raise ValueError("extremum must be > 0 elementwise")

        sign = 1.0 if k == "max" else -1.0
        adjusted = ext * np.exp(sign * _BGK_BETA * sigma * np.sqrt(dt))
        return float(adjusted) if adjusted.ndim == 0 else adjusted

    def __repr__(self) -> str:
        extra = f", K={self.K}" if self._strike_type == "fixed" else ""
        return (
            f"LookbackOption(option_type='{self._option_type}', "
            f"strike_type='{self._strike_type}'{extra})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LookbackOption):
            return NotImplemented
        return (
            self._option_type == other._option_type
            and self._strike_type == other._strike_type
            and self.K == other.K
        )

    def __hash__(self) -> int:
        return hash((self._option_type, self._strike_type, self.K))
