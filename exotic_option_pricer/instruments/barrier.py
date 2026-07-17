"""
Exotic Option Pricer — exotic_option_pricer/instruments/barrier.py

Single-barrier options: knock-in / knock-out on an up or down barrier,
for both calls and puts, under Black-Scholes / GBM dynamics.

Barrier options are the second-most liquid path-dependent derivatives
after Asians. They trade actively in FX (knock-out calls/puts as a way
to cut premium), in equity structured products (bonus certificates,
turbo warrants, autocallable coupons), and across OTC desks as building
blocks for exotic payoffs.

Payoff Taxonomy
---------------
Let ``K`` be the strike, ``H`` the barrier level, ``S_t`` the price path,
``m = min_{0 <= t <= T} S_t`` the running minimum and
``M = max_{0 <= t <= T} S_t`` the running maximum.

    down-and-out call : max(S_T - K, 0) * 1{m  > H}
    down-and-in  call : max(S_T - K, 0) * 1{m <= H}
    up-and-out   call : max(S_T - K, 0) * 1{M  < H}
    up-and-in    call : max(S_T - K, 0) * 1{M >= H}
    down-and-out put  : max(K - S_T, 0) * 1{m  > H}
    down-and-in  put  : max(K - S_T, 0) * 1{m <= H}
    up-and-out   put  : max(K - S_T, 0) * 1{M  < H}
    up-and-in    put  : max(K - S_T, 0) * 1{M >= H}

The knock-in / knock-out events are complementary, giving the
fundamental **in-out parity**::

    V_in(H) + V_out(H) = V_vanilla

for the same direction (up/down), type (call/put) and barrier level.
This identity is exact and is the single most important consistency
check for any barrier implementation.

Reiner-Rubinstein (1991) Analytical Formula — continuous monitoring
-------------------------------------------------------------------
Under GBM with rate ``r``, dividend yield ``q`` and volatility ``sigma``,
the joint distribution of ``(S_T, max/min_t S_t)`` admits a closed form
via the reflection principle. Introducing

    phi    = +1 (call) or -1 (put)
    eta    = +1 (down barrier) or -1 (up barrier)
    lambda = (r - q + sigma^2 / 2) / sigma^2
    sqrtT  = sigma * sqrt(T)

    x1 = ln(S / K)        / sqrtT + lambda * sqrtT
    x2 = ln(S / H)        / sqrtT + lambda * sqrtT
    y1 = ln(H^2 / (S*K))  / sqrtT + lambda * sqrtT
    y2 = ln(H / S)        / sqrtT + lambda * sqrtT

the Reiner-Rubinstein components are

    A = phi*S*e^{-qT}*N(phi*x1) - phi*K*e^{-rT}*N(phi*(x1 - sqrtT))

    B = phi*S*e^{-qT}*N(phi*x2) - phi*K*e^{-rT}*N(phi*(x2 - sqrtT))

    C = phi*S*e^{-qT}*(H/S)^{2*lambda}    *N(eta*y1)
      - phi*K*e^{-rT}*(H/S)^{2*lambda - 2}*N(eta*(y1 - sqrtT))

    D = phi*S*e^{-qT}*(H/S)^{2*lambda}    *N(eta*y2)
      - phi*K*e^{-rT}*(H/S)^{2*lambda - 2}*N(eta*(y2 - sqrtT))

    E = rebate * e^{-rT} * ( N(eta*(x2 - sqrtT))
                            - (H/S)^{2*lambda - 2} * N(eta*(y2 - sqrtT)) )

    F = rebate * ( (H/S)^{a + b} * N(eta*z)
                 + (H/S)^{a - b} * N(eta*(z - 2*b*sqrtT)) )
      with a = lambda - 1, b = sqrt(a^2 + 2*r/sigma^2),
           z = ln(H/S)/sqrtT + b*sqrtT

``E`` is the present value of a rebate paid at expiry ``T`` to the holder
of a knock-in option that never knocks in. ``F`` is the present value of
a rebate paid at the knock-out time to the holder of a knock-out option
that knocks out. Both are zero when ``rebate = 0``.

The eight option values are obtained by selecting the correct
combination of A-F depending on (barrier_type, option_type) and the
relative position of ``H`` vs ``K``. See :meth:`analytical_price` for
the lookup table, which is a transcription of Haug (2007, Table 4-11).

Broadie-Glasserman-Yor (1997) Continuity Correction
---------------------------------------------------
Monte Carlo simulation monitors the barrier at a finite set of times
``t_0 < t_1 < ... < t_n = T`` rather than continuously. Between two
consecutive monitoring dates the path may cross ``H`` undetected,
which biases discretely-monitored prices systematically:

- knock-out MC *overestimates* the price (misses knock-outs)
- knock-in  MC *underestimates* the price (misses knock-ins)

The bias is ``O(1 / sqrt(n))`` in the number of time steps and does
*not* vanish as ``n_paths -> infinity``. Broadie, Glasserman and Kou
(1997) showed that the discretely-monitored price equals the
continuously-monitored price evaluated at a shifted barrier

    H_eff = H * exp(+/- beta * sigma * sqrt(dt))

with ``beta = -zeta(1/2)/sqrt(2*pi) ~= 0.5826``, the sign being **+**
for up barriers and **-** for down barriers. The intuition is that a
down barrier at ``H`` is effectively pushed lower (harder to hit) when
sampled only at discrete times. The correction brings the discretely-
monitored MC price back into O(1/n) agreement with the continuous
Reiner-Rubinstein formula.

Design Decisions
----------------
- ``payoff()`` does *not* attempt to model rebates. Modelling a rebate
  paid at the knock-out time requires the discount factor at the hit
  time, which depends on (T, r, dt) and would break the clean
  ``payoff(paths) -> payoffs`` contract. Use :meth:`analytical_price`
  with ``rebate > 0`` when a rebate is present, and restrict MC
  cross-validation to the ``rebate = 0`` case.
- The strict versus weak inequality at ``S_t = H`` is materially
  irrelevant under continuous distributions (probability zero), but
  we use the market-standard definition: a knock-out survives as long
  as ``min > H`` / ``max < H`` (strict), and a knock-in activates when
  ``min <= H`` / ``max >= H`` (weak). The two conventions are
  complementary by construction.
- The initial spot ``paths[:, 0] = S_0`` is included in the running
  extremum. If the barrier is already violated at ``t = 0``, a
  knock-in is immediately active and a knock-out is immediately dead.
  This is the contractually correct behaviour.

References
----------
.. [1] Reiner, E. & Rubinstein, M. (1991). "Breaking Down the Barriers."
   Risk 4(8).
.. [2] Broadie, M., Glasserman, P. & Kou, S. (1997). "A Continuity
   Correction for Discrete Barrier Options." Mathematical Finance 7(4),
   325-349.
.. [3] Haug, E. G. (2007). The Complete Guide to Option Pricing Formulas,
   2nd ed. McGraw-Hill, Chapter 4 (barrier options tables).
.. [4] Hull, J. (2018). Options, Futures & Other Derivatives, 10th ed.
   Chapter 26.
.. [5] Merton, R. C. (1973). "Theory of Rational Option Pricing."
   Bell Journal of Economics 4(1), 141-183. (Original down-and-out call.)
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from .base import ExoticOption

# Broadie-Glasserman-Yor universal constant: -zeta(1/2) / sqrt(2*pi)
_BGY_BETA: float = 0.5826

_VALID_BARRIER_TYPES: frozenset[str] = frozenset(
    {"up-and-out", "up-and-in", "down-and-out", "down-and-in"}
)


def _reiner_rubinstein_components(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    H: float,
    phi: float,
    eta: float,
    q: float,
    rebate: float,
) -> tuple[float, float, float, float, float, float]:
    """
    Evaluate the six Reiner-Rubinstein components A, B, C, D, E, F.

    Returns a tuple ``(A, B, C, D, E, F)`` of floats. The components
    already incorporate ``phi`` (call/put) and ``eta`` (down/up) in
    their definitions — callers select the correct linear combination
    according to the (barrier_type, option_type, H vs K) table.
    """
    sigma2 = sigma * sigma
    lam = (r - q + 0.5 * sigma2) / sigma2
    sqrt_T = sigma * np.sqrt(T)

    x1 = np.log(S / K) / sqrt_T + lam * sqrt_T
    x2 = np.log(S / H) / sqrt_T + lam * sqrt_T
    y1 = np.log(H * H / (S * K)) / sqrt_T + lam * sqrt_T
    y2 = np.log(H / S) / sqrt_T + lam * sqrt_T

    HS_2lam = (H / S) ** (2.0 * lam)
    HS_2lam_m2 = (H / S) ** (2.0 * lam - 2.0)

    A = (
        phi * S * np.exp(-q * T) * ndtr(phi * x1)
        - phi * K * np.exp(-r * T) * ndtr(phi * (x1 - sqrt_T))
    )
    B = (
        phi * S * np.exp(-q * T) * ndtr(phi * x2)
        - phi * K * np.exp(-r * T) * ndtr(phi * (x2 - sqrt_T))
    )
    C = (
        phi * S * np.exp(-q * T) * HS_2lam * ndtr(eta * y1)
        - phi * K * np.exp(-r * T) * HS_2lam_m2 * ndtr(eta * (y1 - sqrt_T))
    )
    D = (
        phi * S * np.exp(-q * T) * HS_2lam * ndtr(eta * y2)
        - phi * K * np.exp(-r * T) * HS_2lam_m2 * ndtr(eta * (y2 - sqrt_T))
    )

    if rebate != 0.0:
        E = rebate * np.exp(-r * T) * (
            ndtr(eta * (x2 - sqrt_T)) - HS_2lam_m2 * ndtr(eta * (y2 - sqrt_T))
        )
        a = lam - 1.0
        b = np.sqrt(a * a + 2.0 * r / sigma2)
        z = np.log(H / S) / sqrt_T + b * sqrt_T
        F = rebate * (
            (H / S) ** (a + b) * ndtr(eta * z)
            + (H / S) ** (a - b) * ndtr(eta * (z - 2.0 * b * sqrt_T))
        )
    else:
        E = 0.0
        F = 0.0

    return float(A), float(B), float(C), float(D), float(E), float(F)


class BarrierOption(ExoticOption):
    """
    Single-barrier option instrument (knock-in / knock-out).

    Parameters
    ----------
    K : float
        Strike. Must be > 0.
    barrier : float
        Barrier level ``H``. Must be > 0.
    barrier_type : {'up-and-out', 'up-and-in', 'down-and-out', 'down-and-in'}
        The direction of the barrier (up or down) and whether breaching
        it activates (in) or extinguishes (out) the option.
    option_type : {'call', 'put', 'c', 'p'}, default 'call'
        Payoff direction.
    rebate : float, default 0.0
        Cash paid when the contract terminates because of a knock event
        (for knock-ins: paid at expiry if no knock-in; for knock-outs:
        paid at the hit time if knocked out). Used by
        :meth:`analytical_price` only — :meth:`payoff` assumes
        ``rebate = 0`` and the Monte Carlo engine models only the
        vanilla component.

    Notes
    -----
    The constructor does not validate ``barrier`` versus the spot
    because spot is not a property of the instrument. Consistency is
    enforced in :meth:`analytical_price` where ``S`` is provided.

    Examples
    --------
    >>> from exotic_option_pricer.instruments.barrier import BarrierOption
    >>> from exotic_option_pricer.engines.monte_carlo import MonteCarloEngine
    >>> bar = BarrierOption(K=100, barrier=120, barrier_type='up-and-out')
    >>> mc = MonteCarloEngine(n_paths=200_000, seed=42)
    >>> paths = mc.simulate_gbm(100, 1.0, 0.05, 0.20, n_steps=252)
    >>> result = mc.price(bar.payoff, paths, 0.05, 1.0)
    """

    def __init__(
        self,
        K: float,
        barrier: float,
        barrier_type: str,
        option_type: str = "call",
        rebate: float = 0.0,
    ) -> None:
        if K <= 0:
            raise ValueError(f"K must be > 0, got {K}")
        if barrier <= 0:
            raise ValueError(f"barrier must be > 0, got {barrier}")

        bt = barrier_type.strip().lower()
        if bt not in _VALID_BARRIER_TYPES:
            raise ValueError(
                f"barrier_type must be one of {sorted(_VALID_BARRIER_TYPES)}, "
                f"got '{barrier_type}'"
            )

        opt = option_type.strip().lower()
        if opt == "c":
            opt = "call"
        elif opt == "p":
            opt = "put"
        if opt not in ("call", "put"):
            raise ValueError(
                f"option_type must be 'call' or 'put', got '{option_type}'"
            )

        if rebate < 0:
            raise ValueError(f"rebate must be >= 0, got {rebate}")

        self.K = float(K)
        self.barrier = float(barrier)
        self._barrier_type = bt
        self._option_type = opt
        self.rebate = float(rebate)

    @property
    def option_type(self) -> str:
        """'call' or 'put'."""
        return self._option_type

    @property
    def barrier_type(self) -> str:
        """One of ``'up-and-out'``, ``'up-and-in'``, ``'down-and-out'``, ``'down-and-in'``."""
        return self._barrier_type

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute undiscounted barrier payoff from discretely monitored paths.

        Parameters
        ----------
        paths : np.ndarray, shape (n_paths, n_steps + 1)
            Simulated price paths. The barrier is monitored at every
            column, including the initial spot ``paths[:, 0] = S_0``.

        Returns
        -------
        np.ndarray, shape (n_paths,)
            Undiscounted payoffs. The running extremum is taken over the
            full path including S_0, following the standard contract.

        Notes
        -----
        This implementation assumes ``rebate = 0``. If the instrument
        carries a non-zero rebate, the MC price will be **wrong by the
        rebate component** — use :meth:`analytical_price` instead.
        """
        is_down = self._barrier_type.startswith("down")
        is_in = self._barrier_type.endswith("in")

        if is_down:
            extremum = np.min(paths, axis=1)
            knocked = extremum <= self.barrier
        else:
            extremum = np.max(paths, axis=1)
            knocked = extremum >= self.barrier

        S_T = paths[:, -1]
        if self._option_type == "call":
            vanilla = np.maximum(S_T - self.K, 0.0)
        else:
            vanilla = np.maximum(self.K - S_T, 0.0)

        if is_in:
            indicator = knocked.astype(np.float64)
        else:
            indicator = (~knocked).astype(np.float64)

        return np.asarray(vanilla * indicator)

    @staticmethod
    def analytical_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        barrier: float,
        barrier_type: str,
        option_type: str = "call",
        q: float = 0.0,
        rebate: float = 0.0,
    ) -> float:
        """
        Reiner-Rubinstein (1991) closed-form price for a continuously
        monitored single-barrier option under GBM.

        Parameters
        ----------
        S : float
            Spot price. Must be > 0.
        K : float
            Strike. Must be > 0.
        T : float
            Time to expiry in years. Must be > 0.
        r : float
            Risk-free rate.
        sigma : float
            Annualized volatility. Must be > 0.
        barrier : float
            Barrier level ``H``. Must be > 0.
        barrier_type : {'up-and-out', 'up-and-in', 'down-and-out', 'down-and-in'}
        option_type : {'call', 'put', 'c', 'p'}, default 'call'
        q : float, default 0.0
            Continuous dividend yield.
        rebate : float, default 0.0
            Rebate cash amount. See the module docstring for the
            discount convention (knock-in rebate is paid at expiry,
            knock-out rebate at the hit time).

        Returns
        -------
        float
            Fair value under continuous monitoring.

        Notes
        -----
        If the barrier is already breached at ``t = 0`` (``S <= H`` for
        a down barrier or ``S >= H`` for an up barrier) the option is
        effectively a vanilla (for knock-ins) or a pure rebate (for
        knock-outs). These degenerate cases are handled explicitly.

        References
        ----------
        .. [1] Reiner & Rubinstein (1991). Risk 4(8).
        .. [2] Haug (2007), Chapter 4.
        """
        if S <= 0:
            raise ValueError(f"S must be > 0, got {S}")
        if K <= 0:
            raise ValueError(f"K must be > 0, got {K}")
        if T <= 0:
            raise ValueError(f"T must be > 0, got {T}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if barrier <= 0:
            raise ValueError(f"barrier must be > 0, got {barrier}")
        if rebate < 0:
            raise ValueError(f"rebate must be >= 0, got {rebate}")

        bt = barrier_type.strip().lower()
        if bt not in _VALID_BARRIER_TYPES:
            raise ValueError(
                f"barrier_type must be one of {sorted(_VALID_BARRIER_TYPES)}, "
                f"got '{barrier_type}'"
            )

        opt = option_type.strip().lower()
        if opt == "c":
            opt = "call"
        elif opt == "p":
            opt = "put"
        if opt not in ("call", "put"):
            raise ValueError(
                f"option_type must be 'call' or 'put', got '{option_type}'"
            )

        H = float(barrier)
        is_down = bt.startswith("down")
        is_in = bt.endswith("in")

        # Degenerate cases — barrier already breached at t=0
        already_breached = (is_down and S <= H) or (not is_down and S >= H)
        if already_breached:
            if is_in:
                # Knock-in is active: option is effectively vanilla
                # (rebate would never be paid because breach already occurred)
                return _vanilla_bs(S, K, T, r, sigma, opt, q)
            # Knock-out: option is already dead, only rebate remains.
            # Paid at t=0 (immediately), so no discount.
            return float(rebate)

        phi = 1.0 if opt == "call" else -1.0
        eta = 1.0 if is_down else -1.0

        A, B, C, D, E, F = _reiner_rubinstein_components(
            S, K, T, r, sigma, H, phi, eta, q, rebate,
        )

        # Lookup table (Haug 2007, Table 4-11): select component combination
        if is_in:
            if is_down and opt == "call":        # down-and-in call
                value = (C + E) if H <= K else (A - B + D + E)
            elif not is_down and opt == "call":  # up-and-in call
                value = (A + E) if H <= K else (B - C + D + E)
            elif is_down and opt == "put":       # down-and-in put
                value = (B - C + D + E) if H <= K else (A + E)
            else:                                # up-and-in put
                value = (A - B + D + E) if H <= K else (C + E)
        else:  # knock-out
            if is_down and opt == "call":        # down-and-out call
                value = (A - C + F) if H <= K else (B - D + F)
            elif not is_down and opt == "call":  # up-and-out call
                value = F if H <= K else (A - B + C - D + F)
            elif is_down and opt == "put":       # down-and-out put
                value = (A - B + C - D + F) if H <= K else F
            else:                                # up-and-out put
                value = (B - D + F) if H <= K else (A - C + F)

        return float(value)

    @staticmethod
    def continuity_correction(
        barrier: float,
        sigma: float,
        dt: float,
        direction: str,
    ) -> float:
        """
        Broadie-Glasserman-Yor (1997) adjusted barrier for discrete monitoring.

        The continuous-monitoring Reiner-Rubinstein price of a discretely
        monitored barrier option is obtained by replacing ``H`` with

            H_eff = H * exp( +/- beta * sigma * sqrt(dt) )

        where ``beta ~= 0.5826``, the sign is **+** for up barriers
        (push them higher, so harder to hit discretely → correction
        increases price of knock-outs) and **-** for down barriers.

        Parameters
        ----------
        barrier : float
            Nominal contractual barrier level ``H``. Must be > 0.
        sigma : float
            Annualized volatility.
        dt : float
            Monitoring interval ``T / n_steps`` (in years).
        direction : {'up', 'down'}
            Whether the barrier is monitored from below (up) or above
            (down).

        Returns
        -------
        float
            Adjusted barrier level ``H_eff``.

        Examples
        --------
        >>> BarrierOption.continuity_correction(120.0, 0.20, 1/252, 'up')
        120.883...
        >>> BarrierOption.continuity_correction(80.0, 0.20, 1/252, 'down')
        79.415...

        References
        ----------
        .. [1] Broadie, Glasserman & Kou (1997). Math. Finance 7(4).
        """
        if barrier <= 0:
            raise ValueError(f"barrier must be > 0, got {barrier}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")

        d = direction.strip().lower()
        if d not in ("up", "down"):
            raise ValueError(f"direction must be 'up' or 'down', got '{direction}'")

        sign = 1.0 if d == "up" else -1.0
        return float(barrier * np.exp(sign * _BGY_BETA * sigma * np.sqrt(dt)))

    def __repr__(self) -> str:
        extra = f", rebate={self.rebate}" if self.rebate else ""
        return (
            f"BarrierOption(K={self.K}, barrier={self.barrier}, "
            f"barrier_type='{self._barrier_type}', "
            f"option_type='{self._option_type}'{extra})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BarrierOption):
            return NotImplemented
        return (
            self.K == other.K
            and self.barrier == other.barrier
            and self._barrier_type == other._barrier_type
            and self._option_type == other._option_type
            and self.rebate == other.rebate
        )

    def __hash__(self) -> int:
        return hash(
            (self.K, self.barrier, self._barrier_type,
             self._option_type, self.rebate)
        )


def _vanilla_bs(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str, q: float,
) -> float:
    """
    Black-Scholes-Merton price for a vanilla European call or put.

    Implemented inline (rather than importing BlackScholesModel) to keep
    ``instruments/`` decoupled from ``models/``, matching the policy used
    by :class:`AsianOption` and :class:`DigitalOption`.
    """
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if option_type == "call":
        return float(
            S * np.exp(-q * T) * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2)
        )
    return float(
        K * np.exp(-r * T) * ndtr(-d2) - S * np.exp(-q * T) * ndtr(-d1)
    )
