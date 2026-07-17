"""
Exotic Option Pricer — exotic_option_pricer/utils/greeks.py

Bump-and-revalue numerical Greeks for exotic derivatives priced by
Monte Carlo. Uses Common Random Numbers (CRN) to suppress the variance
of the finite-difference estimator, together with an exact GBM path
rescaling for Delta and Gamma.

Motivation
----------
Exotic options (barriers, lookbacks, Asians, digitals) generally do
not admit closed-form Greeks under Black-Scholes. The only method that
is always applicable is **bump-and-revalue**: perturb an input, re-price,
and approximate the sensitivity by a central difference

    dV/dx  ~=  (V(x + h) - V(x - h)) / (2 h)     + O(h^2)

When ``V`` is a Monte Carlo estimate, the two bumped prices are noisy,
and the *variance* of the difference dominates unless the two estimators
are strongly positively correlated.

Common Random Numbers (CRN)
---------------------------
Let ``V_up`` and ``V_dn`` be MC estimates of ``V(x + h)`` and
``V(x - h)`` computed from the *same* underlying standard normal draws.
The variance of the finite-difference estimator is::

    Var[(V_up - V_dn) / (2 h)]
        = (Var[V_up] + Var[V_dn] - 2 * Cov[V_up, V_dn]) / (4 h^2)

With independent random numbers, ``Cov = 0`` and the variance explodes
as ``1 / h^2``. With CRN (identical draws), the two estimators are
highly correlated — for smooth payoffs and small ``h`` the correlation
typically exceeds ``0.99``, reducing the variance by two orders of
magnitude.

Operationally, CRN is implemented by calling ``engine.reset(seed)``
before each bumped simulation, which reinitialises the engine's internal
:class:`numpy.random.Generator` to the same state.

Exact GBM Path Rescaling (Delta, Gamma only)
--------------------------------------------
Under the exact GBM scheme, every simulated path satisfies

    S_t  =  S_0 * exp[ (r - q - sigma^2/2) * t + sigma * W_t ]

which is *linear* in ``S_0``. Bumping the spot therefore multiplies the
entire path by the constant ``(S_0 + h) / S_0`` without ever touching
the random numbers::

    S_t^{(+)}  =  ((S_0 + h) / S_0)  *  S_t

This is **stronger than CRN**: the bumped paths are not just
correlated, they are an analytic function of the base path. The
correlation of ``V(S_0 + h)`` and ``V(S_0 - h)`` is one (up to payoff
non-linearity). Only one simulation is needed per Delta/Gamma call.

For Vega, Theta and Rho the drift or diffusion coefficients of the SDE
change, so no analogous rescaling exists — we fall back to CRN via
``engine.reset(seed)``.

Bump Size
---------
The total error of the central-difference FD estimator decomposes into
a deterministic truncation term and a stochastic MC term::

    total_error  ~  C * h^2  +  sigma_MC / (h * sqrt(N))

Minimising over ``h`` gives ``h* = O(N^{-1/4})``. The defaults below
are pragmatic and cover most production settings; they can be
overridden by passing ``bump=...`` explicitly.

Defaults (per Glasserman 2003, Table 7.1):

- Delta / Gamma  :  ``h = 0.01 * S0``         (1% of spot)
- Vega           :  ``h = 0.01``              (1 vol point, absolute)
- Theta          :  ``h = 1 / 365``           (1 calendar day)
- Rho            :  ``h = 1e-4``              (1 basis point)

Sign Conventions
----------------
- Delta, Gamma, Vega, Rho are defined as derivatives of the option
  value with respect to the named parameter (positive or negative
  as usual).
- **Theta** is ``dV / dt = - dV / dT``: the derivative with respect to
  *calendar* time, not maturity. A long option held over time loses
  value, so theta is typically negative. The sign flip happens inside
  :func:`numerical_theta`.

API
---
- :func:`numerical_delta`, :func:`numerical_gamma`  — path rescaling
- :func:`numerical_vega`, :func:`numerical_theta`, :func:`numerical_rho`
  — CRN via seed reset
- :func:`numerical_greeks` — convenience wrapper returning a dict

All functions take an already-constructed :class:`MonteCarloEngine`
whose ``seed`` attribute must be non-``None`` (CRN requires a known
starting state).

References
----------
- Glasserman (2003), *Monte Carlo Methods in Financial Engineering*,
  chapters 7.1 (finite differences) and 7.2 (CRN).
- Broadie & Glasserman (1996), *Estimating security price derivatives
  using simulation*, Management Science 42(2), 269-285.
- Hull (2018), *Options, Futures, and Other Derivatives*, 10th ed.,
  chapter 20 (Greek letters).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Literal

import numpy as np

from exotic_option_pricer.engines.monte_carlo import MonteCarloEngine

__all__ = [
    "numerical_delta",
    "numerical_gamma",
    "numerical_vega",
    "numerical_theta",
    "numerical_rho",
    "numerical_greeks",
]


PayoffFn = Callable[[np.ndarray], np.ndarray]
GreekName = Literal["delta", "gamma", "vega", "theta", "rho"]

_DEFAULT_BUMPS: dict[str, Callable[[float], float]] = {
    "delta": lambda S0: 0.01 * S0,
    "gamma": lambda S0: 0.01 * S0,
    "vega": lambda _S0: 0.01,
    "theta": lambda _S0: 1.0 / 365.0,
    "rho": lambda _S0: 1.0e-4,
}

_VALID_GREEKS: frozenset[str] = frozenset(_DEFAULT_BUMPS.keys())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_seed(engine: MonteCarloEngine) -> int:
    """Return ``engine.seed`` or raise if it is ``None``.

    Numerical Greeks rely on Common Random Numbers, which requires the
    engine to be instantiated with a known seed. We refuse to silently
    mutate an unseeded engine by picking a default — forcing the user
    to opt in catches reproducibility bugs at construction time rather
    than after a subtle run-to-run drift.
    """
    if engine.seed is None:
        raise ValueError(
            "Numerical Greeks require a seeded MonteCarloEngine so that "
            "Common Random Numbers can be applied consistently across "
            "bumped simulations. Construct the engine as "
            "MonteCarloEngine(n_paths=..., seed=<int>)."
        )
    return engine.seed


def _validate_core(S0: float, T: float, sigma: float) -> None:
    """Validate the Greeks' shared numerical preconditions."""
    if S0 <= 0:
        raise ValueError(f"S0 must be > 0, got {S0}")
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")


def _simulate_crn(
    engine: MonteCarloEngine,
    seed: int,
    S0: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    n_steps: int,
) -> np.ndarray:
    """Reset the engine to ``seed`` and simulate fresh GBM paths.

    Each call produces the same underlying standard-normal draws, so
    consecutive calls with different ``(T, r, sigma, q)`` share random
    numbers — this is the Common Random Numbers variance reduction.
    """
    engine.reset(seed)
    return engine.simulate_gbm(S0, T, r, sigma, q, n_steps=n_steps)


def _mc_price(
    payoff_fn: PayoffFn,
    paths: np.ndarray,
    r: float,
    T: float,
) -> float:
    """Plain discounted-mean estimator (no variance reduction).

    Antithetic variates and control variates are *not* used here. The
    rescaling trick (Delta / Gamma) already gives a perfect correlation
    between the up and down samples — antithetic would add complexity
    without measurable gain. For Vega / Theta / Rho the CRN seed reset
    plays the same role. See the module docstring for the full
    argument.
    """
    discount = np.exp(-r * T)
    payoffs = payoff_fn(paths)
    return float(np.mean(discount * payoffs))


# ---------------------------------------------------------------------------
# Delta and Gamma: exact GBM path rescaling
# ---------------------------------------------------------------------------


def numerical_delta(
    engine: MonteCarloEngine,
    payoff_fn: PayoffFn,
    S0: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    bump: float | None = None,
    n_steps: int = 252,
) -> float:
    r"""Central-difference Delta via exact GBM path rescaling.

    Computes

    .. math::

        \Delta = \frac{\partial V}{\partial S_0}
               \approx \frac{V(S_0 + h) - V(S_0 - h)}{2 h}

    using a *single* simulation of the baseline paths. Under exact GBM,

    .. math::

        S_t^{(S_0 + h)} = \frac{S_0 + h}{S_0} \, S_t,

    so the bumped paths are obtained by scalar multiplication — no new
    random numbers are drawn. The finite-difference estimator is then
    exact in the correlation structure, leaving only the ``O(h^2)``
    truncation bias and the sampling variance of the bumped prices.

    Parameters
    ----------
    engine : MonteCarloEngine
        Must have a non-None seed.
    payoff_fn : callable
        Maps a paths array of shape ``(n_paths, n_steps + 1)`` to a
        one-dimensional array of *undiscounted* payoffs. Typically
        ``instrument.payoff`` for an :class:`ExoticOption` instance.
    S0, T, r, sigma : float
        GBM parameters. All must be strictly positive (``r`` may be
        zero or negative; the others must satisfy ``> 0``).
    q : float, default 0.0
        Continuous dividend yield.
    bump : float or None, default None
        Absolute spot bump ``h``. If ``None``, defaults to
        ``0.01 * S0``. Must satisfy ``0 < bump < S0``.
    n_steps : int, default 252
        Number of time steps in the simulated paths.

    Returns
    -------
    float
        The central-difference Delta estimate.

    Raises
    ------
    ValueError
        If the engine is unseeded, if any of ``S0``, ``T``, ``sigma``
        is non-positive, or if ``bump`` falls outside ``(0, S0)``.

    Notes
    -----
    Because the bumped paths are obtained by multiplying the base paths
    by a constant, there is no regeneration cost: Delta adds only two
    extra payoff evaluations over the baseline simulation. Compared to
    CRN via seed reset, this approach is faster *and* has strictly
    lower variance (the paths are perfectly correlated, not merely
    strongly correlated).

    Examples
    --------
    >>> from exotic_option_pricer.engines.monte_carlo import MonteCarloEngine
    >>> from exotic_option_pricer.instruments.asian import AsianOption
    >>> mc = MonteCarloEngine(n_paths=500_000, seed=42)
    >>> asian = AsianOption(K=100.0)
    >>> delta = numerical_delta(mc, asian.payoff, 100.0, 1.0, 0.05, 0.20)
    >>> 0.4 < delta < 0.7
    True
    """
    _validate_core(S0, T, sigma)
    seed = _require_seed(engine)

    h = bump if bump is not None else _DEFAULT_BUMPS["delta"](S0)
    if h <= 0:
        raise ValueError(f"bump must be > 0, got {h}")
    if h >= S0:
        raise ValueError(f"bump {h} must be < S0 {S0}")

    paths = _simulate_crn(engine, seed, S0, T, r, sigma, q, n_steps)
    factor_up = (S0 + h) / S0
    factor_dn = (S0 - h) / S0
    paths_up = paths * factor_up
    paths_dn = paths * factor_dn

    v_up = _mc_price(payoff_fn, paths_up, r, T)
    v_dn = _mc_price(payoff_fn, paths_dn, r, T)
    return (v_up - v_dn) / (2.0 * h)


def numerical_gamma(
    engine: MonteCarloEngine,
    payoff_fn: PayoffFn,
    S0: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    bump: float | None = None,
    n_steps: int = 252,
) -> float:
    r"""Second-derivative Gamma via 3-point stencil and GBM rescaling.

    .. math::

        \Gamma = \frac{\partial^2 V}{\partial S_0^2}
               \approx \frac{V(S_0 + h) - 2 V(S_0) + V(S_0 - h)}{h^2}

    Uses a single baseline simulation: the mid, up and down prices are
    all evaluated on rescaled versions of the same paths (see
    :func:`numerical_delta` for the rescaling identity).

    Parameters
    ----------
    engine, payoff_fn, S0, T, r, sigma, q, bump, n_steps
        Identical semantics to :func:`numerical_delta`. The default
        ``bump`` is the same ``0.01 * S0`` — using a common bump for
        Delta and Gamma is standard, even though the noise-optimal
        ``h`` for a second-derivative estimator is larger (the
        ``1/h^2`` amplification of MC error hurts Gamma more).

    Returns
    -------
    float
        The central-difference Gamma estimate.

    Notes
    -----
    Gamma is noisier than Delta by a factor of ``1 / h``. If you need
    high precision, pass a larger ``bump`` (e.g. ``0.05 * S0``) and/or
    increase ``engine.n_paths``. For a smoother estimator on vanilla or
    near-vanilla exotics, a likelihood-ratio / Malliavin approach would
    beat bump-and-revalue, but it is payoff-specific and out of scope.
    """
    _validate_core(S0, T, sigma)
    seed = _require_seed(engine)

    h = bump if bump is not None else _DEFAULT_BUMPS["gamma"](S0)
    if h <= 0:
        raise ValueError(f"bump must be > 0, got {h}")
    if h >= S0:
        raise ValueError(f"bump {h} must be < S0 {S0}")

    paths = _simulate_crn(engine, seed, S0, T, r, sigma, q, n_steps)
    paths_up = paths * ((S0 + h) / S0)
    paths_dn = paths * ((S0 - h) / S0)

    v_up = _mc_price(payoff_fn, paths_up, r, T)
    v_mid = _mc_price(payoff_fn, paths, r, T)
    v_dn = _mc_price(payoff_fn, paths_dn, r, T)
    return (v_up - 2.0 * v_mid + v_dn) / (h * h)


# ---------------------------------------------------------------------------
# Vega, Theta, Rho: CRN via seed reset
# ---------------------------------------------------------------------------


def numerical_vega(
    engine: MonteCarloEngine,
    payoff_fn: PayoffFn,
    S0: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    bump: float | None = None,
    n_steps: int = 252,
) -> float:
    r"""Central-difference Vega with Common Random Numbers.

    .. math::

        \mathcal{V} = \frac{\partial V}{\partial \sigma}
                    \approx \frac{V(\sigma + h) - V(\sigma - h)}{2 h}

    Unlike Delta, changing ``sigma`` alters both the drift correction
    (``- sigma^2 / 2``) and the diffusion coefficient of the GBM SDE,
    so no analogous rescaling exists — the bumped paths must be
    regenerated from the same underlying normals. :meth:`MonteCarloEngine.reset`
    restores the RNG state, delivering CRN.

    Parameters
    ----------
    engine, payoff_fn, S0, T, r, sigma, q, n_steps
        As in :func:`numerical_delta`.
    bump : float or None, default None
        Absolute volatility bump. Defaults to ``0.01`` (1 vol point).
        Must satisfy ``0 < bump < sigma``.

    Returns
    -------
    float
        The central-difference Vega estimate.
    """
    _validate_core(S0, T, sigma)
    seed = _require_seed(engine)

    h = bump if bump is not None else _DEFAULT_BUMPS["vega"](S0)
    if h <= 0:
        raise ValueError(f"bump must be > 0, got {h}")
    if sigma - h <= 0:
        raise ValueError(
            f"bump {h} must be < sigma {sigma} (bumped sigma must stay positive)"
        )

    paths_up = _simulate_crn(engine, seed, S0, T, r, sigma + h, q, n_steps)
    paths_dn = _simulate_crn(engine, seed, S0, T, r, sigma - h, q, n_steps)

    v_up = _mc_price(payoff_fn, paths_up, r, T)
    v_dn = _mc_price(payoff_fn, paths_dn, r, T)
    return (v_up - v_dn) / (2.0 * h)


def numerical_theta(
    engine: MonteCarloEngine,
    payoff_fn: PayoffFn,
    S0: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    bump: float | None = None,
    n_steps: int = 252,
) -> float:
    r"""Central-difference Theta with Common Random Numbers.

    Theta is the derivative with respect to *calendar* time, so

    .. math::

        \Theta = \frac{\partial V}{\partial t}
               = -\frac{\partial V}{\partial T}
               \approx -\frac{V(T + h) - V(T - h)}{2 h}.

    Note the sign flip: longer maturity usually *increases* option
    value, so ``dV/dT > 0`` and ``Theta < 0`` for long positions.

    The engine's ``n_steps`` is held fixed across the two simulations,
    so bumping ``T`` changes ``dt = T / n_steps`` slightly. This is the
    standard convention; keeping ``n_steps`` constant avoids changing
    the discretisation granularity between the two legs of the CRN.

    Parameters
    ----------
    engine, payoff_fn, S0, T, r, sigma, q, n_steps
        As in :func:`numerical_delta`.
    bump : float or None, default None
        Maturity bump in years. Defaults to ``1 / 365`` (one calendar
        day). Must satisfy ``0 < bump < T``.

    Returns
    -------
    float
        The central-difference Theta estimate, with the calendar-time
        sign applied.
    """
    _validate_core(S0, T, sigma)
    seed = _require_seed(engine)

    h = bump if bump is not None else _DEFAULT_BUMPS["theta"](S0)
    if h <= 0:
        raise ValueError(f"bump must be > 0, got {h}")
    if T - h <= 0:
        raise ValueError(
            f"bump {h} must be < T {T} (bumped maturity must stay positive)"
        )

    paths_up = _simulate_crn(engine, seed, S0, T + h, r, sigma, q, n_steps)
    paths_dn = _simulate_crn(engine, seed, S0, T - h, r, sigma, q, n_steps)

    v_up = _mc_price(payoff_fn, paths_up, r, T + h)
    v_dn = _mc_price(payoff_fn, paths_dn, r, T - h)

    dv_dT = (v_up - v_dn) / (2.0 * h)
    return -dv_dT


def numerical_rho(
    engine: MonteCarloEngine,
    payoff_fn: PayoffFn,
    S0: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    bump: float | None = None,
    n_steps: int = 252,
) -> float:
    r"""Central-difference Rho with Common Random Numbers.

    .. math::

        \rho = \frac{\partial V}{\partial r}
             \approx \frac{V(r + h) - V(r - h)}{2 h}

    The rate enters both the drift of the SDE and the discount factor
    ``exp(-r T)``. Both effects are captured by regenerating the paths
    with the bumped rate and using the matching discount during
    payoff averaging.

    Parameters
    ----------
    engine, payoff_fn, S0, T, r, sigma, q, n_steps
        As in :func:`numerical_delta`.
    bump : float or None, default None
        Absolute rate bump. Defaults to ``1e-4`` (1 basis point).

    Returns
    -------
    float
        The central-difference Rho estimate.
    """
    _validate_core(S0, T, sigma)
    seed = _require_seed(engine)

    h = bump if bump is not None else _DEFAULT_BUMPS["rho"](S0)
    if h <= 0:
        raise ValueError(f"bump must be > 0, got {h}")

    paths_up = _simulate_crn(engine, seed, S0, T, r + h, sigma, q, n_steps)
    paths_dn = _simulate_crn(engine, seed, S0, T, r - h, sigma, q, n_steps)

    v_up = _mc_price(payoff_fn, paths_up, r + h, T)
    v_dn = _mc_price(payoff_fn, paths_dn, r - h, T)
    return (v_up - v_dn) / (2.0 * h)


# ---------------------------------------------------------------------------
# Bundled API
# ---------------------------------------------------------------------------


def numerical_greeks(
    engine: MonteCarloEngine,
    payoff_fn: PayoffFn,
    S0: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    n_steps: int = 252,
    greeks: Iterable[GreekName] | None = None,
    bumps: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute a collection of Greeks in one call.

    Parameters
    ----------
    engine, payoff_fn, S0, T, r, sigma, q, n_steps
        As in :func:`numerical_delta`.
    greeks : iterable of {'delta', 'gamma', 'vega', 'theta', 'rho'} or None
        Which Greeks to compute. ``None`` is a shortcut for *all five*.
    bumps : dict or None
        Optional per-Greek overrides, e.g. ``{'vega': 0.02}``. Any
        Greek not present falls back to the default in
        ``_DEFAULT_BUMPS``.

    Returns
    -------
    dict
        A mapping from Greek name to its numerical estimate, iterating
        the requested Greeks in the canonical order ``delta, gamma,
        vega, theta, rho``.

    Raises
    ------
    ValueError
        If ``greeks`` contains an unknown name, or if ``bumps`` contains
        an unknown key.
    """
    canonical: tuple[GreekName, ...] = ("delta", "gamma", "vega", "theta", "rho")
    requested: list[GreekName]
    if greeks is None:
        requested = list(canonical)
    else:
        requested_set: set[str] = {str(g) for g in greeks}
        invalid = requested_set - _VALID_GREEKS
        if invalid:
            raise ValueError(
                f"Unknown greeks: {sorted(invalid)}. "
                f"Valid names: {sorted(_VALID_GREEKS)}."
            )
        requested = [g for g in canonical if g in requested_set]

    bumps = bumps or {}
    invalid_bumps = set(bumps.keys()) - _VALID_GREEKS
    if invalid_bumps:
        raise ValueError(
            f"Unknown bump keys: {sorted(invalid_bumps)}. "
            f"Valid names: {sorted(_VALID_GREEKS)}."
        )

    dispatch: dict[str, Callable[[float | None], float]] = {
        "delta": lambda h: numerical_delta(
            engine, payoff_fn, S0, T, r, sigma, q, bump=h, n_steps=n_steps
        ),
        "gamma": lambda h: numerical_gamma(
            engine, payoff_fn, S0, T, r, sigma, q, bump=h, n_steps=n_steps
        ),
        "vega": lambda h: numerical_vega(
            engine, payoff_fn, S0, T, r, sigma, q, bump=h, n_steps=n_steps
        ),
        "theta": lambda h: numerical_theta(
            engine, payoff_fn, S0, T, r, sigma, q, bump=h, n_steps=n_steps
        ),
        "rho": lambda h: numerical_rho(
            engine, payoff_fn, S0, T, r, sigma, q, bump=h, n_steps=n_steps
        ),
    }

    return {name: dispatch[name](bumps.get(name)) for name in requested}
