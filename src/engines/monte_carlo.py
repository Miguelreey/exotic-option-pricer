"""
Exotic Option Pricer — src/engines/monte_carlo.py

Production-grade Monte Carlo pricing engine for European and exotic derivatives.

Implements:
- GBM simulation: exact (log-space), Euler-Maruyama, Milstein
- Quasi-Monte Carlo: scrambled Sobol sequences for O(1/N) convergence
- European option pricing with cross-validation against BS analytical
- Generic payoff pricing (extensible to any path-dependent derivative)
- Batch pricing: multiple payoffs on shared paths (desk-style efficiency)
- Variance reduction: antithetic variates, control variates, importance sampling
- Euler absorption at zero for non-GBM SDEs (Heston, rBergomi)
- Convergence analysis with statistical diagnostics
- Reproducible results via numpy.random.Generator (not legacy global state)

The engine is designed to be the workhorse for Phases 3-5:
- Phase 3: Exotic payoff functions plug into price() via payoff_fn
- Phase 4: Heston paths replace simulate_gbm() with simulate_heston()
- Phase 5: Rough Bergomi paths via fractional Brownian motion

Mathematical Background
-----------------------
Under the risk-neutral measure Q, the discounted price process is a
martingale. For GBM:

    dS = (r - q) S dt + sigma S dW^Q

The MC estimator:

    V_hat = e^{-rT} * (1/N) * sum_{i=1}^{N} payoff(S_T^{(i)})

converges to the true price V_0 = e^{-rT} E^Q[payoff(S_T)] by the
Strong Law of Large Numbers, with CLT-based confidence intervals:

    V_hat +/- z_{alpha/2} * sigma_hat / sqrt(N)

Three SDE discretization schemes are implemented:

1. **Exact (log-space)**: No discretization error. Uses the analytical
   solution of the GBM SDE via Ito's lemma on log(S):
       S_{t+dt} = S_t * exp((r - q - sigma^2/2)*dt + sigma*sqrt(dt)*Z)

2. **Euler-Maruyama**: First-order scheme for general SDEs:
       S_{t+dt} = S_t * (1 + (r-q)*dt + sigma*sqrt(dt)*Z)
   Weak convergence O(dt), strong convergence O(sqrt(dt)).

3. **Milstein**: Adds the Ito-Taylor correction term:
       S_{t+dt} = S_t * (1 + (r-q)*dt + sigma*sqrt(dt)*Z + 0.5*sigma^2*(Z^2-1)*dt)
   Strong convergence O(dt). For GBM specifically, Milstein is numerically
   equivalent to the exact scheme (the correction completes the Taylor expansion).

References
----------
.. [1] Glasserman (2003). Monte Carlo Methods in Financial Engineering.
.. [2] Kloeden & Platen (1992). Numerical Solution of SDEs.
.. [3] Hull (2018). Options, Futures & Other Derivatives, 10th ed. Ch. 21.
.. [4] Jaeckel (2002). Monte Carlo Methods in Finance. Wiley.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, overload

import numpy as np
from scipy.stats import norm as _norm_dist
from scipy.stats.qmc import Sobol as _Sobol

from .variance_reduction import (
    control_variate_adjust,
    importance_sampling_likelihood,
    importance_sampling_shift,
)

# z_{0.025} for 95% confidence intervals
_Z_95 = 1.959964


@dataclass(frozen=True, slots=True)
class MCResult:
    """
    Container for Monte Carlo pricing results.

    All fields are populated by MonteCarloEngine methods. The frozen dataclass
    ensures immutability — results are never silently mutated after creation.

    Attributes
    ----------
    price : float
        Estimated option price (mean of discounted payoffs).
    std_error : float
        Standard error of the price estimator = std / sqrt(N_effective).
        For antithetic variates, N_effective = N (number of pairs),
        not 2N (total paths), because pairs are averaged before computing SE.
    ci_lower : float
        Lower bound of 95% confidence interval.
    ci_upper : float
        Upper bound of 95% confidence interval.
    n_paths : int
        Number of independent paths (or pairs for antithetic).
    variance_reduction : str
        Method used: 'none', 'antithetic', 'control', 'antithetic+control'.

    Notes
    -----
    The confidence interval is based on the CLT:
        price +/- 1.96 * std_error

    This is valid for N >= ~30 by CLT. For small N, consider bootstrap CIs.
    The 95% CI should contain the true price in >= 90% of repeated experiments
    (accounting for finite-sample CLT approximation error).
    """

    price: float
    std_error: float
    ci_lower: float
    ci_upper: float
    n_paths: int
    variance_reduction: str

    def __repr__(self) -> str:
        return (
            f"MCResult(price={self.price:.6f}, std_error={self.std_error:.6f}, "
            f"CI=[{self.ci_lower:.6f}, {self.ci_upper:.6f}], "
            f"n_paths={self.n_paths:,}, vr='{self.variance_reduction}')"
        )


def _build_result(payoffs: np.ndarray, n_paths: int, vr_label: str) -> MCResult:
    """Build MCResult from an array of (already discounted) payoffs."""
    price = float(np.mean(payoffs))
    std_error = float(np.std(payoffs, ddof=1) / np.sqrt(len(payoffs)))
    return MCResult(
        price=price,
        std_error=std_error,
        ci_lower=price - _Z_95 * std_error,
        ci_upper=price + _Z_95 * std_error,
        n_paths=n_paths,
        variance_reduction=vr_label,
    )


class MonteCarloEngine:
    """
    Monte Carlo engine for derivative pricing under GBM dynamics.

    Parameters
    ----------
    n_paths : int, default 100_000
        Number of independent simulation paths. For antithetic variates,
        this is the number of pairs (total paths simulated = 2 * n_paths).
    n_steps : int, default 252
        Number of time steps per path. 252 = trading days per year.
        For European options with exact GBM, n_steps=1 suffices (terminal
        value only). For path-dependent exotics (Phase 3), the full path
        is needed.
    seed : int or None, default None
        Random seed for reproducibility. Uses numpy.random.Generator
        (not the deprecated numpy.random.seed global state).

    Examples
    --------
    >>> from src.engines.monte_carlo import MonteCarloEngine
    >>> mc = MonteCarloEngine(n_paths=500_000, seed=42)
    >>> result = mc.price_european(100, 100, 1.0, 0.05, 0.20,
    ...                            option_type='call', antithetic=True,
    ...                            control_variate=True)
    >>> abs(result.price - 10.4506) < 0.05
    True
    """

    def __init__(
        self,
        n_paths: int = 100_000,
        n_steps: int = 252,
        seed: int | None = None,
    ) -> None:
        if n_paths < 2:
            raise ValueError(
                f"n_paths must be >= 2, got {n_paths}. "
                f"At least 2 paths are needed to estimate standard error."
            )
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")

        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        """
        Reset the internal RNG to a known state.

        After calling simulate_gbm() or price_european(), the RNG state
        has advanced. This method reinitializes it, restoring exact
        reproducibility for subsequent operations.

        Parameters
        ----------
        seed : int or None, default None
            New seed. If None, reuses the seed from construction.
            If the engine was created without a seed (seed=None),
            calling reset(seed=None) creates a new non-reproducible RNG.

        Examples
        --------
        >>> mc = MonteCarloEngine(n_paths=1000, seed=42)
        >>> r1 = mc.price_european(100, 100, 1, 0.05, 0.2, 'call')
        >>> mc.reset()
        >>> r2 = mc.price_european(100, 100, 1, 0.05, 0.2, 'call')
        >>> r1.price == r2.price
        True
        """
        effective_seed = seed if seed is not None else self.seed
        self._rng = np.random.default_rng(effective_seed)
        if seed is not None:
            self.seed = seed

    # ──────────────────────────────────────────────
    # Path simulation
    # ──────────────────────────────────────────────

    def _build_paths_from_normals(
        self,
        S0: float,
        T: float,
        r: float,
        sigma: float,
        q: float,
        scheme: Literal["exact", "euler", "milstein"],
        Z: np.ndarray,
        absorb: bool = False,
    ) -> np.ndarray:
        """Build GBM paths from pre-generated standard normal increments.

        This is the single code path for all GBM simulation — both
        simulate_gbm() and price_european() delegate here, eliminating
        duplication between the two.

        Parameters
        ----------
        S0, T, r, sigma, q : float
            GBM parameters (see simulate_gbm).
        scheme : Literal['exact', 'euler', 'milstein']
            Discretization scheme.
        Z : np.ndarray, shape (n_paths, n_steps)
            Standard normal increments. **Modified in-place** for the exact
            scheme to minimize memory allocation.
        absorb : bool, default False
            If True, clamp S_t = max(S_t, 0) after each Euler step.
            Only applies to the Euler scheme — exact and Milstein produce
            strictly positive paths by construction (log-space).

        Returns
        -------
        paths : np.ndarray, shape (n_paths, n_steps + 1)
        """
        n_paths, n_steps = Z.shape
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = S0

        if scheme == "exact":
            # Exact log-space solution: zero discretization error
            # S_{n+1} = S_n * exp((r - q - sigma^2/2)*dt + sigma*sqrt(dt)*Z_n)
            drift = (r - q - 0.5 * sigma * sigma) * dt
            diffusion = sigma * sqrt_dt
            Z *= diffusion
            Z += drift
            np.cumsum(Z, axis=1, out=Z)
            np.exp(Z, out=Z)
            Z *= S0
            paths[:, 1:] = Z

        elif scheme == "euler":
            # Euler-Maruyama: S_{n+1} = S_n * (1 + (r-q)*dt + sigma*sqrt(dt)*Z_n)
            # Weak convergence: O(dt). Can produce S < 0 for large dt.
            #
            # Unlike exact and Milstein, Euler CANNOT be vectorized via
            # log-space cumsum because the multiplicative factor
            # (1 + mu*dt + sigma*sqrt(dt)*Z) can be negative, making log
            # undefined. The sequential loop is unavoidable here. For GBM
            # pricing, use the exact scheme instead; Euler exists for
            # validation of weak convergence theory and as a reference
            # for non-GBM SDEs (Heston, Phase 4) where no closed-form
            # step exists.
            drift_coeff = (r - q) * dt
            diff_coeff = sigma * sqrt_dt
            for step in range(n_steps):
                paths[:, step + 1] = paths[:, step] * (
                    1.0 + drift_coeff + diff_coeff * Z[:, step]
                )
                if absorb:
                    np.maximum(paths[:, step + 1], 0.0, out=paths[:, step + 1])

        elif scheme == "milstein":
            # Milstein for GBM: vectorized via log-space cumsum.
            #
            # The per-step multiplier is:
            #   M_n = 1 + (r-q)*dt + sigma*sqrt(dt)*Z + 0.5*sigma^2*dt*(Z^2 - 1)
            #
            # For GBM, b(S) = sigma*S and b'(S) = sigma, so the Milstein
            # correction is 0.5 * sigma^2 * S * (Z^2 - 1) * dt. Since M_n
            # depends only on Z_n (not on S_n beyond the multiplicative
            # structure), we have S_N = S_0 * prod(M_n), and:
            #   log(S_N/S_0) = sum(log(M_n))
            #
            # This is mathematically equivalent to the exact scheme for GBM
            # (the Milstein correction completes the Ito-Taylor expansion),
            # but we keep it as a separate code path for:
            # (a) validating strong convergence theory in tests
            # (b) serving as template for non-GBM Milstein (Phase 4+)
            #
            # Note: log(M_n) requires M_n > 0. For GBM, this holds when
            # sigma^2 * dt < 1, which is always true with reasonable
            # discretization (e.g., dt = 1/252 with sigma < 15.9).
            # Extreme parameters (sigma > 1/sqrt(dt)) can produce NaN —
            # use the exact scheme for such cases.
            drift_coeff = (r - q) * dt
            diff_coeff = sigma * sqrt_dt
            milstein_coeff = 0.5 * sigma * sigma * dt
            log_multipliers = np.log(
                1.0 + drift_coeff + diff_coeff * Z + milstein_coeff * (Z * Z - 1.0)
            )
            np.cumsum(log_multipliers, axis=1, out=log_multipliers)
            np.exp(log_multipliers, out=log_multipliers)
            log_multipliers *= S0
            paths[:, 1:] = log_multipliers

        return paths

    @overload
    def simulate_gbm(
        self,
        S0: float,
        T: float,
        r: float,
        sigma: float,
        q: float = ...,
        scheme: Literal["exact", "euler", "milstein"] = ...,
        *,
        n_steps: int | None = ...,
        antithetic: Literal[False] = ...,
        quasi: bool = ...,
        absorb: bool = ...,
    ) -> np.ndarray: ...

    @overload
    def simulate_gbm(
        self,
        S0: float,
        T: float,
        r: float,
        sigma: float,
        q: float = ...,
        scheme: Literal["exact", "euler", "milstein"] = ...,
        *,
        n_steps: int | None = ...,
        antithetic: Literal[True],
        quasi: bool = ...,
        absorb: bool = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def simulate_gbm(
        self,
        S0: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        scheme: Literal["exact", "euler", "milstein"] = "exact",
        *,
        n_steps: int | None = None,
        antithetic: bool = False,
        quasi: bool = False,
        absorb: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Simulate GBM paths under the risk-neutral measure Q.

        dS = (r - q) S dt + sigma S dW^Q

        Parameters
        ----------
        S0 : float
            Initial spot price. Must be > 0.
        T : float
            Time horizon in years. Must be > 0.
        r : float
            Risk-free rate (annualized, continuous compounding).
        sigma : float
            Annualized volatility (decimal). Must be > 0.
        q : float, default 0.0
            Continuous dividend yield.
        scheme : {'exact', 'euler', 'milstein'}, default 'exact'
            SDE discretization scheme:
            - 'exact': Log-space analytical solution (no discretization error)
            - 'euler': Euler-Maruyama (weak O(dt), strong O(sqrt(dt)))
            - 'milstein': Milstein (strong O(dt), equivalent to exact for GBM)
        n_steps : int or None, default None
            Number of time steps. If None, uses the engine's n_steps
            (set at construction). Pass n_steps=1 for European options
            where only the terminal value S_T is needed.
        antithetic : bool, default False
            If True, also generate paths from negated normals (-Z).
            Returns a tuple (paths, paths_anti) instead of a single array.
            The antithetic paths preserve negative correlation with the
            originals, enabling variance reduction when averaged per pair.
        quasi : bool, default False
            If True, use Quasi-Monte Carlo with scrambled Sobol sequences
            instead of pseudo-random normals. Sobol sequences fill the
            sample space more uniformly, achieving O(1/N) convergence
            (up to log factors) vs O(1/sqrt(N)) for standard MC.
            Most effective for low-dimensional problems (n_steps <= ~40).
            Uses Owen scrambling for unbiased estimates with valid CIs.
        absorb : bool, default False
            If True, clamp S_t = max(S_t, 0) after each Euler step to
            prevent negative prices. Only affects the 'euler' scheme —
            'exact' and 'milstein' produce strictly positive paths by
            construction (log-space). Essential for non-GBM SDEs (Heston,
            rBergomi) where Euler can produce negative values.

        Returns
        -------
        paths : np.ndarray, shape (n_paths, n_steps + 1)
            When antithetic=False. paths[:, 0] == S0 for all paths.
            paths[:, -1] is the terminal value S_T.
        (paths, paths_anti) : tuple of np.ndarray
            When antithetic=True. Both arrays have shape (n_paths, n_steps + 1).
            paths uses the original normals Z; paths_anti uses -Z.

        Raises
        ------
        ValueError
            If S0 <= 0, T <= 0, sigma <= 0, or scheme is unknown.

        Notes
        -----
        The exact scheme applies Ito's lemma to log(S):
            d(log S) = (r - q - sigma^2/2) dt + sigma dW
        which has solution:
            log S_t = log S_0 + (r - q - sigma^2/2)*t + sigma*W_t

        This means each step is:
            S_{t+dt} = S_t * exp((r - q - sigma^2/2)*dt + sigma*sqrt(dt)*Z)

        The -sigma^2/2 term (Ito correction) is critical. Without it,
        E[S_T] != S_0 * exp((r-q)*T), introducing systematic bias.
        This is the #1 implementation error in student MC engines.
        """
        if S0 <= 0:
            raise ValueError(f"S0 must be > 0, got {S0}")
        if T <= 0:
            raise ValueError(f"T must be > 0, got {T}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")

        if scheme not in ("exact", "euler", "milstein"):
            raise ValueError(f"scheme must be 'exact', 'euler', or 'milstein', got '{scheme}'")

        n_steps_actual = n_steps if n_steps is not None else self.n_steps
        if n_steps_actual < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps_actual}")

        if quasi:
            # Scrambled Sobol: low-discrepancy sequence with Owen scrambling
            # for unbiased estimates. Convergence O(1/N · (log N)^d) vs O(1/√N).
            # n_paths is rounded up to the next power of 2 for optimal
            # Sobol properties, then trimmed to the requested count.
            sampler = _Sobol(d=n_steps_actual, scramble=True, seed=self.seed)
            m = int(np.ceil(np.log2(max(self.n_paths, 2))))
            uniforms = sampler.random(2**m)[:self.n_paths]
            # Inverse CDF: uniform -> normal. Clip to avoid inf at boundaries.
            Z = _norm_dist.ppf(np.clip(uniforms, 1e-10, 1 - 1e-10))
        else:
            Z = self._rng.standard_normal((self.n_paths, n_steps_actual))

        if antithetic:
            Z_anti = -Z  # new array — unaffected by in-place ops on Z
            paths_pos = self._build_paths_from_normals(
                S0, T, r, sigma, q, scheme, Z, absorb=absorb,
            )
            paths_neg = self._build_paths_from_normals(
                S0, T, r, sigma, q, scheme, Z_anti, absorb=absorb,
            )
            return paths_pos, paths_neg

        return self._build_paths_from_normals(
            S0, T, r, sigma, q, scheme, Z, absorb=absorb,
        )

    # ──────────────────────────────────────────────
    # European option pricing
    # ──────────────────────────────────────────────

    def price_european(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        q: float = 0.0,
        antithetic: bool = False,
        control_variate: bool = False,
        importance_sampling: bool = False,
    ) -> MCResult:
        """
        Price a European vanilla option via Monte Carlo.

        Convenience method that generates single-step GBM paths (exact scheme)
        and delegates to price() with optional variance reduction.

        Parameters
        ----------
        S0 : float
            Spot price. Must be > 0.
        K : float
            Strike price. Must be >= 0.
        T : float
            Time to expiry in years. Must be > 0.
        r : float
            Risk-free rate (annualized, continuous compounding).
        sigma : float
            Annualized volatility (decimal). Must be > 0.
        option_type : str, default 'call'
            'call' or 'put'.
        q : float, default 0.0
            Continuous dividend yield.
        antithetic : bool, default False
            Use antithetic variates.
        control_variate : bool, default False
            Use control variates with discounted S_T as control.
        importance_sampling : bool, default False
            Use importance sampling with optimal drift shift. Centers
            the terminal distribution at the strike K, dramatically
            reducing variance for deep OTM options (where standard MC
            produces mostly zero payoffs). Incompatible with antithetic
            variates; compatible with control variates.

        Returns
        -------
        MCResult
            Pricing result with price, std_error, CI, and metadata.

        Raises
        ------
        ValueError
            If importance_sampling and antithetic are both True.

        Notes
        -----
        For European options, only the terminal value S_T matters, so we use
        a single-step exact simulation internally for maximum efficiency.

        The control variate is the discounted terminal stock price
        e^{-rT}*S_T, whose expectation under Q is S_0*e^{-qT}. This avoids
        dependency on BlackScholesModel while achieving |rho| > 0.85 for ATM
        options (variance reduction > 70%).
        """
        if S0 <= 0:
            raise ValueError(f"S0 must be > 0, got {S0}")
        if K < 0:
            raise ValueError(f"K must be >= 0, got {K}")
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

        if importance_sampling and antithetic:
            raise ValueError(
                "importance_sampling and antithetic cannot be used together. "
                "IS changes the sampling distribution, breaking antithetic symmetry."
            )

        # --- Importance sampling path (separate code path for clarity) ---
        if importance_sampling:
            theta = importance_sampling_shift(S0, K, T, r, sigma, q)
            Z = self._rng.standard_normal(self.n_paths)
            Z_shifted = Z + theta

            # Compute S_T with shifted normals
            mu_T = (r - q - 0.5 * sigma * sigma) * T
            S_T = S0 * np.exp(mu_T + sigma * np.sqrt(T) * Z_shifted)

            # Payoff (undiscounted)
            if opt == "call":
                raw_payoffs = np.maximum(S_T - K, 0.0)
            else:
                raw_payoffs = np.maximum(K - S_T, 0.0)

            # Likelihood ratio correction
            lr = importance_sampling_likelihood(Z, theta)
            payoffs = np.exp(-r * T) * raw_payoffs * lr

            # Optional: combine with control variate
            if control_variate:
                discount = np.exp(-r * T)
                cv_expected = S0 * np.exp(-q * T)
                control_vals = discount * S_T * lr
                payoffs, _ = control_variate_adjust(
                    payoffs, control_vals, cv_expected,
                )

            vr_label = "importance" + ("+control" if control_variate else "")
            return _build_result(payoffs, self.n_paths, vr_label)

        # --- Standard path ---
        if opt == "call":
            def payoff_fn(p: np.ndarray) -> np.ndarray:
                return np.maximum(p[:, -1] - K, 0.0)
        else:
            def payoff_fn(p: np.ndarray) -> np.ndarray:
                return np.maximum(K - p[:, -1], 0.0)

        # Single-step exact simulation — no discretization error for Europeans
        paths_anti: np.ndarray | None = None
        if antithetic:
            paths, paths_anti = self.simulate_gbm(
                S0, T, r, sigma, q=q, n_steps=1, antithetic=True,
            )
        else:
            paths = self.simulate_gbm(S0, T, r, sigma, q=q, n_steps=1)

        # Control variate: e^{-rT} S_T with E^Q[e^{-rT}S_T] = S_0 e^{-qT}
        if control_variate:
            discount = np.exp(-r * T)
            cv_expected = S0 * np.exp(-q * T)

            def cv_fn(p: np.ndarray) -> tuple[np.ndarray, float]:
                return discount * p[:, -1], cv_expected

            return self.price(
                payoff_fn, paths, r, T,
                paths_anti=paths_anti, control_fn=cv_fn,
            )

        return self.price(payoff_fn, paths, r, T, paths_anti=paths_anti)

    # ──────────────────────────────────────────────
    # Generic payoff pricing (for exotics, Phase 3+)
    # ──────────────────────────────────────────────

    def price(
        self,
        payoff_fn: Callable[[np.ndarray], np.ndarray],
        paths: np.ndarray,
        r: float,
        T: float,
        paths_anti: np.ndarray | None = None,
        control_fn: Callable[[np.ndarray], tuple[np.ndarray, float]] | None = None,
    ) -> MCResult:
        """
        Price a derivative with an arbitrary payoff function.

        This is the general-purpose pricing method. Any payoff that can be
        expressed as a function of the simulated paths can be priced here.
        Supports antithetic variates and control variates for variance
        reduction on path-dependent exotics.

        Parameters
        ----------
        payoff_fn : callable
            Function mapping paths to **undiscounted** payoff values.
            Signature: payoff_fn(paths: ndarray shape (n, m)) -> ndarray shape (n,)
            where n = number of paths, m = number of time points (n_steps + 1).
        paths : np.ndarray, shape (n_paths, n_steps + 1)
            Simulated price paths (from simulate_gbm or other SDE solver).
        r : float
            Risk-free rate for discounting.
        T : float
            Time to maturity (for discounting).
        paths_anti : np.ndarray or None, default None
            Antithetic paths (from simulate_gbm with antithetic=True).
            If provided, payoffs are averaged per pair before statistics,
            preserving the negative correlation for variance reduction.
        control_fn : callable or None, default None
            Control variate function.
            Signature: control_fn(paths: ndarray) -> (control_values: ndarray, E[C]: float)
            The known expectation E[C] is used to adjust payoffs via
            Y_cv = Y - beta*(C - E[C]) with optimal beta.

        Returns
        -------
        MCResult

        Examples
        --------
        >>> mc = MonteCarloEngine(n_paths=100_000, seed=42)
        >>> paths = mc.simulate_gbm(100, 1.0, 0.05, 0.20)
        >>> # European call payoff
        >>> result = mc.price(lambda p: np.maximum(p[:, -1] - 100, 0), paths, 0.05, 1.0)

        >>> # With antithetic variates (Phase 3+ exotics)
        >>> paths, paths_a = mc.simulate_gbm(100, 1.0, 0.05, 0.20, antithetic=True)
        >>> result = mc.price(lambda p: np.maximum(p[:, -1] - 100, 0), paths, 0.05, 1.0,
        ...                   paths_anti=paths_a)
        """
        raw_payoffs = payoff_fn(paths)
        if raw_payoffs.shape[0] != paths.shape[0]:
            raise ValueError(
                f"payoff_fn returned {raw_payoffs.shape[0]} values for "
                f"{paths.shape[0]} paths"
            )

        discount = np.exp(-r * T)

        # Determine VR label
        if paths_anti is not None and control_fn is not None:
            vr_label = "antithetic+control"
        elif paths_anti is not None:
            vr_label = "antithetic"
        elif control_fn is not None:
            vr_label = "control"
        else:
            vr_label = "none"

        if paths_anti is not None:
            raw_payoffs_anti = payoff_fn(paths_anti)
            if raw_payoffs_anti.shape[0] != paths_anti.shape[0]:
                raise ValueError(
                    f"payoff_fn returned {raw_payoffs_anti.shape[0]} values for "
                    f"{paths_anti.shape[0]} antithetic paths"
                )
            # Per-pair averaging preserves the negative correlation
            payoffs = discount * 0.5 * (raw_payoffs + raw_payoffs_anti)
        else:
            payoffs = discount * raw_payoffs

        if control_fn is not None:
            if paths_anti is not None:
                control_vals, control_exp = control_fn(paths)
                control_vals_anti, _ = control_fn(paths_anti)
                control_avg = 0.5 * (control_vals + control_vals_anti)
                payoffs, _ = control_variate_adjust(
                    payoffs, control_avg, control_exp,
                )
            else:
                control_vals, control_exp = control_fn(paths)
                payoffs, _ = control_variate_adjust(
                    payoffs, control_vals, control_exp,
                )

        return _build_result(payoffs, paths.shape[0], vr_label)

    # ──────────────────────────────────────────────
    # Batch pricing (reuse paths for multiple payoffs)
    # ──────────────────────────────────────────────

    def price_batch(
        self,
        payoff_fns: list[Callable[[np.ndarray], np.ndarray]],
        paths: np.ndarray,
        r: float,
        T: float,
        paths_anti: np.ndarray | None = None,
        control_fn: Callable[[np.ndarray], tuple[np.ndarray, float]] | None = None,
    ) -> list[MCResult]:
        """
        Price multiple derivatives on the same simulated paths.

        In a real desk, thousands of options (different strikes, maturities)
        share the same underlying — regenerating paths per option is wasteful.
        This method evaluates all payoffs on a single set of paths, amortizing
        the simulation cost.

        Parameters
        ----------
        payoff_fns : list of callable
            Each callable maps paths -> undiscounted payoffs (same signature
            as payoff_fn in price()).
        paths : np.ndarray, shape (n_paths, n_steps + 1)
            Simulated price paths (shared across all payoffs).
        r : float
            Risk-free rate for discounting.
        T : float
            Time to maturity.
        paths_anti : np.ndarray or None, default None
            Antithetic paths for variance reduction (shared).
        control_fn : callable or None, default None
            Control variate function (shared). Applied independently
            to each payoff's values.

        Returns
        -------
        list of MCResult
            One result per payoff function, in the same order as payoff_fns.

        Examples
        --------
        >>> mc = MonteCarloEngine(n_paths=100_000, seed=42)
        >>> paths = mc.simulate_gbm(100, 1.0, 0.05, 0.20, n_steps=1)
        >>> strikes = [90, 95, 100, 105, 110]
        >>> payoffs = [lambda p, K=K: np.maximum(p[:, -1] - K, 0) for K in strikes]
        >>> results = mc.price_batch(payoffs, paths, 0.05, 1.0)
        """
        return [
            self.price(
                pf, paths, r, T,
                paths_anti=paths_anti,
                control_fn=control_fn,
            )
            for pf in payoff_fns
        ]

    # ──────────────────────────────────────────────
    # Convergence analysis
    # ──────────────────────────────────────────────

    def convergence_analysis(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        q: float = 0.0,
        path_counts: list[int] | None = None,
        reference_price: float | None = None,
        antithetic: bool = False,
        control_variate: bool = False,
    ) -> dict:
        """
        Analyze MC convergence as a function of the number of paths.

        For each path count N, runs a fresh MC simulation and records
        the price, std_error, and absolute error vs reference.

        Parameters
        ----------
        S0, K, T, r, sigma : float
            Option parameters.
        option_type : str, default 'call'
        q : float, default 0.0
        path_counts : list of int, optional
            Path counts to test. Default: [1000, 5000, 10000, 50000, 100000, 500000].
        reference_price : float, optional
            Analytical or benchmark price for error computation.
            If None, uses the estimate from the largest path count.
        antithetic : bool, default False
            Use antithetic variates at each path count.
        control_variate : bool, default False
            Use control variates at each path count.

        Returns
        -------
        dict with keys:
            'path_counts': list[int]
            'prices': list[float]
            'std_errors': list[float]
            'errors': list[float]  (absolute error vs reference)
            'reference_price': float
            'variance_reduction': str
        """
        if path_counts is None:
            path_counts = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
        if len(path_counts) == 0:
            raise ValueError("path_counts must not be empty")

        prices: list[float] = []
        std_errors: list[float] = []

        # Same seed for all path counts: the first N paths of a larger run
        # are identical to an N-path run. This makes the convergence plot
        # smoother (prices are correlated across N values), which is the
        # standard approach in convergence analysis (Glasserman 2003, §2.3).
        base_seed = self.seed if self.seed is not None else 12345

        for n in path_counts:
            engine = MonteCarloEngine(n_paths=n, n_steps=1, seed=base_seed)
            result = engine.price_european(
                S0, K, T, r, sigma, option_type, q,
                antithetic=antithetic, control_variate=control_variate,
            )
            prices.append(result.price)
            std_errors.append(result.std_error)

        if reference_price is None:
            reference_price = prices[-1]

        errors = [abs(p - reference_price) for p in prices]

        vr_label = result.variance_reduction  # from last iteration

        return {
            "path_counts": path_counts,
            "prices": prices,
            "std_errors": std_errors,
            "errors": errors,
            "reference_price": reference_price,
            "variance_reduction": vr_label,
        }

    def __repr__(self) -> str:
        return (
            f"MonteCarloEngine(n_paths={self.n_paths:,}, "
            f"n_steps={self.n_steps}, seed={self.seed})"
        )
