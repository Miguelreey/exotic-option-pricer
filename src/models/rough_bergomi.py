"""
Exotic Option Pricer — src/models/rough_bergomi.py

Rough Bergomi (Bayer-Friz-Gatheral 2016) rough volatility model.

Implements:
- Exact joint simulation of the Riemann-Liouville Volterra process and its
  Brownian driver via Cholesky factorization of the full covariance matrix
  (reference method, O((2n)^3) — small meshes)
- The hybrid scheme of Bennedsen-Lunde-Pakkanen (2017) with kappa = 1:
  exact treatment of the singular kernel on the most recent interval plus an
  FFT-accelerated discrete convolution for the history (production method,
  O(paths * n log n))
- European pricing via the conditional Black-Scholes ("turbocharged")
  estimator of McCrickerd-Pakkanen (2018): the orthogonal Brownian dimension
  is integrated out analytically and the B-measurable residual is tamed with
  an exact-mean control variate on S_cond — measured ~0.4x the plain-MC
  standard error at the canonical equity set, with an exact (zero-variance)
  Black-Scholes limit as eta -> 0 at rho = 0
- ABC Greeks: delta/gamma by exact path rescaling (rBergomi coefficients do
  not depend on the spot level), vega = dV/d(sqrt(xi0)) via the exact scaling
  v proportional to xi0, theta by re-simulated meshes under common random
  numbers, rho on shared draws
- model_greeks(): dV/d{xi0, eta, H, rho} with exact common random numbers
  (the Gaussian draws do not depend on the parameters)
- iv_surface(): implied volatility surface via the Phase 1 Halley solver

Mathematical Background
-----------------------
Under the risk-neutral measure Q:

    dS_t / S_t = (r - q) dt + sqrt(v_t) (rho dB_t + sqrt(1-rho^2) dB_t^perp)

    v_t = xi0 * exp(eta * V_t - eta^2/2 * t^(2H)),
    V_t = sqrt(2H) * int_0^t (t-s)^(H-1/2) dB_s

with B and B^perp independent standard Brownian motions. V is the
**Riemann-Liouville** (Volterra) fractional process — memory only on [0, t],
NOT the Mandelbrot-Van Ness fBM of the standard ``fbm`` libraries (same H,
different covariance). Crucially, the Volterra driver B is the SAME Brownian
motion that carries the spot-vol correlation: generating the volatility from
an independent Brownian motion and "correlating afterwards" destroys the
skew structure.

Exact moments (test anchors), with gamma = H - 1/2:

    E[V_t^2]   = 2H * int_0^t (t-s)^(2*gamma) ds = t^(2H)
    E[v_t]     = xi0                       (lognormal mean correction, exact)
    E[v_t^2]   = xi0^2 * exp(eta^2 t^(2H))
    E[V_t B_s] = sqrt(2H)/(H+1/2) * (t^(H+1/2) - (t - min(s,t))^(H+1/2))
    E[V_s V_t] = 2H * int_0^s (s-u)^gamma (t-u)^gamma du   (s <= t;
                 hypergeometric — computed by quadrature)

Why rough volatility: realized volatility is empirically rough (Gatheral-
Jaisson-Rosenbaum 2018, H ~ 0.1) and the short-dated ATM implied-vol skew
follows the power law psi(T) ~ T^(H - 1/2) (Fukasawa 2011), which explodes
as T -> 0. Markovian stochastic-volatility models (Heston) produce a skew
that flattens to a constant at short maturities: the Phase 4 SPX calibration
needed kappa = 13, xi = 3 chasing the short skew and still missed the short
wings. rBergomi reproduces the power law with 4 parameters.

rBergomi is NOT Markovian (v_t depends on the whole history of B through
the power-law kernel): there is no finite-dimensional PDE and no
characteristic function — pricing is Monte Carlo. There is also no external
benchmark (no QuantLib support): validation is by two mathematically
independent implementations that must agree (Cholesky vs hybrid), exact
analytical anchors, and the scaling laws above — the same standard used in
the literature.

References
----------
.. [1] Bayer, Friz, Gatheral (2016). Pricing under Rough Volatility.
       Quantitative Finance 16(6), 887-904.
.. [2] Bennedsen, Lunde, Pakkanen (2017). Hybrid Scheme for Brownian
       Semistationary Processes. Finance & Stochastics 21(4), 931-965.
.. [3] McCrickerd, Pakkanen (2018). Turbocharging Monte Carlo Pricing for
       the Rough Bergomi Model. Quantitative Finance 18(11), 1877-1886.
.. [4] Gatheral, Jaisson, Rosenbaum (2018). Volatility is Rough.
       Quantitative Finance 18(6), 933-949.
.. [5] Fukasawa (2011). Asymptotic Analysis for Stochastic Volatility:
       Martingale Expansion. Finance & Stochastics 15, 635-654.
"""

from typing import Callable, Dict, Literal, Union, overload

import numpy as np
from scipy.integrate import quad
from scipy.signal import fftconvolve
from scipy.special import ndtr

from ..engines.monte_carlo import MCResult, MonteCarloEngine
from ..engines.variance_reduction import control_variate_adjust
from .base import Numeric, PricingModel
from .black_scholes import BlackScholesModel

# (A, int_var): A = sum sqrt(v_i) dB_i (stochastic integral WITHOUT the rho
# factor) and int_var = sum v_i * dt (integrated variance, "I" in the
# derivations) — the per-path sufficient statistics of the conditional
# Black-Scholes estimator.
_CondDraws = tuple[np.ndarray, np.ndarray]


class RoughBergomiModel(PricingModel):
    """
    Rough Bergomi model for European options via conditional Monte Carlo.

    Parameters
    ----------
    xi0 : float
        Initial forward variance. Must be > 0. Flat curve in v1:
        E[v_t] = xi0 for all t (the API is ready for a term structure
        xi0(t) from variance-swap stripping, out of scope until the
        calibration extension). Note: variance, not volatility — for 20%
        forward vol pass 0.04.
    eta : float
        Volatility of volatility. Must be > 0. Equity-typical: 1.5-2.5.
    H : float
        Hurst exponent, roughness of the volatility path. Must be in
        (0, 0.5]; H < 0.5 is "rough" (empirical equity: 0.05-0.15) and
        H = 0.5 recovers a classical lognormal-volatility model
        (V_t = B_t exactly). H > 0.5 (smooth vol) is rejected: no
        empirical interest and the hybrid-scheme weights are derived
        for the singular kernel.
    rho : float
        Correlation between the spot and the Volterra DRIVER B. Must be
        in (-1, 1). Negative rho produces the equity skew.
    mc_paths : int, keyword-only, default 131072
        Monte Carlo paths used by ``price()`` and the Greeks. This is a
        numerical setting, not a model parameter: it does not enter
        ``__eq__``/``__hash__``.
    mc_steps : int, keyword-only, default 256
        Time steps per year-equivalent mesh used by ``price()``. Numerical
        setting, not a model parameter.
    seed : int, keyword-only, default 42
        Seed for the internal generator. ``price()`` is deterministic
        given the model instance (fresh ``numpy.random.Generator`` seeded
        with this value on every call). Numerical setting, not a model
        parameter.

    Attributes
    ----------
    xi0, eta, H : float
        As passed to the constructor.
    rho_sv : float
        Spot-vol correlation rho. Stored as ``rho_sv`` because the
        PricingModel ABC reserves ``rho()`` for the interest-rate Greek.
    mc_paths, mc_steps, seed : int
        Numerical settings.

    Notes
    -----
    ``price()`` uses the conditional Black-Scholes estimator (see
    ``price_european_conditional``), which is unbiased for the left-point
    discretization, has typically 10-50x lower variance than plain payoff
    Monte Carlo, and collapses to the deterministic Black-Scholes price
    when eta -> 0 and rho = 0. Plain payoff pricing on simulated paths is
    available as ``price_mc()`` (cross-validation, exotics).

    Examples
    --------
    >>> model = RoughBergomiModel(xi0=0.04, eta=1.9, H=0.1, rho=-0.9)
    >>> price = model.price(100, 100, 1.0, 0.05, 'call')
    >>> 8.0 < price < 12.0
    True

    Exotics from Phase 3 consume rBergomi paths without modification:

    >>> from src.engines.monte_carlo import MonteCarloEngine
    >>> engine = MonteCarloEngine(n_paths=50_000, seed=1)
    >>> paths = model.simulate(100, 1.0, 0.05, n_paths=50_000, seed=1)
    >>> res = engine.price(lambda p: np.maximum(p[:, -1] - 100, 0), paths, 0.05, 1.0)
    """

    def __init__(self, xi0: float, eta: float, H: float, rho: float, *,
                 mc_paths: int = 131072, mc_steps: int = 256,
                 seed: int = 42) -> None:
        params = {"xi0": xi0, "eta": eta, "H": H, "rho": rho}
        for name, value in params.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if xi0 <= 0:
            raise ValueError(f"xi0 must be > 0, got {xi0}. "
                             f"Note: xi0 is a variance — for 20% vol use 0.04.")
        if eta <= 0:
            raise ValueError(f"eta must be > 0, got {eta}")
        if not 0.0 < H <= 0.5:
            raise ValueError(f"H must be in (0, 0.5], got {H}. "
                             f"H < 0.5 is rough vol; H = 0.5 is classical "
                             f"lognormal vol; H > 0.5 is not supported.")
        if not -1.0 < rho < 1.0:
            raise ValueError(f"rho must be in (-1, 1), got {rho}")
        if mc_paths < 2:
            raise ValueError(f"mc_paths must be >= 2, got {mc_paths}")
        if mc_steps < 1:
            raise ValueError(f"mc_steps must be >= 1, got {mc_steps}")

        self.xi0: float = float(xi0)
        self.eta: float = float(eta)
        self.H: float = float(H)
        self.rho_sv: float = float(rho)
        self.mc_paths: int = int(mc_paths)
        self.mc_steps: int = int(mc_steps)
        self.seed: int = int(seed)

    # Input validation (_validate_inputs / _validate_option_type) is
    # inherited from the PricingModel ABC.

    # ──────────────────────────────────────────────
    # Volterra covariance (quadrature with exact anchors)
    # ──────────────────────────────────────────────

    def _volterra_covariance(self, s: float, t: float) -> float:
        """
        Exact covariance E[V_s V_t] of the Riemann-Liouville Volterra
        process, gamma = H - 1/2:

            E[V_s V_t] = 2H * int_0^min(s,t) (s-u)^gamma (t-u)^gamma du

        The integral is hypergeometric in general — evaluated by QAWS
        quadrature (``quad`` with ``weight='alg'``), which absorbs the
        algebraic endpoint singularity (min(s,t)-u)^gamma into the
        quadrature weight and integrates it exactly (machine precision,
        vs ~1e-10 for a plain adaptive rule with a breakpoint hint).

        Exact anchors used by the tests:
        - s = t:   E[V_t^2] = 2H * t^(2H)/(2H) = t^(2H)
        - H = 1/2: kernel identically 1, so E[V_s V_t] = min(s, t)
          (the covariance of a standard Brownian motion)

        Parameters
        ----------
        s, t : float
            Times, >= 0. Symmetric in (s, t). Returns 0.0 if either is 0
            (V_0 = 0 almost surely).

        Returns
        -------
        float
        """
        lo, hi = (s, t) if s <= t else (t, s)
        if lo <= 0.0:
            return 0.0
        gamma = self.H - 0.5
        if lo == hi:
            # weight (t-u)^(2*gamma), integrand 1 — QAWS handles the
            # (integrable) singularity 2*gamma in (-1, 0) exactly
            val, _ = quad(lambda u: 1.0, 0.0, lo,
                          weight='alg', wvar=(0.0, 2.0 * gamma))
        else:
            # weight (lo-u)^gamma; (hi-u)^gamma is smooth on [0, lo]
            val, _ = quad(lambda u: (hi - u) ** gamma, 0.0, lo,
                          weight='alg', wvar=(0.0, gamma))
        return float(2.0 * self.H * val)

    # ──────────────────────────────────────────────
    # Volterra simulation — exact Cholesky (reference)
    # ──────────────────────────────────────────────

    def _simulate_volterra_cholesky(
        self, T: float, n_steps: int, n_paths: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Exact-in-distribution joint simulation of (V_{t_1..t_n}, B_{t_1..t_n})
        by Cholesky factorization of the full (2n x 2n) covariance matrix.

        Blocks: V-V from ``_volterra_covariance`` (quadrature), V-B from the
        closed form

            E[V_t B_s] = sqrt(2H)/(H+1/2) * (t^(H+1/2) - (t-min(s,t))^(H+1/2))

        and B-B = min(s, t). Cost O((2n)^3) setup + O(paths * (2n)^2)
        sampling: the gold-standard reference for small meshes (n <= 64 in
        the tests), against which the hybrid scheme is cross-validated.

        Returns
        -------
        (V_tilde, dB) : tuple of np.ndarray, each (n_paths, n_steps)
            V_tilde[:, i] = V at t_{i+1}; dB[:, i] = B_{t_{i+1}} - B_{t_i}.
        """
        n = n_steps
        dt = T / n
        t = dt * np.arange(1, n + 1, dtype=np.float64)

        Sigma = np.empty((2 * n, 2 * n), dtype=np.float64)
        for i in range(n):
            for j in range(i, n):
                c = self._volterra_covariance(t[i], t[j])
                Sigma[i, j] = Sigma[j, i] = c
        Hp = self.H + 0.5
        cov_vb = (np.sqrt(2.0 * self.H) / Hp) * (
            t[:, None] ** Hp - np.maximum(t[:, None] - t[None, :], 0.0) ** Hp
        )
        Sigma[:n, n:] = cov_vb
        Sigma[n:, :n] = cov_vb.T
        Sigma[n:, n:] = np.minimum(t[:, None], t[None, :])

        # Jitter, documented and not silent: the singular kernel plus
        # quadrature noise (~1e-13) can leave Sigma numerically non-PSD,
        # and at H = 0.5 the joint law is exactly degenerate (V = B makes
        # Sigma rank n). 1e-12 * int_var restores positive definiteness while
        # perturbing each marginal variance by at most 1e-12.
        L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(2 * n))

        Z = rng.standard_normal((n_paths, 2 * n))
        X = Z @ L.T
        V_tilde = X[:, :n]
        dB = np.diff(X[:, n:], axis=1, prepend=0.0)
        return V_tilde, dB

    # ──────────────────────────────────────────────
    # Volterra simulation — hybrid scheme (production)
    # ──────────────────────────────────────────────

    def _simulate_volterra_hybrid(
        self, T: float, n_steps: int, n_paths: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Hybrid scheme of Bennedsen-Lunde-Pakkanen (2017) with kappa = 1.

        In V_{t_i} = sqrt(2H) int_0^{t_i} (t_i - s)^gamma dB_s the most
        recent interval [t_{i-1}, t_i] — where the kernel is singular — is
        integrated EXACTLY as a Gaussian correlated with dB_{i-1}:

            W1_{j+1} = int_{t_j}^{t_{j+1}} (t_{j+1} - s)^gamma dB_s

            Var[dB]      = dt
            Var[W1]      = int_0^dt s^(2*gamma) ds = dt^(2H) / (2H)
            Cov[dB, W1]  = int_0^dt s^gamma ds     = dt^(H+1/2) / (H+1/2)

        generated per step with the closed 2x2 Cholesky
        W1 = (Cov/Var[dB]) dB + sqrt(Var[W1] - Cov^2/Var[dB]) Z2. The rest
        of the history is approximated by evaluating the kernel at the
        L2-optimal points (BLP Prop. 2.8)

            b_k = ((k^(gamma+1) - (k-1)^(gamma+1)) / (gamma+1))^(1/gamma),
            G_k = (b_k * dt)^gamma,     k = 2..n

        so that

            V_{t_i} ~= sqrt(2H) * (W1_i + sum_{k=2}^{i} G_k dB_{i-k})

        The sum is a causal discrete convolution, evaluated for all i at
        once by FFT: O(paths * n log n). Index contract (verified against
        the exact Cholesky simulation in the tests — off-by-one here is
        the classic error, producing plausible prices with a wrong skew):
        with conv = fftconvolve(dB, [G_2..G_n]), V_{t_i} takes
        conv[:, i-2] for i >= 2 and the bare W1 term for i = 1.

        Special branch H = 0.5: gamma = 0 makes b_k^(1/gamma) an
        indeterminate 1^inf, but the kernel is identically 1, so V = B
        exactly. Z2 is still drawn (unused) to keep the generator stream
        aligned with the general branch — common-random-number continuity
        of H -> paths at the H = 0.5 boundary.

        Returns
        -------
        (V_tilde, dB) : tuple of np.ndarray, each (n_paths, n_steps)
            Same contract as ``_simulate_volterra_cholesky``. dB is REUSED
            by the caller for the correlated part of the spot.
        """
        n = n_steps
        dt = T / n
        H = self.H

        Z1 = rng.standard_normal((n_paths, n))
        Z2 = rng.standard_normal((n_paths, n))
        dB = np.sqrt(dt) * Z1

        if H == 0.5:
            return np.cumsum(dB, axis=1), dB

        gamma = H - 0.5
        var_dB = dt
        var_W1 = dt ** (2.0 * H) / (2.0 * H)
        cov = dt ** (H + 0.5) / (H + 0.5)
        # Conditional-normal residual variance; clipped at 0 because the
        # exact zero at H = 0.5 can round negative in floating point
        resid = max(var_W1 - cov * cov / var_dB, 0.0)
        W1 = (cov / var_dB) * dB + np.sqrt(resid) * Z2

        V_tilde = np.sqrt(2.0 * H) * W1
        if n >= 2:
            k = np.arange(2, n + 1, dtype=np.float64)
            b = ((k ** (gamma + 1.0) - (k - 1.0) ** (gamma + 1.0))
                 / (gamma + 1.0)) ** (1.0 / gamma)
            G = (b * dt) ** gamma
            conv = fftconvolve(dB, G[None, :], axes=1)
            V_tilde[:, 1:] += np.sqrt(2.0 * H) * conv[:, : n - 1]
        return V_tilde, dB

    def _volterra(
        self, T: float, n_steps: int, n_paths: int, method: str,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Dispatch to the chosen Volterra simulation method."""
        if method == 'hybrid':
            return self._simulate_volterra_hybrid(T, n_steps, n_paths, rng)
        elif method == 'cholesky':
            return self._simulate_volterra_cholesky(T, n_steps, n_paths, rng)
        raise ValueError(f"method must be 'hybrid' or 'cholesky', got '{method}'")

    # ──────────────────────────────────────────────
    # From Volterra to variance and spot paths
    # ──────────────────────────────────────────────

    def _variance_path(self, V_tilde: np.ndarray, T: float, *,
                       xi0: float | None = None,
                       eta: float | None = None) -> np.ndarray:
        """
        Variance on the full grid t_0..t_n from the Volterra values at
        t_1..t_n:

            v[:, 0] = xi0,
            v[:, i] = xi0 * exp(eta * V_{t_i} - eta^2/2 * t_i^(2H))

        The -eta^2/2 t^(2H) term is the exact lognormal mean correction:
        V_t ~ N(0, t^(2H)) makes E[v_t] = xi0 for every t. Omitting it (or
        the sqrt(2H) in V) biases E[v_t] and everything downstream — the
        anchors E[v_t] = xi0 and Var[V_t] = t^(2H) are the first tests.

        xi0/eta overrides support the finite-difference model Greeks on
        common random numbers (same V_tilde, bumped parameter).
        """
        xi0 = self.xi0 if xi0 is None else xi0
        eta = self.eta if eta is None else eta
        n = V_tilde.shape[1]
        t = (T / n) * np.arange(1, n + 1, dtype=np.float64)
        v = np.empty((V_tilde.shape[0], n + 1), dtype=np.float64)
        v[:, 0] = xi0
        v[:, 1:] = xi0 * np.exp(eta * V_tilde
                                - 0.5 * eta * eta * t ** (2.0 * self.H))
        return v

    def _spot_from_volterra(
        self, S0: float, T: float, r: float, q: float,
        V_tilde: np.ndarray, dB: np.ndarray, dB_perp: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Left-point log-Euler spot from the Volterra draws:

            ln S_{i+1} = ln S_i + (r-q) dt - v_i dt / 2
                         + sqrt(v_i) (rho dB_i + sqrt(1-rho^2) dB_i^perp)

        with v_i F_{t_i}-measurable and (dB_i, dB_i^perp) independent of
        F_{t_i}. This scheme is an EXACT martingale step by step: given
        F_{t_i} the exponent is Gaussian with mean -v_i dt/2 and variance
        v_i dt, so E[e^{-(r-q)dt} S_{i+1}/S_i | F_{t_i}] = 1 for ANY n —
        no Andersen-style correction needed. Discretization error affects
        the shape of the terminal distribution (option prices), not the
        forward. A midpoint/trapezoidal rule in sqrt(v) dB would break
        this exactness and add a Stratonovich-type correlation bias.

        Returns
        -------
        (paths, v) : spot paths (n_paths, n+1) with paths[:, 0] = S0, and
            variance paths (n_paths, n+1) with v[:, 0] = xi0.
        """
        n = V_tilde.shape[1]
        dt = T / n
        rho = self.rho_sv
        v = self._variance_path(V_tilde, T)
        v_left = v[:, :-1]
        increments = ((r - q) * dt - 0.5 * v_left * dt
                      + np.sqrt(v_left) * (rho * dB
                                           + np.sqrt(1.0 - rho * rho) * dB_perp))
        log_S = np.log(S0) + np.cumsum(increments, axis=1)
        paths = np.empty((V_tilde.shape[0], n + 1), dtype=np.float64)
        paths[:, 0] = S0
        paths[:, 1:] = np.exp(log_S)
        return paths, v

    # ──────────────────────────────────────────────
    # Public simulation API (Phase 3 exotics consume this)
    # ──────────────────────────────────────────────

    @overload
    def simulate(self, S0: float, T: float, r: float, q: float = ..., *,
                 n_paths: int | None = ..., n_steps: int | None = ...,
                 method: str = ..., antithetic: Literal[False] = ...,
                 return_variance: Literal[False] = ...,
                 seed: int | None = ...) -> np.ndarray: ...

    @overload
    def simulate(self, S0: float, T: float, r: float, q: float = ..., *,
                 n_paths: int | None = ..., n_steps: int | None = ...,
                 method: str = ..., antithetic: Literal[True],
                 return_variance: Literal[False] = ...,
                 seed: int | None = ...) -> tuple[np.ndarray, np.ndarray]: ...

    @overload
    def simulate(self, S0: float, T: float, r: float, q: float = ..., *,
                 n_paths: int | None = ..., n_steps: int | None = ...,
                 method: str = ..., antithetic: Literal[False] = ...,
                 return_variance: Literal[True],
                 seed: int | None = ...) -> tuple[np.ndarray, np.ndarray]: ...

    @overload
    def simulate(self, S0: float, T: float, r: float, q: float = ..., *,
                 n_paths: int | None = ..., n_steps: int | None = ...,
                 method: str = ..., antithetic: Literal[True],
                 return_variance: Literal[True],
                 seed: int | None = ...) -> tuple[tuple[np.ndarray, np.ndarray],
                                                  tuple[np.ndarray, np.ndarray]]: ...

    def simulate(self, S0: float, T: float, r: float, q: float = 0.0, *,
                 n_paths: int | None = None, n_steps: int | None = None,
                 method: str = 'hybrid', antithetic: bool = False,
                 return_variance: bool = False, seed: int | None = None,
                 ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray],
                            tuple[tuple[np.ndarray, np.ndarray],
                                  tuple[np.ndarray, np.ndarray]]]:
        """
        Simulate rBergomi spot paths under Q.

        The simulation lives in the model (not in ``MonteCarloEngine``)
        because — unlike the Markovian, stateless-per-step Heston QE — the
        Volterra process requires mesh-dependent structures (kernel
        weights, local 2x2 Cholesky, full covariance factorization in the
        exact method). The output is drop-in compatible with the Phase 2
        engine and Phase 3 exotics:

            engine.price(exotic.payoff, model.simulate(...), r, T)

        Parameters
        ----------
        S0 : float
            Initial spot. Must be > 0 and finite.
        T : float
            Horizon in years. Must be > 0 and finite.
        r : float
            Risk-free rate (annualized, continuous compounding).
        q : float, default 0.0
            Continuous dividend yield.
        n_paths : int or None, keyword-only
            Paths; defaults to the model's ``mc_paths``. Must be >= 2.
        n_steps : int or None, keyword-only
            Time steps; defaults to the model's ``mc_steps``. Must be >= 1.
            The weak error of the left-point scheme is O(dt), so exotics
            need n_steps >> 1 (the forward is exact for any n).
        method : {'hybrid', 'cholesky'}, keyword-only
            'hybrid' (default): BLP 2017 hybrid scheme, O(paths n log n) —
            production. 'cholesky': exact joint Gaussian, O((2n)^3) setup
            — reference for small meshes (n <= 64); the covariance
            assembly runs one quadrature per time pair.
        antithetic : bool, keyword-only, default False
            If True, also return antithetic paths built by negating EVERY
            Gaussian draw. Valid here — in contrast to QE-Heston (Phase 4),
            where the nonlinear branch-switching map destroys the
            antithetic correlation — because the map from Gaussians to
            (V, B, B_perp) is LINEAR: V(-Z) = -V(Z) exactly.
        return_variance : bool, keyword-only, default False
            If True, also return the variance paths.
        seed : int or None, keyword-only
            Seed for this simulation. None (default) uses the model's
            ``seed`` — reproducible by default, consistent with the
            deterministic-given-the-instance contract of ``price()``.

        Returns
        -------
        paths : np.ndarray, shape (n_paths, n_steps + 1)
            Spot paths, paths[:, 0] = S0. Strictly positive.
        (paths, paths_anti) : tuple
            When antithetic=True.
        (paths, variance) : tuple
            When return_variance=True. variance[:, 0] = xi0; variance > 0
            everywhere by construction (exponential — no absorption
            branches needed, unlike CIR).
        ((paths, paths_anti), (variance, variance_anti)) : nested tuples
            When both flags are True.
        """
        if not np.isfinite(S0) or S0 <= 0:
            raise ValueError(f"S0 must be finite and > 0, got {S0}")
        if not np.isfinite(T) or T <= 0:
            raise ValueError(f"T must be finite and > 0, got {T}")
        if not np.isfinite(r) or not np.isfinite(q):
            raise ValueError(f"r and q must be finite, got r={r}, q={q}")
        n_paths = self.mc_paths if n_paths is None else int(n_paths)
        n_steps = self.mc_steps if n_steps is None else int(n_steps)
        if n_paths < 2:
            raise ValueError(f"n_paths must be >= 2, got {n_paths}")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")

        rng = np.random.default_rng(self.seed if seed is None else seed)
        V_tilde, dB = self._volterra(T, n_steps, n_paths, method, rng)
        dB_perp = np.sqrt(T / n_steps) * rng.standard_normal((n_paths, n_steps))

        paths, v = self._spot_from_volterra(S0, T, r, q, V_tilde, dB, dB_perp)
        if not antithetic:
            return (paths, v) if return_variance else paths

        # Antithetic: negating all draws is exact by linearity (see above)
        paths_a, v_a = self._spot_from_volterra(S0, T, r, q,
                                                -V_tilde, -dB, -dB_perp)
        if return_variance:
            return (paths, paths_a), (v, v_a)
        return paths, paths_a

    # ──────────────────────────────────────────────
    # Conditional Black-Scholes estimator (McCrickerd-Pakkanen 2018)
    # ──────────────────────────────────────────────

    def _conditional_terms(self, V_tilde: np.ndarray, dB: np.ndarray,
                           T: float, *, xi0: float | None = None,
                           eta: float | None = None) -> _CondDraws:
        """
        Per-path sufficient statistics of the conditional estimator, with
        the SAME left-point discretization as the path scheme:

            A = sum_i sqrt(v_i) dB_i     (rho NOT applied — kept factored
                                          so the rho-Greek can bump rho on
                                          identical draws)
            I = sum_i v_i * dt           (integrated variance)

        xi0/eta overrides recompute (A, I) from the same Volterra draws
        with a bumped parameter (common random numbers for model Greeks).
        """
        v_left = self._variance_path(V_tilde, T, xi0=xi0, eta=eta)[:, :-1]
        dt = T / V_tilde.shape[1]
        A = np.einsum('ij,ij->i', np.sqrt(v_left), dB)
        integrated_var = v_left.sum(axis=1) * dt
        return A, integrated_var

    def _conditional_draws(self, T: float, *, n_paths: int | None = None,
                           n_steps: int | None = None,
                           method: str = 'hybrid',
                           seed: int | None = None) -> _CondDraws:
        """Simulate the Volterra layer and reduce it to (A, I)."""
        n_paths = self.mc_paths if n_paths is None else int(n_paths)
        n_steps = self.mc_steps if n_steps is None else int(n_steps)
        rng = np.random.default_rng(self.seed if seed is None else seed)
        V_tilde, dB = self._volterra(T, n_steps, n_paths, method, rng)
        return self._conditional_terms(V_tilde, dB, T)

    def _conditional_values(self, S: float, K: float, T: float, r: float,
                            opt: str, q: float, A: np.ndarray,
                            int_var: np.ndarray, *,
                            rho: float | None = None) -> np.ndarray:
        """
        Per-path DISCOUNTED conditional Black-Scholes values. Derivation:
        splitting the log-spot into the B-measurable part and the part
        driven by the orthogonal Brownian motion,

            ln S_T = ln S_0 + (r-q)T
                     - rho^2/2 I + rho A          (B-measurable)
                     - (1-rho^2)/2 I + sqrt(1-rho^2) int sqrt(v) dB_perp
                                                   (conditionally Gaussian)

        so conditional on the path of B (which determines v), with

            S_cond  = S_0 exp((r-q)T + rho A - rho^2/2 I)
            Sigma^2 = (1 - rho^2) I

        the terminal spot is lognormal, S_T | B = S_cond e^{-Sigma^2/2
        + Sigma Z}, and the conditional call price is EXACT Black-Scholes:

            E[(S_T - K)^+ | B] = S_cond N(d1) - K N(d2),
            d_{1,2} = (ln(S_cond / K) +- Sigma^2/2) / Sigma

        This decomposition is exact for the discrete left-point scheme
        (not only in the continuous limit): the estimator carries the same
        O(dt) discretization bias as plain Monte Carlo on the same mesh,
        but ZERO sampling noise from the B_perp dimension. Sigma^2 > 0
        always (rho in (-1,1), I > 0 by the exponential variance), so no
        degenerate paths need special-casing.

        Puts by exact parity per path: put = call - S e^{-qT} + K e^{-rT}
        (a constant shift — identical standard error). Unbiased because
        E[e^{-(r-q)T} S_cond] = S_0 exactly for any n (the left-point
        martingale property, taking E[exp(rho sqrt(v_i) dB_i
        - rho^2 v_i dt / 2) | F_i] = 1 step by step).

        Assumes K > 0 — callers handle K = 0 (prepaid forward) upstream.
        """
        rho = self.rho_sv if rho is None else rho
        S_cond = S * np.exp((r - q) * T + rho * A - 0.5 * rho * rho * int_var)
        Sigma = np.sqrt((1.0 - rho * rho) * int_var)
        d1 = (np.log(S_cond / K) + 0.5 * Sigma * Sigma) / Sigma
        d2 = d1 - Sigma
        call_vals: np.ndarray = np.exp(-r * T) * (S_cond * ndtr(d1) - K * ndtr(d2))
        if opt == 'call':
            return call_vals
        put_vals: np.ndarray = call_vals - S * np.exp(-q * T) + K * np.exp(-r * T)
        return put_vals

    def _control_adjust(self, vals: np.ndarray, S: float, T: float,
                        r: float, q: float, A: np.ndarray, int_var: np.ndarray,
                        *, rho: float | None = None) -> np.ndarray:
        """
        Control-variate completion of the conditional estimator
        (the "turbocharging" of McCrickerd-Pakkanen 2018).

        The conditional estimator alone integrates out only the B_perp
        noise, whose share of the total variance is (1 - rho^2): the
        reduction is total at rho = 0 (zero variance) but VANISHES as
        |rho| -> 1, where almost all the randomness lives in the
        B-measurable S_cond. The fix is the control

            C = e^{-rT} S_cond,   E[C] = S e^{-qT}   (EXACT for any mesh,
                                   by the left-point martingale property)

        with the sample-optimal beta (Phase 2 ``control_variate_adjust``;
        beta = 0 fallback when Var[C] ~ 0, e.g. rho = 0, keeps the
        deterministic Black-Scholes limit intact). The conditional call
        value is a smooth increasing function of S_cond, so the residual
        correlation is high precisely in the high-|rho| regime where the
        conditional step alone is weak. Measured at the canonical set
        (xi0=0.04, eta=1.9, H=0.1, rho=-0.9): conditional alone ~0.77x
        the plain-MC standard error; with the control ~0.4x.
        """
        rho = self.rho_sv if rho is None else rho
        control = S * np.exp(-q * T + rho * A - 0.5 * rho * rho * int_var)
        adjusted, _ = control_variate_adjust(vals, control,
                                             float(S * np.exp(-q * T)))
        return adjusted

    def _conditional_price(self, S: float, K: float, T: float, r: float,
                           opt: str, q: float, A: np.ndarray, int_var: np.ndarray,
                           *, rho: float | None = None,
                           control_variate: bool = True) -> float:
        """
        Mean of the per-path conditional values, control-variate adjusted
        by default. Assumes K > 0 like ``_conditional_values`` — every
        caller (price/Greeks kernels, iv_surface) intercepts K = 0 with
        its deterministic closed form before reaching here.
        """
        vals = self._conditional_values(S, K, T, r, opt, q, A, int_var, rho=rho)
        if control_variate:
            vals = self._control_adjust(vals, S, T, r, q, A, int_var, rho=rho)
        return float(np.mean(vals))

    def price_european_conditional(
        self, S: float, K: float, T: float, r: float,
        option_type: str = 'call', q: float = 0.0, *,
        n_paths: int | None = None, n_steps: int | None = None,
        method: str = 'hybrid', seed: int | None = None,
        control_variate: bool = True,
    ) -> tuple[float, float]:
        """
        European price and standard error via the conditional estimator —
        the engine behind ``price()``, exposed with its Monte Carlo
        diagnostics and the choice of Volterra method.

        Scalar inputs only (one simulation prices one contract; for
        strike/maturity grids use ``price()`` or ``iv_surface``, which
        share draws across strikes at each maturity).

        Parameters
        ----------
        S, K, T, r : float
            Spot, strike, maturity, rate.
        option_type : str, default 'call'
        q : float, default 0.0
        n_paths, n_steps : int or None, keyword-only
            Default to the model's numerical settings.
        method : {'hybrid', 'cholesky'}, keyword-only
        seed : int or None, keyword-only
            None uses the model's seed.
        control_variate : bool, keyword-only, default True
            Apply the e^{-rT} S_cond control variate (see
            ``_control_adjust``). With False the estimator is the bare
            conditional mean, whose sample mean is EXACTLY monotone in K
            and exactly parity-consistent on shared draws (the in-sample
            beta of the control varies with K and breaks pathwise
            monotonicity by O(SE) — parity remains exact either way).

        Returns
        -------
        (price, std_error) : tuple of float
            std_error is 0.0 for the deterministic K = 0 branch.
        """
        S, K, T, r = float(S), float(K), float(T), float(r)
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)
        if K == 0.0:
            return (float(S * np.exp(-q * T)) if opt == 'call' else 0.0, 0.0)
        A, int_var = self._conditional_draws(T, n_paths=n_paths, n_steps=n_steps,
                                       method=method, seed=seed)
        vals = self._conditional_values(S, K, T, r, opt, q, A, int_var)
        if control_variate:
            vals = self._control_adjust(vals, S, T, r, q, A, int_var)
        se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        return float(np.mean(vals)), se

    # ──────────────────────────────────────────────
    # ABC price and Greeks
    # ──────────────────────────────────────────────

    def _vectorize(self, kernel: Callable[[float, float, float, float], float],
                   S: Numeric, K: Numeric, T: Numeric,
                   r: Numeric) -> Numeric:
        """Broadcast (S, K, T, r) and apply a scalar kernel elementwise."""
        S_b, K_b, T_b, r_b = np.broadcast_arrays(
            np.asarray(S, dtype=np.float64), np.asarray(K, dtype=np.float64),
            np.asarray(T, dtype=np.float64), np.asarray(r, dtype=np.float64),
        )
        if S_b.ndim == 0:
            return kernel(float(S_b), float(K_b), float(T_b), float(r_b))
        flat = [
            kernel(float(s), float(k), float(t), float(rr))
            for s, k, t, rr in zip(S_b.ravel(), K_b.ravel(), T_b.ravel(), r_b.ravel())
        ]
        return np.asarray(flat, dtype=np.float64).reshape(S_b.shape)

    def _draws(self, T: float, cache: dict[float, _CondDraws]) -> _CondDraws:
        """
        (A, I) at maturity T, memoized per call so that every element of a
        broadcast request (and every Greek of ``greeks()``) sharing a
        maturity reuses one simulation. The cache is call-local: no state
        survives on the instance, keeping price() deterministic and
        instances hashable by model parameters only.
        """
        if T not in cache:
            cache[T] = self._conditional_draws(T)
        return cache[T]

    def price(self, S: Numeric, K: Numeric, T: Numeric, r: Numeric,
              option_type: str = 'call', q: float = 0.0) -> Numeric:
        """
        European price via the conditional Black-Scholes estimator.

        Deterministic given the model instance: a fresh generator seeded
        with the model's ``seed`` is used per maturity. Broadcast inputs
        share one simulation per distinct maturity (the Volterra layer
        does not depend on S, K, r or q). For the Monte Carlo standard
        error use ``price_european_conditional``; for plain payoff Monte
        Carlo (cross-validation) use ``price_mc``.
        """
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)
        cache: dict[float, _CondDraws] = {}

        def kernel(s: float, k: float, t: float, rr: float) -> float:
            if k == 0.0:
                return float(s * np.exp(-q * t)) if opt == 'call' else 0.0
            A, int_var = self._draws(t, cache)
            return self._conditional_price(s, k, t, rr, opt, q, A, int_var)

        return self._vectorize(kernel, S, K, T, r)

    def _delta_kernel(self, opt: str, q: float,
                      cache: dict[float, _CondDraws],
                      ) -> Callable[[float, float, float, float], float]:
        """
        Delta by exact path rescaling: the rBergomi coefficients do not
        depend on the spot level, so S_cond is proportional to S_0 with
        (A, I) independent of S_0 — one simulation, central finite
        difference in S on identical draws. The conditional price is a
        smooth function of S (no payoff discontinuity survives the
        conditional-BS smoothing), so h = 1e-3 S is safe.
        """
        def kernel(s: float, k: float, t: float, rr: float) -> float:
            h = 1e-3 * s
            if k == 0.0:
                up = s + h if opt == 'call' else 0.0
                dn = s - h if opt == 'call' else 0.0
                return float((up - dn) * np.exp(-q * t) / (2.0 * h))
            A, int_var = self._draws(t, cache)
            up = self._conditional_price(s + h, k, t, rr, opt, q, A, int_var)
            dn = self._conditional_price(s - h, k, t, rr, opt, q, A, int_var)
            return (up - dn) / (2.0 * h)
        return kernel

    def _gamma_kernel(self, opt: str, q: float,
                      cache: dict[float, _CondDraws],
                      ) -> Callable[[float, float, float, float], float]:
        """Gamma: second central difference in S on identical draws."""
        def kernel(s: float, k: float, t: float, rr: float) -> float:
            if k == 0.0:
                return 0.0  # zero-strike price is linear in S
            h = 1e-3 * s
            A, int_var = self._draws(t, cache)
            up = self._conditional_price(s + h, k, t, rr, opt, q, A, int_var)
            mid = self._conditional_price(s, k, t, rr, opt, q, A, int_var)
            dn = self._conditional_price(s - h, k, t, rr, opt, q, A, int_var)
            return (up - 2.0 * mid + dn) / (h * h)
        return kernel

    def _vega_kernel(self, opt: str, q: float,
                     cache: dict[float, _CondDraws],
                     ) -> Callable[[float, float, float, float], float]:
        """
        Vega = dV/d(sqrt(xi0)) — sensitivity to the initial forward
        volatility, matching the Heston Phase 4 convention: it collapses
        to the FULL Black-Scholes vega as eta -> 0 (xi0 is the entire
        forward-variance curve here). Exact common random numbers: v is
        proportional to xi0 pointwise, so bumping xi0 -> c*xi0 maps the
        SAME draws to (A, I) -> (sqrt(c) A, c I) — no re-simulation.
        For the sensitivity to xi0 itself use ``model_greeks()['xi0']``
        (chain rule: vega = dV/dxi0 * 2 sqrt(xi0), tested).
        """
        s_vol = float(np.sqrt(self.xi0))
        h = 1e-3 * s_vol
        c_up = (s_vol + h) ** 2 / self.xi0
        c_dn = (s_vol - h) ** 2 / self.xi0

        def kernel(s: float, k: float, t: float, rr: float) -> float:
            if k == 0.0:
                return 0.0  # prepaid forward: no vol dependence
            A, int_var = self._draws(t, cache)
            up = self._conditional_price(s, k, t, rr, opt, q,
                                         np.sqrt(c_up) * A, c_up * int_var)
            dn = self._conditional_price(s, k, t, rr, opt, q,
                                         np.sqrt(c_dn) * A, c_dn * int_var)
            return (up - dn) / (2.0 * h)
        return kernel

    def _theta_kernel(self, opt: str, q: float,
                      cache: dict[float, _CondDraws],
                      ) -> Callable[[float, float, float, float], float]:
        """
        Theta = dV/dt (per year) = -dV/dT, central difference in maturity.
        Changing T changes the mesh, so the draws at T-h and T+h are
        RE-SIMULATED with the model's seed: the same Gaussians feed both
        meshes and the Gaussians->paths map is smooth in T with n fixed —
        common random numbers through the seed. The bump h = 1e-2 T is
        coarser than Heston's (1e-3) because the CRN residual here is MC
        noise rather than quadrature noise; the residual standard error
        of the Greek is ~ SE(price) / (2h * sqrt(correlation gain)).
        """
        def kernel(s: float, k: float, t: float, rr: float) -> float:
            h = 1e-2 * t
            if k == 0.0:
                if opt != 'call':
                    return 0.0
                up = s * np.exp(-q * (t - h))
                dn = s * np.exp(-q * (t + h))
                return float((up - dn) / (2.0 * h))
            A_dn, I_dn = self._draws(t - h, cache)
            A_up, I_up = self._draws(t + h, cache)
            p_dn = self._conditional_price(s, k, t - h, rr, opt, q, A_dn, I_dn)
            p_up = self._conditional_price(s, k, t + h, rr, opt, q, A_up, I_up)
            return (p_dn - p_up) / (2.0 * h)
        return kernel

    def _rho_kernel(self, opt: str, q: float,
                    cache: dict[float, _CondDraws],
                    ) -> Callable[[float, float, float, float], float]:
        """
        Rho = dV/dr (interest-rate sensitivity, NOT the spot-vol
        correlation rho_sv — for dV/d(correlation) use
        ``model_greeks()['rho']``). The Volterra draws do not depend on r,
        so this is a central difference on identical (A, I) —
        deterministic given the seed.
        """
        def kernel(s: float, k: float, t: float, rr: float) -> float:
            h = 1e-4
            if k == 0.0:
                return 0.0  # S e^{-qT} has no r-dependence
            A, int_var = self._draws(t, cache)
            up = self._conditional_price(s, k, t, rr + h, opt, q, A, int_var)
            dn = self._conditional_price(s, k, t, rr - h, opt, q, A, int_var)
            return (up - dn) / (2.0 * h)
        return kernel

    def delta(self, S: Numeric, K: Numeric, T: Numeric, r: Numeric,
              option_type: str = 'call', q: float = 0.0) -> Numeric:
        """Delta = dV/dS by exact path rescaling (one simulation)."""
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)
        return self._vectorize(self._delta_kernel(opt, q, {}), S, K, T, r)

    def gamma(self, S: Numeric, K: Numeric, T: Numeric, r: Numeric,
              option_type: str = 'call', q: float = 0.0) -> Numeric:
        """Gamma = d2V/dS2 by exact path rescaling. Call = put (parity)."""
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)
        return self._vectorize(self._gamma_kernel(opt, q, {}), S, K, T, r)

    def vega(self, S: Numeric, K: Numeric, T: Numeric, r: Numeric,
             option_type: str = 'call', q: float = 0.0) -> Numeric:
        """Vega = dV/d(sqrt(xi0)) on identical draws (exact scaling)."""
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)
        return self._vectorize(self._vega_kernel(opt, q, {}), S, K, T, r)

    def theta(self, S: Numeric, K: Numeric, T: Numeric, r: Numeric,
              option_type: str = 'call', q: float = 0.0) -> Numeric:
        """Theta = dV/dt (per year), re-simulated meshes under CRN."""
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)
        return self._vectorize(self._theta_kernel(opt, q, {}), S, K, T, r)

    def rho(self, S: Numeric, K: Numeric, T: Numeric, r: Numeric,
            option_type: str = 'call', q: float = 0.0) -> Numeric:
        """Rho = dV/dr on identical draws."""
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)
        return self._vectorize(self._rho_kernel(opt, q, {}), S, K, T, r)

    def greeks(self, S: Numeric, K: Numeric, T: Numeric, r: Numeric,
               option_type: str = 'call', q: float = 0.0) -> Dict[str, Numeric]:
        """
        All ABC Greeks in one dict: price, delta, gamma, vega, theta, rho.

        One shared draws cache: price/delta/gamma/vega/rho reuse a single
        simulation per distinct maturity; theta adds the two re-simulated
        meshes at T -+ h. For the rBergomi parameter sensitivities a desk
        hedges with, see ``model_greeks()``.
        """
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)
        cache: dict[float, _CondDraws] = {}

        def price_kernel(s: float, k: float, t: float, rr: float) -> float:
            if k == 0.0:
                return float(s * np.exp(-q * t)) if opt == 'call' else 0.0
            A, int_var = self._draws(t, cache)
            return self._conditional_price(s, k, t, rr, opt, q, A, int_var)

        return {
            'price': self._vectorize(price_kernel, S, K, T, r),
            'delta': self._vectorize(self._delta_kernel(opt, q, cache), S, K, T, r),
            'gamma': self._vectorize(self._gamma_kernel(opt, q, cache), S, K, T, r),
            'vega': self._vectorize(self._vega_kernel(opt, q, cache), S, K, T, r),
            'theta': self._vectorize(self._theta_kernel(opt, q, cache), S, K, T, r),
            'rho': self._vectorize(self._rho_kernel(opt, q, cache), S, K, T, r),
        }

    # ──────────────────────────────────────────────
    # Model-parameter Greeks
    # ──────────────────────────────────────────────

    def _bumped(self, **overrides: float) -> "RoughBergomiModel":
        """Copy with some model parameters replaced (numerical settings kept)."""
        params = {"xi0": self.xi0, "eta": self.eta, "H": self.H,
                  "rho": self.rho_sv}
        params.update(overrides)
        return RoughBergomiModel(**params, mc_paths=self.mc_paths,
                                 mc_steps=self.mc_steps, seed=self.seed)

    def model_greeks(self, S: Numeric, K: Numeric, T: Numeric, r: Numeric,
                     option_type: str = 'call', q: float = 0.0,
                     ) -> Dict[str, Numeric]:
        """
        Sensitivities to the four rBergomi parameters by central finite
        differences with EXACT common random numbers — the Gaussian draws
        do not depend on the parameters, so each bump reuses them:

        - 'xi0': v is proportional to xi0, so the bump maps the same draws
          through (A, I) -> (sqrt(c) A, c I) with c = xi0'/xi0 (exact).
        - 'eta': (A, I) recomputed from the SAME Volterra draws with the
          bumped eta (v depends on eta nonlinearly, but V does not).
        - 'H': the kernel itself changes — re-simulated with the model's
          seed (same Gaussians; the Gaussians->paths map is continuous in
          H). Central bump clamped to stay inside (0, 0.5]; at H = 0.5
          exactly, a one-sided backward difference is used (the boundary).
        - 'rho': enters only the conditional-BS formula (A is stored
          without the rho factor) — same draws, bump in the formula.
          Clamped to keep both bumps inside (-1, 1), as in Phase 4.

        Returns
        -------
        dict
            Keys 'xi0', 'eta', 'H', 'rho'. Note: here 'rho' is the
            spot-vol correlation, NOT the interest-rate Greek of
            ``greeks()``.
        """
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)

        volterra_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        ai_cache: dict[tuple[float, str], _CondDraws] = {}

        def volterra(t: float) -> tuple[np.ndarray, np.ndarray]:
            if t not in volterra_cache:
                rng = np.random.default_rng(self.seed)
                volterra_cache[t] = self._simulate_volterra_hybrid(
                    t, self.mc_steps, self.mc_paths, rng)
            return volterra_cache[t]

        def terms(t: float, tag: str, *, eta: float | None = None,
                  model: "RoughBergomiModel | None" = None) -> _CondDraws:
            key = (t, tag)
            if key not in ai_cache:
                if model is not None:
                    # bumped-H model, same seed -> same Gaussians (CRN)
                    ai_cache[key] = model._conditional_draws(t)
                else:
                    V_tilde, dB = volterra(t)
                    ai_cache[key] = self._conditional_terms(V_tilde, dB, t,
                                                            eta=eta)
            return ai_cache[key]

        h_xi0 = 1e-3 * self.xi0
        c_up, c_dn = (self.xi0 + h_xi0) / self.xi0, (self.xi0 - h_xi0) / self.xi0

        def kernel_xi0(s: float, k: float, t: float, rr: float) -> float:
            if k == 0.0:
                return 0.0
            A, int_var = terms(t, 'base')
            up = self._conditional_price(s, k, t, rr, opt, q,
                                         np.sqrt(c_up) * A, c_up * int_var)
            dn = self._conditional_price(s, k, t, rr, opt, q,
                                         np.sqrt(c_dn) * A, c_dn * int_var)
            return (up - dn) / (2.0 * h_xi0)

        h_eta = 1e-3 * self.eta

        def kernel_eta(s: float, k: float, t: float, rr: float) -> float:
            if k == 0.0:
                return 0.0
            A_up, I_up = terms(t, 'eta+', eta=self.eta + h_eta)
            A_dn, I_dn = terms(t, 'eta-', eta=self.eta - h_eta)
            up = self._conditional_price(s, k, t, rr, opt, q, A_up, I_up)
            dn = self._conditional_price(s, k, t, rr, opt, q, A_dn, I_dn)
            return (up - dn) / (2.0 * h_eta)

        h_H = min(1e-3 * self.H, 0.5 * (0.5 - self.H))
        if h_H > 0.0:
            model_H_up = self._bumped(H=self.H + h_H)
            model_H_dn = self._bumped(H=self.H - h_H)

            def kernel_H(s: float, k: float, t: float, rr: float) -> float:
                if k == 0.0:
                    return 0.0
                A_up, I_up = terms(t, 'H+', model=model_H_up)
                A_dn, I_dn = terms(t, 'H-', model=model_H_dn)
                up = self._conditional_price(s, k, t, rr, opt, q, A_up, I_up)
                dn = self._conditional_price(s, k, t, rr, opt, q, A_dn, I_dn)
                return (up - dn) / (2.0 * h_H)
        else:
            # H = 0.5 exactly: backward one-sided difference at the boundary
            h_bwd = 1e-3 * self.H
            model_H_dn = self._bumped(H=self.H - h_bwd)

            def kernel_H(s: float, k: float, t: float, rr: float) -> float:
                if k == 0.0:
                    return 0.0
                A0, I0 = terms(t, 'base')
                A_dn, I_dn = terms(t, 'H-', model=model_H_dn)
                base = self._conditional_price(s, k, t, rr, opt, q, A0, I0)
                dn = self._conditional_price(s, k, t, rr, opt, q, A_dn, I_dn)
                return (base - dn) / h_bwd

        h_rho = min(1e-3 * max(abs(self.rho_sv), 0.1),
                    0.5 * (1.0 - abs(self.rho_sv)))

        def kernel_rho(s: float, k: float, t: float, rr: float) -> float:
            if k == 0.0:
                return 0.0
            A, int_var = terms(t, 'base')
            up = self._conditional_price(s, k, t, rr, opt, q, A, int_var,
                                         rho=self.rho_sv + h_rho)
            dn = self._conditional_price(s, k, t, rr, opt, q, A, int_var,
                                         rho=self.rho_sv - h_rho)
            return (up - dn) / (2.0 * h_rho)

        return {
            'xi0': self._vectorize(kernel_xi0, S, K, T, r),
            'eta': self._vectorize(kernel_eta, S, K, T, r),
            'H': self._vectorize(kernel_H, S, K, T, r),
            'rho': self._vectorize(kernel_rho, S, K, T, r),
        }

    # ──────────────────────────────────────────────
    # Plain payoff Monte Carlo (cross-validation, exotics)
    # ──────────────────────────────────────────────

    def price_mc(self, S: float, K: float, T: float, r: float,
                 option_type: str = 'call', q: float = 0.0, *,
                 n_paths: int | None = None, n_steps: int | None = None,
                 method: str = 'hybrid', antithetic: bool = False,
                 seed: int | None = None) -> MCResult:
        """
        European price by plain payoff Monte Carlo on simulated paths —
        the cross-check of the conditional estimator (same discretization
        bias, ~10-50x more sampling variance) and the template for pricing
        exotics: ``engine.price(exotic.payoff, model.simulate(...), r, T)``.

        Returns
        -------
        MCResult
            Price, standard error, 95% CI, path count, variance-reduction
            label — the Phase 2 container.
        """
        S, K, T, r = float(S), float(K), float(T), float(r)
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)
        n_paths_eff = self.mc_paths if n_paths is None else int(n_paths)
        n_steps_eff = self.mc_steps if n_steps is None else int(n_steps)

        engine = MonteCarloEngine(n_paths=n_paths_eff, n_steps=n_steps_eff)
        if opt == 'call':
            def payoff(p: np.ndarray) -> np.ndarray:
                return np.maximum(p[:, -1] - K, 0.0)
        else:
            def payoff(p: np.ndarray) -> np.ndarray:
                return np.maximum(K - p[:, -1], 0.0)

        if antithetic:
            paths, paths_anti = self.simulate(
                S, T, r, q, n_paths=n_paths_eff, n_steps=n_steps_eff,
                method=method, antithetic=True, seed=seed)
            return engine.price(payoff, paths, r, T, paths_anti=paths_anti)
        paths = self.simulate(S, T, r, q, n_paths=n_paths_eff,
                              n_steps=n_steps_eff, method=method, seed=seed)
        return engine.price(payoff, paths, r, T)

    # ──────────────────────────────────────────────
    # Implied volatility surface
    # ──────────────────────────────────────────────

    def iv_surface(self, S: float, strikes: Numeric, maturities: Numeric,
                   r: float, q: float = 0.0, *,
                   n_paths: int | None = None, n_steps: int | None = None,
                   method: str = 'hybrid',
                   seed: int | None = None) -> np.ndarray:
        """
        Implied volatility surface via the conditional estimator and the
        Phase 1 Halley solver.

        One simulation per maturity, shared across all strikes (the
        Volterra layer does not depend on K). Each price is inverted on
        the OUT-of-the-money side — call for K >= forward, put for
        K < forward: the model's put-call parity is exact by construction,
        so both sides give the same implied vol in exact arithmetic, but
        the OTM side is better conditioned (extrinsic value dominates,
        larger vega-to-price ratio for the root-finder).

        Parameters
        ----------
        S : float
            Spot. Must be > 0.
        strikes : array_like
            Strikes, all > 0 (a zero strike has no implied vol).
        maturities : array_like
            Maturities in years, all > 0.
        r : float
            Risk-free rate.
        q : float, default 0.0
            Continuous dividend yield.
        n_paths, n_steps, method, seed : keyword-only
            Numerical overrides, as in ``price_european_conditional``.

        Returns
        -------
        np.ndarray, shape (len(maturities), len(strikes))
            Implied volatilities (annualized decimals).
        """
        strikes_arr = np.atleast_1d(np.asarray(strikes, dtype=np.float64))
        maturities_arr = np.atleast_1d(np.asarray(maturities, dtype=np.float64))
        self._validate_inputs(S, strikes_arr, maturities_arr, r)
        if np.any(strikes_arr <= 0):
            raise ValueError("iv_surface requires strikes > 0; a zero "
                             "strike has no implied volatility.")

        ivs = np.empty((len(maturities_arr), len(strikes_arr)), dtype=np.float64)
        for i, t in enumerate(maturities_arr):
            A, int_var = self._conditional_draws(float(t), n_paths=n_paths,
                                           n_steps=n_steps, method=method,
                                           seed=seed)
            forward = S * np.exp((r - q) * t)
            for j, k in enumerate(strikes_arr):
                opt = 'call' if k >= forward else 'put'
                p = self._conditional_price(float(S), float(k), float(t),
                                            float(r), opt, q, A, int_var)
                ivs[i, j] = BlackScholesModel.implied_vol(
                    p, float(S), float(k), float(t), float(r), opt, q)
        return ivs

    # ──────────────────────────────────────────────
    # Dunder methods — model parameters only (D1): two instances with
    # different MC resolution are the SAME model.
    # ──────────────────────────────────────────────

    def __repr__(self) -> str:
        return (f"RoughBergomiModel(xi0={self.xi0}, eta={self.eta}, "
                f"H={self.H}, rho={self.rho_sv})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RoughBergomiModel):
            return NotImplemented
        return (self.xi0, self.eta, self.H, self.rho_sv) == \
               (other.xi0, other.eta, other.H, other.rho_sv)

    def __hash__(self) -> int:
        return hash((self.xi0, self.eta, self.H, self.rho_sv))
