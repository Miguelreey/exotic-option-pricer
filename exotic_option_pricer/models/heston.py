"""
Exotic Option Pricer — exotic_option_pricer/models/heston.py

Heston (1993) stochastic volatility model with characteristic-function pricing.

Implements:
- Characteristic function in the numerically stable "Little Heston Trap"
  formulation (Albrecher et al. 2007)
- European vanilla pricing via Gil-Pelaez Fourier inversion with a
  Black-Scholes control variate (Andersen & Piterbarg 2010, §8.7) on a
  vectorized composite Gauss-Legendre grid; adaptive quadrature retained
  as a fallback for pathological corners
- Fast strike-grid pricing via Carr-Madan (1999) FFT with Simpson weights
  (for implied vol surfaces and calibration)
- ABC Greeks: delta and gamma exact via the differentiated characteristic
  function; vega (dV/d sqrt(v0)), theta and rho via central finite differences
  on the deterministic Fourier price (no MC noise)
- model_greeks(): sensitivities to the five Heston parameters
  dV/d{v0, kappa, theta, xi, rho} — what a desk actually hedges with

Mathematical Background
-----------------------
Under the risk-neutral measure Q the spot and its instantaneous variance
follow:

    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW^S_t
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW^v_t
    d<W^S, W^v>_t = rho dt

The variance is a CIR process: mean-reverting at speed kappa to the long-run
level theta, with vol-of-vol xi. The Feller condition 2*kappa*theta >= xi^2
guarantees v_t > 0 strictly; when violated (common in equity calibration)
v_t can touch zero but remains nonnegative.

Heston admits a semi-closed form: the characteristic function
phi(u) = E^Q[exp(i u ln S_T)] is exponential-affine in (ln S_0, v_0).
Vanilla prices follow by Fourier inversion (Gil-Pelaez 1951):

    C = S e^{-qT} P1 - K e^{-rT} P2
    P_j = 1/2 + (1/pi) * int_0^inf Re[ e^{-iu ln K} phi_j(u) / (iu) ] du

with phi_2(u) = phi(u) and phi_1(u) = phi(u - i) / phi(-i) (share measure).

The "Little Heston Trap": Heston's original (1993) formulation of the
log-term in the characteristic function crosses the principal branch cut of
the complex logarithm for long maturities, producing discontinuous (wrong)
prices. The Albrecher et al. (2007) formulation used here keeps |g| <= 1 so
the argument of the logarithm never winds around the origin, and is
continuous for all maturities. This is the #1 implementation error in
Heston pricers.

References
----------
.. [1] Heston (1993). A Closed-Form Solution for Options with Stochastic
       Volatility. Review of Financial Studies 6(2), 327-343.
.. [2] Albrecher, Mayer, Schoutens, Tistaert (2007). The Little Heston
       Trap. Wilmott Magazine, January 2007, 83-92.
.. [3] Carr & Madan (1999). Option Valuation Using the Fast Fourier
       Transform. Journal of Computational Finance 2(4), 61-73.
.. [4] Gil-Pelaez (1951). Note on the Inversion Theorem. Biometrika 38,
       481-482.
.. [5] Gatheral (2006). The Volatility Surface. Wiley.
.. [6] Andersen & Piterbarg (2007). Moment Explosions in Stochastic
       Volatility Models. Finance and Stochastics 11(1), 29-50.
.. [7] Andersen & Piterbarg (2010). Interest Rate Modeling. Atlantic
       Financial Press. §8.7 (Fourier integration with control variates).
"""

import warnings
from functools import lru_cache
from typing import Callable, Dict, Union

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.special import ndtr

from .base import Numeric, PricingModel

# char_func evaluation points: complex arguments are part of the contract
# (Gil-Pelaez uses u - i, Carr-Madan uses u - (alpha+1)i)
ComplexNumeric = Union[float, complex, np.ndarray]

# Adaptive quadrature settings for the Gil-Pelaez FALLBACK path (engaged
# only when the composite Gauss-Legendre budget of _fourier_grid is
# exceeded). epsabs=1e-9 keeps the quadrature error two orders of magnitude
# below the 1e-6 cross-validation tolerance against Carr-Madan FFT and
# QuantLib. limit=400 accommodates the slowly-decaying, fast-oscillating
# integrands of very-low-vol short-maturity corners.
_QUAD_OPTS = {"epsabs": 1e-9, "epsrel": 1e-9, "limit": 400}

# Composite Gauss-Legendre rule for the vectorized Gil-Pelaez path: 16
# nodes per panel with panels no longer than half an estimated oscillation
# period. The high per-panel order is what buys robustness: for oscillatory
# integrands spectral order per panel converges much faster than panel
# subdivision, so even where the analytic phase-rate bound of
# _fourier_grid underestimates the true oscillation severalfold the rule
# keeps ~1e-11 accuracy (empirical worst case over the 89 QuantLib anchors
# plus stress corners: 3e-11, vs 1.7e-6 for 8-node panels of equal length).
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(16)
# Panel budget: 8192 panels = 131k characteristic-function evaluations
# (~ms vectorized). Corners whose oscillation x support product exceeds it
# (Hypothesis-extreme low-vol high-xi deep strikes) fall back to `quad`.
_MAX_PANELS = 8192
# Envelope threshold for truncating the Fourier tail: the neglected mass is
# bounded by _TAIL_EPS * u_max, far below the 1e-9 accuracy target of the
# adaptive path it replaces.
_TAIL_EPS = 1e-12


@lru_cache(maxsize=8)
def _carr_madan_constants(
    n_fft: int, eta: float, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Strike- and model-independent Carr-Madan arrays for (n_fft, eta, alpha).

    A calibration evaluates price_surface ~10^4 times with identical grid
    settings; rebuilding these arrays per call was ~20-30% of its cost.
    Returns (u grid, log-strike grid, combined FFT input weight
    e^{iub} * simpson / denominator, damping e^{-alpha k}/pi), all marked
    read-only so a cache hit can never be mutated by a caller.
    """
    u = eta * np.arange(n_fft)
    lam = 2.0 * np.pi / (n_fft * eta)
    b = 0.5 * n_fft * lam
    k_grid = -b + lam * np.arange(n_fft)

    # Simpson weights eta/3 * [1, 4, 2, 4, ..., 2, 4] (Carr-Madan eq. 24).
    # The final endpoint closure is irrelevant because psi(u_max) ~ 0.
    simpson = np.full(n_fft, eta / 3.0)
    simpson[1::2] *= 4.0
    simpson[2::2] *= 2.0

    denominator = alpha * alpha + alpha - u * u + 1j * (2.0 * alpha + 1.0) * u
    fft_weight = np.exp(1j * u * b) * simpson / denominator
    damp = np.exp(-alpha * k_grid) / np.pi

    for arr in (u, k_grid, fft_weight, damp):
        arr.setflags(write=False)
    return u, k_grid, fft_weight, damp


class HestonModel(PricingModel):
    """
    Heston (1993) stochastic volatility model for European options.

    Pricing is by Fourier inversion of the characteristic function
    (semi-closed form) — fast and exact to quadrature tolerance. Monte Carlo
    simulation of the Heston SDE (for exotics) lives in
    ``MonteCarloEngine.simulate_heston()``, which uses the Andersen QE scheme.

    Parameters
    ----------
    v0 : float
        Initial instantaneous variance. Must be > 0.
        Note: variance, not volatility — for 20% initial vol pass 0.04.
    kappa : float
        Mean-reversion speed of the variance. Must be > 0.
    theta : float
        Long-run variance level. Must be > 0.
    xi : float
        Volatility of variance ("vol-of-vol"). Must be > 0.
        Controls the convexity (smile curvature) of the implied vol surface.
    rho : float
        Instantaneous correlation between spot and variance Brownian
        motions. Must be in (-1, 1). Negative rho (equity leverage effect)
        produces the downward implied-vol skew.

    Attributes
    ----------
    v0, kappa, xi : float
        As passed to the constructor.
    theta_v : float
        Long-run variance theta. Stored as ``theta_v`` (not ``theta``)
        because the PricingModel ABC reserves ``theta()`` for the
        calendar-time Greek.
    rho_sv : float
        Spot-vol correlation rho. Stored as ``rho_sv`` because the ABC
        reserves ``rho()`` for the interest-rate Greek.

    Warns
    -----
    UserWarning
        If the Feller condition 2*kappa*theta >= xi^2 is violated. The model
        remains valid (variance can touch zero but stays nonnegative) and
        this is common for equity calibrations — the warning is informative,
        not an error.

    Examples
    --------
    >>> heston = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
    >>> price = heston.price(100, 100, 1.0, 0.05, 'call')
    >>> 9.0 < price < 11.0
    True

    In the deterministic-variance limit (xi -> 0 with v0 = theta), Heston
    collapses to Black-Scholes with sigma = sqrt(v0):

    >>> bs_like = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=1e-4, rho=0.0)
    >>> round(bs_like.price(100, 100, 1.0, 0.05, 'call'), 3)
    10.451
    """

    def __init__(self, v0: float, kappa: float, theta: float, xi: float, rho: float) -> None:
        params = {"v0": v0, "kappa": kappa, "theta": theta, "xi": xi, "rho": rho}
        for name, value in params.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if v0 <= 0:
            raise ValueError(
                f"v0 must be > 0, got {v0}. Note: v0 is a variance — for 20% vol use 0.04."
            )
        if kappa <= 0:
            raise ValueError(f"kappa must be > 0, got {kappa}")
        if theta <= 0:
            raise ValueError(
                f"theta must be > 0, got {theta}. Note: theta is a variance — for 20% vol use 0.04."
            )
        if xi <= 0:
            raise ValueError(f"xi must be > 0, got {xi}")
        if not -1.0 < rho < 1.0:
            raise ValueError(f"rho must be in (-1, 1), got {rho}")

        if 2.0 * kappa * theta < xi * xi:
            warnings.warn(
                f"Feller condition violated: 2*kappa*theta = {2 * kappa * theta:.6f} "
                f"< xi^2 = {xi * xi:.6f}. The variance process can reach zero. "
                f"This is common in equity calibrations and handled correctly "
                f"by both the characteristic function and the QE simulation "
                f"scheme; informative, not an error.",
                UserWarning,
                stacklevel=2,
            )

        self.v0: float = float(v0)
        self.kappa: float = float(kappa)
        self.theta_v: float = float(theta)
        self.xi: float = float(xi)
        self.rho_sv: float = float(rho)

    # Input validation (_validate_inputs / _validate_option_type) is
    # inherited from the PricingModel ABC.

    # ──────────────────────────────────────────────
    # Characteristic function (Little Heston Trap form)
    # ──────────────────────────────────────────────

    def char_func(
        self, u: ComplexNumeric, T: float, r: float, q: float = 0.0, S0: float = 1.0
    ) -> complex | np.ndarray:
        """
        Characteristic function of ln S_T under Q: phi(u) = E[exp(i u ln S_T)].

        Uses the numerically stable formulation of Albrecher et al. (2007):

            a = kappa - rho*xi*i*u
            d = sqrt(a^2 + xi^2 (i*u + u^2))            (principal branch)
            g = (a - d) / (a + d)                        with |g| <= 1
            C = (kappa*theta/xi^2) [ (a - d) T - 2 ln((1 - g e^{-dT})/(1 - g)) ]
            D = ((a - d)/xi^2) (1 - e^{-dT}) / (1 - g e^{-dT})
            phi(u) = exp( i u [ln S0 + (r - q) T] + C + D v0 )

        Heston's original (1993) paper uses g1 = 1/g (the "+d" form), whose
        complex logarithm crosses the principal branch cut for long
        maturities and corrupts prices. With |g| <= 1 the argument of the
        logarithm never winds around the origin — continuous for all T.

        Parameters
        ----------
        u : float, complex, or ndarray
            Evaluation point(s). Complex arguments are required internally:
            Gil-Pelaez P1 uses phi(u - i) and Carr-Madan uses
            phi(u - (alpha+1)i).
        T : float
            Time to maturity in years. Must be > 0.
        r : float
            Risk-free rate (annualized, continuous compounding).
        q : float, default 0.0
            Continuous dividend yield.
        S0 : float, default 1.0
            Spot price. The default 1.0 gives the characteristic function
            of the log-return process (ln S0 = 0).

        Returns
        -------
        complex or ndarray of complex
            phi(u), scalar if u is scalar.

        Notes
        -----
        Exact values used as test anchors:
        - phi(0) = 1 (a probability distribution integrates to one)
        - phi(-i) = E[S_T] = S0 exp((r - q) T) (Q-martingale condition)

        u = 0 and u = -i are exactly the roots of iu + u^2, where d
        degenerates to sqrt(a^2). At u = -i, a = kappa - rho*xi is real and
        the principal branch gives d = |a|: for kappa < rho*xi this makes
        a + d = 0 and the g-ratio hits 0/0, although the limit of the full
        expression is perfectly finite. Both points are therefore
        special-cased to their exact values — legitimate because the Heston
        spot is a TRUE martingale for every admissible parameter set
        (Keller-Ressel 2011), so phi(-i) = forward always.

        For complex u with Im(u) <= -1 (moments of order > 1), the formula
        is the analytic continuation of E[S_T^p], which equals the moment
        only while the moment is finite — beyond the explosion time
        T*(p) (Andersen & Piterbarg 2007) it returns finite but spurious
        values. ``price_surface`` guards this via
        ``_moment_explosion_time``; P1/P2 pricing only ever uses
        Im(u) in {0, -1}, which never explodes.
        """
        u_arr = np.asarray(u, dtype=np.complex128)
        kappa, theta, xi, rho, v0 = (
            self.kappa,
            self.theta_v,
            self.xi,
            self.rho_sv,
            self.v0,
        )

        iu = 1j * u_arr
        a = kappa - rho * xi * iu
        # errstate: the two removable points u = 0, u = -i can generate
        # 0/0 or x/0 intermediates; they are overwritten with their exact
        # values below. Genuine overflow (moment explosion region) still
        # propagates as inf/nan for the caller-side guards.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # Principal-branch sqrt gives Re(d) >= 0, which together with
            # the (a - d) numerator keeps |g| <= 1 (Little-Trap form).
            d = np.sqrt(a * a + xi * xi * (iu + u_arr * u_arr))
            g = (a - d) / (a + d)

            exp_dT = np.exp(-d * T)
            C = (kappa * theta / (xi * xi)) * (
                (a - d) * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))
            )
            D = ((a - d) / (xi * xi)) * (1.0 - exp_dT) / (1.0 - g * exp_dT)

            phi = np.exp(iu * (np.log(S0) + (r - q) * T) + C + D * v0)

        # Exact values at the removable singularities of the g-ratio
        phi = np.where(u_arr == 0.0, 1.0 + 0.0j, phi)
        phi = np.where(u_arr == -1.0j, S0 * np.exp((r - q) * T) + 0.0j, phi)
        return complex(phi) if phi.ndim == 0 else phi

    def _moment_explosion_time(self, p: float) -> float:
        """
        Explosion time T*(p) of E[S_T^p] for p > 1 (Andersen & Piterbarg
        2007, Prop. 3.1): the moment is finite iff T < T*(p).

        At u = -ip the characteristic-function discriminant becomes

            a = kappa - rho*xi*p,    d^2 = a^2 - xi^2 p(p-1)

        and the moment explodes at the first zero of 1 - g e^{-dT}:

        - d^2 >= 0, a > 0:  |g| < 1, no zero — T* = infinity.
        - d^2 >= 0, a <= 0: g = (a-d)/(a+d) >= 1 real — T* = ln(g)/d
          (limit 2/|a| as d -> 0).
        - d^2 < 0: d = i*delta, |g| = 1 with phase psi = -2 atan2(delta, a)
          — T* = (psi mod 2pi)/delta.
        """
        kappa, xi, rho = self.kappa, self.xi, self.rho_sv
        a = kappa - rho * xi * p
        d2 = a * a - xi * xi * p * (p - 1.0)
        if d2 > 0.0:
            if a > 0.0:
                return float("inf")
            d_real = np.sqrt(d2)
            g_real = (a - d_real) / (a + d_real)
            return float(np.log(g_real) / d_real)
        if d2 == 0.0:
            return float("inf") if a > 0.0 else float(2.0 / abs(a))
        delta = np.sqrt(-d2)
        psi = -2.0 * np.arctan2(delta, a)
        return float(np.mod(psi, 2.0 * np.pi) / delta)

    # ──────────────────────────────────────────────
    # Gil-Pelaez probabilities (BS control variate + composite Gauss-Legendre)
    # ──────────────────────────────────────────────

    def _cv_total_variance(self, T: float) -> float:
        """
        Matched total variance of the Black-Scholes control variate:

            w = E^Q[ int_0^T v_t dt ] = theta T + (v0 - theta)(1 - e^{-kappa T})/kappa

        (closed-form mean of the integrated CIR variance). Matching the
        expected integrated variance makes phi_j^BS track phi_j to leading
        order in the vol-of-vol, so the residual integrand carries only the
        xi- and rho-driven correction. expm1 keeps the kappa*T -> 0 limit
        exact (-> v0 T) without cancellation.
        """
        return float(
            self.theta_v * T - (self.v0 - self.theta_v) * np.expm1(-self.kappa * T) / self.kappa
        )

    def _cv_term(
        self,
        u: ComplexNumeric,
        j: int,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        x: float,
        w: float,
    ) -> np.ndarray:
        """
        Control-variated Gil-Pelaez numerator on a u grid (before any pole
        division):

            e^{-iu ln K} phi_j(u) - phi_j^{BS-term}(u)

        with the Black-Scholes characteristic functions of matched total
        variance w already combined with the e^{-iu ln K} phase: under BS,
        ln S_T is N(m - w/2, w) under Q and N(m + w/2, w) under the share
        measure (m = ln S + (r - q)T), so with x = m - ln K

            e^{-iu ln K} phi_2^BS(u) = exp( iu(x - w/2) - u^2 w/2 )
            e^{-iu ln K} phi_1^BS(u) = exp( iu(x + w/2) - u^2 w/2 ).

        j = 1 is the share-measure term (phi(u - i)/phi(-i)), j = 2 the
        Q-measure term. phi(-i) is substituted by its exact closed form
        S e^{(r-q)T} (true martingale, Keller-Ressel 2011) — identical to
        char_func's special-cased value.
        """
        u_arr = np.asarray(u, dtype=np.float64)
        if j == 1:
            phi = np.asarray(self.char_func(u_arr - 1j, T, r, q, S)) / (S * np.exp((r - q) * T))
            drift = x + 0.5 * w
        else:
            phi = np.asarray(self.char_func(u_arr, T, r, q, S))
            drift = x - 0.5 * w
        phase = np.exp(-1j * u_arr * np.log(K))
        bs = np.exp(1j * u_arr * drift - 0.5 * w * u_arr * u_arr)
        result: np.ndarray = phase * phi - bs
        return result

    def _fourier_grid(
        self, S: float, T: float, r: float, q: float, x: float, w: float, *, pole: bool
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Composite Gauss-Legendre grid (nodes, weights) for the residual
        Gil-Pelaez integrand on [0, u_max], or None when the panel budget
        would exceed _MAX_PANELS (caller falls back to adaptive quad).

        Truncation: u_max starts at the analytic estimate
        min(sqrt(2 ln(1/eps)/w), ln(1/eps)/c_inf) — Gaussian-regime and
        asymptotic-regime cutoffs, with c_inf = sqrt(1-rho^2)(v0 + kappa
        theta T)/xi the Heston tail decay rate (Andersen-Piterbarg 2010) —
        and is then VERIFIED numerically on the actual |phi_j| envelope,
        growing by 1.5x until the envelope (with the 1/u pole decay when
        ``pole``) sits below _TAIL_EPS at both u_max and 2 u_max. Neither
        analytic estimate is trusted on its own: c_inf is asymptotic only
        (it wildly overestimates decay for xi -> 0), and the Gaussian
        cutoff underestimates the support when the slow exponential tail
        dominates (low-vol high-xi corners).

        Panel length: min(pi/omega, 1/sqrt(w), u_max/16) — at most half an
        estimated oscillation period, no coarser than the envelope scale,
        at least 16 panels. omega bounds the phase rate of
        e^{-iu ln K} phi_j(u) = exp(iux + C + D v0 - iu m):
        |x| from the moneyness rotation, w/2 from the BS drift split, and
        the C + D v0 phase slope (kappa theta T + v0)/xi scaled by
        |rho| + (xi - 2 kappa rho)/(2 kappa): the |rho| part is the
        asymptotic slope from Im(a) = -rho xi u, and the second part is the
        near-origin slope of Im(d) — from d^2 = kappa^2 +
        iu xi(xi - 2 kappa rho) + (1-rho^2) xi^2 u^2, d(0) = kappa gives
        Im d'(0) = xi (xi - 2 kappa rho)/(2 kappa), which dominates the
        transition region when xi >> kappa with strongly negative rho and
        is NOT captured by the asymptotic slope alone (the 8-node/no-Im(d)
        first cut of this grid lost 1.7e-6 on exactly that corner of the
        QuantLib anchor set C).
        """
        vol_load = (self.v0 + self.kappa * self.theta_v * T) / self.xi
        im_d_slope = max(self.xi - 2.0 * self.kappa * self.rho_sv, 0.0) / (2.0 * self.kappa)
        omega = abs(x) + 0.5 * w + (abs(self.rho_sv) + im_d_slope) * vol_load
        c_inf = np.sqrt(1.0 - self.rho_sv * self.rho_sv) * vol_load
        log_eps = -np.log(_TAIL_EPS)
        forward = S * np.exp((r - q) * T)

        u_max = float(np.clip(min(np.sqrt(2.0 * log_eps / w), log_eps / c_inf), 10.0, 1.0e7))
        for _ in range(60):
            probe = np.array([u_max, 2.0 * u_max])
            env = (
                np.abs(np.asarray(self.char_func(probe, T, r, q, S)))
                + np.abs(np.asarray(self.char_func(probe - 1j, T, r, q, S))) / forward
            )
            if pole:
                env = env / np.maximum(probe, 1.0)
            if float(np.max(env)) < _TAIL_EPS:
                break
            u_max *= 1.5
            if u_max > 1.0e8:
                return None
        else:
            return None

        panel_len = min(np.pi / omega, 1.0 / np.sqrt(w), u_max / 16.0)
        n_panels = int(np.ceil(u_max / panel_len))
        if n_panels > _MAX_PANELS:
            return None

        edges = np.linspace(0.0, u_max, n_panels + 1)
        half = 0.5 * u_max / n_panels
        mid = 0.5 * (edges[1:] + edges[:-1])
        nodes = (mid[:, None] + half * _GL_NODES[None, :]).ravel()
        weights = np.broadcast_to(half * _GL_WEIGHTS, (n_panels, _GL_WEIGHTS.size)).ravel()
        return nodes, weights

    def _p1_p2(self, S: float, K: float, T: float, r: float, q: float) -> tuple[float, float]:
        """
        Gil-Pelaez exercise probabilities P1 (share measure) and P2 (Q)
        with a Black-Scholes control variate (Andersen-Piterbarg 2010 §8.7):

            P_j = N(d_j) + (1/pi) int_0^inf
                      Re[ e^{-iu ln K} (phi_j(u) - phi_j^BS(u)) / (iu) ] du

        where phi_j^BS is the BS characteristic function with total variance
        w = E[int_0^T v_t dt] matched to the model, and N(d_j) is its exact
        Gil-Pelaez value: under BS the probabilities ARE N(d1), N(d2) with
        d_{1,2} = x/sqrt(w) +- sqrt(w)/2, x = ln(S/K) + (r - q)T. The
        residual integrand is orders of magnitude smaller than the raw one
        (the CV absorbs the bulk of the oscillatory mass), which is what
        lets a moderate fixed grid replace adaptive quadrature.

        Evaluated on the vectorized composite Gauss-Legendre grid of
        ``_fourier_grid``; falls back to the adaptive-quad path (same
        control-variated integrand, scalar callbacks) for corners whose
        oscillation x support product exceeds the panel budget. The
        integrands keep the removable singularity at u = 0 (both phi_j and
        phi_j^BS tend to 1, so the difference vanishes linearly); neither
        Gauss-Legendre nor Gauss-Kronrod nodes sit on the endpoint.
        """
        x = float(np.log(S / K) + (r - q) * T)
        w = self._cv_total_variance(T)
        sqrt_w = np.sqrt(w)
        d1 = x / sqrt_w + 0.5 * sqrt_w
        d2 = d1 - sqrt_w

        grid = self._fourier_grid(S, T, r, q, x, w, pole=True)
        if grid is not None:
            u, wts = grid
            int_p1 = float(wts @ (self._cv_term(u, 1, S, K, T, r, q, x, w) / (1j * u)).real)
            int_p2 = float(wts @ (self._cv_term(u, 2, S, K, T, r, q, x, w) / (1j * u)).real)
        else:

            def integrand_p1(u: float) -> float:
                return float((self._cv_term(u, 1, S, K, T, r, q, x, w) / (1j * u)).real)

            def integrand_p2(u: float) -> float:
                return float((self._cv_term(u, 2, S, K, T, r, q, x, w) / (1j * u)).real)

            int_p1, _ = quad(integrand_p1, 0.0, np.inf, **_QUAD_OPTS)
            int_p2, _ = quad(integrand_p2, 0.0, np.inf, **_QUAD_OPTS)

        # Clip pure quadrature noise (~1e-12) outside [0, 1]; genuine errors
        # are caught by the price-bound and parity tests, not masked here.
        P1 = float(np.clip(float(ndtr(d1)) + int_p1 / np.pi, 0.0, 1.0))
        P2 = float(np.clip(float(ndtr(d2)) + int_p2 / np.pi, 0.0, 1.0))
        return P1, P2

    def _price_delta_scalar(
        self, S: float, K: float, T: float, r: float, opt: str, q: float
    ) -> tuple[float, float]:
        """
        Scalar Gil-Pelaez (price, delta) from ONE (P1, P2) evaluation —
        price needs both probabilities and delta = e^{-qT} P1 is a
        byproduct, so computing them together halves the quadrature work
        of ``greeks()``. K = 0 handled without the log-K integral.
        """
        disc_q = np.exp(-q * T)
        disc_r = np.exp(-r * T)
        if K == 0.0:
            # A zero-strike call pays S_T: worth the prepaid forward with
            # delta e^{-qT}. A zero-strike put pays max(-S_T, 0) = 0.
            return (float(S * disc_q), float(disc_q)) if opt == "call" else (0.0, 0.0)

        P1, P2 = self._p1_p2(S, K, T, r, q)
        call = S * disc_q * P1 - K * disc_r * P2
        # Project onto the European no-arbitrage band
        # max(S e^{-qT} - K e^{-rT}, 0) <= C <= S e^{-qT}. In extreme
        # corners (deep ITM, short T, low vol) the residual quadrature
        # error (~1e-7 on the old adaptive path) can land the raw call just
        # outside the band. Both legs derive from the projected call, so
        # put-call parity is exact by construction; quadrature accuracy is
        # still tested independently against the QuantLib anchors.
        call = min(max(call, S * disc_q - K * disc_r, 0.0), S * disc_q)
        delta_call = disc_q * P1
        if opt == "call":
            return float(call), float(delta_call)
        # Put-call parity: P = C - S e^{-qT} + K e^{-rT}, dP/dS = delta_C - e^{-qT}
        return float(call - S * disc_q + K * disc_r), float(delta_call - disc_q)

    def _price_scalar(self, S: float, K: float, T: float, r: float, opt: str, q: float) -> float:
        """Scalar Gil-Pelaez price. See ``_price_delta_scalar``."""
        return self._price_delta_scalar(S, K, T, r, opt, q)[0]

    # ──────────────────────────────────────────────
    # Price (ABC)
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
        European option price by Gil-Pelaez Fourier inversion.

            C = S e^{-qT} P1 - K e^{-rT} P2,    P = C - S e^{-qT} + K e^{-rT}

        Parameters
        ----------
        S, K, T, r : float or ndarray
            Spot, strike, maturity (years), risk-free rate. Broadcast
            together elementwise.
        option_type : str, default 'call'
            'call' or 'put'.
        q : float, default 0.0
            Continuous dividend yield.

        Returns
        -------
        float or ndarray
            Option price(s).

        Notes
        -----
        Array inputs are broadcast and evaluated elementwise: each element
        gets its own control-variated Gauss-Legendre grid (truncation and
        panel density depend on the strike and maturity), so the
        per-element loop remains — but each element is now a handful of
        vectorized characteristic-function evaluations instead of an
        adaptive scalar quadrature. For many strikes at a single maturity
        ``price_surface()`` (Carr-Madan FFT — one transform prices the
        whole strike grid) is still the faster route and is what the
        calibrator uses.
        """
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)
        return self._vectorize(
            lambda s, k, t, rr: self._price_scalar(s, k, t, rr, opt, q),
            S,
            K,
            T,
            r,
        )

    # ──────────────────────────────────────────────
    # Carr-Madan FFT surface
    # ──────────────────────────────────────────────

    def price_surface(
        self,
        S: float,
        strikes: Numeric,
        T: float,
        r: float,
        q: float = 0.0,
        option_type: str = "call",
        *,
        alpha: float = 1.5,
        eta: float = 0.25,
        n_fft: int = 4096,
    ) -> np.ndarray:
        """
        Price a strike grid at one maturity via Carr-Madan (1999) FFT.

        With damping alpha and k = ln K, the damped call price has Fourier
        transform

            psi(u) = e^{-rT} phi(u - (alpha+1)i)
                     / (alpha^2 + alpha - u^2 + i(2*alpha + 1)u)

        and the call is recovered as

            C(k) = (e^{-alpha k}/pi) int_0^inf Re[ e^{-iuk} psi(u) ] du.

        The integral is evaluated for all strikes simultaneously with one
        FFT (Simpson-weighted, N = n_fft nodes, spacing eta), then cubic
        spline interpolation maps the FFT log-strike grid to the requested
        strikes. This is what makes calibration tractable: each objective
        evaluation needs an entire option chain.

        Parameters
        ----------
        S : float
            Spot price. Must be > 0.
        strikes : float or ndarray
            Strike(s). Must be > 0 (use ``price()`` for the K = 0 edge case).
        T : float
            Time to maturity in years. Must be > 0.
        r : float
            Risk-free rate.
        q : float, default 0.0
            Continuous dividend yield.
        option_type : str, default 'call'
            'call' or 'put'. Puts via put-call parity.
        alpha : float, default 1.5
            Carr-Madan damping factor. Requires E[S_T^(alpha+1)] < infinity;
            Heston moments above 1 can explode for long maturities and
            strongly positive rho (Andersen & Piterbarg 2007). alpha = 1.5
            is safe for equity-style parameters (rho <= 0). A guard raises
            ValueError if the required moment does not exist.
        eta : float, default 0.25
            Spacing of the Fourier grid. The log-strike grid spacing is
            2*pi/(n_fft*eta) ~ 0.6% with the defaults.
        n_fft : int, default 4096
            FFT size (power of 2).

        Returns
        -------
        np.ndarray
            Option prices at the requested strikes (1-D, same length).

        Raises
        ------
        ValueError
            If inputs are invalid or E[S_T^(alpha+1)] diverges (moment
            explosion — reduce alpha).

        Notes
        -----
        Prices are computed on the normalized problem S = 1 (homogeneity:
        C(S, K) = S * C(1, K/S)), which keeps every factor in the FFT O(1)
        and avoids the catastrophic cancellation that raw log-spot offsets
        of order exp((alpha+1) ln S) would introduce.
        """
        self._validate_inputs(S, np.asarray(strikes), T, r)
        opt = self._validate_option_type(option_type)
        strikes_arr = np.atleast_1d(np.asarray(strikes, dtype=np.float64))
        if np.any(strikes_arr <= 0):
            raise ValueError("price_surface requires strikes > 0; use price() for K = 0.")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if eta <= 0:
            raise ValueError(f"eta must be > 0, got {eta}")
        if n_fft < 2 or (n_fft & (n_fft - 1)) != 0:
            raise ValueError(f"n_fft must be a power of 2 >= 2, got {n_fft}")

        # Carr-Madan requires E[S_T^(alpha+1)] < inf. A finiteness check on
        # char_func is NOT sufficient: beyond the explosion time the
        # analytic continuation returns finite but spurious values, so the
        # closed-form explosion time is checked instead.
        t_star = self._moment_explosion_time(alpha + 1.0)
        if T >= t_star:
            raise ValueError(
                f"E[S_T^(alpha+1)] diverges: with alpha={alpha} the moment "
                f"of order {alpha + 1.0} explodes at T* = {t_star:.4f} < T = {T} "
                f"(Andersen-Piterbarg moment explosion). Reduce alpha."
            )

        # Normalized problem: spot = 1, strikes k' = ln(K/S), result * S.
        # Everything that depends only on (n_fft, eta, alpha) — u grid,
        # log-strike grid, Simpson/twiddle/denominator weight, damping —
        # comes from the module-level cache; only the characteristic
        # function and the FFT are per-call work.
        u, k_grid, fft_weight, damp = _carr_madan_constants(n_fft, float(eta), float(alpha))

        phi_vals = self.char_func(u - (alpha + 1.0) * 1j, T, r, q, 1.0)
        fft_vals = np.fft.fft(np.exp(-r * T) * phi_vals * fft_weight)
        calls_grid = damp * fft_vals.real

        spline = CubicSpline(k_grid, calls_grid)
        k_req = np.log(strikes_arr / S)
        if np.any(k_req < k_grid[0]) or np.any(k_req > k_grid[-1]):
            raise ValueError(
                f"log-moneyness outside FFT grid [{k_grid[0]:.2f}, {k_grid[-1]:.2f}]; "
                f"increase n_fft or eta."
            )
        calls = S * np.asarray(spline(k_req), dtype=np.float64)

        if opt == "put":
            # Parity on the normalized problem, rescaled by S
            prices = calls - S * np.exp(-q * T) + strikes_arr * np.exp(-r * T)
        else:
            prices = calls
        result: np.ndarray = np.maximum(prices, 0.0)
        return result

    # ──────────────────────────────────────────────
    # Greeks — delta/gamma exact via the CF, rest by central FD
    # ──────────────────────────────────────────────

    def _delta_scalar(self, S: float, K: float, T: float, r: float, opt: str, q: float) -> float:
        """
        Exact delta. Heston coefficients do not depend on the spot level, so
        S_T = S * M_T with M_T independent of S. Differentiating
        C = e^{-rT} E[(S M_T - K)^+] under the integral sign:

            dC/dS = e^{-rT} E[M_T 1{S_T > K}] = e^{-qT} P1

        (the share-measure exercise probability). Put delta via parity.
        """
        return self._price_delta_scalar(S, K, T, r, opt, q)[1]

    def _gamma_scalar(self, S: float, K: float, T: float, r: float, q: float) -> float:
        """
        Exact gamma = e^{-qT} dP1/dS. Since phi_1 depends on S only through
        exp(iu ln S), term-by-term differentiation of the Gil-Pelaez integral
        gives dphi_1/dS = phi_1 * iu / S, the iu cancels the 1/(iu) pole, and

            gamma = e^{-qT}/(pi S) int_0^inf Re[ e^{-iu ln K} phi_1(u) ] du.

        Same for calls and puts (parity: the linear terms vanish).

        The BS control variate applies verbatim (same phi_1 - phi_1^BS
        residual, no pole division) because the BS term has the closed form

            int_0^inf Re[ e^{-iu ln K} phi_1^BS(u) ] du
                = int_0^inf e^{-u^2 w/2} cos(u(x + w/2)) du
                = (pi/sqrt(w)) n(d1)

        (Gaussian cosine transform int_0^inf e^{-au^2} cos(bu) du =
        (1/2) sqrt(pi/a) e^{-b^2/4a} with a = w/2, b = x + w/2 = d1
        sqrt(w)), which reproduces the Black-Scholes gamma
        e^{-qT} n(d1)/(S sqrt(w)) exactly.
        """
        if K == 0.0:
            return 0.0
        x = float(np.log(S / K) + (r - q) * T)
        w = self._cv_total_variance(T)
        sqrt_w = np.sqrt(w)
        d1 = x / sqrt_w + 0.5 * sqrt_w
        bs_integral = np.pi / sqrt_w * np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)

        grid = self._fourier_grid(S, T, r, q, x, w, pole=False)
        if grid is not None:
            u, wts = grid
            integral = float(wts @ self._cv_term(u, 1, S, K, T, r, q, x, w).real)
        else:

            def integrand(u: float) -> float:
                return float(self._cv_term(u, 1, S, K, T, r, q, x, w).real)

            integral, _ = quad(integrand, 0.0, np.inf, **_QUAD_OPTS)

        return float(np.exp(-q * T) / (np.pi * S) * (bs_integral + integral))

    def _vectorize(
        self,
        kernel: Callable[[float, float, float, float], float],
        S: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
    ) -> Numeric:
        """Broadcast (S, K, T, r) and apply a scalar kernel elementwise."""
        S_b, K_b, T_b, r_b = np.broadcast_arrays(
            np.asarray(S, dtype=np.float64),
            np.asarray(K, dtype=np.float64),
            np.asarray(T, dtype=np.float64),
            np.asarray(r, dtype=np.float64),
        )
        if S_b.ndim == 0:
            return kernel(float(S_b), float(K_b), float(T_b), float(r_b))
        flat = [
            kernel(float(s), float(k), float(t), float(rr))
            for s, k, t, rr in zip(S_b.ravel(), K_b.ravel(), T_b.ravel(), r_b.ravel())
        ]
        return np.asarray(flat, dtype=np.float64).reshape(S_b.shape)

    @staticmethod
    def _central_diff(up_price: Numeric, dn_price: Numeric, h: float) -> Numeric:
        """Central difference (up - dn) / (2h), 0-d results collapsed to float."""
        diff = (np.asarray(up_price) - np.asarray(dn_price)) / (2.0 * h)
        return float(diff) if diff.ndim == 0 else diff

    def _bumped(self, **overrides: float) -> "HestonModel":
        """
        Copy of this model with some parameters replaced, suppressing the
        Feller warning: the user was already warned at construction; internal
        finite-difference bumps must not spam it again.
        """
        params = {
            "v0": self.v0,
            "kappa": self.kappa,
            "theta": self.theta_v,
            "xi": self.xi,
            "rho": self.rho_sv,
        }
        params.update(overrides)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return HestonModel(**params)

    def delta(
        self,
        S: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        option_type: str = "call",
        q: float = 0.0,
    ) -> Numeric:
        """Delta = dV/dS = e^{-qT} P1 (exact, no bumping). Put via parity."""
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)
        return self._vectorize(
            lambda s, k, t, rr: self._delta_scalar(s, k, t, rr, opt, q),
            S,
            K,
            T,
            r,
        )

    def gamma(
        self,
        S: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        option_type: str = "call",
        q: float = 0.0,
    ) -> Numeric:
        """Gamma = d2V/dS2, exact via the differentiated CF. Call = put."""
        self._validate_inputs(S, K, T, r)
        self._validate_option_type(option_type)
        return self._vectorize(
            lambda s, k, t, rr: self._gamma_scalar(s, k, t, rr, q),
            S,
            K,
            T,
            r,
        )

    def vega(
        self,
        S: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        option_type: str = "call",
        q: float = 0.0,
    ) -> Numeric:
        """
        Vega = dV/d(sqrt(v0)) — sensitivity to the initial instantaneous
        volatility sqrt(v0), by central finite difference on the Fourier
        price (deterministic — no MC noise).

        Heston has no single sigma, so the ABC vega is mapped to the
        initial-vol direction: it answers "if today's instantaneous vol
        moves by 1 vol point, how does V move?", matching the BS vega in the
        xi -> 0 limit. For the full parameter sensitivities a desk hedges
        with, use ``model_greeks()``.
        """
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)
        s_vol = float(np.sqrt(self.v0))
        h = 1e-3 * s_vol
        up = self._bumped(v0=(s_vol + h) ** 2)
        dn = self._bumped(v0=(s_vol - h) ** 2)
        return self._central_diff(
            up.price(S, K, T, r, opt, q),
            dn.price(S, K, T, r, opt, q),
            h,
        )

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
        Theta = dV/dt (per year) = -dV/dT, central finite difference in
        maturity. Same sign convention as BlackScholesModel.theta.
        """
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)
        T_arr = np.asarray(T, dtype=np.float64)
        h = 1e-3 * float(np.min(T_arr))
        return self._central_diff(
            self.price(S, K, T_arr - h, r, opt, q),
            self.price(S, K, T_arr + h, r, opt, q),
            h,
        )

    def rho(
        self,
        S: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        option_type: str = "call",
        q: float = 0.0,
    ) -> Numeric:
        """
        Rho = dV/dr (interest-rate sensitivity, NOT the spot-vol correlation
        rho_sv — for dV/d(correlation) use ``model_greeks()['rho']``).
        Central finite difference in r.
        """
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)
        h = 1e-4
        r_arr = np.asarray(r, dtype=np.float64)
        return self._central_diff(
            self.price(S, K, T, r_arr + h, opt, q),
            self.price(S, K, T, r_arr - h, opt, q),
            h,
        )

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
        All ABC Greeks in one dict: price, delta, gamma, vega, theta, rho.

        delta/gamma are exact (differentiated characteristic function);
        vega/theta/rho are central finite differences on the deterministic
        Fourier price. For Heston parameter sensitivities see
        ``model_greeks()``.

        price and delta come from ONE shared (P1, P2) evaluation per
        element (delta = e^{-qT} P1 is a byproduct of the price integrals),
        bit-identical to calling ``price()`` and ``delta()`` separately.
        """
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)

        S_b, K_b, T_b, r_b = np.broadcast_arrays(
            np.asarray(S, dtype=np.float64),
            np.asarray(K, dtype=np.float64),
            np.asarray(T, dtype=np.float64),
            np.asarray(r, dtype=np.float64),
        )
        price: Numeric
        delta: Numeric
        if S_b.ndim == 0:
            price, delta = self._price_delta_scalar(
                float(S_b), float(K_b), float(T_b), float(r_b), opt, q
            )
        else:
            pairs = [
                self._price_delta_scalar(float(s), float(k), float(t), float(rr), opt, q)
                for s, k, t, rr in zip(S_b.ravel(), K_b.ravel(), T_b.ravel(), r_b.ravel())
            ]
            flat = np.asarray(pairs, dtype=np.float64)
            price = flat[:, 0].reshape(S_b.shape)
            delta = flat[:, 1].reshape(S_b.shape)

        return {
            "price": price,
            "delta": delta,
            "gamma": self.gamma(S, K, T, r, opt, q),
            "vega": self.vega(S, K, T, r, opt, q),
            "theta": self.theta(S, K, T, r, opt, q),
            "rho": self.rho(S, K, T, r, opt, q),
        }

    def model_greeks(
        self,
        S: Numeric,
        K: Numeric,
        T: Numeric,
        r: Numeric,
        option_type: str = "call",
        q: float = 0.0,
    ) -> Dict[str, Numeric]:
        """
        Sensitivities to the five Heston parameters by central finite
        differences: dV/dv0, dV/dkappa, dV/dtheta, dV/dxi, dV/drho.

        These (not the ABC Greeks) are what a desk running a Heston book
        actually hedges: v0/theta exposure maps to variance swaps, xi to
        vol-of-vol instruments, rho to skew positions.

        Returns
        -------
        dict
            Keys 'v0', 'kappa', 'theta', 'xi', 'rho' — named after the
            mathematical parameters. Note: here 'theta' and 'rho' are the
            long-run variance and the spot-vol correlation, NOT the
            calendar-time and interest-rate Greeks of ``greeks()``.
        """
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)

        def central_fd(name: str, value: float, h: float) -> Numeric:
            up = self._bumped(**{name: value + h})
            dn = self._bumped(**{name: value - h})
            return self._central_diff(
                up.price(S, K, T, r, opt, q),
                dn.price(S, K, T, r, opt, q),
                h,
            )

        # Relative bumps with a floor; rho clamped so both bumps stay in (-1, 1)
        h_rho = min(1e-3 * max(abs(self.rho_sv), 0.1), 0.5 * (1.0 - abs(self.rho_sv)))
        return {
            "v0": central_fd("v0", self.v0, 1e-3 * self.v0),
            "kappa": central_fd("kappa", self.kappa, 1e-3 * self.kappa),
            "theta": central_fd("theta", self.theta_v, 1e-3 * self.theta_v),
            "xi": central_fd("xi", self.xi, 1e-3 * self.xi),
            "rho": central_fd("rho", self.rho_sv, h_rho),
        }

    # ──────────────────────────────────────────────
    # Dunder methods
    # ──────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"HestonModel(v0={self.v0}, kappa={self.kappa}, "
            f"theta={self.theta_v}, xi={self.xi}, rho={self.rho_sv})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HestonModel):
            return NotImplemented
        return (self.v0, self.kappa, self.theta_v, self.xi, self.rho_sv) == (
            other.v0,
            other.kappa,
            other.theta_v,
            other.xi,
            other.rho_sv,
        )

    def __hash__(self) -> int:
        return hash((self.v0, self.kappa, self.theta_v, self.xi, self.rho_sv))
