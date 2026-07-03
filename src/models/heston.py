"""
Exotic Option Pricer — src/models/heston.py

Heston (1993) stochastic volatility model with characteristic-function pricing.

Implements:
- Characteristic function in the numerically stable "Little Heston Trap"
  formulation (Albrecher et al. 2007)
- European vanilla pricing via Gil-Pelaez Fourier inversion with adaptive
  quadrature (pointwise, robust)
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
"""

import warnings
from typing import Callable, Dict, Union

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

from .base import Numeric, PricingModel

# char_func evaluation points: complex arguments are part of the contract
# (Gil-Pelaez uses u - i, Carr-Madan uses u - (alpha+1)i)
ComplexNumeric = Union[float, complex, np.ndarray]

# Adaptive quadrature settings for Gil-Pelaez inversion. epsabs=1e-9 keeps
# the quadrature error two orders of magnitude below the 1e-6 cross-validation
# tolerance against Carr-Madan FFT and QuantLib. limit=400 accommodates the
# slowly-decaying, fast-oscillating integrands of very-low-vol short-maturity
# corners (effective support ~1/sqrt(v*T) with oscillation period 2*pi/ln(K)).
_QUAD_OPTS = {"epsabs": 1e-9, "epsrel": 1e-9, "limit": 400}


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

    def __init__(self, v0: float, kappa: float, theta: float,
                 xi: float, rho: float) -> None:
        params = {"v0": v0, "kappa": kappa, "theta": theta, "xi": xi, "rho": rho}
        for name, value in params.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if v0 <= 0:
            raise ValueError(f"v0 must be > 0, got {v0}. "
                             f"Note: v0 is a variance — for 20% vol use 0.04.")
        if kappa <= 0:
            raise ValueError(f"kappa must be > 0, got {kappa}")
        if theta <= 0:
            raise ValueError(f"theta must be > 0, got {theta}. "
                             f"Note: theta is a variance — for 20% vol use 0.04.")
        if xi <= 0:
            raise ValueError(f"xi must be > 0, got {xi}")
        if not -1.0 < rho < 1.0:
            raise ValueError(f"rho must be in (-1, 1), got {rho}")

        if 2.0 * kappa * theta < xi * xi:
            warnings.warn(
                f"Feller condition violated: 2*kappa*theta = {2*kappa*theta:.6f} "
                f"< xi^2 = {xi*xi:.6f}. The variance process can reach zero. "
                f"This is common in equity calibrations and handled correctly "
                f"by both the characteristic function and the QE simulation "
                f"scheme; informative, not an error.",
                UserWarning, stacklevel=2,
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

    def char_func(self, u: ComplexNumeric, T: float, r: float,
                  q: float = 0.0, S0: float = 1.0) -> complex | np.ndarray:
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
            self.kappa, self.theta_v, self.xi, self.rho_sv, self.v0,
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
        phi = np.where(u_arr == -1.0j,
                       S0 * np.exp((r - q) * T) + 0.0j, phi)
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
    # Gil-Pelaez probabilities
    # ──────────────────────────────────────────────

    def _p1_p2(self, S: float, K: float, T: float, r: float,
               q: float) -> tuple[float, float]:
        """
        Gil-Pelaez exercise probabilities P1 (share measure) and P2 (Q).

            P_j = 1/2 + (1/pi) int_0^inf Re[ e^{-iu ln K} phi_j(u)/(iu) ] du

        phi_2(u) = phi(u); phi_1(u) = phi(u - i)/phi(-i). The integrands have
        a removable singularity at u = 0 (the 1/(iu) pole is purely
        imaginary; the real part has a finite limit) — Gauss-Kronrod nodes
        never sit on the endpoint, so no special-casing is required.
        """
        log_K = np.log(K)
        # phi(-i) = forward = S e^{(r-q)T}; kept complex (Im ~ 1e-16 noise)
        phi_minus_i = self.char_func(-1j, T, r, q, S)

        def integrand_p2(u: float) -> float:
            val = np.exp(-1j * u * log_K) * self.char_func(u, T, r, q, S) / (1j * u)
            return float(val.real)

        def integrand_p1(u: float) -> float:
            val = (np.exp(-1j * u * log_K)
                   * self.char_func(u - 1j, T, r, q, S)
                   / (1j * u * phi_minus_i))
            return float(val.real)

        int_p1, _ = quad(integrand_p1, 0.0, np.inf, **_QUAD_OPTS)
        int_p2, _ = quad(integrand_p2, 0.0, np.inf, **_QUAD_OPTS)

        # Clip pure quadrature noise (~1e-12) outside [0, 1]; genuine errors
        # are caught by the price-bound and parity tests, not masked here.
        P1 = float(np.clip(0.5 + int_p1 / np.pi, 0.0, 1.0))
        P2 = float(np.clip(0.5 + int_p2 / np.pi, 0.0, 1.0))
        return P1, P2

    def _price_scalar(self, S: float, K: float, T: float, r: float,
                      opt: str, q: float) -> float:
        """Scalar Gil-Pelaez price. K = 0 handled without the log-K integral."""
        disc_q = np.exp(-q * T)
        disc_r = np.exp(-r * T)
        if K == 0.0:
            # A zero-strike call pays S_T: worth the prepaid forward. A
            # zero-strike put pays max(-S_T, 0) = 0.
            return float(S * disc_q) if opt == 'call' else 0.0

        P1, P2 = self._p1_p2(S, K, T, r, q)
        call = S * disc_q * P1 - K * disc_r * P2
        if opt == 'call':
            return float(max(call, 0.0))
        # Put-call parity: P = C - S e^{-qT} + K e^{-rT}
        return float(max(call - S * disc_q + K * disc_r, 0.0))

    # ──────────────────────────────────────────────
    # Price (ABC)
    # ──────────────────────────────────────────────

    def price(self, S: Numeric, K: Numeric, T: Numeric, r: Numeric,
              option_type: str = 'call', q: float = 0.0) -> Numeric:
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
        Array inputs are broadcast and evaluated elementwise: adaptive
        quadrature is inherently scalar, so vectorization cannot remove the
        per-element loop. For many strikes at a single maturity use
        ``price_surface()`` (Carr-Madan FFT — one transform prices the whole
        strike grid), which is what the calibrator does.
        """
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)
        return self._vectorize(
            lambda s, k, t, rr: self._price_scalar(s, k, t, rr, opt, q),
            S, K, T, r,
        )

    # ──────────────────────────────────────────────
    # Carr-Madan FFT surface
    # ──────────────────────────────────────────────

    def price_surface(self, S: float, strikes: Numeric, T: float, r: float,
                      q: float = 0.0, option_type: str = 'call', *,
                      alpha: float = 1.5, eta: float = 0.25,
                      n_fft: int = 4096) -> np.ndarray:
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
            raise ValueError("price_surface requires strikes > 0; "
                             "use price() for K = 0.")
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
        u = eta * np.arange(n_fft)
        lam = 2.0 * np.pi / (n_fft * eta)
        b = 0.5 * n_fft * lam
        k_grid = -b + lam * np.arange(n_fft)

        phi_vals = self.char_func(u - (alpha + 1.0) * 1j, T, r, q, 1.0)
        denominator = (alpha * alpha + alpha - u * u
                       + 1j * (2.0 * alpha + 1.0) * u)
        psi = np.exp(-r * T) * phi_vals / denominator

        # Simpson weights eta/3 * [1, 4, 2, 4, ..., 2, 4] (Carr-Madan eq. 24).
        # The final endpoint closure is irrelevant because psi(u_max) ~ 0.
        simpson = np.full(n_fft, eta / 3.0)
        simpson[1::2] *= 4.0
        simpson[2::2] *= 2.0

        fft_input = np.exp(1j * u * b) * psi * simpson
        fft_vals = np.fft.fft(fft_input)
        calls_grid = np.exp(-alpha * k_grid) / np.pi * fft_vals.real

        spline = CubicSpline(k_grid, calls_grid)
        k_req = np.log(strikes_arr / S)
        if np.any(k_req < k_grid[0]) or np.any(k_req > k_grid[-1]):
            raise ValueError(
                f"log-moneyness outside FFT grid [{k_grid[0]:.2f}, {k_grid[-1]:.2f}]; "
                f"increase n_fft or eta."
            )
        calls = S * np.asarray(spline(k_req), dtype=np.float64)

        if opt == 'put':
            # Parity on the normalized problem, rescaled by S
            prices = calls - S * np.exp(-q * T) + strikes_arr * np.exp(-r * T)
        else:
            prices = calls
        result: np.ndarray = np.maximum(prices, 0.0)
        return result

    # ──────────────────────────────────────────────
    # Greeks — delta/gamma exact via the CF, rest by central FD
    # ──────────────────────────────────────────────

    def _delta_scalar(self, S: float, K: float, T: float, r: float,
                      opt: str, q: float) -> float:
        """
        Exact delta. Heston coefficients do not depend on the spot level, so
        S_T = S * M_T with M_T independent of S. Differentiating
        C = e^{-rT} E[(S M_T - K)^+] under the integral sign:

            dC/dS = e^{-rT} E[M_T 1{S_T > K}] = e^{-qT} P1

        (the share-measure exercise probability). Put delta via parity.
        """
        if K == 0.0:
            return float(np.exp(-q * T)) if opt == 'call' else 0.0
        P1, _ = self._p1_p2(S, K, T, r, q)
        delta_call = np.exp(-q * T) * P1
        if opt == 'call':
            return float(delta_call)
        return float(delta_call - np.exp(-q * T))

    def _gamma_scalar(self, S: float, K: float, T: float, r: float,
                      q: float) -> float:
        """
        Exact gamma = e^{-qT} dP1/dS. Since phi_1 depends on S only through
        exp(iu ln S), term-by-term differentiation of the Gil-Pelaez integral
        gives dphi_1/dS = phi_1 * iu / S, the iu cancels the 1/(iu) pole, and

            gamma = e^{-qT}/(pi S) int_0^inf Re[ e^{-iu ln K} phi_1(u) ] du.

        Same for calls and puts (parity: the linear terms vanish).
        """
        if K == 0.0:
            return 0.0
        log_K = np.log(K)
        phi_minus_i = self.char_func(-1j, T, r, q, S)

        def integrand(u: float) -> float:
            val = (np.exp(-1j * u * log_K)
                   * self.char_func(u - 1j, T, r, q, S) / phi_minus_i)
            return float(val.real)

        integral, _ = quad(integrand, 0.0, np.inf, **_QUAD_OPTS)
        return float(np.exp(-q * T) / (np.pi * S) * integral)

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
        params = {"v0": self.v0, "kappa": self.kappa, "theta": self.theta_v,
                  "xi": self.xi, "rho": self.rho_sv}
        params.update(overrides)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return HestonModel(**params)

    def delta(self, S: Numeric, K: Numeric, T: Numeric, r: Numeric,
              option_type: str = 'call', q: float = 0.0) -> Numeric:
        """Delta = dV/dS = e^{-qT} P1 (exact, no bumping). Put via parity."""
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)
        return self._vectorize(
            lambda s, k, t, rr: self._delta_scalar(s, k, t, rr, opt, q),
            S, K, T, r,
        )

    def gamma(self, S: Numeric, K: Numeric, T: Numeric, r: Numeric,
              option_type: str = 'call', q: float = 0.0) -> Numeric:
        """Gamma = d2V/dS2, exact via the differentiated CF. Call = put."""
        self._validate_inputs(S, K, T, r)
        self._validate_option_type(option_type)
        return self._vectorize(
            lambda s, k, t, rr: self._gamma_scalar(s, k, t, rr, q),
            S, K, T, r,
        )

    def vega(self, S: Numeric, K: Numeric, T: Numeric, r: Numeric,
             option_type: str = 'call', q: float = 0.0) -> Numeric:
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
            up.price(S, K, T, r, opt, q), dn.price(S, K, T, r, opt, q), h,
        )

    def theta(self, S: Numeric, K: Numeric, T: Numeric, r: Numeric,
              option_type: str = 'call', q: float = 0.0) -> Numeric:
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
            self.price(S, K, T_arr + h, r, opt, q), h,
        )

    def rho(self, S: Numeric, K: Numeric, T: Numeric, r: Numeric,
            option_type: str = 'call', q: float = 0.0) -> Numeric:
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
            self.price(S, K, T, r_arr - h, opt, q), h,
        )

    def greeks(self, S: Numeric, K: Numeric, T: Numeric, r: Numeric,
               option_type: str = 'call', q: float = 0.0) -> Dict[str, Numeric]:
        """
        All ABC Greeks in one dict: price, delta, gamma, vega, theta, rho.

        delta/gamma are exact (differentiated characteristic function);
        vega/theta/rho are central finite differences on the deterministic
        Fourier price. For Heston parameter sensitivities see
        ``model_greeks()``.
        """
        self._validate_inputs(S, K, T, r)
        opt = self._validate_option_type(option_type)
        return {
            'price': self.price(S, K, T, r, opt, q),
            'delta': self.delta(S, K, T, r, opt, q),
            'gamma': self.gamma(S, K, T, r, opt, q),
            'vega': self.vega(S, K, T, r, opt, q),
            'theta': self.theta(S, K, T, r, opt, q),
            'rho': self.rho(S, K, T, r, opt, q),
        }

    def model_greeks(self, S: Numeric, K: Numeric, T: Numeric, r: Numeric,
                     option_type: str = 'call', q: float = 0.0,
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
                up.price(S, K, T, r, opt, q), dn.price(S, K, T, r, opt, q), h,
            )

        # Relative bumps with a floor; rho clamped so both bumps stay in (-1, 1)
        h_rho = min(1e-3 * max(abs(self.rho_sv), 0.1),
                    0.5 * (1.0 - abs(self.rho_sv)))
        return {
            'v0': central_fd('v0', self.v0, 1e-3 * self.v0),
            'kappa': central_fd('kappa', self.kappa, 1e-3 * self.kappa),
            'theta': central_fd('theta', self.theta_v, 1e-3 * self.theta_v),
            'xi': central_fd('xi', self.xi, 1e-3 * self.xi),
            'rho': central_fd('rho', self.rho_sv, h_rho),
        }

    # ──────────────────────────────────────────────
    # Dunder methods
    # ──────────────────────────────────────────────

    def __repr__(self) -> str:
        return (f"HestonModel(v0={self.v0}, kappa={self.kappa}, "
                f"theta={self.theta_v}, xi={self.xi}, rho={self.rho_sv})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HestonModel):
            return NotImplemented
        return (self.v0, self.kappa, self.theta_v, self.xi, self.rho_sv) == \
               (other.v0, other.kappa, other.theta_v, other.xi, other.rho_sv)

    def __hash__(self) -> int:
        return hash((self.v0, self.kappa, self.theta_v, self.xi, self.rho_sv))
