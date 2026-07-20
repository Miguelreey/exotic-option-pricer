"""
Exotic Option Pricer — tests/test_heston.py

Phase 4 test suite: Heston stochastic volatility model.

Covers:
- Parameter validation, Feller warning, dunder methods
- Characteristic function anchors: phi(0) = 1, phi(-i) = forward,
  conjugate symmetry, |phi| <= 1
- Black-Scholes limit (xi -> 0, v0 = theta): cross-validation with Phase 1
- Put-call parity and no-arbitrage price bounds
- Gil-Pelaez quadrature vs Carr-Madan FFT (two independent integration
  methods on the same characteristic function)
- External cross-validation: 90 reference prices generated with
  QuantLib 1.42.1 AnalyticHestonEngine (independent implementation),
  including a long-maturity stress set that breaks the original (1993)
  characteristic-function formulation via the complex-log branch cut
- Greeks: exact delta/gamma vs finite differences, BS-limit Greeks,
  model_greeks coherence (chain rule vega <-> dV/dv0)
- QE simulation: exact CIR moments, martingale property (corrected and
  plain), Feller-violated mass at zero, reproducibility, Fourier vs MC
- Phase 3 exotics priced on Heston paths without modification
- Implied volatility skew/smile (rho < 0 -> downward skew)
- Property-based tests (Hypothesis)

References
----------
.. [1] Heston (1993). Review of Financial Studies 6(2), 327-343.
.. [2] Albrecher et al. (2007). The Little Heston Trap. Wilmott.
.. [3] Andersen (2008). J. Computational Finance 11(3), 1-42.
.. [4] Carr & Madan (1999). J. Computational Finance 2(4), 61-73.
"""

import warnings

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from exotic_option_pricer.calibration.heston_calibrator import (
    HestonCalibrator,
    filter_option_quotes,
)
from exotic_option_pricer.calibration.market_data import (
    _implied_dividend_yield,
    _select_expiries,
)
from exotic_option_pricer.engines.monte_carlo import MonteCarloEngine
from exotic_option_pricer.instruments.asian import AsianOption
from exotic_option_pricer.instruments.barrier import BarrierOption
from exotic_option_pricer.models.black_scholes import BlackScholesModel
from exotic_option_pricer.models.heston import HestonModel

# ──────────────────────────────────────────────
# Shared parameters and fixtures
# ──────────────────────────────────────────────

S0 = 100.0

# Equity-style set (Feller violated: 2*2*0.04 = 0.16 < 0.25 = xi^2)
PARAMS_A = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
R_A, Q_A = 0.05, 0.02


def make_model(**kwargs) -> HestonModel:
    """Construct a HestonModel suppressing the (expected) Feller warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return HestonModel(**kwargs)


@pytest.fixture(scope="module")
def model_a() -> HestonModel:
    return make_model(**PARAMS_A)


@pytest.fixture(scope="module")
def qe_paths_a() -> tuple[MonteCarloEngine, np.ndarray, np.ndarray]:
    """
    Shared QE simulation under PARAMS_A: 400k paths, 100 steps, T = 1.

    Module-scoped so the martingale, CIR-moment, exotics and smile tests
    amortize one ~5 s simulation instead of re-running it each time
    (desk-style path reuse, same rationale as price_batch).
    """
    engine = MonteCarloEngine(n_paths=400_000, seed=42)
    paths, v_paths = engine.simulate_heston(
        S0,
        PARAMS_A["v0"],
        1.0,
        R_A,
        kappa=PARAMS_A["kappa"],
        theta=PARAMS_A["theta"],
        xi=PARAMS_A["xi"],
        rho=PARAMS_A["rho"],
        q=Q_A,
        n_steps=100,
        return_variance=True,
    )
    return engine, paths, v_paths


# ──────────────────────────────────────────────
# Validation and construction
# ──────────────────────────────────────────────


class TestHestonValidation:
    def test_valid_construction(self):
        m = HestonModel(v0=0.04, kappa=2.0, theta=0.05, xi=0.3, rho=-0.5)
        assert m.v0 == 0.04
        assert m.kappa == 2.0
        assert m.theta_v == 0.05
        assert m.xi == 0.3
        assert m.rho_sv == -0.5

    @pytest.mark.parametrize("bad", [0.0, -0.04, np.nan, np.inf])
    def test_invalid_v0(self, bad):
        with pytest.raises(ValueError):
            HestonModel(v0=bad, kappa=2.0, theta=0.04, xi=0.3, rho=-0.5)

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
    def test_invalid_kappa(self, bad):
        with pytest.raises(ValueError):
            HestonModel(v0=0.04, kappa=bad, theta=0.04, xi=0.3, rho=-0.5)

    @pytest.mark.parametrize("bad", [0.0, -0.04, np.inf])
    def test_invalid_theta(self, bad):
        with pytest.raises(ValueError):
            HestonModel(v0=0.04, kappa=2.0, theta=bad, xi=0.3, rho=-0.5)

    @pytest.mark.parametrize("bad", [0.0, -0.3, np.nan])
    def test_invalid_xi(self, bad):
        with pytest.raises(ValueError):
            HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=bad, rho=-0.5)

    @pytest.mark.parametrize("bad", [-1.0, 1.0, -1.5, 2.0, np.nan])
    def test_invalid_rho(self, bad):
        with pytest.raises(ValueError):
            HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=bad)

    def test_feller_violation_warns(self):
        # 2*kappa*theta = 0.16 < xi^2 = 0.25
        with pytest.warns(UserWarning, match="Feller"):
            HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)

    def test_feller_satisfied_no_warning(self):
        # 2*3*0.06 = 0.36 > 0.16 = xi^2
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            HestonModel(v0=0.09, kappa=3.0, theta=0.06, xi=0.4, rho=-0.5)

    def test_invalid_option_type(self, model_a):
        with pytest.raises(ValueError, match="option_type"):
            model_a.price(100, 100, 1.0, 0.05, "straddle")

    def test_invalid_market_inputs(self, model_a):
        with pytest.raises(ValueError):
            model_a.price(-100, 100, 1.0, 0.05)
        with pytest.raises(ValueError):
            model_a.price(100, -5, 1.0, 0.05)
        with pytest.raises(ValueError):
            model_a.price(100, 100, 0.0, 0.05)
        with pytest.raises(ValueError):
            model_a.price(np.inf, 100, 1.0, 0.05)

    def test_repr_eq_hash(self):
        m1 = make_model(**PARAMS_A)
        m2 = make_model(**PARAMS_A)
        m3 = make_model(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.6)
        assert m1 == m2
        assert m1 != m3
        assert m1 != "not a model"
        assert hash(m1) == hash(m2)
        assert "HestonModel" in repr(m1)
        assert "0.04" in repr(m1)


# ──────────────────────────────────────────────
# Characteristic function
# ──────────────────────────────────────────────


class TestCharacteristicFunction:
    def test_phi_zero_is_one(self, model_a):
        phi0 = model_a.char_func(0.0, T=1.0, r=R_A, q=Q_A, S0=S0)
        assert abs(phi0 - 1.0) < 1e-12

    @pytest.mark.parametrize("T", [0.1, 1.0, 5.0, 15.0])
    def test_phi_minus_i_is_forward(self, model_a, T):
        # phi(-i) = E[S_T] = S0 exp((r-q)T): the Q-martingale anchor.
        # T = 15 stresses the branch-cut region of the original formulation.
        phi = model_a.char_func(-1j, T=T, r=R_A, q=Q_A, S0=S0)
        forward = S0 * np.exp((R_A - Q_A) * T)
        assert abs(phi - forward) < 1e-10 * forward

    def test_conjugate_symmetry(self, model_a):
        # phi(-u) = conj(phi(u)) for real u (CF of a real random variable)
        u = np.array([0.5, 1.0, 3.7, 10.0, 25.0])
        phi_pos = model_a.char_func(u, T=2.0, r=R_A, q=Q_A, S0=S0)
        phi_neg = model_a.char_func(-u, T=2.0, r=R_A, q=Q_A, S0=S0)
        np.testing.assert_allclose(phi_neg, np.conj(phi_pos), rtol=1e-12)

    def test_modulus_bounded_by_one(self, model_a):
        u = np.linspace(0.1, 60.0, 200)
        phi = model_a.char_func(u, T=1.0, r=R_A, q=Q_A, S0=S0)
        assert np.all(np.abs(phi) <= 1.0 + 1e-12)

    def test_vectorized_matches_scalar(self, model_a):
        u = np.array([0.3, 1.7, 8.0])
        vec = model_a.char_func(u, T=1.0, r=R_A, q=Q_A, S0=S0)
        scalars = [model_a.char_func(float(x), T=1.0, r=R_A, q=Q_A, S0=S0) for x in u]
        np.testing.assert_allclose(vec, scalars, rtol=1e-14)

    def test_scalar_returns_python_complex(self, model_a):
        phi = model_a.char_func(1.0, T=1.0, r=R_A)
        assert isinstance(phi, complex)


# ──────────────────────────────────────────────
# Black-Scholes limit (cross-validation with Phase 1)
# ──────────────────────────────────────────────


class TestBlackScholesLimit:
    """
    With xi -> 0 the variance is deterministic, v_t -> theta exponentially;
    if additionally v0 = theta = sigma^2, the variance is constant and
    Heston degenerates to Black-Scholes exactly. xi = 1e-4 leaves a
    residual O(xi^2) ~ 1e-8, so 1e-6 is a rigorous tolerance.
    """

    @pytest.fixture(scope="class")
    def degenerate(self) -> HestonModel:
        return make_model(v0=0.04, kappa=2.0, theta=0.04, xi=1e-4, rho=0.0)

    @pytest.fixture(scope="class")
    def bs(self) -> BlackScholesModel:
        return BlackScholesModel(sigma=0.20)

    @pytest.mark.parametrize(
        "K,T", [(100.0, 1.0), (90.0, 0.5), (110.0, 2.0), (100.0, 0.1), (60.0, 1.0), (150.0, 1.0)]
    )
    @pytest.mark.parametrize("opt", ["call", "put"])
    def test_price_matches_bs(self, degenerate, bs, K, T, opt):
        heston_p = degenerate.price(S0, K, T, 0.05, opt, q=0.02)
        bs_p = bs.price(S0, K, T, 0.05, opt, q=0.02)
        assert abs(heston_p - bs_p) < 1e-6

    def test_benchmark_value(self, degenerate):
        # BS(100, 100, 1, 5%, sigma=20%) = 10.4506 (Hull benchmark)
        assert abs(degenerate.price(100, 100, 1.0, 0.05, "call") - 10.4506) < 1e-3

    def test_delta_gamma_match_bs(self, degenerate, bs):
        d_h = degenerate.delta(S0, 100, 1.0, 0.05, "call", q=0.02)
        d_b = bs.delta(S0, 100, 1.0, 0.05, "call", q=0.02)
        assert abs(d_h - d_b) < 1e-6
        g_h = degenerate.gamma(S0, 100, 1.0, 0.05, "call", q=0.02)
        g_b = bs.gamma(S0, 100, 1.0, 0.05, "call", q=0.02)
        assert abs(g_h - g_b) < 1e-6

    def test_theta_rho_match_bs(self, degenerate, bs):
        # With v0 = theta the effective vol is T-independent, so the
        # calendar Greek matches BS; rho (rate) matches as well.
        t_h = degenerate.theta(S0, 100, 1.0, 0.05, "call", q=0.02)
        t_b = bs.theta(S0, 100, 1.0, 0.05, "call", q=0.02)
        assert abs(t_h - t_b) < 1e-4
        r_h = degenerate.rho(S0, 100, 1.0, 0.05, "call", q=0.02)
        r_b = bs.rho(S0, 100, 1.0, 0.05, "call", q=0.02)
        assert abs(r_h - r_b) < 1e-4

    def test_vega_matches_bs_scaled(self, degenerate, bs):
        """
        Heston vega is dV/d(sqrt(v0)) with theta held FIXED. In the
        deterministic limit V = BS(sigma_eff), sigma_eff^2 =
        theta + (v0-theta) w with w = (1 - e^{-kappa T})/(kappa T), so at
        v0 = theta: dV/d(sqrt(v0)) = BS_vega * w — NOT the full BS vega
        (which bumps the entire vol curve, not just its short end).
        """
        kappa, T = 2.0, 1.0
        w = (1.0 - np.exp(-kappa * T)) / (kappa * T)
        v_h = degenerate.vega(S0, 100, T, 0.05, "call", q=0.02)
        v_b = bs.vega(S0, 100, T, 0.05, "call", q=0.02)
        assert abs(v_h - v_b * w) < 1e-3 * v_b


# ──────────────────────────────────────────────
# Parity and bounds
# ──────────────────────────────────────────────

PARITY_SETS = [
    dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7),
    dict(v0=0.09, kappa=0.5, theta=0.06, xi=1.0, rho=-0.9),
    dict(v0=0.02, kappa=5.0, theta=0.03, xi=0.3, rho=0.4),
]


class TestParityAndBounds:
    @pytest.mark.parametrize("params", PARITY_SETS)
    def test_put_call_parity(self, params):
        """
        C - P = S e^{-qT} - K e^{-rT}.

        The put is constructed from parity internally, so this is a
        regression contract (protects against a future reimplementation
        breaking it), not an independent check — independence comes from
        the QuantLib benchmark below, where calls AND puts are pinned
        against an external engine.
        """
        m = make_model(**params)
        c = m.price(S0, 105, 1.5, 0.03, "call", q=0.01)
        p = m.price(S0, 105, 1.5, 0.03, "put", q=0.01)
        rhs = S0 * np.exp(-0.01 * 1.5) - 105 * np.exp(-0.03 * 1.5)
        assert abs((c - p) - rhs) < 1e-8

    def test_call_within_no_arbitrage_bounds(self, model_a):
        for K in [60.0, 100.0, 160.0]:
            c = model_a.price(S0, K, 1.0, R_A, "call", q=Q_A)
            lower = max(S0 * np.exp(-Q_A) - K * np.exp(-R_A), 0.0)
            upper = S0 * np.exp(-Q_A)
            assert lower - 1e-9 <= c <= upper + 1e-9

    def test_call_decreasing_in_strike(self, model_a):
        strikes = np.array([70.0, 85.0, 100.0, 115.0, 130.0])
        prices = model_a.price(S0, strikes, 1.0, R_A, "call", q=Q_A)
        assert np.all(np.diff(prices) < 0)

    def test_call_convex_in_strike(self, model_a):
        strikes = np.linspace(60.0, 140.0, 17)
        prices = model_a.price(S0, strikes, 1.0, R_A, "call", q=Q_A)
        second_diff = np.diff(prices, 2)
        # Butterfly spreads have nonnegative value (risk-neutral density >= 0)
        assert np.all(second_diff > -1e-8)

    def test_zero_strike_call_is_prepaid_forward(self, model_a):
        c = model_a.price(S0, 0.0, 1.0, R_A, "call", q=Q_A)
        assert abs(c - S0 * np.exp(-Q_A)) < 1e-12
        p = model_a.price(S0, 0.0, 1.0, R_A, "put", q=Q_A)
        assert p == 0.0

    def test_vectorized_price_matches_scalar(self, model_a):
        strikes = np.array([90.0, 100.0, 110.0])
        vec = model_a.price(S0, strikes, 1.0, R_A, "call", q=Q_A)
        scalars = [model_a.price(S0, float(k), 1.0, R_A, "call", q=Q_A) for k in strikes]
        np.testing.assert_allclose(vec, scalars, rtol=0, atol=1e-14)
        assert vec.shape == (3,)


# ──────────────────────────────────────────────
# Gil-Pelaez quadrature vs Carr-Madan FFT
# ──────────────────────────────────────────────


class TestFourierVsFFT:
    STRIKES = np.array([70.0, 80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 140.0])

    @pytest.mark.parametrize("T", [0.2, 1.0, 5.0])
    def test_call_surface_matches_quadrature(self, model_a, T):
        fft_p = model_a.price_surface(S0, self.STRIKES, T, R_A, q=Q_A)
        quad_p = model_a.price(S0, self.STRIKES, T, R_A, "call", q=Q_A)
        np.testing.assert_allclose(fft_p, quad_p, rtol=0, atol=1e-6)

    def test_put_surface_matches_quadrature(self, model_a):
        fft_p = model_a.price_surface(S0, self.STRIKES, 1.0, R_A, q=Q_A, option_type="put")
        quad_p = model_a.price(S0, self.STRIKES, 1.0, R_A, "put", q=Q_A)
        np.testing.assert_allclose(fft_p, quad_p, rtol=0, atol=1e-6)

    def test_surface_validations(self, model_a):
        with pytest.raises(ValueError, match="strikes > 0"):
            model_a.price_surface(S0, np.array([0.0, 100.0]), 1.0, R_A)
        with pytest.raises(ValueError, match="power of 2"):
            model_a.price_surface(S0, np.array([100.0]), 1.0, R_A, n_fft=1000)
        with pytest.raises(ValueError, match="alpha"):
            model_a.price_surface(S0, np.array([100.0]), 1.0, R_A, alpha=-1.0)
        with pytest.raises(ValueError, match="eta"):
            model_a.price_surface(S0, np.array([100.0]), 1.0, R_A, eta=0.0)
        with pytest.raises(ValueError, match="log-moneyness outside"):
            # ln(1e-9/100) = -25 sits far left of the FFT grid (~[-12.9, 12.9])
            model_a.price_surface(S0, np.array([1.0e-9]), 1.0, R_A)

    def test_moment_explosion_guard(self):
        """
        For strongly positive rho the moment E[S_T^(alpha+1)] explodes in
        finite time (Andersen-Piterbarg 2007); the guard must raise rather
        than silently return the spurious analytic continuation. A
        finiteness check would NOT catch this — the continued formula
        stays finite past T*.
        """
        m = make_model(v0=0.09, kappa=0.3, theta=0.09, xi=1.5, rho=0.9)
        with pytest.raises(ValueError, match="diverges|explosion"):
            m.price_surface(S0, np.array([100.0]), 10.0, 0.02, alpha=5.0)

    def test_moment_explosion_threshold(self):
        """
        Same parameters, sharp threshold: p = 6, a = kappa - rho*xi*p =
        -7.8, d^2 = a^2 - xi^2 p(p-1) = -6.66 < 0, so delta = 2.5807,
        psi = -2 atan2(delta, a) = -5.6442, and
        T* = (psi mod 2pi)/delta = 0.6390/2.5807 = 0.2476.
        """
        m = make_model(v0=0.09, kappa=0.3, theta=0.09, xi=1.5, rho=0.9)
        t_star = m._moment_explosion_time(6.0)
        assert abs(t_star - 0.2476) < 1e-3
        # Below T*: prices fine; above: guard raises
        prices = m.price_surface(S0, np.array([100.0]), 0.9 * t_star, 0.02, alpha=5.0)
        assert np.isfinite(prices).all() and prices[0] > 0
        with pytest.raises(ValueError, match="diverges|explosion"):
            m.price_surface(S0, np.array([100.0]), 1.1 * t_star, 0.02, alpha=5.0)

    def test_moment_no_explosion_equity_params(self, model_a):
        # Equity-style rho < 0 with alpha = 1.5: a = 2.875 > 0,
        # d^2 = 7.33 > 0 -> the moment never explodes.
        assert model_a._moment_explosion_time(2.5) == float("inf")


# ──────────────────────────────────────────────
# QuantLib cross-validation (external reference)
# ──────────────────────────────────────────────

# Reference prices generated once with QuantLib 1.42.1
# AnalyticHestonEngine (relative tolerance 1e-12, max 100k evaluations),
# flat curves, Actual365Fixed with integer-day maturities so the year
# fraction is exact. Generator script documented in the Phase 4 spec note.
QL_PARAM_SETS = {
    "A_equity_feller_violated": dict(
        v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, r=0.05, q=0.02
    ),
    "B_feller_ok": dict(v0=0.09, kappa=3.0, theta=0.06, xi=0.4, rho=-0.5, r=0.03, q=0.0),
    "C_stress_long_T": dict(v0=0.02, kappa=0.5, theta=0.08, xi=1.0, rho=-0.9, r=0.02, q=0.01),
}

QL_REFERENCE_PRICES = {
    ("A_equity_feller_violated", 70.0, 0.2, "call"): 30.3093984373,
    ("A_equity_feller_violated", 70.0, 0.2, "put"): 0.0120878653,
    ("A_equity_feller_violated", 90.0, 0.2, "call"): 11.2027149559,
    ("A_equity_feller_violated", 90.0, 0.2, "put"): 0.7064010590,
    ("A_equity_feller_violated", 100.0, 0.2, "call"): 3.7306200944,
    ("A_equity_feller_violated", 100.0, 0.2, "put"): 3.1348045350,
    ("A_equity_feller_violated", 110.0, 0.2, "call"): 0.3613244326,
    ("A_equity_feller_violated", 110.0, 0.2, "put"): 9.6660072106,
    ("A_equity_feller_violated", 140.0, 0.2, "call"): 0.0000027764,
    ("A_equity_feller_violated", 140.0, 0.2, "put"): 39.0061805669,
    ("A_equity_feller_violated", 70.0, 1.0, "call"): 32.0885935066,
    ("A_equity_feller_violated", 70.0, 1.0, "put"): 0.6547858910,
    ("A_equity_feller_violated", 90.0, 1.0, "call"): 15.4594302381,
    ("A_equity_feller_violated", 90.0, 1.0, "put"): 3.0502111125,
    ("A_equity_feller_violated", 100.0, 1.0, "call"): 8.7525962800,
    ("A_equity_feller_violated", 100.0, 1.0, "put"): 5.8556713994,
    ("A_equity_feller_violated", 110.0, 1.0, "call"): 3.8996854291,
    ("A_equity_feller_violated", 110.0, 1.0, "put"): 10.5150547935,
    ("A_equity_feller_violated", 140.0, 1.0, "call"): 0.0753403259,
    ("A_equity_feller_violated", 140.0, 1.0, "put"): 35.2275924253,
    ("A_equity_feller_violated", 70.0, 5.0, "call"): 38.6547703260,
    ("A_equity_feller_violated", 70.0, 5.0, "put"): 2.6870833374,
    ("A_equity_feller_violated", 90.0, 5.0, "call"): 26.7900545661,
    ("A_equity_feller_violated", 90.0, 5.0, "put"): 6.3983832390,
    ("A_equity_feller_violated", 100.0, 5.0, "call"): 21.7135969828,
    ("A_equity_feller_violated", 100.0, 5.0, "put"): 9.1099334863,
    ("A_equity_feller_violated", 110.0, 5.0, "call"): 17.2620360180,
    ("A_equity_feller_violated", 110.0, 5.0, "put"): 12.4463803523,
    ("A_equity_feller_violated", 140.0, 5.0, "call"): 7.6630171236,
    ("A_equity_feller_violated", 140.0, 5.0, "put"): 26.2113849500,
    ("B_feller_ok", 70.0, 0.2, "call"): 30.4539899148,
    ("B_feller_ok", 70.0, 0.2, "put"): 0.0352473986,
    ("B_feller_ok", 90.0, 0.2, "call"): 11.9867937028,
    ("B_feller_ok", 90.0, 0.2, "put"): 1.4484104677,
    ("B_feller_ok", 100.0, 0.2, "call"): 5.3520629581,
    ("B_feller_ok", 100.0, 0.2, "put"): 4.7538593635,
    ("B_feller_ok", 110.0, 0.2, "call"): 1.6912794269,
    ("B_feller_ok", 110.0, 0.2, "put"): 11.0332554728,
    ("B_feller_ok", 140.0, 0.2, "call"): 0.0067427176,
    ("B_feller_ok", 140.0, 0.2, "put"): 39.1692576851,
    ("B_feller_ok", 70.0, 1.0, "call"): 33.0975609555,
    ("B_feller_ok", 70.0, 1.0, "put"): 1.0287483039,
    ("B_feller_ok", 90.0, 1.0, "call"): 17.5452581772,
    ("B_feller_ok", 90.0, 1.0, "put"): 4.8853561965,
    ("B_feller_ok", 100.0, 1.0, "call"): 11.6656232263,
    ("B_feller_ok", 100.0, 1.0, "put"): 8.7101765811,
    ("B_feller_ok", 110.0, 1.0, "call"): 7.2586963999,
    ("B_feller_ok", 110.0, 1.0, "put"): 14.0077050902,
    ("B_feller_ok", 140.0, 1.0, "call"): 1.2064132857,
    ("B_feller_ok", 140.0, 1.0, "put"): 37.0687879824,
    ("B_feller_ok", 70.0, 5.0, "call"): 44.2277989467,
    ("B_feller_ok", 70.0, 5.0, "put"): 4.4773572965,
    ("B_feller_ok", 90.0, 5.0, "call"): 32.7148064293,
    ("B_feller_ok", 90.0, 5.0, "put"): 10.1785243076,
    ("B_feller_ok", 100.0, 5.0, "call"): 27.9254455442,
    ("B_feller_ok", 100.0, 5.0, "put"): 13.9962431867,
    ("B_feller_ok", 110.0, 5.0, "call"): 23.7413607403,
    ("B_feller_ok", 110.0, 5.0, "put"): 18.4192381470,
    ("B_feller_ok", 140.0, 5.0, "call"): 14.3279644713,
    ("B_feller_ok", 140.0, 5.0, "put"): 34.8270811708,
    ("C_stress_long_T", 70.0, 0.2, "call"): 30.1231847590,
    ("C_stress_long_T", 70.0, 0.2, "put"): 0.0435441464,
    ("C_stress_long_T", 90.0, 0.2, "call"): 10.7496476348,
    ("C_stress_long_T", 90.0, 0.2, "put"): 0.5901668090,
    ("C_stress_long_T", 100.0, 0.2, "call"): 2.0973920144,
    ("C_stress_long_T", 100.0, 0.2, "put"): 1.8979910821,
    ("C_stress_long_T", 110.0, 0.2, "call"): 0.0043150678,
    ("C_stress_long_T", 110.0, 0.2, "put"): 9.7649940289,
    ("C_stress_long_T", 140.0, 0.2, "put"): 39.6409186428,
    ("C_stress_long_T", 70.0, 1.0, "call"): 31.3227293143,
    ("C_stress_long_T", 70.0, 1.0, "put"): 0.9316530709,
    ("C_stress_long_T", 90.0, 1.0, "call"): 13.1418666405,
    ("C_stress_long_T", 90.0, 1.0, "put"): 2.3547638632,
    ("C_stress_long_T", 100.0, 1.0, "call"): 4.8918465654,
    ("C_stress_long_T", 100.0, 1.0, "put"): 3.9067305211,
    ("C_stress_long_T", 110.0, 1.0, "call"): 0.3157077988,
    ("C_stress_long_T", 110.0, 1.0, "put"): 9.1325784876,
    ("C_stress_long_T", 140.0, 1.0, "call"): 0.0016549794,
    ("C_stress_long_T", 140.0, 1.0, "put"): 38.2244858674,
    ("C_stress_long_T", 70.0, 5.0, "call"): 36.1215486595,
    ("C_stress_long_T", 70.0, 5.0, "put"): 4.3372254720,
    ("C_stress_long_T", 90.0, 5.0, "call"): 21.2987682653,
    ("C_stress_long_T", 90.0, 5.0, "put"): 7.6111934385,
    ("C_stress_long_T", 100.0, 5.0, "call"): 14.5811215203,
    ("C_stress_long_T", 100.0, 5.0, "put"): 9.9419208738,
    ("C_stress_long_T", 110.0, 5.0, "call"): 8.6293807548,
    ("C_stress_long_T", 110.0, 5.0, "put"): 13.0385542886,
    ("C_stress_long_T", 140.0, 5.0, "call"): 0.3834264791,
    ("C_stress_long_T", 140.0, 5.0, "put"): 31.9377225540,
}


class TestQuantLibBenchmark:
    """
    External cross-validation: an independent C++ implementation
    (QuantLib AnalyticHestonEngine) agrees to < 2e-8 across 89 contracts.
    A common bug in char_func/quadrature CANNOT hide here.

    The deep-OTM (140, 0.2, call) case of set C (price 1.3e-9) is excluded:
    at that magnitude both engines return quadrature noise and the
    comparison is meaningless in absolute terms; its put is pinned instead
    (parity covers the call).
    """

    _models = {
        name: make_model(**{k: v for k, v in p.items() if k not in ("r", "q")})
        for name, p in QL_PARAM_SETS.items()
    }

    @pytest.mark.parametrize(
        "key",
        sorted(QL_REFERENCE_PRICES),
        ids=lambda k: f"{k[0]}-K{k[1]:.0f}-T{k[2]}-{k[3]}",
    )
    def test_matches_quantlib(self, key):
        set_name, K, T, opt = key
        params = QL_PARAM_SETS[set_name]
        model = self._models[set_name]
        own = model.price(S0, K, T, params["r"], opt, q=params["q"])
        assert abs(own - QL_REFERENCE_PRICES[key]) < 1e-6


# ──────────────────────────────────────────────
# Control-variated Gil-Pelaez quadrature
# ──────────────────────────────────────────────


class TestControlVariateQuadrature:
    """
    The Gil-Pelaez path integrates phi_j - phi_j^BS (matched total
    variance) on a composite Gauss-Legendre grid and adds back the exact
    N(d_j); the adaptive-quad fallback shares the same control-variated
    integrand. These tests pin the two paths against each other and the
    corners that used to exhaust the adaptive subdivision limit.
    """

    @pytest.mark.parametrize("params", PARITY_SETS)
    @pytest.mark.parametrize("K,T", [(70.0, 0.2), (100.0, 1.0), (140.0, 5.0), (95.0, 0.05)])
    def test_grid_matches_quad_fallback(self, params, K, T, monkeypatch):
        """Same integrand, two independent quadratures: GL grid vs quad."""
        m = make_model(**params)
        p_grid = m.price(S0, K, T, 0.03, "call", q=0.01)
        monkeypatch.setattr(HestonModel, "_fourier_grid", lambda self, *a, **kw: None)
        p_quad = m.price(S0, K, T, 0.03, "call", q=0.01)
        assert abs(p_grid - p_quad) < 1e-8

    def test_gamma_grid_matches_quad_fallback(self, model_a, monkeypatch):
        """Gamma's fallback path pinned against its grid path too."""
        g_grid = model_a.gamma(S0, 105.0, 1.0, R_A, "call", q=Q_A)
        monkeypatch.setattr(HestonModel, "_fourier_grid", lambda self, *a, **kw: None)
        g_quad = model_a.gamma(S0, 105.0, 1.0, R_A, "call", q=Q_A)
        assert abs(g_grid - g_quad) < 1e-9

    def test_grid_path_engages_for_standard_parameters(self):
        """
        Fast-path regression guard: if a future change made _fourier_grid
        decline ordinary parameters, every price would silently take the
        ~15x slower adaptive fallback with all accuracy tests still green.
        Pin that the grid engages across the QuantLib anchor sets.
        """
        for name, p in QL_PARAM_SETS.items():
            m = make_model(**{k: v for k, v in p.items() if k not in ("r", "q")})
            for T in (0.2, 1.0, 5.0):
                w = m._cv_total_variance(T)
                for K in (70.0, 100.0, 140.0):
                    x = float(np.log(S0 / K) + (p["r"] - p["q"]) * T)
                    spec = m._fourier_grid(S0, T, p["r"], p["q"], x, w, pole=True)
                    assert spec is not None, f"{name}, K={K}, T={T} fell off the grid path"

    def test_fallback_engages_beyond_panel_budget(self):
        """
        Parameters engineered so the oscillation x support product exceeds
        the panel budget (w ~ 2e-5 gives a very slow tail, deep strike a
        fast phase): _fourier_grid must decline (None) and price() must
        still return a sane value via the adaptive fallback plus the
        no-arbitrage projection, with parity exact. These parameters sit
        outside the Hypothesis strategy ranges, so this is the only test
        exercising the budget escape hatch naturally.
        """
        m = make_model(v0=0.001, kappa=0.1, theta=0.001, xi=2.0, rho=0.0)
        K, T = 40.0, 0.02
        x = float(np.log(S0 / K))
        w = m._cv_total_variance(T)
        assert m._fourier_grid(S0, T, 0.0, 0.0, x, w, pole=True) is None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # quad may legitimately warn here
            c = m.price(S0, K, T, 0.0, "call")
            p = m.price(S0, K, T, 0.0, "put")
        assert abs((c - p) - (S0 - K)) < 1e-10
        assert S0 - K - 1e-7 <= c <= S0

    def test_resolution_check_reports_nonconvergence(self, model_a):
        """
        A never-agreeing integrand must come back converged=False (the
        escape hatch to quad), not as a silently accepted wrong number.
        sin(1e7 u) aliases on every affordable resolution.
        """
        _, ok = model_a._checked_gl_integrals(lambda u: (np.sin(1.0e7 * u),), 50.0, 16)
        assert not ok

    def test_ladder_exhaustion_returns_none(self, model_a, monkeypatch):
        """
        With an unsatisfiable tail threshold the envelope ladder must give
        up and hand the option to the adaptive fallback instead of looping
        forever or returning a truncated grid.
        """
        from exotic_option_pricer.models import heston as heston_mod

        monkeypatch.setattr(heston_mod, "_TAIL_EPS", 0.0)
        # -log(0) -> inf is intentional here: the analytic estimates blow
        # up, the envelope can never beat a zero threshold, and the ladder
        # must exhaust deterministically.
        with np.errstate(divide="ignore"):
            assert model_a._fourier_grid(S0, 1.0, R_A, Q_A, 0.0, 0.04, pole=True) is None

    def test_zero_strike_greeks(self, model_a):
        """K = 0: call delta = e^{-qT} (prepaid forward), put delta = 0,
        gamma = 0 — handled without the log-K integral."""
        assert model_a.delta(S0, 0.0, 1.0, R_A, "call", q=Q_A) == pytest.approx(
            np.exp(-Q_A), abs=1e-15
        )
        assert model_a.delta(S0, 0.0, 1.0, R_A, "put", q=Q_A) == 0.0
        assert model_a.gamma(S0, 0.0, 1.0, R_A, "call", q=Q_A) == 0.0

    def test_extreme_corner_prices_without_warnings(self):
        """
        The 2026-07-17 Hypothesis corner (v0 = 0.005, xi = 1.0, T = 1/16):
        the raw integrand's exponential tail decays at rate ~5e-3, which
        exhausted quad's 400 subdivisions and left ~4e-7 errors. The GL
        grid must price it warning-free with parity exact by construction.
        """
        m = make_model(v0=0.005, kappa=1.0, theta=0.015625, xi=1.0, rho=-0.875)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            c = m.price(S0, 50.0, 0.0625, 0.0, "call")
            p = m.price(S0, 50.0, 0.0625, 0.0, "put")
            m.greeks(S0, 50.0, 0.0625, 0.0, "call")
        assert abs((c - p) - 50.0) < 1e-10
        assert c >= 50.0 - 1e-10

    def test_cv_total_variance_closed_form(self):
        """w = theta T + (v0 - theta)(1 - e^{-kappa T})/kappa, exactly."""
        m = make_model(**PARAMS_A)  # v0 = theta: w = v0 T with no transient
        assert abs(m._cv_total_variance(2.0) - 0.08) < 1e-15
        m2 = make_model(v0=0.09, kappa=0.5, theta=0.04, xi=0.3, rho=-0.5)
        expected = 0.04 * 1.7 + (0.09 - 0.04) * (1.0 - np.exp(-0.5 * 1.7)) / 0.5
        assert abs(m2._cv_total_variance(1.7) - expected) < 1e-15

    def test_bs_control_variate_is_exact_in_bs_limit(self):
        """
        xi -> 0 with v0 = theta: phi_j -> phi_j^BS, the residual integral
        vanishes and P_j collapse to N(d_j). The price must match
        Black-Scholes far tighter than the generic 1e-6 limit tolerance —
        this pins that the CV terms use the right measure shifts (a sign
        error in the +-w/2 drifts would shift P1 vs P2 by ~n(d1) sqrt(w)
        and fail by orders of magnitude).

        xi = 1e-4, not smaller: the Little-Trap CF computes
        (kappa theta/xi^2) * [cancelling terms], so its round-off noise
        grows as eps*kappa*theta/xi^2 while the true residual shrinks as
        xi^2 — below xi ~ 1e-4 the CF's own conditioning floor (not the
        quadrature) dominates and the observed error RISES (measured:
        2e-8 at xi=1e-4, 1.3e-5 at xi=1e-6).
        """
        m = make_model(v0=0.04, kappa=2.0, theta=0.04, xi=1e-4, rho=0.0)
        bs = BlackScholesModel(sigma=0.20)
        for K, opt in [(80.0, "call"), (100.0, "call"), (125.0, "put")]:
            assert (
                abs(
                    m.price(S0, K, 1.0, 0.05, opt, q=0.02) - bs.price(S0, K, 1.0, 0.05, opt, q=0.02)
                )
                < 1e-7
            )

    def test_greeks_price_delta_share_quadrature(self, model_a):
        """greeks() price/delta must be bit-identical to the standalone calls."""
        strikes = np.array([80.0, 100.0, 125.0])
        g = model_a.greeks(S0, strikes, 1.0, R_A, "put", q=Q_A)
        np.testing.assert_array_equal(
            g["price"], model_a.price(S0, strikes, 1.0, R_A, "put", q=Q_A)
        )
        np.testing.assert_array_equal(
            g["delta"], model_a.delta(S0, strikes, 1.0, R_A, "put", q=Q_A)
        )

    def test_price_surface_cache_isolation(self, model_a):
        """The (n_fft, eta, alpha) cache must not leak state across models."""
        strikes = np.array([80.0, 100.0, 120.0])
        first = model_a.price_surface(S0, strikes, 1.0, R_A, q=Q_A)
        other = make_model(v0=0.09, kappa=1.0, theta=0.09, xi=0.8, rho=-0.3)
        different = other.price_surface(S0, strikes, 1.0, R_A, q=Q_A)
        again = model_a.price_surface(S0, strikes, 1.0, R_A, q=Q_A)
        np.testing.assert_array_equal(first, again)
        assert not np.allclose(first, different)


# ──────────────────────────────────────────────
# Greeks
# ──────────────────────────────────────────────


class TestHestonGreeks:
    def test_delta_matches_finite_difference(self, model_a):
        h = 1e-2
        for K, opt in [(100.0, "call"), (110.0, "put"), (80.0, "call")]:
            d = model_a.delta(S0, K, 1.0, R_A, opt, q=Q_A)
            p_up = model_a.price(S0 + h, K, 1.0, R_A, opt, q=Q_A)
            p_dn = model_a.price(S0 - h, K, 1.0, R_A, opt, q=Q_A)
            assert abs(d - (p_up - p_dn) / (2 * h)) < 1e-6

    def test_gamma_matches_finite_difference(self, model_a):
        h = 0.05
        for K in (95.0, 105.0):
            g = model_a.gamma(S0, K, 1.0, R_A, "call", q=Q_A)
            p_up = model_a.price(S0 + h, K, 1.0, R_A, "call", q=Q_A)
            p_0 = model_a.price(S0, K, 1.0, R_A, "call", q=Q_A)
            p_dn = model_a.price(S0 - h, K, 1.0, R_A, "call", q=Q_A)
            assert abs(g - (p_up - 2 * p_0 + p_dn) / (h * h)) < 1e-5

    def test_put_call_delta_relation(self, model_a):
        # From parity: delta_call - delta_put = e^{-qT}
        dc = model_a.delta(S0, 100, 1.0, R_A, "call", q=Q_A)
        dp = model_a.delta(S0, 100, 1.0, R_A, "put", q=Q_A)
        assert abs((dc - dp) - np.exp(-Q_A)) < 1e-10

    def test_gamma_same_call_put_and_positive(self, model_a):
        gc = model_a.gamma(S0, 100, 1.0, R_A, "call", q=Q_A)
        gp = model_a.gamma(S0, 100, 1.0, R_A, "put", q=Q_A)
        assert gc == gp
        assert gc > 0

    def test_vega_positive_and_theta_negative_atm(self, model_a):
        assert model_a.vega(S0, 100, 1.0, R_A, "call", q=Q_A) > 0
        # ATM call theta is negative for these (standard) parameters
        assert model_a.theta(S0, 100, 1.0, R_A, "call", q=Q_A) < 0

    def test_rho_signs(self, model_a):
        assert model_a.rho(S0, 100, 1.0, R_A, "call", q=Q_A) > 0
        assert model_a.rho(S0, 100, 1.0, R_A, "put", q=Q_A) < 0

    def test_theta_matches_finite_difference(self, model_a):
        h = 1e-4
        th = model_a.theta(S0, 100, 1.0, R_A, "call", q=Q_A)
        fd = (
            model_a.price(S0, 100, 1.0 - h, R_A, "call", q=Q_A)
            - model_a.price(S0, 100, 1.0 + h, R_A, "call", q=Q_A)
        ) / (2 * h)
        assert abs(th - fd) < 1e-4

    def test_greeks_dict_complete(self, model_a):
        g = model_a.greeks(S0, 100, 1.0, R_A, "call", q=Q_A)
        assert set(g) == {"price", "delta", "gamma", "vega", "theta", "rho"}
        assert abs(g["price"] - model_a.price(S0, 100, 1.0, R_A, "call", q=Q_A)) < 1e-12
        assert abs(g["delta"] - model_a.delta(S0, 100, 1.0, R_A, "call", q=Q_A)) < 1e-12

    def test_model_greeks_chain_rule_coherence(self, model_a):
        # vega = dV/d sqrt(v0) and model_greeks v0 = dV/dv0 are computed by
        # two different FD paths; the exact chain rule links them:
        # dV/d sqrt(v0) = dV/dv0 * 2 sqrt(v0)
        vega = model_a.vega(S0, 100, 1.0, R_A, "call", q=Q_A)
        dv0 = model_a.model_greeks(S0, 100, 1.0, R_A, "call", q=Q_A)["v0"]
        assert abs(vega - dv0 * 2 * np.sqrt(model_a.v0)) < 1e-3 * abs(vega)

    def test_model_greeks_signs(self, model_a):
        mg = model_a.model_greeks(S0, 100, 1.0, R_A, "call", q=Q_A)
        assert set(mg) == {"v0", "kappa", "theta", "xi", "rho"}
        # More initial variance / long-run variance -> higher option value
        assert mg["v0"] > 0
        assert mg["theta"] > 0

    def test_model_greeks_match_direct_bumps(self, model_a):
        # Independent recomputation of dV/dxi with a coarser bump
        h = 1e-3 * PARAMS_A["xi"]
        up = make_model(**{**PARAMS_A, "xi": PARAMS_A["xi"] + h})
        dn = make_model(**{**PARAMS_A, "xi": PARAMS_A["xi"] - h})
        expected = (
            up.price(S0, 100, 1.0, R_A, "call", q=Q_A) - dn.price(S0, 100, 1.0, R_A, "call", q=Q_A)
        ) / (2 * h)
        got = model_a.model_greeks(S0, 100, 1.0, R_A, "call", q=Q_A)["xi"]
        assert abs(got - expected) < 1e-8

    def test_greeks_vectorized(self, model_a):
        strikes = np.array([90.0, 100.0, 110.0])
        d = model_a.delta(S0, strikes, 1.0, R_A, "call", q=Q_A)
        assert d.shape == (3,)
        assert np.all(np.diff(d) < 0)  # call delta decreasing in strike


# ──────────────────────────────────────────────
# QE simulation
# ──────────────────────────────────────────────


class TestQESimulation:
    def test_parameter_validation(self):
        mc = MonteCarloEngine(n_paths=100, seed=1)
        base = dict(S0=100, v0=0.04, T=1.0, r=0.05, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
        for field, bad in [
            ("S0", -1.0),
            ("v0", 0.0),
            ("T", 0.0),
            ("kappa", 0.0),
            ("theta", -0.1),
            ("xi", 0.0),
            ("rho", 1.0),
            ("rho", -1.5),
        ]:
            kwargs = {**base, field: bad}
            with pytest.raises(ValueError):
                mc.simulate_heston(**kwargs)
        with pytest.raises(ValueError, match="psi_c"):
            mc.simulate_heston(**base, psi_c=0.5)
        with pytest.raises(ValueError, match="n_steps"):
            mc.simulate_heston(**base, n_steps=0)

    def test_shapes_and_initial_values(self, qe_paths_a):
        _, paths, v_paths = qe_paths_a
        assert paths.shape == (400_000, 101)
        assert v_paths.shape == (400_000, 101)
        assert np.all(paths[:, 0] == S0)
        assert np.all(v_paths[:, 0] == PARAMS_A["v0"])

    def test_paths_positive_variance_nonnegative(self, qe_paths_a):
        _, paths, v_paths = qe_paths_a
        assert np.all(paths > 0)
        assert np.all(v_paths >= 0)
        assert np.all(np.isfinite(paths))

    def test_martingale_property(self, qe_paths_a):
        # E[e^{-(r-q)T} S_T] = S0; with the Andersen martingale correction
        # this holds exactly in expectation, so only MC noise remains.
        _, paths, _ = qe_paths_a
        x = np.exp(-(R_A - Q_A) * 1.0) * paths[:, -1]
        se = x.std(ddof=1) / np.sqrt(len(x))
        assert abs(x.mean() - S0) < 3 * se

    def test_martingale_coarse_steps(self):
        # dt = 0.25 stresses the drift approximation; the corrected scheme
        # must stay unbiased even here.
        mc = MonteCarloEngine(n_paths=300_000, seed=7)
        p = mc.simulate_heston(
            S0, 0.04, 1.0, 0.05, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, n_steps=4
        )
        x = np.exp(-0.05) * p[:, -1]
        se = x.std(ddof=1) / np.sqrt(len(x))
        assert abs(x.mean() - S0) < 3 * se

    def test_plain_scheme_runs_and_close(self):
        # The uncorrected textbook scheme remains available and unbiased
        # within statistical resolution at fine steps.
        mc = MonteCarloEngine(n_paths=200_000, seed=11)
        p = mc.simulate_heston(
            S0,
            0.04,
            1.0,
            0.05,
            kappa=2.0,
            theta=0.04,
            xi=0.5,
            rho=-0.7,
            n_steps=100,
            martingale_correction=False,
        )
        x = np.exp(-0.05) * p[:, -1]
        se = x.std(ddof=1) / np.sqrt(len(x))
        assert abs(x.mean() - S0) < 3 * se

    def test_cir_exact_mean(self, qe_paths_a):
        # E[v_T | v_0] = theta + (v0 - theta) e^{-kappa T} (exact CIR moment)
        _, _, v_paths = qe_paths_a
        v0, kappa, theta = (PARAMS_A["v0"], PARAMS_A["kappa"], PARAMS_A["theta"])
        v_T = v_paths[:, -1]
        exact = theta + (v0 - theta) * np.exp(-kappa * 1.0)
        se = v_T.std(ddof=1) / np.sqrt(len(v_T))
        assert abs(v_T.mean() - exact) < 3 * se

    def test_cir_exact_variance(self, qe_paths_a):
        # Var[v_T | v_0] = v0 xi^2 e^{-kT}(1-e^{-kT})/k
        #                  + theta xi^2 (1-e^{-kT})^2 / (2k)
        # Sampling error of the sample variance at N = 400k is ~0.3%;
        # 1.5% tolerance leaves a 5x margin while still failing for any
        # systematic moment error.
        _, _, v_paths = qe_paths_a
        v0, kappa, theta, xi = (
            PARAMS_A["v0"],
            PARAMS_A["kappa"],
            PARAMS_A["theta"],
            PARAMS_A["xi"],
        )
        e = np.exp(-kappa * 1.0)
        exact = v0 * xi**2 * e * (1 - e) / kappa + theta * xi**2 * (1 - e) ** 2 / (2 * kappa)
        assert abs(v_paths[:, -1].var(ddof=1) - exact) < 0.015 * exact

    def test_feller_violated_mass_at_zero(self, qe_paths_a):
        # PARAMS_A violates Feller (0.16 < 0.25): the exponential branch
        # must place strictly positive mass at v = 0. A scheme that cannot
        # reach zero (e.g. naive lognormal matching) fails this.
        _, _, v_paths = qe_paths_a
        assert np.any(v_paths == 0.0)

    def test_reproducibility_via_reset(self):
        mc = MonteCarloEngine(n_paths=5_000, seed=99)
        p1 = mc.simulate_heston(
            S0, 0.04, 0.5, 0.05, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, n_steps=10
        )
        mc.reset()
        p2 = mc.simulate_heston(
            S0, 0.04, 0.5, 0.05, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, n_steps=10
        )
        np.testing.assert_array_equal(p1, p2)

    def test_default_returns_only_spot(self):
        mc = MonteCarloEngine(n_paths=100, seed=1)
        out = mc.simulate_heston(
            S0, 0.04, 0.5, 0.05, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, n_steps=5
        )
        assert isinstance(out, np.ndarray)
        assert out.shape == (100, 6)

    def test_uses_engine_default_steps(self):
        mc = MonteCarloEngine(n_paths=50, n_steps=37, seed=1)
        out = mc.simulate_heston(S0, 0.04, 0.5, 0.05, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
        assert out.shape == (50, 38)

    def test_zero_correlation_terminal_moment(self):
        # With rho = 0: E[ln S_T] = ln S0 + (r-q)T - 0.5 * E[integral v dt],
        # and E[integral v dt] = theta T + (v0-theta)(1-e^{-kT})/k exactly.
        mc = MonteCarloEngine(n_paths=400_000, seed=21)
        v0, kappa, theta, xi = 0.09, 3.0, 0.06, 0.4
        p = mc.simulate_heston(
            S0, v0, 2.0, 0.03, kappa=kappa, theta=theta, xi=xi, rho=0.0, n_steps=100
        )
        log_ST = np.log(p[:, -1])
        int_v = theta * 2.0 + (v0 - theta) * (1 - np.exp(-kappa * 2.0)) / kappa
        exact = np.log(S0) + 0.03 * 2.0 - 0.5 * int_v
        se = log_ST.std(ddof=1) / np.sqrt(len(log_ST))
        assert abs(log_ST.mean() - exact) < 3 * se


class TestFourierVsMC:
    """The two pricing routes must agree: semi-closed form vs simulation."""

    @pytest.fixture(scope="class")
    def mc_setup(self):
        engine = MonteCarloEngine(n_paths=300_000, seed=5)
        paths = engine.simulate_heston(
            S0,
            PARAMS_A["v0"],
            1.0,
            R_A,
            kappa=PARAMS_A["kappa"],
            theta=PARAMS_A["theta"],
            xi=PARAMS_A["xi"],
            rho=PARAMS_A["rho"],
            q=Q_A,
            n_steps=200,
        )
        return engine, paths

    @pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
    def test_european_call(self, model_a, mc_setup, K):
        engine, paths = mc_setup
        fourier = model_a.price(S0, K, 1.0, R_A, "call", q=Q_A)
        res = engine.price(lambda p: np.maximum(p[:, -1] - K, 0.0), paths, R_A, 1.0)
        assert abs(res.price - fourier) < 3 * res.std_error

    def test_european_put(self, model_a, mc_setup):
        engine, paths = mc_setup
        fourier = model_a.price(S0, 100.0, 1.0, R_A, "put", q=Q_A)
        res = engine.price(lambda p: np.maximum(100.0 - p[:, -1], 0.0), paths, R_A, 1.0)
        assert abs(res.price - fourier) < 3 * res.std_error


class TestExoticsUnderHeston:
    """
    Phase 3 instruments price on Heston paths without touching a line:
    ExoticOption.payoff(paths) is model-agnostic by design.
    """

    def test_asian_prices_and_differs_from_gbm(self, qe_paths_a):
        engine, paths, _ = qe_paths_a
        asian = AsianOption(K=100.0, option_type="call", avg_type="arithmetic")
        res_h = engine.price(asian.payoff, paths, R_A, 1.0)
        assert res_h.price > 0

        # GBM with sigma = sqrt(theta) = long-run vol: stochastic vol must
        # move the Asian price by far more than the combined noise (the
        # smile/path-dependence interaction is real, not noise).
        gbm_engine = MonteCarloEngine(n_paths=400_000, seed=43)
        gbm_paths = gbm_engine.simulate_gbm(
            S0,
            1.0,
            R_A,
            np.sqrt(PARAMS_A["theta"]),
            q=Q_A,
            n_steps=100,
        )
        res_g = gbm_engine.price(asian.payoff, gbm_paths, R_A, 1.0)
        combined_se = float(np.hypot(res_h.std_error, res_g.std_error))
        assert abs(res_h.price - res_g.price) > 3 * combined_se

    def test_barrier_under_heston_within_bounds(self, qe_paths_a, model_a):
        engine, paths, _ = qe_paths_a
        barrier = BarrierOption(
            K=100.0, barrier=80.0, barrier_type="down-and-out", option_type="call"
        )
        res = engine.price(barrier.payoff, paths, R_A, 1.0)
        vanilla = model_a.price(S0, 100.0, 1.0, R_A, "call", q=Q_A)
        # Knock-out is worth less than vanilla, more than zero
        assert 0.0 < res.price < vanilla

    def test_barrier_in_out_parity_under_heston(self, qe_paths_a, model_a):
        # KI + KO = vanilla holds path-by-path for ANY model, so it must
        # hold exactly (same paths) up to the MC error of the vanilla leg.
        engine, paths, _ = qe_paths_a
        ko = BarrierOption(K=100.0, barrier=80.0, barrier_type="down-and-out", option_type="call")
        ki = BarrierOption(K=100.0, barrier=80.0, barrier_type="down-and-in", option_type="call")
        res_ko = engine.price(ko.payoff, paths, R_A, 1.0)
        res_ki = engine.price(ki.payoff, paths, R_A, 1.0)
        res_van = engine.price(
            lambda p: np.maximum(p[:, -1] - 100.0, 0.0),
            paths,
            R_A,
            1.0,
        )
        assert abs((res_ko.price + res_ki.price) - res_van.price) < 1e-10


# ──────────────────────────────────────────────
# Implied volatility smile / skew
# ──────────────────────────────────────────────


class TestVolatilitySmile:
    def _iv_curve(self, model, strikes, T, r, q):
        ivs = []
        for K in strikes:
            price = model.price(S0, K, T, r, "call", q=q)
            ivs.append(BlackScholesModel.implied_vol(price, S0, K, T, r, "call", q=q))
        return np.array(ivs)

    def test_negative_rho_produces_downward_skew(self, model_a):
        # The defining qualitative feature BS cannot reproduce: with
        # rho < 0, implied vol decreases with strike.
        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
        ivs = self._iv_curve(model_a, strikes, 1.0, R_A, Q_A)
        assert np.all(np.diff(ivs) < 0)

    def test_zero_rho_produces_symmetric_smile(self):
        m = make_model(v0=0.04, kappa=1.0, theta=0.04, xi=0.8, rho=0.0)
        # Strikes symmetric in log-moneyness around the forward
        F = S0 * np.exp((0.0 - 0.0) * 1.0)
        ks = F * np.exp(np.array([-0.25, -0.10, 0.0, 0.10, 0.25]))
        ivs = self._iv_curve(m, ks, 1.0, 0.0, 0.0)
        # Convex smile: wings above the ATM-forward vol
        assert ivs[0] > ivs[2] and ivs[4] > ivs[2]
        # rho = 0 makes the smile symmetric to first order
        assert abs(ivs[0] - ivs[4]) < 0.1 * (ivs[0] - ivs[2])
        assert abs(ivs[1] - ivs[3]) < 0.1 * (ivs[0] - ivs[2])

    def test_smile_flattens_with_maturity(self, model_a):
        # Mean reversion flattens the skew at long maturities
        strikes = [90.0, 110.0]
        skew_short = np.diff(self._iv_curve(model_a, strikes, 0.25, R_A, Q_A))[0]
        skew_long = np.diff(self._iv_curve(model_a, strikes, 5.0, R_A, Q_A))[0]
        assert abs(skew_long) < abs(skew_short)


# ──────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────


class TestFilterOptionQuotes:
    def _base_quote(self, **overrides):
        quote = dict(
            strikes=np.array([100.0]),
            maturities=np.array([0.5]),
            bids=np.array([5.0]),
            asks=np.array([5.2]),
            volumes=np.array([50.0]),
        )
        quote.update(overrides)
        return quote

    def test_good_quote_kept(self):
        mask = filter_option_quotes(S0=100.0, **self._base_quote())
        assert mask.tolist() == [True]

    @pytest.mark.parametrize(
        "overrides",
        [
            dict(bids=np.array([0.0])),  # no bid
            dict(asks=np.array([4.0])),  # crossed market
            dict(volumes=np.array([0.0])),  # no volume
            dict(bids=np.array([0.5]), asks=np.array([5.0])),  # huge spread
            dict(strikes=np.array([50.0])),  # outside moneyness band
            dict(maturities=np.array([0.005])),  # below 1-week expiry
            dict(maturities=np.array([5.0])),  # beyond LEAPS band
        ],
    )
    def test_bad_quotes_dropped(self, overrides):
        mask = filter_option_quotes(S0=100.0, **self._base_quote(**overrides))
        assert mask.tolist() == [False]

    def test_validations(self):
        q = self._base_quote()
        with pytest.raises(ValueError, match="S0"):
            filter_option_quotes(S0=-1.0, **q)
        with pytest.raises(ValueError, match="equal length"):
            filter_option_quotes(S0=100.0, **self._base_quote(volumes=np.array([1.0, 2.0])))


class TestMarketDataHelpers:
    """Pure helpers of market_data (no yfinance, no network)."""

    def test_select_expiries_samples_the_band(self):
        import datetime as dt

        today = dt.date(2026, 6, 11)
        # Daily expiries for 3 weeks, then monthlies out to 2 years —
        # the SPX failure mode: naive [:8] returns a 2-week wall.
        listed = [str(today + dt.timedelta(days=d)) for d in range(1, 22)]
        listed += [str(today + dt.timedelta(days=30 * m)) for m in range(1, 25)]
        chosen = _select_expiries(listed, today, (0.02, 2.5), 8)
        ts = [t for _, t in chosen]
        assert len(ts) == 8
        assert ts == sorted(ts)
        # Must reach deep into the band, not cluster at the short end
        assert ts[0] < 0.1
        assert ts[-1] > 1.5

    def test_select_expiries_keeps_all_when_few(self):
        import datetime as dt

        today = dt.date(2026, 6, 11)
        listed = [str(today + dt.timedelta(days=d)) for d in (30, 90, 365)]
        chosen = _select_expiries(listed, today, (0.02, 2.5), 8)
        assert len(chosen) == 3

    def test_implied_dividend_yield_round_trip(self):
        # Build synthetic two-sided quotes from BS with known q: parity
        # must recover it regardless of the vol used to generate prices.
        true_q, r, T = 0.024, 0.04, 0.5
        bs = BlackScholesModel(sigma=0.22)
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        calls = {"strike": [], "bid": [], "ask": []}
        puts = {"strike": [], "bid": [], "ask": []}
        for K in strikes:
            c = bs.price(100.0, K, T, r, "call", q=true_q)
            p = bs.price(100.0, K, T, r, "put", q=true_q)
            calls["strike"].append(K)
            calls["bid"].append(c - 0.01)
            calls["ask"].append(c + 0.01)
            puts["strike"].append(K)
            puts["bid"].append(p - 0.01)
            puts["ask"].append(p + 0.01)
        got = _implied_dividend_yield(calls, puts, 100.0, T, r, fallback=0.0)
        assert abs(got - true_q) < 1e-10

    def test_implied_dividend_yield_fallback(self):
        empty = {"strike": [], "bid": [], "ask": []}
        got = _implied_dividend_yield(empty, empty, 100.0, 0.5, 0.04, fallback=0.013)
        assert got == 0.013
        # One-sided book (no put quotes) must also fall back
        calls = {"strike": [100.0], "bid": [5.0], "ask": [5.2]}
        got = _implied_dividend_yield(calls, empty, 100.0, 0.5, 0.04, fallback=0.013)
        assert got == 0.013


class TestHestonCalibrator:
    TRUE = dict(v0=0.04, kappa=1.5, theta=0.05, xi=0.6, rho=-0.65)
    R, Q = 0.03, 0.01

    @pytest.fixture(scope="class")
    def surface(self):
        """Synthetic surface generated from a known parameter set."""
        cal = HestonCalibrator(S0, self.R, self.Q)
        Ks = np.array([80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0])
        Ts = [0.25, 1.0, 2.0]
        strikes = np.tile(Ks, len(Ts))
        maturities = np.repeat(Ts, len(Ks))
        ivs = cal.model_ivs(make_model(**self.TRUE), strikes, maturities)
        assert np.isfinite(ivs).all()
        return cal, strikes, maturities, ivs

    def test_constructor_validation(self):
        with pytest.raises(ValueError):
            HestonCalibrator(S0=-100.0, r=0.03)
        with pytest.raises(ValueError):
            HestonCalibrator(S0=100.0, r=np.nan)

    def test_input_validation(self, surface):
        cal, strikes, maturities, ivs = surface
        with pytest.raises(ValueError, match="equal length"):
            cal.calibrate(strikes, maturities[:-1], ivs)
        with pytest.raises(ValueError, match="at least 5"):
            cal.calibrate(strikes[:3], maturities[:3], ivs[:3])
        with pytest.raises(ValueError, match="market_ivs"):
            cal.calibrate(strikes, maturities, np.full_like(ivs, -0.2))
        with pytest.raises(ValueError, match="weights"):
            cal.calibrate(strikes, maturities, ivs, weights=ivs[:-1])
        with pytest.raises(ValueError, match="feller_penalty"):
            cal.calibrate(strikes, maturities, ivs, feller_penalty=-1.0)

    def test_round_trip_exact_recovery(self, surface):
        """
        Noise-free surface: the global minimum is exactly the generating
        parameters. Measured recovery is at machine precision (1e-12);
        the tolerances below are 1e8 looser yet still pin every parameter
        far more tightly than any market calibration could.
        """
        cal, strikes, maturities, ivs = surface
        res = cal.calibrate(strikes, maturities, ivs, n_starts=2)
        assert res.success
        assert res.rmse_iv < 1e-6
        assert abs(res.params["v0"] - self.TRUE["v0"]) < 1e-4
        assert abs(res.params["theta"] - self.TRUE["theta"]) < 1e-4
        assert abs(res.params["xi"] - self.TRUE["xi"]) < 1e-3
        assert abs(res.params["rho"] - self.TRUE["rho"]) < 1e-3
        assert abs(res.params["kappa"] - self.TRUE["kappa"]) < 1e-2
        assert not res.feller_satisfied  # 2*1.5*0.05 = 0.15 < 0.36
        assert "rmse" in repr(res)

    def test_round_trip_with_noise(self, surface):
        # 0.2 vol pts of quote noise: fit error must stay at noise level
        # and the well-identified parameters (v0, rho) must stay close.
        cal, strikes, maturities, ivs = surface
        rng = np.random.default_rng(7)
        noisy = ivs + rng.normal(0.0, 0.002, ivs.shape)
        res = cal.calibrate(strikes, maturities, noisy, n_starts=2)
        assert res.rmse_iv < 0.004
        assert abs(res.params["v0"] - self.TRUE["v0"]) < 0.005
        assert abs(res.params["rho"] - self.TRUE["rho"]) < 0.05

    def test_feller_penalty_enforces_condition(self, surface):
        # The unpenalized optimum violates Feller (by construction of
        # TRUE); a strong soft penalty must push the fit into the
        # admissible region, trading fit quality for it.
        cal, strikes, maturities, ivs = surface
        plain = cal.calibrate(strikes, maturities, ivs, n_starts=1)
        penalized = cal.calibrate(strikes, maturities, ivs, n_starts=1, feller_penalty=10.0)
        assert not plain.feller_satisfied
        assert penalized.feller_satisfied
        assert penalized.rmse_iv > plain.rmse_iv

    def test_zero_weights_exclude_options(self, surface):
        # Corrupt two quotes but give them weight 0: the fit must ignore
        # them and still recover the true parameters.
        cal, strikes, maturities, ivs = surface
        corrupted = ivs.copy()
        corrupted[3] += 0.30
        corrupted[15] -= 0.10
        w = np.ones_like(ivs)
        w[[3, 15]] = 0.0
        res = cal.calibrate(strikes, maturities, corrupted, weights=w, n_starts=1)
        assert abs(res.params["v0"] - self.TRUE["v0"]) < 1e-4
        assert abs(res.params["rho"] - self.TRUE["rho"]) < 1e-3

    def test_explicit_starts(self, surface):
        cal, strikes, maturities, ivs = surface
        res = cal.calibrate(
            strikes,
            maturities,
            ivs,
            starts=[dict(v0=0.09, kappa=4.0, theta=0.09, xi=0.2, rho=-0.1)],
            n_starts=1,
        )
        assert res.n_starts == 1
        assert res.rmse_iv < 1e-5

    def test_model_ivs_nan_for_unpriceable_maturities(self):
        # rho = 0.9, xi = 1.5, kappa = 0.3: T*(2.5) = ln(g)/d = 0.676,
        # so T = 5 cannot be priced by Carr-Madan with alpha = 1.5 while
        # T = 0.2 can. model_ivs must mark, not crash.
        cal = HestonCalibrator(S0, 0.02, 0.0)
        m = make_model(v0=0.09, kappa=0.3, theta=0.09, xi=1.5, rho=0.9)
        ivs = cal.model_ivs(m, np.array([100.0, 100.0]), np.array([0.2, 5.0]))
        assert np.isfinite(ivs[0])
        assert np.isnan(ivs[1])

    def test_model_ivs_matches_scalar_inversion(self, model_a):
        """
        The whole-chain batch inversion must reproduce a per-option scalar
        FFT-price -> implied_vol loop exactly (same solver trajectory).
        """
        cal = HestonCalibrator(S0, R_A, Q_A)
        strikes = np.tile(np.linspace(80.0, 120.0, 9), 3)
        maturities = np.repeat([0.2, 1.0, 2.0], 9)
        ivs = cal.model_ivs(model_a, strikes, maturities)

        ref = np.full(strikes.shape, np.nan)
        for T in np.unique(maturities):
            idx = np.flatnonzero(maturities == T)
            prices = model_a.price_surface(S0, strikes[idx], float(T), R_A, q=Q_A)
            for j, price in zip(idx, prices):
                ref[j] = BlackScholesModel.implied_vol(
                    float(price), S0, float(strikes[j]), float(T), R_A, "call", q=Q_A
                )
        assert np.isfinite(ivs).all()
        # 1-2 ulp: NumPy SIMD array kernels vs the scalar path
        np.testing.assert_allclose(ivs, ref, rtol=1e-14, atol=0)


# ──────────────────────────────────────────────
# Property-based tests (Hypothesis)
# ──────────────────────────────────────────────

heston_params = st.fixed_dictionaries(
    {
        "v0": st.floats(0.005, 0.25),
        "kappa": st.floats(0.2, 8.0),
        "theta": st.floats(0.005, 0.25),
        "xi": st.floats(0.05, 1.2),
        # rho capped at 0.5: strongly positive rho with long maturities sits in
        # the moment-explosion region (Andersen-Piterbarg 2007) — a genuine
        # model property, not an implementation artifact, and irrelevant for
        # equity (rho < 0).
        "rho": st.floats(-0.95, 0.5),
    }
)
market = st.fixed_dictionaries(
    {
        "K": st.floats(50.0, 200.0),
        "T": st.floats(0.05, 3.0),
        "r": st.floats(-0.02, 0.10),
        "q": st.floats(0.0, 0.06),
    }
)


class TestHestonProperties:
    @given(params=heston_params, mkt=market)
    @settings(max_examples=500, deadline=None)
    # Pinned regression (found by Hypothesis 2026-07-17): low vol + short T
    # makes the Gil-Pelaez integrand exhaust the subdivision limit; the
    # ~4e-7 quadrature error pushed the raw call below intrinsic and the
    # put floor at zero then broke parity. _price_scalar now projects the
    # call onto the no-arbitrage band and derives the put from the
    # projected value, so parity is exact by construction — the tolerance
    # only absorbs float roundoff.
    @example(
        params={"v0": 0.005, "kappa": 1.0, "theta": 0.015625, "xi": 1.0, "rho": -0.875},
        mkt={"K": 50.0, "T": 0.0625, "r": 0.0, "q": 0.0},
    )
    def test_put_call_parity_random(self, params, mkt):
        m = make_model(**params)
        c = m.price(S0, mkt["K"], mkt["T"], mkt["r"], "call", q=mkt["q"])
        p = m.price(S0, mkt["K"], mkt["T"], mkt["r"], "put", q=mkt["q"])
        rhs = S0 * np.exp(-mkt["q"] * mkt["T"]) - mkt["K"] * np.exp(-mkt["r"] * mkt["T"])
        assert abs((c - p) - rhs) < 1e-8

    @given(params=heston_params, mkt=market)
    @settings(max_examples=300, deadline=None)
    def test_price_within_bounds_random(self, params, mkt):
        m = make_model(**params)
        c = m.price(S0, mkt["K"], mkt["T"], mkt["r"], "call", q=mkt["q"])
        lower = max(
            S0 * np.exp(-mkt["q"] * mkt["T"]) - mkt["K"] * np.exp(-mkt["r"] * mkt["T"]), 0.0
        )
        upper = S0 * np.exp(-mkt["q"] * mkt["T"])
        assert lower - 1e-7 <= c <= upper + 1e-7

    @given(params=heston_params, T=st.floats(0.05, 5.0), u=st.floats(0.01, 50.0))
    @settings(max_examples=500, deadline=None)
    def test_cf_modulus_bounded_random(self, params, T, u):
        m = make_model(**params)
        phi = m.char_func(u, T=T, r=0.03, q=0.01, S0=S0)
        assert abs(phi) <= 1.0 + 1e-10

    @given(params=heston_params, T=st.floats(0.05, 5.0))
    @settings(max_examples=500, deadline=None)
    def test_cf_martingale_anchor_random(self, params, T):
        m = make_model(**params)
        phi = m.char_func(-1j, T=T, r=0.03, q=0.01, S0=S0)
        forward = S0 * np.exp((0.03 - 0.01) * T)
        assert abs(phi - forward) < 1e-8 * forward

    @given(params=heston_params)
    @settings(max_examples=200, deadline=None)
    def test_call_decreasing_in_strike_random(self, params):
        m = make_model(**params)
        c_low = m.price(S0, 90.0, 1.0, 0.03, "call")
        c_high = m.price(S0, 110.0, 1.0, 0.03, "call")
        assert c_low > c_high
