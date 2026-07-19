"""
Exotic Option Pricer - tests/test_pricing.py

Comprehensive test suite for BlackScholesModel.

Covers:
1. Hull benchmarks                    8. Homogeneity of degree 1
2. Put-Call Parity                    9. Price bounds (no-arbitrage)
3. Boundary conditions               10. Vectorization
4. Greek signs and bounds             11. Input validation
5. Greek symmetry (PCP derivatives)   12. Dividend yield
6. Finite-difference first-order      13. Implied volatility
7. BS PDE satisfaction                14. Second-order Greeks (FD)
                                      15. Dual Greeks (FD)
                                      16. Monte Carlo cross-validation
                                      17. Extreme parameters
                                      18. Batch Greeks consistency

Reference: S=100, K=100, T=1, r=0.05, sigma=0.20
d1=0.35, d2=0.15, C=10.4506, P=5.5735
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exotic_option_pricer.models.black_scholes import BlackScholesModel

# ============================================================================
# Standard test parameters
# ============================================================================
HULL_S = 100.0
HULL_K = 100.0
HULL_T = 1.0
HULL_R = 0.05
HULL_SIGMA = 0.20

HULL_D1 = 0.3500
HULL_D2 = 0.1500
HULL_CALL = 10.4506
HULL_PUT = 5.5735


@pytest.fixture
def bs():
    return BlackScholesModel(sigma=HULL_SIGMA)


# ============================================================================
# 1. Hull Benchmarks
# ============================================================================
class TestHullBenchmarks:
    def test_d1_d2_values(self, bs):
        d1, d2 = bs._compute_d1_d2(HULL_S, HULL_K, HULL_T, HULL_R)
        assert abs(float(d1) - HULL_D1) < 0.0001
        assert abs(float(d2) - HULL_D2) < 0.0001

    def test_atm_call(self, bs):
        assert abs(bs.price(HULL_S, HULL_K, HULL_T, HULL_R, "call") - HULL_CALL) < 0.001

    def test_atm_put(self, bs):
        assert abs(bs.price(HULL_S, HULL_K, HULL_T, HULL_R, "put") - HULL_PUT) < 0.001


# ============================================================================
# 2. Put-Call Parity
# ============================================================================
class TestPutCallParity:
    @pytest.mark.parametrize(
        "S,K,T,r",
        [
            (100, 100, 1.0, 0.05),
            (100, 80, 0.5, 0.03),
            (100, 120, 2.0, 0.08),
            (50, 50, 0.25, 0.01),
            (200, 100, 0.01, 0.10),
            (100, 100, 5.0, 0.02),
            (100, 100, 1.0, 0.0),
            (150, 130, 0.25, 0.02),
        ],
    )
    def test_pcp(self, bs, S, K, T, r):
        call = bs.price(S, K, T, r, "call")
        put = bs.price(S, K, T, r, "put")
        assert abs((call - put) - (S - K * np.exp(-r * T))) < 1e-10

    @pytest.mark.parametrize(
        "S,K,T,r,q",
        [
            (100, 100, 1.0, 0.05, 0.02),
            (100, 80, 0.5, 0.03, 0.05),
            (120, 100, 2.0, 0.08, 0.01),
        ],
    )
    def test_pcp_with_dividends(self, bs, S, K, T, r, q):
        """C - P = S*exp(-qT) - K*exp(-rT) with dividends."""
        call = bs.price(S, K, T, r, "call", q=q)
        put = bs.price(S, K, T, r, "put", q=q)
        fwd = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert abs((call - put) - fwd) < 1e-10


# ============================================================================
# 3. Boundary Conditions
# ============================================================================
class TestBoundaryConditions:
    def test_deep_itm_call(self, bs):
        call = bs.price(1000.0, 100.0, HULL_T, HULL_R, "call")
        fwd = 1000.0 - 100.0 * np.exp(-HULL_R * HULL_T)
        assert abs(call - fwd) / fwd < 1e-6

    def test_deep_otm_call(self, bs):
        assert bs.price(10.0, 1000.0, HULL_T, HULL_R, "call") < 1e-10

    def test_deep_itm_put(self, bs):
        put = bs.price(10.0, 1000.0, HULL_T, HULL_R, "put")
        fwd = 1000.0 * np.exp(-HULL_R * HULL_T) - 10.0
        assert abs(put - fwd) / fwd < 1e-6

    def test_zero_strike_call(self, bs):
        assert abs(bs.price(HULL_S, 0.0, HULL_T, HULL_R, "call") - HULL_S) < 1e-6

    def test_zero_strike_put(self, bs):
        assert abs(bs.price(HULL_S, 0.0, HULL_T, HULL_R, "put")) < 1e-10


# ============================================================================
# 4. Greek Signs and Bounds
# ============================================================================
class TestGreekSigns:
    def test_call_delta_bounds(self, bs):
        d = bs.delta(np.linspace(50, 200, 100), HULL_K, HULL_T, HULL_R, "call")
        assert np.all(d >= -1e-15) and np.all(d <= 1.0 + 1e-15)

    def test_put_delta_bounds(self, bs):
        d = bs.delta(np.linspace(50, 200, 100), HULL_K, HULL_T, HULL_R, "put")
        assert np.all(d >= -1.0 - 1e-15) and np.all(d <= 1e-15)

    def test_gamma_positive(self, bs):
        assert np.all(bs.gamma(np.linspace(50, 200, 100), HULL_K, HULL_T, HULL_R, "call") > 0)

    def test_vega_positive(self, bs):
        assert np.all(bs.vega(np.linspace(50, 200, 100), HULL_K, HULL_T, HULL_R, "call") >= -1e-15)

    def test_call_theta_negative(self, bs):
        assert bs.theta(HULL_S, HULL_K, HULL_T, HULL_R, "call") < 0

    def test_call_rho_positive(self, bs):
        assert bs.rho(HULL_S, HULL_K, HULL_T, HULL_R, "call") > 0

    def test_put_rho_negative(self, bs):
        assert bs.rho(HULL_S, HULL_K, HULL_T, HULL_R, "put") < 0


# ============================================================================
# 5. Greek Symmetry (PCP derivatives)
# ============================================================================
class TestGreekSymmetry:
    @pytest.mark.parametrize(
        "S,K,T,r",
        [
            (100, 100, 1.0, 0.05),
            (120, 100, 0.5, 0.03),
            (80, 100, 2.0, 0.08),
            (50, 100, 0.1, 0.01),
        ],
    )
    def test_gamma_call_equals_put(self, bs, S, K, T, r):
        assert np.isclose(bs.gamma(S, K, T, r, "call"), bs.gamma(S, K, T, r, "put"), rtol=1e-12)

    @pytest.mark.parametrize(
        "S,K,T,r",
        [
            (100, 100, 1.0, 0.05),
            (120, 100, 0.5, 0.03),
            (80, 100, 2.0, 0.08),
            (50, 100, 0.1, 0.01),
        ],
    )
    def test_vega_call_equals_put(self, bs, S, K, T, r):
        assert np.isclose(bs.vega(S, K, T, r, "call"), bs.vega(S, K, T, r, "put"), rtol=1e-12)

    @pytest.mark.parametrize(
        "S,K,T,r",
        [
            (100, 100, 1.0, 0.05),
            (120, 100, 0.5, 0.03),
            (80, 100, 2.0, 0.08),
        ],
    )
    def test_delta_call_minus_put_equals_one(self, bs, S, K, T, r):
        assert np.isclose(
            bs.delta(S, K, T, r, "call") - bs.delta(S, K, T, r, "put"), 1.0, atol=1e-12
        )

    @pytest.mark.parametrize(
        "S,K,T,r,q",
        [
            (100, 100, 1.0, 0.05, 0.02),
            (120, 100, 0.5, 0.03, 0.04),
        ],
    )
    def test_delta_pcp_with_dividends(self, bs, S, K, T, r, q):
        """Delta_call - Delta_put = exp(-qT)."""
        diff = bs.delta(S, K, T, r, "call", q=q) - bs.delta(S, K, T, r, "put", q=q)
        assert np.isclose(diff, np.exp(-q * T), atol=1e-12)


# ============================================================================
# 6. Finite-Difference First-Order Greeks
# ============================================================================
class TestGreeksFiniteDifference:
    @pytest.fixture
    def params(self):
        return dict(S=100.0, K=100.0, T=1.0, r=0.05)

    def test_delta_fd(self, bs, params):
        h = 0.01
        S = params["S"]
        fd = (
            bs.price(S + h, params["K"], params["T"], params["r"], "call")
            - bs.price(S - h, params["K"], params["T"], params["r"], "call")
        ) / (2 * h)
        assert abs(fd - bs.delta(**params, option_type="call")) < 1e-6

    def test_gamma_fd(self, bs, params):
        h = 0.01
        S = params["S"]
        fd = (
            bs.price(S + h, params["K"], params["T"], params["r"], "call")
            - 2 * bs.price(S, params["K"], params["T"], params["r"], "call")
            + bs.price(S - h, params["K"], params["T"], params["r"], "call")
        ) / h**2
        assert abs(fd - bs.gamma(**params, option_type="call")) < 1e-4

    def test_vega_fd(self, bs, params):
        h = 0.0001
        fd = (
            BlackScholesModel(sigma=bs.sigma + h).price(**params, option_type="call")
            - BlackScholesModel(sigma=bs.sigma - h).price(**params, option_type="call")
        ) / (2 * h)
        assert abs(fd - bs.vega(**params, option_type="call")) < 0.01

    def test_theta_fd(self, bs, params):
        h = 1.0 / 365.0
        T = params["T"]
        fd = (
            bs.price(params["S"], params["K"], T - h, params["r"], "call")
            - bs.price(params["S"], params["K"], T, params["r"], "call")
        ) / h
        assert abs(fd - bs.theta(**params, option_type="call")) < 0.05

    def test_rho_fd(self, bs, params):
        h = 0.0001
        fd = (
            bs.price(params["S"], params["K"], params["T"], params["r"] + h, "call")
            - bs.price(params["S"], params["K"], params["T"], params["r"] - h, "call")
        ) / (2 * h)
        assert abs(fd - bs.rho(**params, option_type="call")) < 0.01


# ============================================================================
# 7. BS PDE Satisfaction
# ============================================================================
class TestBSPDE:
    """theta + r*S*delta + 0.5*sigma^2*S^2*gamma = r*V (exactly)."""

    @pytest.mark.parametrize(
        "S,K,T,r,otype",
        [
            (100, 100, 1.0, 0.05, "call"),
            (100, 100, 1.0, 0.05, "put"),
            (120, 100, 0.5, 0.03, "call"),
            (80, 100, 2.0, 0.08, "put"),
            (50, 50, 0.25, 0.01, "call"),
        ],
    )
    def test_pde(self, bs, S, K, T, r, otype):
        V = bs.price(S, K, T, r, otype)
        lhs = (
            bs.theta(S, K, T, r, otype)
            + r * S * bs.delta(S, K, T, r, otype)
            + 0.5 * bs.sigma**2 * S**2 * bs.gamma(S, K, T, r, otype)
        )
        assert abs(lhs - r * V) < 1e-10

    @pytest.mark.parametrize(
        "S,K,T,r,q,otype",
        [
            (100, 100, 1.0, 0.05, 0.03, "call"),
            (100, 100, 1.0, 0.05, 0.03, "put"),
        ],
    )
    def test_pde_with_dividends(self, bs, S, K, T, r, q, otype):
        """theta + (r-q)*S*delta + 0.5*sigma^2*S^2*gamma = r*V with dividends."""
        V = bs.price(S, K, T, r, otype, q=q)
        lhs = (
            bs.theta(S, K, T, r, otype, q=q)
            + (r - q) * S * bs.delta(S, K, T, r, otype, q=q)
            + 0.5 * bs.sigma**2 * S**2 * bs.gamma(S, K, T, r, otype, q=q)
        )
        assert abs(lhs - r * V) < 1e-10


# ============================================================================
# 8. Homogeneity
# ============================================================================
class TestHomogeneity:
    """C(aS, aK) = a * C(S, K)."""

    @pytest.mark.parametrize("scale", [0.5, 2.0, 10.0])
    def test_call_homogeneity(self, bs, scale):
        base = bs.price(HULL_S, HULL_K, HULL_T, HULL_R, "call")
        assert np.isclose(
            bs.price(scale * HULL_S, scale * HULL_K, HULL_T, HULL_R, "call"),
            scale * base,
            rtol=1e-12,
        )

    @pytest.mark.parametrize("scale", [0.5, 2.0, 10.0])
    def test_put_homogeneity(self, bs, scale):
        base = bs.price(HULL_S, HULL_K, HULL_T, HULL_R, "put")
        assert np.isclose(
            bs.price(scale * HULL_S, scale * HULL_K, HULL_T, HULL_R, "put"),
            scale * base,
            rtol=1e-12,
        )


# ============================================================================
# 9. Price Bounds
# ============================================================================
class TestPriceBounds:
    @pytest.mark.parametrize(
        "S,K,T,r",
        [
            (100, 100, 1.0, 0.05),
            (120, 80, 0.5, 0.03),
            (80, 120, 2.0, 0.08),
        ],
    )
    def test_call_bounds(self, bs, S, K, T, r):
        call = bs.price(S, K, T, r, "call")
        assert call <= S + 1e-10
        assert call >= max(S - K * np.exp(-r * T), 0.0) - 1e-10

    @pytest.mark.parametrize(
        "S,K,T,r",
        [
            (100, 100, 1.0, 0.05),
            (120, 80, 0.5, 0.03),
            (80, 120, 2.0, 0.08),
        ],
    )
    def test_put_bounds(self, bs, S, K, T, r):
        put = bs.price(S, K, T, r, "put")
        assert put <= K * np.exp(-r * T) + 1e-10
        assert put >= max(K * np.exp(-r * T) - S, 0.0) - 1e-10


# ============================================================================
# 10. Vectorization
# ============================================================================
class TestVectorization:
    def test_price_vector_matches_scalar(self, bs):
        spots = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        vec = bs.price(spots, HULL_K, HULL_T, HULL_R, "call")
        for i, S in enumerate(spots):
            assert np.isclose(vec[i], bs.price(S, HULL_K, HULL_T, HULL_R, "call"), rtol=1e-14)

    def test_greeks_vector_matches_scalar(self, bs):
        spots = np.array([80.0, 100.0, 120.0])
        for name in ("delta", "gamma", "vega", "theta", "rho"):
            method = getattr(bs, name)
            vec = method(spots, HULL_K, HULL_T, HULL_R, "call")
            for i, S in enumerate(spots):
                assert np.isclose(vec[i], method(S, HULL_K, HULL_T, HULL_R, "call"), rtol=1e-14)

    def test_returns_scalar_for_scalar_input(self, bs):
        assert isinstance(bs.price(100.0, 100.0, 1.0, 0.05, "call"), float)

    def test_returns_array_for_array_input(self, bs):
        result = bs.price(np.array([80, 100, 120.0]), 100.0, 1.0, 0.05, "call")
        assert isinstance(result, np.ndarray)


# ============================================================================
# 11. Input Validation
# ============================================================================
class TestInputValidation:
    def test_negative_sigma(self):
        with pytest.raises(ValueError):
            BlackScholesModel(sigma=-0.20)

    def test_zero_sigma(self):
        with pytest.raises(ValueError):
            BlackScholesModel(sigma=0.0)

    def test_negative_spot(self, bs):
        with pytest.raises(ValueError):
            bs.price(-100.0, 100.0, 1.0, 0.05, "call")

    def test_negative_time(self, bs):
        with pytest.raises(ValueError):
            bs.price(100.0, 100.0, -1.0, 0.05, "call")

    def test_zero_time(self, bs):
        with pytest.raises(ValueError):
            bs.price(100.0, 100.0, 0.0, 0.05, "call")

    def test_invalid_option_type(self, bs):
        with pytest.raises(ValueError):
            bs.price(100.0, 100.0, 1.0, 0.05, "forward")

    def test_negative_strike(self, bs):
        with pytest.raises(ValueError):
            bs.price(100.0, -10.0, 1.0, 0.05, "call")

    def test_inf_spot_raises(self, bs):
        with pytest.raises(ValueError):
            bs.price(np.inf, 100.0, 1.0, 0.05, "call")

    def test_nan_strike_raises(self, bs):
        with pytest.raises(ValueError):
            bs.price(100.0, np.nan, 1.0, 0.05, "call")

    def test_inf_time_raises(self, bs):
        with pytest.raises(ValueError):
            bs.price(100.0, 100.0, np.inf, 0.05, "call")

    def test_nan_sigma_raises(self):
        with pytest.raises(ValueError):
            BlackScholesModel(sigma=np.nan)

    def test_inf_sigma_raises(self):
        with pytest.raises(ValueError):
            BlackScholesModel(sigma=np.inf)

    def test_large_sigma_warns(self):
        """sigma > 5.0 should emit a UserWarning suggesting decimal."""
        with pytest.warns(UserWarning, match="unusually large"):
            BlackScholesModel(sigma=20.0)

    def test_shorthand_option_types(self, bs):
        """'c' and 'p' shortcuts must work."""
        assert np.isclose(bs.price(100, 100, 1.0, 0.05, "c"), bs.price(100, 100, 1.0, 0.05, "call"))
        assert np.isclose(bs.price(100, 100, 1.0, 0.05, "p"), bs.price(100, 100, 1.0, 0.05, "put"))


# ============================================================================
# 12. Dividend Yield
# ============================================================================
class TestDividendYield:
    def test_dividend_reduces_call_price(self, bs):
        """Higher q -> lower call price."""
        c0 = bs.price(100, 100, 1.0, 0.05, "call", q=0.0)
        c3 = bs.price(100, 100, 1.0, 0.05, "call", q=0.03)
        assert c3 < c0

    def test_dividend_increases_put_price(self, bs):
        """Higher q -> higher put price."""
        p0 = bs.price(100, 100, 1.0, 0.05, "put", q=0.0)
        p3 = bs.price(100, 100, 1.0, 0.05, "put", q=0.03)
        assert p3 > p0

    def test_delta_reduced_by_dividend(self, bs):
        """Call delta < 1 even deep ITM when q > 0 (discounted by exp(-qT))."""
        d = bs.delta(1000, 100, 1.0, 0.05, "call", q=0.05)
        assert d < 1.0
        assert np.isclose(d, np.exp(-0.05), atol=0.001)

    def test_pde_with_dividends(self, bs):
        """theta + (r-q)*S*delta + 0.5*sigma^2*S^2*gamma = r*V."""
        S, K, T, r, q = 100, 100, 1.0, 0.05, 0.03
        V = bs.price(S, K, T, r, "call", q=q)
        lhs = (
            bs.theta(S, K, T, r, "call", q=q)
            + (r - q) * S * bs.delta(S, K, T, r, "call", q=q)
            + 0.5 * bs.sigma**2 * S**2 * bs.gamma(S, K, T, r, "call", q=q)
        )
        assert abs(lhs - r * V) < 1e-10

    def test_known_dividend_price(self):
        """Known value: S=100, K=100, T=1, r=5%, q=3%, sigma=20%."""
        bs = BlackScholesModel(sigma=0.20)
        # Verified against QuantLib/Hull tables
        call = bs.price(100, 100, 1.0, 0.05, "call", q=0.03)
        assert abs(call - 8.6525) < 0.01  # Merton model with q=3%


# ============================================================================
# 13. Implied Volatility
# ============================================================================
class TestImpliedVol:
    def test_roundtrip_atm(self, bs):
        """Price -> IV -> Price should recover."""
        price = bs.price(100, 100, 1.0, 0.05, "call")
        iv = BlackScholesModel.implied_vol(price, 100, 100, 1.0, 0.05, "call")
        assert abs(iv - 0.20) < 1e-10

    def test_roundtrip_itm(self, bs):
        price = bs.price(120, 100, 1.0, 0.05, "call")
        iv = BlackScholesModel.implied_vol(price, 120, 100, 1.0, 0.05, "call")
        assert abs(iv - 0.20) < 1e-10

    def test_roundtrip_otm(self, bs):
        price = bs.price(80, 100, 1.0, 0.05, "call")
        iv = BlackScholesModel.implied_vol(price, 80, 100, 1.0, 0.05, "call")
        assert abs(iv - 0.20) < 1e-8

    def test_roundtrip_put(self, bs):
        price = bs.price(100, 100, 1.0, 0.05, "put")
        iv = BlackScholesModel.implied_vol(price, 100, 100, 1.0, 0.05, "put")
        assert abs(iv - 0.20) < 1e-10

    def test_roundtrip_with_dividends(self):
        bs = BlackScholesModel(sigma=0.30)
        price = bs.price(100, 100, 1.0, 0.05, "call", q=0.03)
        iv = BlackScholesModel.implied_vol(price, 100, 100, 1.0, 0.05, "call", q=0.03)
        assert abs(iv - 0.30) < 1e-10

    def test_put_call_iv_parity(self, bs):
        """IV from call == IV from put for same strike."""
        call_p = bs.price(100, 100, 1.0, 0.05, "call")
        put_p = bs.price(100, 100, 1.0, 0.05, "put")
        iv_call = BlackScholesModel.implied_vol(call_p, 100, 100, 1.0, 0.05, "call")
        iv_put = BlackScholesModel.implied_vol(put_p, 100, 100, 1.0, 0.05, "put")
        assert abs(iv_call - iv_put) < 1e-10

    def test_high_vol(self):
        """Should handle high IV correctly."""
        bs = BlackScholesModel(sigma=2.0)
        price = bs.price(100, 100, 1.0, 0.05, "call")
        iv = BlackScholesModel.implied_vol(price, 100, 100, 1.0, 0.05, "call")
        assert abs(iv - 2.0) < 1e-8

    def test_low_vol(self):
        """Should handle low IV correctly."""
        bs = BlackScholesModel(sigma=0.01)
        price = bs.price(100, 100, 1.0, 0.05, "call")
        iv = BlackScholesModel.implied_vol(price, 100, 100, 1.0, 0.05, "call")
        assert abs(iv - 0.01) < 1e-8

    def test_below_lower_bound_raises(self):
        """Deep ITM call: price below intrinsic forward value should raise."""
        with pytest.raises(ValueError):
            # Intrinsic forward = 120 - 100*exp(-0.05) ~ 24.88. Price 1.0 is below.
            BlackScholesModel.implied_vol(1.0, 120, 100, 1.0, 0.05, "call")

    def test_above_upper_bound_raises(self):
        with pytest.raises(ValueError):
            BlackScholesModel.implied_vol(101.0, 100, 100, 1.0, 0.05, "call")

    def test_roundtrip_deep_otm(self):
        """Far-from-ATM uses asymptotic initial guess (sqrt(2|x|) / sqrt(T)).

        Requires |x| = |ln(F/K)| >= 0.5 to trigger the far-from-ATM branch.
        S=30, K=100 gives x ≈ -1.15.
        """
        bs = BlackScholesModel(sigma=0.40)
        price = bs.price(30, 100, 1.0, 0.05, "call")
        iv = BlackScholesModel.implied_vol(price, 30, 100, 1.0, 0.05, "call")
        assert abs(iv - 0.40) < 1e-4

    def test_roundtrip_deep_itm_put(self):
        """Deep ITM put: |ln(F/K)| >> 0.5, exercises far-from-ATM branch."""
        bs = BlackScholesModel(sigma=0.35)
        price = bs.price(30, 100, 1.0, 0.05, "put")
        iv = BlackScholesModel.implied_vol(price, 30, 100, 1.0, 0.05, "put")
        assert abs(iv - 0.35) < 1e-4

    def test_brent_fallback(self):
        """Force Halley to fail by using max_iter=0, triggering Brent."""
        bs = BlackScholesModel(sigma=0.20)
        price = bs.price(100, 100, 1.0, 0.05, "call")
        iv = BlackScholesModel.implied_vol(price, 100, 100, 1.0, 0.05, "call", max_iter=0)
        assert abs(iv - 0.20) < 1e-6


# ============================================================================
# 14. Second-Order Greeks (Finite Difference)
# ============================================================================
class TestSecondOrderGreeks:
    """Validate all second-order Greeks against finite differences."""

    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0

    def test_vanna_fd(self, bs):
        """Vanna = dVega/dS."""
        h = 0.01
        fd = (
            bs.vega(self.S + h, self.K, self.T, self.r, "call")
            - bs.vega(self.S - h, self.K, self.T, self.r, "call")
        ) / (2 * h)
        assert abs(fd - bs.vanna(self.S, self.K, self.T, self.r, "call")) < 1e-4

    def test_volga_fd(self, bs):
        """Volga = dVega/dsigma."""
        h = 0.0001
        v_up = BlackScholesModel(sigma=bs.sigma + h).vega(self.S, self.K, self.T, self.r, "call")
        v_dn = BlackScholesModel(sigma=bs.sigma - h).vega(self.S, self.K, self.T, self.r, "call")
        fd = (v_up - v_dn) / (2 * h)
        assert abs(fd - bs.volga(self.S, self.K, self.T, self.r, "call")) < 0.1

    def test_charm_fd(self, bs):
        """Charm = dDelta/dt. As calendar time advances by h, T decreases by h."""
        h = 1.0 / 365.0
        T = self.T
        fd = (
            bs.delta(self.S, self.K, T - h, self.r, "call")
            - bs.delta(self.S, self.K, T, self.r, "call")
        ) / h
        assert abs(fd - bs.charm(self.S, self.K, T, self.r, "call")) < 0.01

    def test_speed_fd(self, bs):
        """Speed = dGamma/dS."""
        h = 0.01
        fd = (
            bs.gamma(self.S + h, self.K, self.T, self.r, "call")
            - bs.gamma(self.S - h, self.K, self.T, self.r, "call")
        ) / (2 * h)
        assert abs(fd - bs.speed(self.S, self.K, self.T, self.r, "call")) < 1e-6

    def test_zomma_fd(self, bs):
        """Zomma = dGamma/dsigma."""
        h = 0.0001
        g_up = BlackScholesModel(sigma=bs.sigma + h).gamma(self.S, self.K, self.T, self.r, "call")
        g_dn = BlackScholesModel(sigma=bs.sigma - h).gamma(self.S, self.K, self.T, self.r, "call")
        fd = (g_up - g_dn) / (2 * h)
        assert abs(fd - bs.zomma(self.S, self.K, self.T, self.r, "call")) < 1e-4

    def test_color_fd(self, bs):
        """Color = dGamma/dt = -dGamma/dT."""
        h = 1.0 / 365.0
        T = self.T
        fd = (
            -(
                bs.gamma(self.S, self.K, T - h, self.r, "call")
                - bs.gamma(self.S, self.K, T, self.r, "call")
            )
            / h
        )
        assert abs(fd - bs.color(self.S, self.K, T, self.r, "call")) < 0.001

    def test_vanna_same_for_call_put(self, bs):
        assert np.isclose(
            bs.vanna(self.S, self.K, self.T, self.r, "call"),
            bs.vanna(self.S, self.K, self.T, self.r, "put"),
            rtol=1e-12,
        )

    def test_volga_same_for_call_put(self, bs):
        assert np.isclose(
            bs.volga(self.S, self.K, self.T, self.r, "call"),
            bs.volga(self.S, self.K, self.T, self.r, "put"),
            rtol=1e-12,
        )

    # --- Second-order Greeks with dividends (q > 0) ---

    @pytest.mark.parametrize("q", [0.02, 0.05])
    def test_vanna_fd_with_dividends(self, bs, q):
        """Vanna = dVega/dS with continuous dividend yield."""
        h = 0.01
        fd = (
            bs.vega(self.S + h, self.K, self.T, self.r, "call", q=q)
            - bs.vega(self.S - h, self.K, self.T, self.r, "call", q=q)
        ) / (2 * h)
        assert abs(fd - bs.vanna(self.S, self.K, self.T, self.r, "call", q=q)) < 1e-4

    @pytest.mark.parametrize("q", [0.02, 0.05])
    def test_charm_fd_with_dividends(self, bs, q):
        """Charm = dDelta/dt with continuous dividend yield."""
        h = 1.0 / 365.0
        fd = (
            bs.delta(self.S, self.K, self.T - h, self.r, "call", q=q)
            - bs.delta(self.S, self.K, self.T, self.r, "call", q=q)
        ) / h
        assert abs(fd - bs.charm(self.S, self.K, self.T, self.r, "call", q=q)) < 0.01

    @pytest.mark.parametrize("q", [0.02, 0.05])
    def test_speed_fd_with_dividends(self, bs, q):
        """Speed = dGamma/dS with continuous dividend yield."""
        h = 0.01
        fd = (
            bs.gamma(self.S + h, self.K, self.T, self.r, "call", q=q)
            - bs.gamma(self.S - h, self.K, self.T, self.r, "call", q=q)
        ) / (2 * h)
        assert abs(fd - bs.speed(self.S, self.K, self.T, self.r, "call", q=q)) < 1e-6

    @pytest.mark.parametrize("q", [0.02, 0.05])
    def test_color_fd_with_dividends(self, bs, q):
        """Color = -dGamma/dT with continuous dividend yield."""
        h = 1.0 / 365.0
        fd = (
            -(
                bs.gamma(self.S, self.K, self.T - h, self.r, "call", q=q)
                - bs.gamma(self.S, self.K, self.T, self.r, "call", q=q)
            )
            / h
        )
        assert abs(fd - bs.color(self.S, self.K, self.T, self.r, "call", q=q)) < 0.001


# ============================================================================
# 15. Dual Greeks (FD)
# ============================================================================
class TestDualGreeks:
    S, K, T, r = 100.0, 100.0, 1.0, 0.05

    def test_dual_delta_fd(self, bs):
        """Dual delta = dV/dK."""
        h = 0.01
        fd = (
            bs.price(self.S, self.K + h, self.T, self.r, "call")
            - bs.price(self.S, self.K - h, self.T, self.r, "call")
        ) / (2 * h)
        assert abs(fd - bs.dual_delta(self.S, self.K, self.T, self.r, "call")) < 1e-6

    def test_dual_gamma_fd(self, bs):
        """Dual gamma = d²V/dK²."""
        h = 0.01
        fd = (
            bs.price(self.S, self.K + h, self.T, self.r, "call")
            - 2 * bs.price(self.S, self.K, self.T, self.r, "call")
            + bs.price(self.S, self.K - h, self.T, self.r, "call")
        ) / h**2
        assert abs(fd - bs.dual_gamma(self.S, self.K, self.T, self.r, "call")) < 1e-4

    def test_dual_delta_call_negative(self, bs):
        """dC/dK < 0 (higher strike -> lower call value)."""
        assert bs.dual_delta(self.S, self.K, self.T, self.r, "call") < 0

    def test_dual_delta_put_positive(self, bs):
        """dP/dK > 0 (higher strike -> higher put value)."""
        assert bs.dual_delta(self.S, self.K, self.T, self.r, "put") > 0

    def test_dual_gamma_positive(self, bs):
        """d²V/dK² > 0 (convexity in strike)."""
        assert bs.dual_gamma(self.S, self.K, self.T, self.r, "call") > 0

    def test_breeden_litzenberger(self, bs):
        """Dual gamma = exp(-rT) * risk-neutral density at K."""
        dg = bs.dual_gamma(self.S, self.K, self.T, self.r, "call")
        prob_itm = bs.probability_itm(self.S, self.K, self.T, self.r, "call")
        # Dual gamma should be positive where density is positive
        assert dg > 0
        assert 0 < prob_itm < 1


# ============================================================================
# 16. Monte Carlo Cross-Validation
# ============================================================================
class TestMonteCarloCrossValidation:
    """MC prices must converge to BS analytical prices."""

    def _mc_price(self, S, K, T, r, sigma, option_type, q=0.0, N=2_000_000, seed=42):
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal(N)
        ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        if option_type == "call":
            payoff = np.maximum(ST - K, 0)
        else:
            payoff = np.maximum(K - ST, 0)
        mc = np.exp(-r * T) * np.mean(payoff)
        se = np.exp(-r * T) * np.std(payoff) / np.sqrt(N)
        return mc, se

    def test_mc_call_atm(self, bs):
        mc, se = self._mc_price(100, 100, 1.0, 0.05, 0.20, "call")
        bs_price = bs.price(100, 100, 1.0, 0.05, "call")
        assert abs(mc - bs_price) < 4 * se

    def test_mc_put_atm(self, bs):
        mc, se = self._mc_price(100, 100, 1.0, 0.05, 0.20, "put")
        bs_price = bs.price(100, 100, 1.0, 0.05, "put")
        assert abs(mc - bs_price) < 4 * se

    def test_mc_call_otm(self, bs):
        mc, se = self._mc_price(80, 100, 1.0, 0.05, 0.20, "call")
        bs_price = bs.price(80, 100, 1.0, 0.05, "call")
        assert abs(mc - bs_price) < 4 * se

    def test_mc_call_with_dividends(self):
        bs = BlackScholesModel(sigma=0.20)
        mc, se = self._mc_price(100, 100, 1.0, 0.05, 0.20, "call", q=0.03)
        bs_price = bs.price(100, 100, 1.0, 0.05, "call", q=0.03)
        assert abs(mc - bs_price) < 4 * se

    def test_mc_put_with_dividends(self):
        bs = BlackScholesModel(sigma=0.20)
        mc, se = self._mc_price(100, 100, 1.0, 0.05, 0.20, "put", q=0.03)
        bs_price = bs.price(100, 100, 1.0, 0.05, "put", q=0.03)
        assert abs(mc - bs_price) < 4 * se


# ============================================================================
# 17. Extreme Parameters
# ============================================================================
class TestExtremeParameters:
    def test_very_high_vol(self):
        """sigma=3.0 (300%) should still give valid prices."""
        bs = BlackScholesModel(sigma=3.0)
        call = bs.price(100, 100, 1.0, 0.05, "call")
        assert 0 < call <= 100
        assert np.isfinite(call)

    def test_very_low_vol(self):
        """sigma=0.001 (0.1%) should approximate intrinsic."""
        bs = BlackScholesModel(sigma=0.001)
        call = bs.price(110, 100, 1.0, 0.05, "call")
        intrinsic = 110 - 100 * np.exp(-0.05)
        assert abs(call - intrinsic) < 0.1

    def test_very_short_expiry(self, bs):
        """T=1/365/24 (1 hour). ITM call ~ intrinsic."""
        T = 1.0 / 365.0 / 24.0
        call = bs.price(110, 100, T, 0.05, "call")
        assert abs(call - 10.0) < 0.5

    def test_very_long_expiry(self, bs):
        """T=30 years. Call should be between intrinsic and S."""
        call = bs.price(100, 100, 30.0, 0.05, "call")
        assert 0 < call <= 100
        assert np.isfinite(call)

    def test_zero_rate(self, bs):
        """r=0 should still produce valid prices satisfying PCP."""
        call = bs.price(100, 100, 1.0, 0.0, "call")
        put = bs.price(100, 100, 1.0, 0.0, "put")
        assert abs((call - put) - 0.0) < 1e-10  # S - K*exp(0) = 0 for ATM

    def test_all_greeks_finite(self, bs):
        """All Greeks should be finite for standard parameters."""
        g = bs.greeks(100, 100, 1.0, 0.05, "call")
        for name, val in g.items():
            assert np.isfinite(val), f"{name} is not finite: {val}"


# ============================================================================
# 18. Batch Greeks Consistency
# ============================================================================
class TestBatchGreeks:
    """greeks() batch method must match individual method calls."""

    @pytest.mark.parametrize("otype", ["call", "put"])
    def test_batch_matches_individual(self, bs, otype):
        S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0
        g = bs.greeks(S, K, T, r, otype, q=q)

        assert np.isclose(g["price"], bs.price(S, K, T, r, otype, q=q), rtol=1e-14)
        assert np.isclose(g["delta"], bs.delta(S, K, T, r, otype, q=q), rtol=1e-14)
        assert np.isclose(g["gamma"], bs.gamma(S, K, T, r, otype, q=q), rtol=1e-14)
        assert np.isclose(g["vega"], bs.vega(S, K, T, r, otype, q=q), rtol=1e-14)
        assert np.isclose(g["theta"], bs.theta(S, K, T, r, otype, q=q), rtol=1e-14)
        assert np.isclose(g["rho"], bs.rho(S, K, T, r, otype, q=q), rtol=1e-14)
        assert np.isclose(g["vanna"], bs.vanna(S, K, T, r, otype, q=q), rtol=1e-14)
        assert np.isclose(g["volga"], bs.volga(S, K, T, r, otype, q=q), rtol=1e-14)
        assert np.isclose(g["charm"], bs.charm(S, K, T, r, otype, q=q), rtol=1e-14)
        assert np.isclose(g["speed"], bs.speed(S, K, T, r, otype, q=q), rtol=1e-14)
        assert np.isclose(g["zomma"], bs.zomma(S, K, T, r, otype, q=q), rtol=1e-14)
        assert np.isclose(g["color"], bs.color(S, K, T, r, otype, q=q), rtol=1e-14)
        assert np.isclose(g["dual_delta"], bs.dual_delta(S, K, T, r, otype, q=q), rtol=1e-14)
        assert np.isclose(g["dual_gamma"], bs.dual_gamma(S, K, T, r, otype, q=q), rtol=1e-14)

    def test_batch_with_dividends(self, bs):
        g = bs.greeks(100, 100, 1.0, 0.05, "call", q=0.03)
        assert np.isclose(g["price"], bs.price(100, 100, 1.0, 0.05, "call", q=0.03), rtol=1e-14)
        assert np.isclose(g["vanna"], bs.vanna(100, 100, 1.0, 0.05, "call", q=0.03), rtol=1e-14)

    def test_batch_returns_dict(self, bs):
        g = bs.greeks(100, 100, 1.0, 0.05, "call")
        assert isinstance(g, dict)
        expected_keys = {
            "price",
            "delta",
            "gamma",
            "vega",
            "theta",
            "rho",
            "vanna",
            "volga",
            "charm",
            "speed",
            "zomma",
            "color",
            "dual_delta",
            "dual_gamma",
            "probability_itm",
            "elasticity",
        }
        assert expected_keys == set(g.keys())


# ============================================================================
# 19. Elasticity
# ============================================================================
class TestElasticity:
    """Lambda = Delta * S / V."""

    def test_elasticity_call_atm(self, bs):
        e = bs.elasticity(100, 100, 1.0, 0.05, "call")
        delta = bs.delta(100, 100, 1.0, 0.05, "call")
        price = bs.price(100, 100, 1.0, 0.05, "call")
        assert np.isclose(e, delta * 100.0 / price, rtol=1e-12)

    def test_elasticity_deep_otm_is_nan(self, bs):
        """Deep OTM: price ≈ 0 -> elasticity is NaN (division by zero)."""
        e = bs.elasticity(10, 1000, 0.01, 0.05, "call")
        assert np.isnan(e)

    def test_elasticity_leverage_positive(self, bs):
        """Call elasticity > 1 (leveraged exposure)."""
        e = bs.elasticity(100, 100, 1.0, 0.05, "call")
        assert e > 1.0


# ============================================================================
# 20. Dual Greeks with Dividends (FD)
# ============================================================================
class TestDualGreeksDividends:
    S, K, T, r = 100.0, 100.0, 1.0, 0.05

    @pytest.mark.parametrize("q", [0.02, 0.05])
    def test_dual_delta_fd_with_dividends(self, bs, q):
        h = 0.01
        fd = (
            bs.price(self.S, self.K + h, self.T, self.r, "call", q=q)
            - bs.price(self.S, self.K - h, self.T, self.r, "call", q=q)
        ) / (2 * h)
        assert abs(fd - bs.dual_delta(self.S, self.K, self.T, self.r, "call", q=q)) < 1e-6

    @pytest.mark.parametrize("q", [0.02, 0.05])
    def test_dual_gamma_fd_with_dividends(self, bs, q):
        h = 0.01
        fd = (
            bs.price(self.S, self.K + h, self.T, self.r, "call", q=q)
            - 2 * bs.price(self.S, self.K, self.T, self.r, "call", q=q)
            + bs.price(self.S, self.K - h, self.T, self.r, "call", q=q)
        ) / h**2
        assert abs(fd - bs.dual_gamma(self.S, self.K, self.T, self.r, "call", q=q)) < 1e-4


# ============================================================================
# 21. Model Identity
# ============================================================================
class TestModelIdentity:
    """__repr__, __eq__, __hash__ for interoperability."""

    def test_repr(self):
        bs = BlackScholesModel(sigma=0.20)
        assert repr(bs) == "BlackScholesModel(sigma=0.2)"

    def test_eq_same_sigma(self):
        assert BlackScholesModel(sigma=0.20) == BlackScholesModel(sigma=0.20)

    def test_eq_different_sigma(self):
        assert BlackScholesModel(sigma=0.20) != BlackScholesModel(sigma=0.30)

    def test_eq_different_type(self):
        assert BlackScholesModel(sigma=0.20) != "not a model"

    def test_hash_consistency(self):
        """Equal objects must have equal hashes (dict/set compatibility)."""
        a = BlackScholesModel(sigma=0.20)
        b = BlackScholesModel(sigma=0.20)
        assert hash(a) == hash(b)

    def test_usable_as_dict_key(self):
        bs1 = BlackScholesModel(sigma=0.20)
        bs2 = BlackScholesModel(sigma=0.30)
        d = {bs1: "vol_20", bs2: "vol_30"}
        assert d[BlackScholesModel(sigma=0.20)] == "vol_20"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
