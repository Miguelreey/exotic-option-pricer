"""
Exotic Option Pricer — tests/test_properties.py

Property-based tests using Hypothesis.
Verifies mathematical invariants hold for ALL valid parameter combinations,
not just hand-picked test cases.

References
----------
.. [1] MacKinlay & Lo (1997). "The Econometrics of Financial Markets."
.. [2] Hull (2018). Options, Futures & Other Derivatives, 10th ed.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hypothesis import given, settings
from hypothesis.strategies import floats

from exotic_option_pricer.models.black_scholes import BlackScholesModel

# ============================================================================
# Hypothesis strategies for valid financial parameters
# ============================================================================
spot = floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False)
strike = floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False)
time_to_exp = floats(min_value=0.01, max_value=30.0, allow_nan=False, allow_infinity=False)
rate = floats(min_value=-0.05, max_value=0.30, allow_nan=False, allow_infinity=False)
vol = floats(min_value=0.01, max_value=3.0, allow_nan=False, allow_infinity=False)
div_yield = floats(min_value=0.0, max_value=0.15, allow_nan=False, allow_infinity=False)
scale_factor = floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)


# ============================================================================
# 1. Put-Call Parity — must hold universally
# ============================================================================
class TestPCPProperty:
    """C - P = S*exp(-qT) - K*exp(-rT) for all valid (S, K, T, r, sigma, q)."""

    @given(S=spot, K=strike, T=time_to_exp, r=rate, sigma=vol)
    @settings(max_examples=500)
    def test_pcp_no_dividends(self, S, K, T, r, sigma):
        bs = BlackScholesModel(sigma=sigma)
        call = bs.price(S, K, T, r, 'call')
        put = bs.price(S, K, T, r, 'put')
        parity = S - K * np.exp(-r * T)
        assert abs((call - put) - parity) < 1e-8

    @given(S=spot, K=strike, T=time_to_exp, r=rate, sigma=vol, q=div_yield)
    @settings(max_examples=500)
    def test_pcp_with_dividends(self, S, K, T, r, sigma, q):
        bs = BlackScholesModel(sigma=sigma)
        call = bs.price(S, K, T, r, 'call', q=q)
        put = bs.price(S, K, T, r, 'put', q=q)
        parity = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert abs((call - put) - parity) < 1e-8


# ============================================================================
# 2. No-Arbitrage Price Bounds
# ============================================================================
class TestBoundsProperty:
    """Prices must respect no-arbitrage bounds for all parameters."""

    @given(S=spot, K=strike, T=time_to_exp, r=rate, sigma=vol)
    @settings(max_examples=500)
    def test_call_bounds(self, S, K, T, r, sigma):
        bs = BlackScholesModel(sigma=sigma)
        call = bs.price(S, K, T, r, 'call')
        assert call >= -1e-10
        assert call <= S + 1e-10
        assert call >= max(S - K * np.exp(-r * T), 0.0) - 1e-10

    @given(S=spot, K=strike, T=time_to_exp, r=rate, sigma=vol)
    @settings(max_examples=500)
    def test_put_bounds(self, S, K, T, r, sigma):
        bs = BlackScholesModel(sigma=sigma)
        put = bs.price(S, K, T, r, 'put')
        assert put >= -1e-10
        assert put <= K * np.exp(-r * T) + 1e-10
        assert put >= max(K * np.exp(-r * T) - S, 0.0) - 1e-10


# ============================================================================
# 3. Greek Bounds — universal constraints
# ============================================================================
class TestGreekBoundsProperty:

    @given(S=spot, K=strike, T=time_to_exp, r=rate, sigma=vol)
    @settings(max_examples=500)
    def test_call_delta_in_01(self, S, K, T, r, sigma):
        bs = BlackScholesModel(sigma=sigma)
        d = bs.delta(S, K, T, r, 'call')
        assert -1e-10 <= d <= 1.0 + 1e-10

    @given(S=spot, K=strike, T=time_to_exp, r=rate, sigma=vol)
    @settings(max_examples=500)
    def test_put_delta_in_neg1_0(self, S, K, T, r, sigma):
        bs = BlackScholesModel(sigma=sigma)
        d = bs.delta(S, K, T, r, 'put')
        assert -1.0 - 1e-10 <= d <= 1e-10

    @given(S=spot, K=strike, T=time_to_exp, r=rate, sigma=vol)
    @settings(max_examples=500)
    def test_gamma_nonnegative(self, S, K, T, r, sigma):
        bs = BlackScholesModel(sigma=sigma)
        assert bs.gamma(S, K, T, r, 'call') >= -1e-15

    @given(S=spot, K=strike, T=time_to_exp, r=rate, sigma=vol)
    @settings(max_examples=500)
    def test_vega_nonnegative(self, S, K, T, r, sigma):
        bs = BlackScholesModel(sigma=sigma)
        assert bs.vega(S, K, T, r, 'call') >= -1e-15


# ============================================================================
# 4. Greek Symmetry (PCP derivatives)
# ============================================================================
class TestSymmetryProperty:
    """Gamma and vega must be identical for calls and puts."""

    @given(S=spot, K=strike, T=time_to_exp, r=rate, sigma=vol)
    @settings(max_examples=500)
    def test_gamma_call_equals_put(self, S, K, T, r, sigma):
        bs = BlackScholesModel(sigma=sigma)
        assert np.isclose(bs.gamma(S, K, T, r, 'call'),
                          bs.gamma(S, K, T, r, 'put'), rtol=1e-10)

    @given(S=spot, K=strike, T=time_to_exp, r=rate, sigma=vol)
    @settings(max_examples=500)
    def test_vega_call_equals_put(self, S, K, T, r, sigma):
        bs = BlackScholesModel(sigma=sigma)
        assert np.isclose(bs.vega(S, K, T, r, 'call'),
                          bs.vega(S, K, T, r, 'put'), rtol=1e-10)

    @given(S=spot, K=strike, T=time_to_exp, r=rate, sigma=vol)
    @settings(max_examples=500)
    def test_delta_call_minus_put(self, S, K, T, r, sigma):
        """Delta_call - Delta_put = 1 (no dividends)."""
        bs = BlackScholesModel(sigma=sigma)
        diff = bs.delta(S, K, T, r, 'call') - bs.delta(S, K, T, r, 'put')
        assert np.isclose(diff, 1.0, atol=1e-10)

    @given(S=spot, K=strike, T=time_to_exp, r=rate, sigma=vol, q=div_yield)
    @settings(max_examples=500)
    def test_delta_pcp_with_dividends(self, S, K, T, r, sigma, q):
        """Delta_call - Delta_put = exp(-qT) with dividends."""
        bs = BlackScholesModel(sigma=sigma)
        diff = bs.delta(S, K, T, r, 'call', q=q) - bs.delta(S, K, T, r, 'put', q=q)
        assert np.isclose(diff, np.exp(-q * T), atol=1e-10)


# ============================================================================
# 5. Homogeneity of Degree 1
# ============================================================================
class TestHomogeneityProperty:
    """C(aS, aK) = a * C(S, K) for all a > 0."""

    @given(S=spot, K=strike, T=time_to_exp, r=rate, sigma=vol, a=scale_factor)
    @settings(max_examples=500)
    def test_call_homogeneity(self, S, K, T, r, sigma, a):
        bs = BlackScholesModel(sigma=sigma)
        base = bs.price(S, K, T, r, 'call')
        scaled = bs.price(a * S, a * K, T, r, 'call')
        assert np.isclose(scaled, a * base, rtol=1e-10)

    @given(S=spot, K=strike, T=time_to_exp, r=rate, sigma=vol, a=scale_factor)
    @settings(max_examples=500)
    def test_put_homogeneity(self, S, K, T, r, sigma, a):
        bs = BlackScholesModel(sigma=sigma)
        base = bs.price(S, K, T, r, 'put')
        scaled = bs.price(a * S, a * K, T, r, 'put')
        assert np.isclose(scaled, a * base, rtol=1e-10)


# ============================================================================
# 6. BS PDE Satisfaction
# ============================================================================
class TestPDEProperty:
    """theta + r*S*delta + 0.5*sigma^2*S^2*gamma = r*V for all parameters."""

    @given(S=spot, K=strike, T=time_to_exp, r=rate, sigma=vol)
    @settings(max_examples=500)
    def test_pde_no_dividends(self, S, K, T, r, sigma):
        bs = BlackScholesModel(sigma=sigma)
        for otype in ('call', 'put'):
            V = bs.price(S, K, T, r, otype)
            lhs = (bs.theta(S, K, T, r, otype)
                   + r * S * bs.delta(S, K, T, r, otype)
                   + 0.5 * sigma**2 * S**2 * bs.gamma(S, K, T, r, otype))
            assert abs(lhs - r * V) < 1e-6 * max(abs(V), 1.0)

    @given(S=spot, K=strike, T=time_to_exp, r=rate, sigma=vol, q=div_yield)
    @settings(max_examples=500)
    def test_pde_with_dividends(self, S, K, T, r, sigma, q):
        bs = BlackScholesModel(sigma=sigma)
        for otype in ('call', 'put'):
            V = bs.price(S, K, T, r, otype, q=q)
            lhs = (bs.theta(S, K, T, r, otype, q=q)
                   + (r - q) * S * bs.delta(S, K, T, r, otype, q=q)
                   + 0.5 * sigma**2 * S**2 * bs.gamma(S, K, T, r, otype, q=q))
            assert abs(lhs - r * V) < 1e-6 * max(abs(V), 1.0)


# ============================================================================
# 7. Implied Vol Roundtrip
# ============================================================================
class TestImpliedVolProperty:

    @given(S=floats(min_value=50.0, max_value=200.0,
                    allow_nan=False, allow_infinity=False),
           moneyness=floats(min_value=0.85, max_value=1.15,
                            allow_nan=False, allow_infinity=False),
           T=floats(min_value=0.25, max_value=3.0,
                    allow_nan=False, allow_infinity=False),
           r=floats(min_value=0.0, max_value=0.10,
                    allow_nan=False, allow_infinity=False),
           sigma=floats(min_value=0.15, max_value=1.0,
                        allow_nan=False, allow_infinity=False))
    @settings(max_examples=500)
    def test_roundtrip(self, S, moneyness, T, r, sigma):
        """price(sigma) -> implied_vol -> sigma for reasonable params.

        Constrains moneyness to 85%-115% and T >= 0.25y to ensure sufficient
        time value — the regime where IV extraction is well-conditioned.
        """
        K = S * moneyness
        bs = BlackScholesModel(sigma=sigma)
        price = bs.price(S, K, T, r, 'call')
        iv = BlackScholesModel.implied_vol(price, S, K, T, r, 'call')
        assert abs(iv - sigma) / sigma < 0.01  # 1% relative tolerance

    @given(S=floats(min_value=50.0, max_value=200.0,
                    allow_nan=False, allow_infinity=False),
           moneyness=floats(min_value=0.85, max_value=1.15,
                            allow_nan=False, allow_infinity=False),
           T=floats(min_value=0.25, max_value=3.0,
                    allow_nan=False, allow_infinity=False),
           r=floats(min_value=0.0, max_value=0.10,
                    allow_nan=False, allow_infinity=False),
           sigma=floats(min_value=0.15, max_value=1.0,
                        allow_nan=False, allow_infinity=False))
    @settings(max_examples=500)
    def test_roundtrip_put(self, S, moneyness, T, r, sigma):
        K = S * moneyness
        bs = BlackScholesModel(sigma=sigma)
        price = bs.price(S, K, T, r, 'put')
        iv = BlackScholesModel.implied_vol(price, S, K, T, r, 'put')
        assert abs(iv - sigma) / sigma < 0.01  # 1% relative tolerance


# ============================================================================
# 8. All Greeks Finite
# ============================================================================
class TestFiniteProperty:
    """All outputs must be finite for all valid inputs."""

    @given(S=spot, K=strike, T=time_to_exp, r=rate, sigma=vol)
    @settings(max_examples=500)
    def test_all_greeks_finite(self, S, K, T, r, sigma):
        bs = BlackScholesModel(sigma=sigma)
        g = bs.greeks(S, K, T, r, 'call')
        # Elasticity is NaN by design for deep OTM options (price ≈ 0 → division by zero)
        for name, val in g.items():
            if name == 'elasticity':
                continue
            assert np.isfinite(val), (
                f"{name} not finite for S={S}, K={K}, T={T}, r={r}, sigma={sigma}"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
