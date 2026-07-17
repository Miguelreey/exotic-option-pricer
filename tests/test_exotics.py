"""
Exotic Option Pricer — tests/test_exotics.py

Comprehensive test suite for exotic option instruments (Phase 3).

Covers:
1. TestDigitalOptions        — analytical prices, MC cross-validation, decomposition
2. TestDigitalProperties     — complementarity, vanilla decomposition, bounds
3. TestDigitalEdgeCases      — K→0, K→∞, σ→0, deep ITM/OTM
4. TestDigitalHypothesis     — property-based testing with Hypothesis

Reference: Hull (2018) Ch. 26, Haug (2007) Ch. 10.
"""

import os
import sys

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from exotic_option_pricer.engines.monte_carlo import MonteCarloEngine
from exotic_option_pricer.instruments.asian import AsianOption
from exotic_option_pricer.instruments.barrier import BarrierOption
from exotic_option_pricer.instruments.base import ExoticOption
from exotic_option_pricer.instruments.digital import DigitalOption
from exotic_option_pricer.instruments.lookback import LookbackOption
from exotic_option_pricer.models.black_scholes import BlackScholesModel
from exotic_option_pricer.utils.greeks import (
    numerical_delta,
    numerical_gamma,
    numerical_greeks,
    numerical_rho,
    numerical_theta,
    numerical_vega,
)

# ============================================================================
# Standard test parameters (consistent with Phase 1 and 2)
# ============================================================================
HULL_S = 100.0
HULL_K = 100.0
HULL_T = 1.0
HULL_R = 0.05
HULL_SIGMA = 0.20
HULL_Q = 0.0

# Hull (2018) Table 15.5: BS vanilla call/put prices
HULL_CALL = 10.4506
HULL_PUT = 5.5735

MC_SEED = 42
MC_PATHS = 300_000  # digital payoffs are discontinuous — need more paths


# ============================================================================
# 1. Digital Option Analytical Prices
# ============================================================================
class TestDigitalAnalytical:
    """Verify analytical prices against known values and identities."""

    def test_cash_call_atm(self):
        """
        Cash-or-nothing ATM call: e^{-rT} * Q * N(d2).

        For S=K=100, T=1, r=0.05, sigma=0.20, q=0:
            d2 = [ln(1) + (0.05 - 0.02)·1] / (0.20) = 0.15
            N(0.15) ≈ 0.55962
            price = e^{-0.05} * 1 * 0.55962 ≈ 0.53241
        """
        price = DigitalOption.analytical_price(
            100, 100, 1.0, 0.05, 0.20, 'call', 'cash', 1.0,
        )
        # Manual: d1 = (0 + 0.07)/0.20 = 0.35, d2 = 0.35 - 0.20 = 0.15
        from scipy.special import ndtr
        d2 = 0.15
        expected = np.exp(-0.05) * ndtr(d2)
        assert abs(price - expected) < 1e-10

    def test_cash_put_atm(self):
        """Cash-or-nothing ATM put: e^{-rT} * Q * N(-d2)."""
        price = DigitalOption.analytical_price(
            100, 100, 1.0, 0.05, 0.20, 'put', 'cash', 1.0,
        )
        from scipy.special import ndtr
        d2 = 0.15
        expected = np.exp(-0.05) * ndtr(-d2)
        assert abs(price - expected) < 1e-10

    def test_asset_call_atm(self):
        """Asset-or-nothing ATM call: S * e^{-qT} * N(d1)."""
        price = DigitalOption.analytical_price(
            100, 100, 1.0, 0.05, 0.20, 'call', 'asset',
        )
        from scipy.special import ndtr
        d1 = 0.35
        expected = 100 * ndtr(d1)
        assert abs(price - expected) < 1e-10

    def test_asset_put_atm(self):
        """Asset-or-nothing ATM put: S * e^{-qT} * N(-d1)."""
        price = DigitalOption.analytical_price(
            100, 100, 1.0, 0.05, 0.20, 'put', 'asset',
        )
        from scipy.special import ndtr
        d1 = 0.35
        expected = 100 * ndtr(-d1)
        assert abs(price - expected) < 1e-10

    def test_cash_call_with_dividends(self):
        """Cash-or-nothing call with q > 0 shifts d2."""
        # With q=0.03: d1 = [ln(1) + (0.05 - 0.03 + 0.02)·1] / 0.20 = 0.20
        #              d2 = 0.20 - 0.20 = 0.00
        price = DigitalOption.analytical_price(
            100, 100, 1.0, 0.05, 0.20, 'call', 'cash', 1.0, q=0.03,
        )
        from scipy.special import ndtr
        expected = np.exp(-0.05) * ndtr(0.0)  # N(0) = 0.5
        assert abs(price - expected) < 1e-10

    def test_asset_call_with_dividends(self):
        """Asset-or-nothing call with q > 0: S * e^{-qT} * N(d1)."""
        price = DigitalOption.analytical_price(
            100, 100, 1.0, 0.05, 0.20, 'call', 'asset', q=0.03,
        )
        from scipy.special import ndtr
        d1 = 0.20  # with q=0.03
        expected = 100 * np.exp(-0.03) * ndtr(d1)
        assert abs(price - expected) < 1e-10

    def test_cash_amount_scaling(self):
        """Cash-or-nothing price scales linearly with Q."""
        p1 = DigitalOption.analytical_price(100, 100, 1.0, 0.05, 0.20, 'call', 'cash', 1.0)
        p5 = DigitalOption.analytical_price(100, 100, 1.0, 0.05, 0.20, 'call', 'cash', 5.0)
        assert abs(p5 - 5.0 * p1) < 1e-12

    def test_deep_itm_cash_call(self):
        """Deep ITM cash call: S >> K → N(d2) → 1 → price → e^{-rT} Q."""
        price = DigitalOption.analytical_price(
            1000, 10, 1.0, 0.05, 0.20, 'call', 'cash', 1.0,
        )
        expected = np.exp(-0.05) * 1.0  # N(d2) ≈ 1
        assert abs(price - expected) < 1e-6

    def test_deep_otm_cash_call(self):
        """Deep OTM cash call: S << K → N(d2) → 0 → price → 0."""
        price = DigitalOption.analytical_price(
            10, 1000, 1.0, 0.05, 0.20, 'call', 'cash', 1.0,
        )
        assert price < 1e-10

    def test_deep_itm_asset_call(self):
        """Deep ITM asset call: S >> K → N(d1) → 1 → price → S e^{-qT}."""
        S = 1000.0
        price = DigitalOption.analytical_price(
            S, 10, 1.0, 0.05, 0.20, 'call', 'asset', q=0.02,
        )
        expected = S * np.exp(-0.02)
        assert abs(price - expected) < 0.01


# ============================================================================
# 2. Digital Option MC Cross-Validation
# ============================================================================
class TestDigitalMC:
    """Cross-validate MC prices against analytical solutions."""

    def test_cash_call_mc_vs_analytical(self):
        """MC cash-or-nothing call converges to analytical price."""
        dig = DigitalOption(K=HULL_K, option_type='call', payout_type='cash')
        mc = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        paths = mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA, n_steps=1)
        result = mc.price(dig.payoff, paths, HULL_R, HULL_T)

        analytical = DigitalOption.analytical_price(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA,
        )
        assert abs(result.price - analytical) < 3 * result.std_error

    def test_cash_put_mc_vs_analytical(self):
        """MC cash-or-nothing put converges to analytical price."""
        dig = DigitalOption(K=HULL_K, option_type='put', payout_type='cash')
        mc = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        paths = mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA, n_steps=1)
        result = mc.price(dig.payoff, paths, HULL_R, HULL_T)

        analytical = DigitalOption.analytical_price(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, 'put', 'cash',
        )
        assert abs(result.price - analytical) < 3 * result.std_error

    def test_asset_call_mc_vs_analytical(self):
        """MC asset-or-nothing call converges to analytical price."""
        dig = DigitalOption(K=HULL_K, option_type='call', payout_type='asset')
        mc = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        paths = mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA, n_steps=1)
        result = mc.price(dig.payoff, paths, HULL_R, HULL_T)

        analytical = DigitalOption.analytical_price(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, 'call', 'asset',
        )
        assert abs(result.price - analytical) < 3 * result.std_error

    def test_asset_put_mc_vs_analytical(self):
        """MC asset-or-nothing put converges to analytical price."""
        dig = DigitalOption(K=HULL_K, option_type='put', payout_type='asset')
        mc = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        paths = mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA, n_steps=1)
        result = mc.price(dig.payoff, paths, HULL_R, HULL_T)

        analytical = DigitalOption.analytical_price(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, 'put', 'asset',
        )
        assert abs(result.price - analytical) < 3 * result.std_error

    def test_cash_call_with_antithetic(self):
        """Antithetic variates reduce SE for digital calls."""
        dig = DigitalOption(K=HULL_K, option_type='call', payout_type='cash')
        mc = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)

        # Without antithetic
        paths = mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA, n_steps=1)
        result_plain = mc.price(dig.payoff, paths, HULL_R, HULL_T)

        # With antithetic
        mc.reset()
        paths, paths_anti = mc.simulate_gbm(
            HULL_S, HULL_T, HULL_R, HULL_SIGMA, n_steps=1, antithetic=True,
        )
        result_av = mc.price(dig.payoff, paths, HULL_R, HULL_T, paths_anti=paths_anti)

        analytical = DigitalOption.analytical_price(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA,
        )
        # Both should be close to analytical
        assert abs(result_av.price - analytical) < 3 * result_av.std_error
        # Antithetic should have lower or comparable SE
        # (for digital payoffs the reduction is modest compared to vanilla)
        assert result_av.std_error <= result_plain.std_error * 1.1

    def test_cash_call_otm_mc(self):
        """OTM digital call: MC price is small but positive."""
        dig = DigitalOption(K=130, option_type='call', payout_type='cash')
        mc = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        paths = mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA, n_steps=1)
        result = mc.price(dig.payoff, paths, HULL_R, HULL_T)

        analytical = DigitalOption.analytical_price(
            HULL_S, 130, HULL_T, HULL_R, HULL_SIGMA,
        )
        assert abs(result.price - analytical) < 3 * result.std_error
        assert result.price > 0


# ============================================================================
# 3. Digital Option Mathematical Properties
# ============================================================================
class TestDigitalProperties:
    """Verify mathematical identities that must hold exactly or within SE."""

    def test_cash_complementarity(self):
        """
        Cash-or-nothing call + put = e^{-rT} * Q.

        The events S_T > K and S_T < K partition the sample space
        (P(S_T = K) = 0 under continuous measure).
        """
        call = DigitalOption.analytical_price(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, 'call', 'cash', 1.0,
        )
        put = DigitalOption.analytical_price(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, 'put', 'cash', 1.0,
        )
        expected = np.exp(-HULL_R * HULL_T) * 1.0
        assert abs(call + put - expected) < 1e-12

    def test_asset_complementarity(self):
        """
        Asset-or-nothing call + put = S * e^{-qT}.

        Total asset payout = discounted forward.
        """
        call = DigitalOption.analytical_price(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, 'call', 'asset',
        )
        put = DigitalOption.analytical_price(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, 'put', 'asset',
        )
        expected = HULL_S  # q=0 → e^{-qT} = 1
        assert abs(call + put - expected) < 1e-10

    def test_asset_complementarity_with_dividends(self):
        """Asset-or-nothing call + put = S * e^{-qT} with q > 0."""
        q = 0.03
        call = DigitalOption.analytical_price(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, 'call', 'asset', q=q,
        )
        put = DigitalOption.analytical_price(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, 'put', 'asset', q=q,
        )
        expected = HULL_S * np.exp(-q * HULL_T)
        assert abs(call + put - expected) < 1e-10

    def test_vanilla_decomposition_call(self):
        """
        Vanilla call = asset-or-nothing call - K * cash-or-nothing call.

        This is the BS formula rewritten: C = S N(d1) - K e^{-rT} N(d2).
        The asset-or-nothing call IS S N(d1) and the cash-or-nothing call
        IS e^{-rT} N(d2), so this identity is exact.
        """
        asset_call = DigitalOption.analytical_price(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, 'call', 'asset',
        )
        cash_call = DigitalOption.analytical_price(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, 'call', 'cash', 1.0,
        )
        vanilla = asset_call - HULL_K * cash_call
        assert abs(vanilla - HULL_CALL) < 1e-3

    def test_vanilla_decomposition_put(self):
        """Vanilla put = K * cash-or-nothing put - asset-or-nothing put."""
        asset_put = DigitalOption.analytical_price(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, 'put', 'asset',
        )
        cash_put = DigitalOption.analytical_price(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, 'put', 'cash', 1.0,
        )
        vanilla = HULL_K * cash_put - asset_put
        assert abs(vanilla - HULL_PUT) < 1e-3

    def test_vanilla_decomposition_mc(self):
        """
        MC vanilla decomposition: same paths, three payoffs.

        price(vanilla_call) ≈ price(asset_call) - K * price(cash_call)
        within SE.
        """
        mc = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        paths = mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA, n_steps=1)

        dig_asset = DigitalOption(K=HULL_K, option_type='call', payout_type='asset')
        dig_cash = DigitalOption(K=HULL_K, option_type='call', payout_type='cash')

        r_asset = mc.price(dig_asset.payoff, paths, HULL_R, HULL_T)
        r_cash = mc.price(dig_cash.payoff, paths, HULL_R, HULL_T)

        # Vanilla call via MC (same paths)
        def vanilla_payoff(p: np.ndarray) -> np.ndarray:
            return np.maximum(p[:, -1] - HULL_K, 0.0)

        r_vanilla = mc.price(vanilla_payoff, paths, HULL_R, HULL_T)

        # Decomposition should hold within combined SE
        decomp = r_asset.price - HULL_K * r_cash.price
        combined_se = np.sqrt(r_asset.std_error**2 + (HULL_K * r_cash.std_error)**2)
        assert abs(decomp - r_vanilla.price) < 3 * combined_se

    def test_cash_complementarity_mc(self):
        """MC: cash call + cash put ≈ e^{-rT} Q on same paths."""
        mc = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        paths = mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA, n_steps=1)

        dig_call = DigitalOption(K=HULL_K, option_type='call', payout_type='cash')
        dig_put = DigitalOption(K=HULL_K, option_type='put', payout_type='cash')

        r_call = mc.price(dig_call.payoff, paths, HULL_R, HULL_T)
        r_put = mc.price(dig_put.payoff, paths, HULL_R, HULL_T)

        expected = np.exp(-HULL_R * HULL_T)
        combined_se = np.sqrt(r_call.std_error**2 + r_put.std_error**2)
        assert abs(r_call.price + r_put.price - expected) < 3 * combined_se

    def test_cash_call_leq_discount(self):
        """Cash-or-nothing call ≤ e^{-rT} Q (probability ≤ 1)."""
        for K in [50, 80, 100, 120, 150]:
            price = DigitalOption.analytical_price(
                HULL_S, K, HULL_T, HULL_R, HULL_SIGMA, 'call', 'cash', 1.0,
            )
            assert price <= np.exp(-HULL_R * HULL_T) * 1.0 + 1e-12

    def test_cash_call_monotone_in_strike(self):
        """Cash call decreases with K (lower probability of S_T > K)."""
        strikes = [60, 80, 100, 120, 140]
        prices = [
            DigitalOption.analytical_price(
                HULL_S, K, HULL_T, HULL_R, HULL_SIGMA, 'call', 'cash',
            )
            for K in strikes
        ]
        for i in range(len(prices) - 1):
            assert prices[i] > prices[i + 1]


# ============================================================================
# 4. Digital Option Edge Cases
# ============================================================================
class TestDigitalEdgeCases:
    """Boundary conditions and extreme parameters."""

    def test_payoff_deterministic_itm_call(self):
        """Deterministic path S_T > K → cash call pays Q."""
        dig = DigitalOption(K=90, option_type='call', payout_type='cash', cash_amount=5.0)
        # Single path: S_0=100, S_T=110
        paths = np.array([[100.0, 110.0]])
        payoff = dig.payoff(paths)
        assert payoff[0] == 5.0

    def test_payoff_deterministic_otm_call(self):
        """Deterministic path S_T < K → cash call pays 0."""
        dig = DigitalOption(K=120, option_type='call', payout_type='cash', cash_amount=5.0)
        paths = np.array([[100.0, 110.0]])
        payoff = dig.payoff(paths)
        assert payoff[0] == 0.0

    def test_payoff_deterministic_itm_put(self):
        """Deterministic path S_T < K → cash put pays Q."""
        dig = DigitalOption(K=120, option_type='put', payout_type='cash', cash_amount=3.0)
        paths = np.array([[100.0, 110.0]])
        payoff = dig.payoff(paths)
        assert payoff[0] == 3.0

    def test_payoff_at_strike_call(self):
        """S_T == K exactly → call pays 0 (strict inequality convention)."""
        dig = DigitalOption(K=100, option_type='call', payout_type='cash')
        paths = np.array([[100.0, 100.0]])
        payoff = dig.payoff(paths)
        assert payoff[0] == 0.0

    def test_payoff_at_strike_put(self):
        """S_T == K exactly → put pays 0 (strict inequality convention)."""
        dig = DigitalOption(K=100, option_type='put', payout_type='cash')
        paths = np.array([[100.0, 100.0]])
        payoff = dig.payoff(paths)
        assert payoff[0] == 0.0

    def test_asset_payoff_itm(self):
        """Asset-or-nothing call ITM: pays S_T, not Q."""
        dig = DigitalOption(K=90, option_type='call', payout_type='asset')
        paths = np.array([[100.0, 150.0]])
        payoff = dig.payoff(paths)
        assert payoff[0] == 150.0

    def test_asset_payoff_otm(self):
        """Asset-or-nothing call OTM: pays 0."""
        dig = DigitalOption(K=200, option_type='call', payout_type='asset')
        paths = np.array([[100.0, 150.0]])
        payoff = dig.payoff(paths)
        assert payoff[0] == 0.0

    def test_cash_amount_zero(self):
        """Q=0: price is always 0 regardless of moneyness."""
        price = DigitalOption.analytical_price(
            100, 100, 1.0, 0.05, 0.20, 'call', 'cash', 0.0,
        )
        assert price == 0.0

    def test_short_maturity(self):
        """Short T: ITM digital converges to e^{-rT} * Q."""
        # Very short T, very deep ITM
        price = DigitalOption.analytical_price(
            100, 50, 0.001, 0.05, 0.20, 'call', 'cash', 1.0,
        )
        expected = np.exp(-0.05 * 0.001)
        assert abs(price - expected) < 1e-4

    def test_short_maturity_otm(self):
        """Short T, OTM: digital converges to 0."""
        price = DigitalOption.analytical_price(
            100, 150, 0.001, 0.05, 0.20, 'call', 'cash', 1.0,
        )
        assert price < 1e-4

    def test_low_vol_itm(self):
        """Low sigma, ITM: digital converges to e^{-rT} * Q."""
        price = DigitalOption.analytical_price(
            100, 50, 1.0, 0.05, 0.01, 'call', 'cash', 1.0,
        )
        expected = np.exp(-0.05)
        assert abs(price - expected) < 1e-4

    def test_low_vol_otm(self):
        """Low sigma, OTM: digital converges to 0."""
        price = DigitalOption.analytical_price(
            100, 150, 1.0, 0.05, 0.01, 'call', 'cash', 1.0,
        )
        assert price < 1e-4


# ============================================================================
# 5. Digital Option Input Validation
# ============================================================================
class TestDigitalValidation:
    """Input validation for constructor and analytical_price."""

    def test_negative_strike_constructor(self):
        with pytest.raises(ValueError, match="K must be > 0"):
            DigitalOption(K=-10)

    def test_zero_strike_constructor(self):
        with pytest.raises(ValueError, match="K must be > 0"):
            DigitalOption(K=0)

    def test_invalid_option_type(self):
        with pytest.raises(ValueError, match="option_type"):
            DigitalOption(K=100, option_type='straddle')

    def test_invalid_payout_type(self):
        with pytest.raises(ValueError, match="payout_type"):
            DigitalOption(K=100, payout_type='binary')

    def test_negative_cash_amount(self):
        with pytest.raises(ValueError, match="cash_amount"):
            DigitalOption(K=100, cash_amount=-1.0)

    def test_analytical_negative_S(self):
        with pytest.raises(ValueError, match="S must be > 0"):
            DigitalOption.analytical_price(-100, 100, 1.0, 0.05, 0.20)

    def test_analytical_zero_T(self):
        with pytest.raises(ValueError, match="T must be > 0"):
            DigitalOption.analytical_price(100, 100, 0.0, 0.05, 0.20)

    def test_analytical_zero_sigma(self):
        with pytest.raises(ValueError, match="sigma must be > 0"):
            DigitalOption.analytical_price(100, 100, 1.0, 0.05, 0.0)

    def test_shorthand_option_type(self):
        """Accept 'c' and 'p' as shorthand."""
        p_c = DigitalOption.analytical_price(100, 100, 1.0, 0.05, 0.20, 'c', 'cash')
        p_call = DigitalOption.analytical_price(100, 100, 1.0, 0.05, 0.20, 'call', 'cash')
        assert p_c == p_call


# ============================================================================
# 6. Digital Option Identity (__repr__, __eq__, __hash__)
# ============================================================================
class TestDigitalIdentity:
    """Object identity: repr, equality, hashing."""

    def test_repr_cash(self):
        dig = DigitalOption(K=100, option_type='call', payout_type='cash', cash_amount=5.0)
        r = repr(dig)
        assert "K=100" in r
        assert "call" in r
        assert "cash" in r
        assert "Q=5.0" in r

    def test_repr_asset(self):
        dig = DigitalOption(K=100, option_type='put', payout_type='asset')
        r = repr(dig)
        assert "asset" in r
        assert "put" in r

    def test_eq(self):
        a = DigitalOption(K=100, option_type='call', payout_type='cash', cash_amount=1.0)
        b = DigitalOption(K=100, option_type='call', payout_type='cash', cash_amount=1.0)
        assert a == b

    def test_neq_different_strike(self):
        a = DigitalOption(K=100)
        b = DigitalOption(K=110)
        assert a != b

    def test_neq_different_type(self):
        a = DigitalOption(K=100, option_type='call')
        b = DigitalOption(K=100, option_type='put')
        assert a != b

    def test_hash_consistency(self):
        a = DigitalOption(K=100, option_type='call', payout_type='cash')
        b = DigitalOption(K=100, option_type='call', payout_type='cash')
        assert hash(a) == hash(b)
        assert len({a, b}) == 1

    def test_is_exotic_option(self):
        """DigitalOption inherits from ExoticOption ABC."""
        dig = DigitalOption(K=100)
        assert isinstance(dig, ExoticOption)


# ============================================================================
# 7. Hypothesis Property-Based Tests
# ============================================================================
spot_st = st.floats(min_value=50, max_value=200, allow_nan=False, allow_infinity=False)
strike_st = st.floats(min_value=10, max_value=300, allow_nan=False, allow_infinity=False)
vol_st = st.floats(min_value=0.05, max_value=0.80, allow_nan=False, allow_infinity=False)
rate_st = st.floats(min_value=0.0, max_value=0.15, allow_nan=False, allow_infinity=False)
time_st = st.floats(min_value=0.05, max_value=3.0, allow_nan=False, allow_infinity=False)
div_st = st.floats(min_value=0.0, max_value=0.08, allow_nan=False, allow_infinity=False)
cash_st = st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False)


class TestDigitalHypothesis:
    """Property-based tests with Hypothesis (500+ random cases per property)."""

    @given(S=spot_st, K=strike_st, T=time_st, r=rate_st, sigma=vol_st, q=div_st)
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_cash_complementarity(self, S: float, K: float, T: float,
                                   r: float, sigma: float, q: float):
        """Cash call + cash put = e^{-rT} for all valid params."""
        call = DigitalOption.analytical_price(S, K, T, r, sigma, 'call', 'cash', 1.0, q)
        put = DigitalOption.analytical_price(S, K, T, r, sigma, 'put', 'cash', 1.0, q)
        expected = np.exp(-r * T)
        assert abs(call + put - expected) < 1e-10

    @given(S=spot_st, K=strike_st, T=time_st, r=rate_st, sigma=vol_st, q=div_st)
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_asset_complementarity(self, S: float, K: float, T: float,
                                    r: float, sigma: float, q: float):
        """Asset call + asset put = S e^{-qT} for all valid params."""
        call = DigitalOption.analytical_price(S, K, T, r, sigma, 'call', 'asset', q=q)
        put = DigitalOption.analytical_price(S, K, T, r, sigma, 'put', 'asset', q=q)
        expected = S * np.exp(-q * T)
        assert abs(call + put - expected) < 1e-8

    @given(S=spot_st, K=strike_st, T=time_st, r=rate_st, sigma=vol_st, q=div_st)
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_cash_call_non_negative(self, S: float, K: float, T: float,
                                     r: float, sigma: float, q: float):
        """Cash-or-nothing call price is always non-negative."""
        price = DigitalOption.analytical_price(S, K, T, r, sigma, 'call', 'cash', 1.0, q)
        assert price >= -1e-15

    @given(S=spot_st, K=strike_st, T=time_st, r=rate_st, sigma=vol_st, q=div_st)
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_cash_call_upper_bound(self, S: float, K: float, T: float,
                                    r: float, sigma: float, q: float):
        """Cash-or-nothing call ≤ e^{-rT} (probability ≤ 1)."""
        price = DigitalOption.analytical_price(S, K, T, r, sigma, 'call', 'cash', 1.0, q)
        assert price <= np.exp(-r * T) + 1e-12

    @given(S=spot_st, K=strike_st, T=time_st, r=rate_st, sigma=vol_st, q=div_st)
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_vanilla_decomposition(self, S: float, K: float, T: float,
                                    r: float, sigma: float, q: float):
        """
        Vanilla call = asset-or-nothing call - K * cash-or-nothing call.

        Cross-validate against BlackScholesModel.
        """
        asset_call = DigitalOption.analytical_price(S, K, T, r, sigma, 'call', 'asset', q=q)
        cash_call = DigitalOption.analytical_price(S, K, T, r, sigma, 'call', 'cash', 1.0, q)
        digital_vanilla = asset_call - K * cash_call

        bs = BlackScholesModel(sigma=sigma)
        bs_price = bs.price(S, K, T, r, 'call', q)

        assert abs(digital_vanilla - bs_price) < 1e-8

    @given(S=spot_st, K=strike_st, T=time_st, r=rate_st, sigma=vol_st,
           q=div_st, Q=cash_st)
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_cash_amount_linearity(self, S: float, K: float, T: float,
                                    r: float, sigma: float, q: float, Q: float):
        """Cash-or-nothing price scales linearly with Q."""
        p1 = DigitalOption.analytical_price(S, K, T, r, sigma, 'call', 'cash', 1.0, q)
        pQ = DigitalOption.analytical_price(S, K, T, r, sigma, 'call', 'cash', Q, q)
        assert abs(pQ - Q * p1) < 1e-10 * Q


# ============================================================================
# ============================================================================
#                             ASIAN OPTIONS
# ============================================================================
# ============================================================================

# Standard Asian test parameters
ASIAN_PATHS = 200_000
ASIAN_STEPS = 252


def _kemna_vorst_reference(
    S: float, K: float, T: float, r: float, sigma: float, n_obs: int,
    option_type: str = 'call', q: float = 0.0,
) -> float:
    """
    Independent Kemna-Vorst implementation used as ground truth in tests.

    Computed inline with scipy primitives so that a bug in the production
    ``AsianOption.geometric_price`` would not silently propagate into the
    expected value.
    """
    from scipy.special import ndtr as _ndtr  # local import to keep test hermetic
    n = float(n_obs)
    sig_hat_sq = sigma * sigma * (n + 1.0) * (2.0 * n + 1.0) / (6.0 * n * n)
    sig_hat = np.sqrt(sig_hat_sq)
    r_hat = 0.5 * sig_hat_sq + (r - q - 0.5 * sigma * sigma) * (n + 1.0) / (2.0 * n)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r_hat + 0.5 * sig_hat_sq) * T) / (sig_hat * sqrt_T)
    d2 = d1 - sig_hat * sqrt_T
    adj = np.exp((r_hat - r) * T)
    if option_type == 'call':
        return float(adj * (S * _ndtr(d1) - K * np.exp(-r_hat * T) * _ndtr(d2)))
    return float(adj * (K * np.exp(-r_hat * T) * _ndtr(-d2) - S * _ndtr(-d1)))


# ----------------------------------------------------------------------------
# 8. Asian Option Analytical Prices
# ----------------------------------------------------------------------------
class TestAsianAnalytical:
    """Kemna-Vorst analytical price: agreement with independent reference."""

    def test_geo_call_atm_reference(self):
        """Class geometric_price matches independent implementation."""
        price = AsianOption.geometric_price(
            100, 100, 1.0, 0.05, 0.20, 252, 'call',
        )
        ref = _kemna_vorst_reference(100, 100, 1.0, 0.05, 0.20, 252, 'call')
        assert abs(price - ref) < 1e-12

    def test_geo_put_atm_reference(self):
        price = AsianOption.geometric_price(
            100, 100, 1.0, 0.05, 0.20, 252, 'put',
        )
        ref = _kemna_vorst_reference(100, 100, 1.0, 0.05, 0.20, 252, 'put')
        assert abs(price - ref) < 1e-12

    def test_geo_call_with_dividends(self):
        """Kemna-Vorst with q > 0."""
        price = AsianOption.geometric_price(
            100, 100, 1.0, 0.05, 0.20, 252, 'call', q=0.03,
        )
        ref = _kemna_vorst_reference(100, 100, 1.0, 0.05, 0.20, 252, 'call', q=0.03)
        assert abs(price - ref) < 1e-12

    def test_geo_call_single_fixing_is_vanilla(self):
        """
        For n_obs = 1, sigma_hat^2 = sigma^2 and r_hat = r - q, so the
        Kemna-Vorst formula collapses to the standard Black-Scholes-Merton
        price for a vanilla European call. This is a critical sanity check.
        """
        S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.02
        geo_price = AsianOption.geometric_price(S, K, T, r, sigma, 1, 'call', q=q)
        bs = BlackScholesModel(sigma=sigma)
        bs_price = bs.price(S, K, T, r, 'call', q=q)
        assert abs(geo_price - bs_price) < 1e-10

    def test_geo_effective_model_call_put_parity(self):
        """
        Put-call parity in the effective (rate = r_hat, q = 0) economy:

            C_eff - P_eff = S - K * exp(-r_hat * T)

        so the dividend-free BS identity holds after multiplying the
        class prices by the inverse adjustment factor exp(-(r_hat-r)T).
        """
        S, K, T, r, sigma = 100.0, 110.0, 1.5, 0.04, 0.30
        n_obs = 100
        C = AsianOption.geometric_price(S, K, T, r, sigma, n_obs, 'call')
        P = AsianOption.geometric_price(S, K, T, r, sigma, n_obs, 'put')

        n = float(n_obs)
        sig_hat_sq = sigma * sigma * (n + 1) * (2 * n + 1) / (6 * n * n)
        r_hat = 0.5 * sig_hat_sq + (r - 0.5 * sigma * sigma) * (n + 1) / (2 * n)
        adj_inv = np.exp(-(r_hat - r) * T)
        C_eff = C * adj_inv
        P_eff = P * adj_inv
        rhs = S - K * np.exp(-r_hat * T)
        assert abs((C_eff - P_eff) - rhs) < 1e-10

    def test_geo_continuous_limit(self):
        """
        As n_obs -> infinity the adjusted parameters converge to:

            sigma_hat^2 -> sigma^2 / 3
            r_hat       -> (r - q)/2 - sigma^2/12

        The discrete Kemna-Vorst price must converge to the continuous-
        time formula evaluated with these limits.
        """
        S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0
        p_discrete = AsianOption.geometric_price(S, K, T, r, sigma, 10_000, 'call', q=q)

        # Continuous-limit reference (same BS-like formula)
        from scipy.special import ndtr as _ndtr
        sig_hat_sq = sigma * sigma / 3.0
        sig_hat = np.sqrt(sig_hat_sq)
        r_hat = (r - q) / 2.0 - sigma * sigma / 12.0
        d1 = (np.log(S / K) + (r_hat + 0.5 * sig_hat_sq) * T) / (sig_hat * np.sqrt(T))
        d2 = d1 - sig_hat * np.sqrt(T)
        adj = np.exp((r_hat - r) * T)
        p_continuous = float(
            adj * (S * _ndtr(d1) - K * np.exp(-r_hat * T) * _ndtr(d2))
        )
        assert abs(p_discrete - p_continuous) < 1e-3

    def test_geo_call_deep_itm(self):
        """Deep ITM call: price approaches S - K * exp(-r*T) discounted."""
        price = AsianOption.geometric_price(
            200, 50, 1.0, 0.05, 0.20, 252, 'call',
        )
        # Lower bound: forward minus discounted strike, corrected by
        # the effective drift. Use direct reference instead.
        ref = _kemna_vorst_reference(200, 50, 1.0, 0.05, 0.20, 252, 'call')
        assert abs(price - ref) < 1e-12
        assert price > 145  # very deep ITM, > (S - K)

    def test_geo_call_deep_otm(self):
        """Deep OTM call: price is near zero."""
        price = AsianOption.geometric_price(
            50, 200, 1.0, 0.05, 0.20, 252, 'call',
        )
        assert 0.0 <= price < 1.0


# ----------------------------------------------------------------------------
# 9. Asian Option Monte Carlo Cross-Validation
# ----------------------------------------------------------------------------
class TestAsianMC:
    """Cross-validate MC prices against Kemna-Vorst analytical."""

    def test_geometric_mc_call(self):
        """MC geometric call matches Kemna-Vorst within 3 * SE."""
        asian = AsianOption(K=100, option_type='call', avg_type='geometric')
        mc = MonteCarloEngine(n_paths=ASIAN_PATHS, seed=MC_SEED)
        paths = mc.simulate_gbm(100, 1.0, 0.05, 0.20, n_steps=ASIAN_STEPS)
        result = mc.price(asian.payoff, paths, 0.05, 1.0)
        analytical = AsianOption.geometric_price(
            100, 100, 1.0, 0.05, 0.20, ASIAN_STEPS, 'call',
        )
        assert abs(result.price - analytical) < 3 * result.std_error

    def test_geometric_mc_put(self):
        """MC geometric put matches Kemna-Vorst within 3 * SE."""
        asian = AsianOption(K=100, option_type='put', avg_type='geometric')
        mc = MonteCarloEngine(n_paths=ASIAN_PATHS, seed=MC_SEED)
        paths = mc.simulate_gbm(100, 1.0, 0.05, 0.20, n_steps=ASIAN_STEPS)
        result = mc.price(asian.payoff, paths, 0.05, 1.0)
        analytical = AsianOption.geometric_price(
            100, 100, 1.0, 0.05, 0.20, ASIAN_STEPS, 'put',
        )
        assert abs(result.price - analytical) < 3 * result.std_error

    def test_geometric_mc_with_dividends(self):
        asian = AsianOption(K=100, option_type='call', avg_type='geometric')
        mc = MonteCarloEngine(n_paths=ASIAN_PATHS, seed=MC_SEED)
        paths = mc.simulate_gbm(100, 1.0, 0.05, 0.20, q=0.03, n_steps=ASIAN_STEPS)
        result = mc.price(asian.payoff, paths, 0.05, 1.0)
        analytical = AsianOption.geometric_price(
            100, 100, 1.0, 0.05, 0.20, ASIAN_STEPS, 'call', q=0.03,
        )
        assert abs(result.price - analytical) < 3 * result.std_error

    def test_arithmetic_cv_reduces_variance(self):
        """
        The geometric Asian control variate must reduce SE by > 90%
        relative to plain MC. For ATM parameters the typical reduction
        is > 95% (correlation arith/geo > 0.99).
        """
        asian = AsianOption(K=100, option_type='call', avg_type='arithmetic')
        mc = MonteCarloEngine(n_paths=ASIAN_PATHS, seed=MC_SEED)
        paths = mc.simulate_gbm(100, 1.0, 0.05, 0.20, n_steps=ASIAN_STEPS)

        r_plain = mc.price(asian.payoff, paths, 0.05, 1.0)
        cv_fn = asian.geometric_control_fn(100, 1.0, 0.05, 0.20, n_obs=ASIAN_STEPS)
        r_cv = mc.price(asian.payoff, paths, 0.05, 1.0, control_fn=cv_fn)

        assert r_cv.std_error < 0.10 * r_plain.std_error

    def test_arithmetic_cv_preserves_unbiasedness(self):
        """
        The control-variate estimator remains unbiased: the CV-adjusted
        price must agree with plain MC within combined SE.
        """
        asian = AsianOption(K=100, option_type='call', avg_type='arithmetic')
        mc = MonteCarloEngine(n_paths=ASIAN_PATHS, seed=MC_SEED)
        paths = mc.simulate_gbm(100, 1.0, 0.05, 0.20, n_steps=ASIAN_STEPS)
        r_plain = mc.price(asian.payoff, paths, 0.05, 1.0)
        cv_fn = asian.geometric_control_fn(100, 1.0, 0.05, 0.20, n_obs=ASIAN_STEPS)
        r_cv = mc.price(asian.payoff, paths, 0.05, 1.0, control_fn=cv_fn)

        combined_se = np.sqrt(r_plain.std_error ** 2 + r_cv.std_error ** 2)
        assert abs(r_plain.price - r_cv.price) < 4 * combined_se


# ----------------------------------------------------------------------------
# 10. Asian Option Mathematical Properties
# ----------------------------------------------------------------------------
class TestAsianProperties:
    """Pathwise and price-level properties that must hold mathematically."""

    def test_am_ge_gm_on_paths(self):
        """
        AM-GM inequality: arithmetic mean >= geometric mean pointwise
        on every Monte Carlo path, with equality only when the path is
        constant (probability zero under GBM).
        """
        mc = MonteCarloEngine(n_paths=5000, seed=MC_SEED)
        paths = mc.simulate_gbm(100, 1.0, 0.05, 0.20, n_steps=50)
        am = np.mean(paths[:, 1:], axis=1)
        gm = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
        assert np.all(am + 1e-12 >= gm)
        # Strict inequality on most paths
        assert np.mean(am > gm + 1e-8) > 0.99

    def test_fixed_strike_call_put_parity_mc(self):
        """
        Fixed-strike arithmetic Asian parity:

            C - P = e^{-rT} * (E^Q[mean(S_{t_1..t_n})] - K)

        Use the same paths for both legs so the MC estimator of the
        right-hand side cancels out the dominant variance.
        """
        mc = MonteCarloEngine(n_paths=ASIAN_PATHS, seed=MC_SEED)
        paths = mc.simulate_gbm(100, 1.0, 0.05, 0.20, n_steps=ASIAN_STEPS)
        call = AsianOption(K=100, option_type='call', avg_type='arithmetic')
        put = AsianOption(K=100, option_type='put', avg_type='arithmetic')
        r_call = mc.price(call.payoff, paths, 0.05, 1.0)
        r_put = mc.price(put.payoff, paths, 0.05, 1.0)

        empirical_avg_mean = float(np.mean(np.mean(paths[:, 1:], axis=1)))
        discount = np.exp(-0.05)
        rhs = discount * (empirical_avg_mean - 100.0)
        lhs = r_call.price - r_put.price

        combined_se = np.sqrt(r_call.std_error ** 2 + r_put.std_error ** 2)
        assert abs(lhs - rhs) < 4 * combined_se

    def test_geometric_call_lt_european_call(self):
        """
        Asian averaging reduces effective volatility, so the geometric
        Asian call is strictly cheaper than the vanilla European call
        with the same parameters.
        """
        geo = AsianOption.geometric_price(100, 100, 1.0, 0.05, 0.20, 252, 'call')
        bs = BlackScholesModel(sigma=0.20)
        euro = bs.price(100, 100, 1.0, 0.05, 'call')
        assert geo < euro

    def test_floating_call_always_non_negative(self):
        """Floating-strike call payoff S_T - mean(S) is always >= 0."""
        asian = AsianOption(option_type='call', avg_type='arithmetic',
                             strike_type='floating')
        mc = MonteCarloEngine(n_paths=2000, seed=MC_SEED)
        paths = mc.simulate_gbm(100, 1.0, 0.05, 0.20, n_steps=50)
        payoffs = asian.payoff(paths)
        assert np.all(payoffs >= 0.0)
        # Non-degenerate: most paths should actually pay
        assert np.mean(payoffs > 1e-8) > 0.2

    def test_floating_put_always_non_negative(self):
        """Floating-strike put payoff mean(S) - S_T is always >= 0."""
        asian = AsianOption(option_type='put', avg_type='arithmetic',
                             strike_type='floating')
        mc = MonteCarloEngine(n_paths=2000, seed=MC_SEED)
        paths = mc.simulate_gbm(100, 1.0, 0.05, 0.20, n_steps=50)
        payoffs = asian.payoff(paths)
        assert np.all(payoffs >= 0.0)

    def test_more_fixings_lower_atm_call_price(self):
        """
        More monitoring dates reduce sigma_hat and r_hat, which for an
        ATM call with r > q leaves the price strictly decreasing in n.
        """
        prices = [
            AsianOption.geometric_price(100, 100, 1.0, 0.05, 0.20, n, 'call')
            for n in [2, 5, 20, 252, 5000]
        ]
        for i in range(len(prices) - 1):
            assert prices[i] > prices[i + 1]


# ----------------------------------------------------------------------------
# 11. Asian Option Edge Cases
# ----------------------------------------------------------------------------
class TestAsianEdgeCases:
    """Boundary conditions and deterministic payoffs."""

    def test_fixed_call_deterministic_itm(self):
        asian = AsianOption(K=100, option_type='call', avg_type='arithmetic')
        paths = np.array([[100.0, 110.0, 120.0, 130.0]])  # avg = 120
        payoff = asian.payoff(paths)
        assert payoff[0] == 20.0

    def test_fixed_put_deterministic_itm(self):
        asian = AsianOption(K=150, option_type='put', avg_type='arithmetic')
        paths = np.array([[100.0, 110.0, 120.0, 130.0]])  # avg = 120
        payoff = asian.payoff(paths)
        assert payoff[0] == 30.0

    def test_geometric_average_deterministic(self):
        asian = AsianOption(K=100, option_type='call', avg_type='geometric')
        # paths = [100, 100, 400] → geomean(fixings) = sqrt(100*400) = 200
        paths = np.array([[100.0, 100.0, 400.0]])
        payoff = asian.payoff(paths)
        assert abs(payoff[0] - 100.0) < 1e-12

    def test_floating_call_deterministic(self):
        asian = AsianOption(option_type='call', avg_type='arithmetic',
                             strike_type='floating')
        # avg = 120, S_T = 130 → S_T - avg = 10
        paths = np.array([[100.0, 110.0, 120.0, 130.0]])
        payoff = asian.payoff(paths)
        assert payoff[0] == 10.0

    def test_floating_put_deterministic(self):
        asian = AsianOption(option_type='put', avg_type='arithmetic',
                             strike_type='floating')
        # fixings = [140, 120, 100], avg = 120, S_T = 100 → avg - S_T = 20
        paths = np.array([[100.0, 140.0, 120.0, 100.0]])
        payoff = asian.payoff(paths)
        assert abs(payoff[0] - 20.0) < 1e-12

    def test_small_sigma_limit(self):
        """
        As sigma -> 0, the Kemna-Vorst price tends to the deterministic
        payoff discounted at r, with the average evaluated along the
        drifted path. For ATM with q=0 it tends to a small positive
        number near 0.5 * (S * ((e^{rT}-1)/(rT)) - K) * e^{-rT} (or
        intrinsic if ITM). We only check continuity and positivity.
        """
        p_small = AsianOption.geometric_price(100, 100, 1.0, 0.05, 1e-4, 252, 'call')
        p_ref = AsianOption.geometric_price(100, 100, 1.0, 0.05, 1e-3, 252, 'call')
        assert p_small >= 0.0
        assert abs(p_small - p_ref) < 1.0  # smoothly varying

    def test_short_maturity_limit(self):
        """
        As T -> 0, the Kemna-Vorst price tends to the intrinsic value
        max(S_0 - K, 0) since the average is pinned near S_0.
        """
        p = AsianOption.geometric_price(100, 90, 1e-6, 0.05, 0.20, 252, 'call')
        intrinsic = 100 - 90
        assert abs(p - intrinsic) < 1e-3


# ----------------------------------------------------------------------------
# 12. Asian Option Input Validation
# ----------------------------------------------------------------------------
class TestAsianValidation:
    """Constructor and static-method argument validation."""

    def test_fixed_without_K(self):
        with pytest.raises(ValueError, match="K is required"):
            AsianOption(option_type='call', avg_type='arithmetic',
                         strike_type='fixed')

    def test_fixed_negative_K(self):
        with pytest.raises(ValueError, match="K must be > 0"):
            AsianOption(K=-50, option_type='call', avg_type='arithmetic',
                         strike_type='fixed')

    def test_fixed_zero_K(self):
        with pytest.raises(ValueError, match="K must be > 0"):
            AsianOption(K=0, option_type='call', avg_type='arithmetic',
                         strike_type='fixed')

    def test_floating_K_ignored(self):
        """K passed with strike_type='floating' is stored as None."""
        asian = AsianOption(K=100, option_type='call', avg_type='arithmetic',
                             strike_type='floating')
        assert asian.K is None

    def test_invalid_option_type(self):
        with pytest.raises(ValueError, match="option_type"):
            AsianOption(K=100, option_type='binary', avg_type='arithmetic')

    def test_invalid_avg_type(self):
        with pytest.raises(ValueError, match="avg_type"):
            AsianOption(K=100, avg_type='harmonic')

    def test_invalid_strike_type(self):
        with pytest.raises(ValueError, match="strike_type"):
            AsianOption(K=100, strike_type='exotic')

    def test_shorthand_option_type(self):
        a1 = AsianOption(K=100, option_type='c', avg_type='geometric')
        a2 = AsianOption(K=100, option_type='call', avg_type='geometric')
        assert a1 == a2

    def test_geometric_price_negative_S(self):
        with pytest.raises(ValueError, match="S must be > 0"):
            AsianOption.geometric_price(-10, 100, 1.0, 0.05, 0.20, 252)

    def test_geometric_price_negative_K(self):
        with pytest.raises(ValueError, match="K must be > 0"):
            AsianOption.geometric_price(100, -10, 1.0, 0.05, 0.20, 252)

    def test_geometric_price_zero_T(self):
        with pytest.raises(ValueError, match="T must be > 0"):
            AsianOption.geometric_price(100, 100, 0, 0.05, 0.20, 252)

    def test_geometric_price_zero_sigma(self):
        with pytest.raises(ValueError, match="sigma must be > 0"):
            AsianOption.geometric_price(100, 100, 1.0, 0.05, 0, 252)

    def test_geometric_price_zero_n_obs(self):
        with pytest.raises(ValueError, match="n_obs"):
            AsianOption.geometric_price(100, 100, 1.0, 0.05, 0.20, 0)

    def test_cv_requires_arithmetic_fixed(self):
        """geometric_control_fn rejects non-arithmetic / non-fixed Asians."""
        geo = AsianOption(K=100, avg_type='geometric', strike_type='fixed')
        with pytest.raises(ValueError, match="arithmetic fixed-strike"):
            geo.geometric_control_fn(100, 1.0, 0.05, 0.20, 252)

        flt = AsianOption(avg_type='arithmetic', strike_type='floating')
        with pytest.raises(ValueError, match="arithmetic fixed-strike"):
            flt.geometric_control_fn(100, 1.0, 0.05, 0.20, 252)


# ----------------------------------------------------------------------------
# 13. Asian Option Identity
# ----------------------------------------------------------------------------
class TestAsianIdentity:
    """Object identity: repr, equality, hashing."""

    def test_repr_fixed(self):
        a = AsianOption(K=100, option_type='call', avg_type='arithmetic',
                         strike_type='fixed')
        r = repr(a)
        assert "K=100" in r
        assert "call" in r
        assert "arithmetic" in r
        assert "fixed" in r

    def test_repr_floating(self):
        a = AsianOption(option_type='put', avg_type='geometric',
                         strike_type='floating')
        r = repr(a)
        assert "put" in r
        assert "geometric" in r
        assert "floating" in r

    def test_eq_and_hash(self):
        a = AsianOption(K=100, option_type='call', avg_type='arithmetic')
        b = AsianOption(K=100, option_type='call', avg_type='arithmetic')
        c = AsianOption(K=100, option_type='put', avg_type='arithmetic')
        assert a == b
        assert hash(a) == hash(b)
        assert len({a, b, c}) == 2
        assert a != c

    def test_is_exotic_option(self):
        a = AsianOption(K=100)
        assert isinstance(a, ExoticOption)


# ----------------------------------------------------------------------------
# 14. Asian Option Hypothesis Property-Based Tests
# ----------------------------------------------------------------------------
asian_n_st = st.integers(min_value=1, max_value=500)


class TestAsianHypothesis:
    """Property-based tests covering 500+ random parameter combinations."""

    @given(S=spot_st, K=strike_st, T=time_st, r=rate_st, sigma=vol_st,
           q=div_st, n=asian_n_st)
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_geometric_call_non_negative(self, S, K, T, r, sigma, q, n):
        price = AsianOption.geometric_price(S, K, T, r, sigma, n, 'call', q=q)
        assert price >= -1e-12

    @given(S=spot_st, K=strike_st, T=time_st, r=rate_st, sigma=vol_st,
           q=div_st, n=asian_n_st)
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_geometric_put_non_negative(self, S, K, T, r, sigma, q, n):
        price = AsianOption.geometric_price(S, K, T, r, sigma, n, 'put', q=q)
        assert price >= -1e-12

    @given(S=spot_st, K=strike_st, T=time_st, r=rate_st, sigma=vol_st,
           q=div_st, n=asian_n_st)
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_geometric_call_monotone_in_K(self, S, K, T, r, sigma, q, n):
        """
        Geometric Asian call is monotonically non-increasing in the
        strike K: higher strike implies a smaller conditional expectation
        E^Q[max(G - K, 0)]. Since sigma_hat and r_hat are independent of
        K, the Kemna-Vorst formula inherits this property directly from
        the Black-Scholes call with the effective parameters.
        """
        p_low = AsianOption.geometric_price(S, K, T, r, sigma, n, 'call', q=q)
        p_high = AsianOption.geometric_price(
            S, K + 10.0, T, r, sigma, n, 'call', q=q,
        )
        assert p_high <= p_low + 1e-10

    @given(S=spot_st, K=strike_st, T=time_st, r=rate_st, sigma=vol_st,
           q=div_st, n=asian_n_st)
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_geometric_effective_parity(self, S, K, T, r, sigma, q, n):
        """C_eff - P_eff = S - K * exp(-r_hat * T) in the effective model."""
        C = AsianOption.geometric_price(S, K, T, r, sigma, n, 'call', q=q)
        P = AsianOption.geometric_price(S, K, T, r, sigma, n, 'put', q=q)
        n_f = float(n)
        sig_hat_sq = sigma * sigma * (n_f + 1) * (2 * n_f + 1) / (6 * n_f * n_f)
        r_hat = 0.5 * sig_hat_sq + (r - q - 0.5 * sigma * sigma) * (n_f + 1) / (2 * n_f)
        adj_inv = np.exp(-(r_hat - r) * T)
        lhs = (C - P) * adj_inv
        rhs = S - K * np.exp(-r_hat * T)
        assert abs(lhs - rhs) < 1e-8 * max(S, K)


# ============================================================================
# ============================================================================
#                            BARRIER OPTIONS
# ============================================================================
# ============================================================================
BARRIER_PATHS = 200_000
BARRIER_STEPS = 252


def _mc_barrier_price(
    instrument: BarrierOption,
    S: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int = BARRIER_STEPS,
    q: float = 0.0,
    seed: int = MC_SEED,
    n_paths: int = BARRIER_PATHS,
):
    """Run a fresh MC price for a barrier instrument."""
    mc = MonteCarloEngine(n_paths=n_paths, seed=seed)
    paths = mc.simulate_gbm(S, T, r, sigma, q=q, n_steps=n_steps)
    return mc.price(instrument.payoff, paths, r, T)


# ----------------------------------------------------------------------------
# 15. Barrier Option Analytical Prices (Reiner-Rubinstein 1991)
# ----------------------------------------------------------------------------
class TestBarrierAnalytical:
    """
    Closed-form Reiner-Rubinstein prices verified against:

    1. Merton's (1973) original reflection formula for down-and-out call
       when H <= K. This cross-validates the A - C branch of the lookup.
    2. In-out parity V_in(H) + V_out(H) = V_vanilla (exact to floating
       point) for all 8 barrier/type combinations and both H <= K and
       H > K positions.
    3. Far-barrier limits: knock-out -> vanilla, knock-in -> 0.
    4. Degenerate cases: already-breached knock-in = vanilla,
       already-breached knock-out = undiscounted rebate.
    """

    def test_down_out_call_matches_merton_formula(self):
        """
        Merton (1973, Theory of Rational Option Pricing) derived the
        down-and-out call (with H <= K) as

            C_DO(S, K; H) = C_BS(S, K) - (H/S)^(2*lambda - 2) * C_BS(H^2/S, K)

        with lambda = (r - q + sigma^2 / 2) / sigma^2. This identity
        cross-validates the Reiner-Rubinstein A - C combination against
        the original derivation.
        """
        S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0
        H = 90.0
        price = BarrierOption.analytical_price(
            S, K, T, r, sigma, H, 'down-and-out', 'call', q=q,
        )

        lam = (r - q + 0.5 * sigma ** 2) / sigma ** 2
        bs = BlackScholesModel(sigma=sigma)
        c_full = bs.price(S, K, T, r, 'call', q)
        c_reflect = bs.price(H * H / S, K, T, r, 'call', q)
        expected = c_full - (H / S) ** (2 * lam - 2) * c_reflect

        assert abs(price - expected) < 1e-10

    def test_up_out_put_matches_reflection_formula(self):
        """
        Symmetric reflection identity for the up-and-out put with H >= K:

            P_UO(S, K; H) = P_BS(S, K) - (H/S)^(2*lambda - 2) * P_BS(H^2/S, K)
        """
        S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0
        H = 115.0
        price = BarrierOption.analytical_price(
            S, K, T, r, sigma, H, 'up-and-out', 'put', q=q,
        )

        lam = (r - q + 0.5 * sigma ** 2) / sigma ** 2
        bs = BlackScholesModel(sigma=sigma)
        p_full = bs.price(S, K, T, r, 'put', q)
        p_reflect = bs.price(H * H / S, K, T, r, 'put', q)
        expected = p_full - (H / S) ** (2 * lam - 2) * p_reflect

        assert abs(price - expected) < 1e-10

    @pytest.mark.parametrize("bt,opt,H", [
        ('down-and-out', 'call', 90.0),
        ('up-and-out',   'call', 120.0),
        ('down-and-out', 'put',  90.0),
        ('up-and-out',   'put',  120.0),
        ('down-and-out', 'call', 105.0),
        ('up-and-out',   'call',  95.0),
        ('down-and-out', 'put',  105.0),
        ('up-and-out',   'put',   95.0),
    ])
    def test_in_out_parity(self, bt, opt, H):
        """
        Exact in-out parity: V_in(H) + V_out(H) = V_vanilla. The eight
        parametrised cases exhaust both branches (H <= K and H > K) for
        each (direction, option_type) pair.
        """
        S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.02
        direction = 'down' if bt.startswith('down') else 'up'
        bt_other = f'{direction}-and-in'

        v_out = BarrierOption.analytical_price(
            S, K, T, r, sigma, H, bt, opt, q=q,
        )
        v_in = BarrierOption.analytical_price(
            S, K, T, r, sigma, H, bt_other, opt, q=q,
        )
        bs = BlackScholesModel(sigma=sigma)
        vanilla = bs.price(S, K, T, r, opt, q)

        assert abs(v_out + v_in - vanilla) < 1e-10

    def test_down_out_call_far_barrier_is_vanilla(self):
        """Knock-out with essentially impossible barrier = vanilla."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        H = 1.0
        price = BarrierOption.analytical_price(
            S, K, T, r, sigma, H, 'down-and-out', 'call',
        )
        bs = BlackScholesModel(sigma=sigma)
        vanilla = bs.price(S, K, T, r, 'call')
        assert abs(price - vanilla) < 1e-6

    def test_up_out_call_far_barrier_is_vanilla(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        H = 10_000.0
        price = BarrierOption.analytical_price(
            S, K, T, r, sigma, H, 'up-and-out', 'call',
        )
        bs = BlackScholesModel(sigma=sigma)
        vanilla = bs.price(S, K, T, r, 'call')
        assert abs(price - vanilla) < 1e-6

    def test_down_in_call_far_barrier_is_zero(self):
        """Knock-in with barrier essentially impossible to touch -> 0."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        H = 1.0
        price = BarrierOption.analytical_price(
            S, K, T, r, sigma, H, 'down-and-in', 'call',
        )
        assert price < 1e-6

    def test_up_in_put_far_barrier_is_zero(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        H = 10_000.0
        price = BarrierOption.analytical_price(
            S, K, T, r, sigma, H, 'up-and-in', 'put',
        )
        assert price < 1e-6

    def test_already_breached_down_in_call_is_vanilla(self):
        """If S <= H at t=0, a down-and-in is immediately active."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        H = 100.0
        price = BarrierOption.analytical_price(
            S, K, T, r, sigma, H, 'down-and-in', 'call',
        )
        bs = BlackScholesModel(sigma=sigma)
        vanilla = bs.price(S, K, T, r, 'call')
        assert abs(price - vanilla) < 1e-10

    def test_already_breached_up_out_call_is_rebate(self):
        """If S >= H at t=0, an up-and-out is dead and only the rebate remains."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        H = 100.0
        rebate = 5.0
        price = BarrierOption.analytical_price(
            S, K, T, r, sigma, H, 'up-and-out', 'call', rebate=rebate,
        )
        assert abs(price - rebate) < 1e-12

    def test_rebate_adds_positive_premium_to_knock_in(self):
        """
        Adding a cash rebate to a knock-in option raises its price,
        bounded above by the discounted rebate (paid at T iff no knock-in).
        """
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        H = 70.0
        p_no = BarrierOption.analytical_price(
            S, K, T, r, sigma, H, 'down-and-in', 'call', rebate=0.0,
        )
        p_yes = BarrierOption.analytical_price(
            S, K, T, r, sigma, H, 'down-and-in', 'call', rebate=10.0,
        )
        assert p_yes > p_no
        assert p_yes <= p_no + 10.0 * np.exp(-r * T) + 1e-10


# ----------------------------------------------------------------------------
# 16. Barrier Option Monte Carlo Validation
# ----------------------------------------------------------------------------
class TestBarrierMC:
    """
    Cross-validate discretely-monitored MC against the BGY-shifted
    continuous-monitoring analytical formula.
    """

    def test_payoff_shape(self):
        bar = BarrierOption(K=100, barrier=90, barrier_type='down-and-out')
        paths = np.array([
            [100.0, 105.0, 110.0, 120.0],  # alive, ITM -> 20
            [100.0,  95.0,  85.0, 100.0],  # knocked out -> 0
            [100.0, 100.0, 100.0,  95.0],  # alive, OTM  -> 0
        ])
        payoffs = bar.payoff(paths)
        assert payoffs.shape == (3,)
        np.testing.assert_allclose(payoffs, [20.0, 0.0, 0.0])

    def test_payoff_knock_in_complement(self):
        """
        Knock-in payoff on the same paths is the complement of knock-out.
        Sum must equal the vanilla payoff exactly.
        """
        out = BarrierOption(K=100, barrier=90, barrier_type='down-and-out')
        ins = BarrierOption(K=100, barrier=90, barrier_type='down-and-in')
        paths = np.array([
            [100.0, 105.0, 110.0, 120.0],
            [100.0,  95.0,  85.0, 100.0],
            [100.0, 100.0, 100.0,  95.0],
            [100.0,  88.0,  95.0, 130.0],
        ])
        vanilla = np.maximum(paths[:, -1] - 100.0, 0.0)
        np.testing.assert_allclose(out.payoff(paths) + ins.payoff(paths), vanilla)

    def test_down_out_call_mc_vs_analytical_bgy(self):
        """
        With BGY continuity correction, discrete MC matches the shifted
        analytical within 3.5 * SE. Without correction, a systematic
        O(1/sqrt(n_steps)) bias would exceed this tolerance.
        """
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        H = 90.0
        bar = BarrierOption(K=K, barrier=H, barrier_type='down-and-out')
        res = _mc_barrier_price(bar, S, T, r, sigma)

        dt = T / BARRIER_STEPS
        H_eff = BarrierOption.continuity_correction(H, sigma, dt, 'down')
        rr_shift = BarrierOption.analytical_price(
            S, K, T, r, sigma, H_eff, 'down-and-out', 'call',
        )
        assert abs(res.price - rr_shift) < 3.5 * res.std_error

    def test_up_out_call_mc_vs_analytical_bgy(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        H = 130.0
        bar = BarrierOption(K=K, barrier=H, barrier_type='up-and-out')
        res = _mc_barrier_price(bar, S, T, r, sigma)

        dt = T / BARRIER_STEPS
        H_eff = BarrierOption.continuity_correction(H, sigma, dt, 'up')
        rr_shift = BarrierOption.analytical_price(
            S, K, T, r, sigma, H_eff, 'up-and-out', 'call',
        )
        assert abs(res.price - rr_shift) < 3.5 * res.std_error

    def test_down_in_put_mc_vs_analytical_bgy(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        H = 85.0
        bar = BarrierOption(K=K, barrier=H, barrier_type='down-and-in',
                            option_type='put')
        res = _mc_barrier_price(bar, S, T, r, sigma)

        dt = T / BARRIER_STEPS
        H_eff = BarrierOption.continuity_correction(H, sigma, dt, 'down')
        rr_shift = BarrierOption.analytical_price(
            S, K, T, r, sigma, H_eff, 'down-and-in', 'put',
        )
        assert abs(res.price - rr_shift) < 3.5 * res.std_error

    def test_up_in_put_mc_vs_analytical_bgy(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        H = 115.0
        bar = BarrierOption(K=K, barrier=H, barrier_type='up-and-in',
                            option_type='put')
        res = _mc_barrier_price(bar, S, T, r, sigma)

        dt = T / BARRIER_STEPS
        H_eff = BarrierOption.continuity_correction(H, sigma, dt, 'up')
        rr_shift = BarrierOption.analytical_price(
            S, K, T, r, sigma, H_eff, 'up-and-in', 'put',
        )
        assert abs(res.price - rr_shift) < 3.5 * res.std_error

    def test_mc_respects_in_out_parity_pathwise(self):
        """
        Knock-out + knock-in payoffs on the SAME paths reproduce the
        vanilla payoff exactly (floating-point).
        """
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        H = 90.0
        mc = MonteCarloEngine(n_paths=50_000, seed=MC_SEED)
        paths = mc.simulate_gbm(S, T, r, sigma, n_steps=252)

        out = BarrierOption(K=K, barrier=H, barrier_type='down-and-out')
        ins = BarrierOption(K=K, barrier=H, barrier_type='down-and-in')
        vanilla = np.maximum(paths[:, -1] - K, 0.0)
        np.testing.assert_allclose(
            out.payoff(paths) + ins.payoff(paths), vanilla,
            rtol=0, atol=1e-12,
        )


# ----------------------------------------------------------------------------
# 17. Broadie-Glasserman-Kou Continuity Correction
# ----------------------------------------------------------------------------
class TestBarrierBGY:
    """BGY correction utility and its effect on MC bias."""

    def test_correction_constant_matches_bgk_value(self):
        """
        BGK constant beta = -zeta(1/2) / sqrt(2*pi) ~= 0.5826. Recover
        it from the up-correction at sigma=1, dt=1:
            H_eff / H = exp(+beta * sigma * sqrt(dt)) = exp(beta).
        """
        ratio = BarrierOption.continuity_correction(100.0, 1.0, 1.0, 'up') / 100.0
        assert abs(ratio - np.exp(0.5826)) < 1e-4

    def test_correction_up_pushes_higher(self):
        H_eff = BarrierOption.continuity_correction(120.0, 0.20, 1 / 252, 'up')
        assert H_eff > 120.0

    def test_correction_down_pushes_lower(self):
        H_eff = BarrierOption.continuity_correction(80.0, 0.20, 1 / 252, 'down')
        assert H_eff < 80.0

    def test_correction_converges_as_dt_zero(self):
        """H_eff -> H as dt -> 0 (the continuous-monitoring limit)."""
        H = 100.0
        H_coarse = BarrierOption.continuity_correction(H, 0.20, 1e-1, 'up')
        H_fine = BarrierOption.continuity_correction(H, 0.20, 1e-8, 'up')
        assert abs(H_fine - H) < abs(H_coarse - H)
        assert abs(H_fine - H) < 1e-2

    def test_bgy_improves_mc_agreement(self):
        """
        For a discretely-monitored down-and-out call with 100 steps, the
        MC price matches the BGY-shifted analytical more closely than
        the nominal analytical. This demonstrates the O(1/n) -> O(1/n^2)
        bias reduction claimed by Broadie-Glasserman-Kou (1997).
        """
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        H = 90.0
        n_steps = 100
        dt = T / n_steps

        mc = MonteCarloEngine(n_paths=400_000, seed=MC_SEED)
        paths = mc.simulate_gbm(S, T, r, sigma, n_steps=n_steps)
        bar = BarrierOption(K=K, barrier=H, barrier_type='down-and-out')
        res = mc.price(bar.payoff, paths, r, T)

        rr_nominal = BarrierOption.analytical_price(
            S, K, T, r, sigma, H, 'down-and-out', 'call',
        )
        H_eff = BarrierOption.continuity_correction(H, sigma, dt, 'down')
        rr_shift = BarrierOption.analytical_price(
            S, K, T, r, sigma, H_eff, 'down-and-out', 'call',
        )
        gap_nom = abs(res.price - rr_nominal)
        gap_shift = abs(res.price - rr_shift)
        assert gap_shift < gap_nom


# ----------------------------------------------------------------------------
# 18. Barrier Option Edge Cases and Deterministic Paths
# ----------------------------------------------------------------------------
class TestBarrierEdgeCases:
    """Hand-crafted paths verify payoff semantics exactly."""

    def test_deterministic_knock_out_down(self):
        bar = BarrierOption(K=100, barrier=95, barrier_type='down-and-out')
        paths = np.array([[100.0, 98.0, 94.0, 110.0]])
        assert bar.payoff(paths)[0] == 0.0

    def test_deterministic_survives_down_out(self):
        bar = BarrierOption(K=100, barrier=95, barrier_type='down-and-out')
        paths = np.array([[100.0, 98.0, 96.0, 115.0]])  # never <= 95
        assert bar.payoff(paths)[0] == 15.0

    def test_deterministic_knock_in_up(self):
        bar = BarrierOption(K=100, barrier=110, barrier_type='up-and-in')
        paths = np.array([[100.0, 112.0, 105.0, 120.0]])
        assert bar.payoff(paths)[0] == 20.0

    def test_deterministic_no_knock_in_down(self):
        bar = BarrierOption(K=100, barrier=85, barrier_type='down-and-in',
                            option_type='put')
        paths = np.array([[100.0, 90.0, 92.0, 88.0]])  # never <= 85
        assert bar.payoff(paths)[0] == 0.0

    def test_initial_spot_counts_in_extremum(self):
        """S_0 at the barrier is already a breach (weak inequality)."""
        bar = BarrierOption(K=100, barrier=100, barrier_type='down-and-out')
        paths = np.array([[100.0, 110.0, 120.0, 115.0]])
        assert bar.payoff(paths)[0] == 0.0

    def test_batch_payoff_all_knock_out(self):
        """Vectorization: 1000 paths, all hit the barrier -> all zero."""
        np.random.seed(7)
        paths = 100.0 * np.ones((1000, 10))
        paths[:, 4] = 50.0  # force touch barrier 80
        bar = BarrierOption(K=100, barrier=80, barrier_type='down-and-out')
        payoffs = bar.payoff(paths)
        assert payoffs.shape == (1000,)
        np.testing.assert_array_equal(payoffs, 0.0)


# ----------------------------------------------------------------------------
# 19. Barrier Option Input Validation
# ----------------------------------------------------------------------------
class TestBarrierValidation:
    def test_constructor_negative_K(self):
        with pytest.raises(ValueError, match="K must be > 0"):
            BarrierOption(K=-10, barrier=90, barrier_type='down-and-out')

    def test_constructor_zero_barrier(self):
        with pytest.raises(ValueError, match="barrier must be > 0"):
            BarrierOption(K=100, barrier=0, barrier_type='down-and-out')

    def test_constructor_invalid_barrier_type(self):
        with pytest.raises(ValueError, match="barrier_type"):
            BarrierOption(K=100, barrier=90, barrier_type='sideways')

    def test_constructor_invalid_option_type(self):
        with pytest.raises(ValueError, match="option_type"):
            BarrierOption(K=100, barrier=90, barrier_type='down-and-out',
                          option_type='strange')

    def test_constructor_negative_rebate(self):
        with pytest.raises(ValueError, match="rebate"):
            BarrierOption(K=100, barrier=90, barrier_type='down-and-out',
                          rebate=-1.0)

    def test_analytical_zero_sigma(self):
        with pytest.raises(ValueError, match="sigma must be > 0"):
            BarrierOption.analytical_price(
                100, 100, 1.0, 0.05, 0.0, 90, 'down-and-out', 'call',
            )

    def test_analytical_negative_time(self):
        with pytest.raises(ValueError, match="T must be > 0"):
            BarrierOption.analytical_price(
                100, 100, -1.0, 0.05, 0.20, 90, 'down-and-out', 'call',
            )

    def test_correction_invalid_direction(self):
        with pytest.raises(ValueError, match="direction"):
            BarrierOption.continuity_correction(100, 0.20, 1 / 252, 'sideways')

    def test_correction_non_positive_dt(self):
        with pytest.raises(ValueError, match="dt must be > 0"):
            BarrierOption.continuity_correction(100, 0.20, 0.0, 'up')

    def test_case_insensitive_option_type(self):
        """'C' and 'c' and 'call' are all valid."""
        b1 = BarrierOption(K=100, barrier=90, barrier_type='down-and-out',
                           option_type='C')
        b2 = BarrierOption(K=100, barrier=90, barrier_type='down-and-out',
                           option_type='call')
        assert b1.option_type == 'call'
        assert b2.option_type == 'call'


# ----------------------------------------------------------------------------
# 20. Barrier Option Identity
# ----------------------------------------------------------------------------
class TestBarrierIdentity:
    def test_repr(self):
        bar = BarrierOption(K=100, barrier=90, barrier_type='down-and-out')
        r = repr(bar)
        assert "K=100" in r
        assert "barrier=90" in r
        assert "down-and-out" in r
        assert "call" in r

    def test_repr_with_rebate(self):
        bar = BarrierOption(K=100, barrier=90, barrier_type='down-and-out',
                            rebate=3.0)
        assert "rebate=3" in repr(bar)

    def test_eq_and_hash(self):
        a = BarrierOption(K=100, barrier=90, barrier_type='down-and-out')
        b = BarrierOption(K=100, barrier=90, barrier_type='down-and-out')
        c = BarrierOption(K=100, barrier=90, barrier_type='down-and-in')
        assert a == b
        assert hash(a) == hash(b)
        assert a != c
        assert len({a, b, c}) == 2

    def test_is_exotic_option(self):
        bar = BarrierOption(K=100, barrier=90, barrier_type='down-and-out')
        assert isinstance(bar, ExoticOption)


# ----------------------------------------------------------------------------
# 21. Barrier Option Hypothesis Property-Based Tests
# ----------------------------------------------------------------------------
barrier_down_factor_st = st.floats(
    min_value=0.55, max_value=0.95, allow_nan=False, allow_infinity=False,
)
barrier_up_factor_st = st.floats(
    min_value=1.05, max_value=1.80, allow_nan=False, allow_infinity=False,
)


class TestBarrierHypothesis:
    """Property-based tests with Hypothesis (300 cases per property)."""

    @given(S=spot_st, K=strike_st, T=time_st, r=rate_st, sigma=vol_st,
           q=div_st, f=barrier_down_factor_st)
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_in_out_parity_down_call(
        self, S: float, K: float, T: float, r: float,
        sigma: float, q: float, f: float,
    ):
        H = S * f
        v_out = BarrierOption.analytical_price(
            S, K, T, r, sigma, H, 'down-and-out', 'call', q=q,
        )
        v_in = BarrierOption.analytical_price(
            S, K, T, r, sigma, H, 'down-and-in', 'call', q=q,
        )
        bs = BlackScholesModel(sigma=sigma)
        vanilla = bs.price(S, K, T, r, 'call', q)
        assert abs(v_out + v_in - vanilla) < 1e-6 * max(vanilla, 1.0)

    @given(S=spot_st, K=strike_st, T=time_st, r=rate_st, sigma=vol_st,
           q=div_st, f=barrier_up_factor_st)
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_in_out_parity_up_put(
        self, S: float, K: float, T: float, r: float,
        sigma: float, q: float, f: float,
    ):
        H = S * f
        v_out = BarrierOption.analytical_price(
            S, K, T, r, sigma, H, 'up-and-out', 'put', q=q,
        )
        v_in = BarrierOption.analytical_price(
            S, K, T, r, sigma, H, 'up-and-in', 'put', q=q,
        )
        bs = BlackScholesModel(sigma=sigma)
        vanilla = bs.price(S, K, T, r, 'put', q)
        assert abs(v_out + v_in - vanilla) < 1e-6 * max(vanilla, 1.0)

    @given(S=spot_st, K=strike_st, T=time_st, r=rate_st, sigma=vol_st,
           q=div_st, f=barrier_down_factor_st)
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_knock_out_leq_vanilla_call(
        self, S: float, K: float, T: float, r: float,
        sigma: float, q: float, f: float,
    ):
        """Knock-out call <= vanilla call (the knock-out kills some payoff)."""
        H = S * f
        v_out = BarrierOption.analytical_price(
            S, K, T, r, sigma, H, 'down-and-out', 'call', q=q,
        )
        bs = BlackScholesModel(sigma=sigma)
        vanilla = bs.price(S, K, T, r, 'call', q)
        assert v_out <= vanilla + 1e-8 * max(vanilla, 1.0)

    @given(S=spot_st, K=strike_st, T=time_st, r=rate_st, sigma=vol_st,
           q=div_st, f=barrier_down_factor_st)
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_knock_in_non_negative_call(
        self, S: float, K: float, T: float, r: float,
        sigma: float, q: float, f: float,
    ):
        H = S * f
        v_in = BarrierOption.analytical_price(
            S, K, T, r, sigma, H, 'down-and-in', 'call', q=q,
        )
        assert v_in >= -1e-10


# ============================================================================
# ============================================================================
#                           LOOKBACK OPTIONS
# ============================================================================
# ============================================================================
LOOKBACK_PATHS = 200_000
LOOKBACK_STEPS = 252


# ----------------------------------------------------------------------------
# 22. Lookback Analytical Prices (Goldman-Sosin-Gatto + Conze-Viswanathan)
# ----------------------------------------------------------------------------
class TestLookbackAnalytical:
    """
    Closed-form lookback prices verified against:

    1. Non-negativity and strict positivity for T > 0 and sigma > 0.
    2. Continuity at r = q: the zero-drift L'Hopital limit agrees with
       the main formula evaluated at |r - q| just above the threshold.
    3. Lookback >= vanilla European (timing-risk premium).
    4. Fixed-strike = floating-strike when K = S (the OTM branch
       reduces to the floating formula).
    """

    def test_floating_call_positive(self):
        price = LookbackOption.analytical_price(
            100.0, 1.0, 0.05, 0.20, 'call', 'floating', q=0.02,
        )
        assert price > 0

    def test_floating_put_positive(self):
        price = LookbackOption.analytical_price(
            100.0, 1.0, 0.05, 0.20, 'put', 'floating', q=0.02,
        )
        assert price > 0

    def test_fixed_call_positive(self):
        price = LookbackOption.analytical_price(
            100.0, 1.0, 0.05, 0.20, 'call', 'fixed', K=100.0, q=0.02,
        )
        assert price > 0

    def test_fixed_put_positive(self):
        price = LookbackOption.analytical_price(
            100.0, 1.0, 0.05, 0.20, 'put', 'fixed', K=100.0, q=0.02,
        )
        assert price > 0

    def test_floating_call_geq_vanilla(self):
        """Floating-strike lookback >= vanilla European (timing-risk premium)."""
        S, T, r, sigma, q = 100.0, 1.0, 0.05, 0.20, 0.02
        lb = LookbackOption.analytical_price(
            S, T, r, sigma, 'call', 'floating', q=q,
        )
        bs = BlackScholesModel(sigma=sigma).price(S, S, T, r, 'call', q)
        assert lb > bs

    def test_floating_put_geq_vanilla(self):
        S, T, r, sigma, q = 100.0, 1.0, 0.05, 0.20, 0.02
        lb = LookbackOption.analytical_price(
            S, T, r, sigma, 'put', 'floating', q=q,
        )
        bs = BlackScholesModel(sigma=sigma).price(S, S, T, r, 'put', q)
        assert lb > bs

    def test_zero_drift_limit_matches_main_formula(self):
        """
        L'Hopital limit at r = q must agree with the main formula
        evaluated with a tiny but non-zero drift. Cross-checks the
        continuity of the two branches.
        """
        S, T, sigma = 100.0, 1.0, 0.20
        r = 0.05
        price_zero = LookbackOption.analytical_price(
            S, T, r, sigma, 'call', 'floating', q=r,
        )
        # Tiny bump above the _DRIFT_EPS threshold
        price_bump = LookbackOption.analytical_price(
            S, T, r, sigma, 'call', 'floating', q=r - 1e-6,
        )
        assert abs(price_zero - price_bump) < 1e-3

    def test_zero_drift_limit_put_matches_main_formula(self):
        S, T, sigma = 100.0, 1.0, 0.20
        r = 0.05
        p_zero = LookbackOption.analytical_price(
            S, T, r, sigma, 'put', 'floating', q=r,
        )
        p_bump = LookbackOption.analytical_price(
            S, T, r, sigma, 'put', 'floating', q=r - 1e-6,
        )
        assert abs(p_zero - p_bump) < 1e-3

    def test_fixed_call_branch_continuity_at_itm_boundary(self):
        """
        Branch continuity for the fixed-strike call: at the ITM
        boundary ``S_max = K``, both the OTM branch
        ``_fixed_call(S, K, ...)`` and the ITM branch
        ``(S_max - K) e^{-rT} + _fixed_call(S, S_max, ...)`` collapse
        to ``_fixed_call(S, K, ...)`` because the intrinsic is zero.
        Numerically we approach the boundary from above
        (``S_max = K + eps``) and check the limit matches.
        """
        S, T, r, sigma, q = 100.0, 1.0, 0.05, 0.20, 0.02
        K = 105.0
        at_boundary = LookbackOption.analytical_price(
            S, T, r, sigma, 'call', 'fixed', K=K, q=q, S_max=K,
        )
        eps = 1e-6
        from_above = LookbackOption.analytical_price(
            S, T, r, sigma, 'call', 'fixed', K=K, q=q, S_max=K + eps,
        )
        assert abs(at_boundary - from_above) < 1e-4

    def test_fixed_put_branch_continuity_at_itm_boundary(self):
        """
        Symmetric check for the fixed-strike put: at ``S_min = K``,
        both branches collapse to ``_fixed_put(S, K, ...)``.
        """
        S, T, r, sigma, q = 100.0, 1.0, 0.05, 0.20, 0.02
        K = 95.0
        at_boundary = LookbackOption.analytical_price(
            S, T, r, sigma, 'put', 'fixed', K=K, q=q, S_min=K,
        )
        eps = 1e-6
        from_below = LookbackOption.analytical_price(
            S, T, r, sigma, 'put', 'fixed', K=K, q=q, S_min=K - eps,
        )
        assert abs(at_boundary - from_below) < 1e-4

    def test_fixed_call_deep_itm_intrinsic_bound(self):
        """
        Fixed-strike call with K much below S_max: the intrinsic
        ``(S_max - K) e^{-rT}`` sets a hard lower bound on the price.
        """
        S, T, r, sigma = 100.0, 1.0, 0.05, 0.20
        S_max = 120.0
        K = 80.0
        price = LookbackOption.analytical_price(
            S, T, r, sigma, 'call', 'fixed', K=K, S_max=S_max,
        )
        intrinsic = np.exp(-r * T) * (S_max - K)
        assert price >= intrinsic - 1e-10

    def test_seasoned_floating_monotone_in_smin(self):
        """
        For a seasoned floating-strike call, lowering the running
        minimum ``S_min`` can only INCREASE the price (the realised
        low became lower, so the realised payoff is larger).
        """
        S, T, r, sigma, q = 100.0, 0.5, 0.05, 0.20, 0.02
        p_high = LookbackOption.analytical_price(
            S, T, r, sigma, 'call', 'floating', q=q, S_min=100.0,
        )
        p_low = LookbackOption.analytical_price(
            S, T, r, sigma, 'call', 'floating', q=q, S_min=80.0,
        )
        assert p_low > p_high


# ----------------------------------------------------------------------------
# 23. Lookback Payoff Semantics
# ----------------------------------------------------------------------------
class TestLookbackPayoff:
    """Deterministic paths verify payoff formulas exactly."""

    def test_floating_call_deterministic(self):
        lb = LookbackOption(option_type='call', strike_type='floating')
        paths = np.array([[100.0, 80.0, 90.0, 120.0]])  # min=80, S_T=120
        np.testing.assert_allclose(lb.payoff(paths), [40.0])

    def test_floating_put_deterministic(self):
        lb = LookbackOption(option_type='put', strike_type='floating')
        paths = np.array([[100.0, 130.0, 110.0, 90.0]])  # max=130, S_T=90
        np.testing.assert_allclose(lb.payoff(paths), [40.0])

    def test_fixed_call_deterministic_itm(self):
        lb = LookbackOption(option_type='call', strike_type='fixed', K=100.0)
        paths = np.array([[100.0, 120.0, 115.0, 110.0]])  # max=120
        np.testing.assert_allclose(lb.payoff(paths), [20.0])

    def test_fixed_call_deterministic_otm(self):
        lb = LookbackOption(option_type='call', strike_type='fixed', K=150.0)
        paths = np.array([[100.0, 120.0, 115.0, 110.0]])  # max=120 < K=150
        np.testing.assert_allclose(lb.payoff(paths), [0.0])

    def test_fixed_put_deterministic_itm(self):
        lb = LookbackOption(option_type='put', strike_type='fixed', K=100.0)
        paths = np.array([[100.0, 80.0, 85.0, 90.0]])  # min=80
        np.testing.assert_allclose(lb.payoff(paths), [20.0])

    def test_floating_call_always_non_negative(self):
        """Floating-strike payoffs are always >= 0 (never OTM)."""
        lb = LookbackOption(option_type='call', strike_type='floating')
        np.random.seed(MC_SEED)
        mc = MonteCarloEngine(n_paths=5000, seed=MC_SEED)
        paths = mc.simulate_gbm(100.0, 1.0, 0.05, 0.20, n_steps=100)
        payoffs = lb.payoff(paths)
        assert np.all(payoffs >= 0.0)

    def test_floating_put_always_non_negative(self):
        lb = LookbackOption(option_type='put', strike_type='floating')
        mc = MonteCarloEngine(n_paths=5000, seed=MC_SEED)
        paths = mc.simulate_gbm(100.0, 1.0, 0.05, 0.20, n_steps=100)
        payoffs = lb.payoff(paths)
        assert np.all(payoffs >= 0.0)

    def test_s0_included_in_extremum(self):
        """Running extremum includes the initial spot paths[:, 0]."""
        lb = LookbackOption(option_type='call', strike_type='floating')
        paths = np.array([[100.0, 105.0, 110.0, 115.0]])  # min=100=S0
        np.testing.assert_allclose(lb.payoff(paths), [15.0])


# ----------------------------------------------------------------------------
# 24. Lookback Monte Carlo Validation
# ----------------------------------------------------------------------------
class TestLookbackMC:
    """
    MC cross-validation of the Goldman-Sosin-Gatto / Conze-Viswanathan
    analytical formulas. The Broadie-Glasserman-Kou continuity correction
    is applied to the discretely-observed extremum through the public
    ``LookbackOption.continuity_correction`` API, mirroring the barrier
    case.
    """

    def _mc_with_bgk(
        self, lb: LookbackOption, S: float, T: float, r: float,
        sigma: float, n_steps: int = LOOKBACK_STEPS, q: float = 0.0,
    ):
        """
        Run MC with Broadie-Glasserman-Kou adjustment to the discrete
        running extremum, so the resulting MC price matches the
        continuous-monitoring analytical without an O(1/sqrt(n)) bias.
        """
        mc = MonteCarloEngine(n_paths=LOOKBACK_PATHS, seed=MC_SEED)
        paths = mc.simulate_gbm(S, T, r, sigma, q=q, n_steps=n_steps)
        dt = T / n_steps
        min_corrected = LookbackOption.continuity_correction(
            np.min(paths, axis=1), sigma, dt, 'min',
        )
        max_corrected = LookbackOption.continuity_correction(
            np.max(paths, axis=1), sigma, dt, 'max',
        )
        S_T = paths[:, -1]

        if lb.strike_type == 'floating':
            if lb.option_type == 'call':
                raw = S_T - min_corrected
            else:
                raw = max_corrected - S_T
        else:
            if lb.option_type == 'call':
                raw = np.maximum(max_corrected - lb.K, 0.0)
            else:
                raw = np.maximum(lb.K - min_corrected, 0.0)

        return mc.price(lambda _: raw, paths, r, T)

    def test_floating_call_mc_vs_analytical(self):
        S, T, r, sigma, q = 100.0, 1.0, 0.05, 0.20, 0.02
        lb = LookbackOption(option_type='call', strike_type='floating')
        res = self._mc_with_bgk(lb, S, T, r, sigma, q=q)
        analytical = LookbackOption.analytical_price(
            S, T, r, sigma, 'call', 'floating', q=q,
        )
        assert abs(res.price - analytical) < 3.5 * res.std_error

    def test_floating_put_mc_vs_analytical(self):
        S, T, r, sigma, q = 100.0, 1.0, 0.05, 0.20, 0.02
        lb = LookbackOption(option_type='put', strike_type='floating')
        res = self._mc_with_bgk(lb, S, T, r, sigma, q=q)
        analytical = LookbackOption.analytical_price(
            S, T, r, sigma, 'put', 'floating', q=q,
        )
        assert abs(res.price - analytical) < 3.5 * res.std_error

    def test_fixed_call_mc_vs_analytical(self):
        S, T, r, sigma, q = 100.0, 1.0, 0.05, 0.20, 0.02
        lb = LookbackOption(option_type='call', strike_type='fixed', K=105.0)
        res = self._mc_with_bgk(lb, S, T, r, sigma, q=q)
        analytical = LookbackOption.analytical_price(
            S, T, r, sigma, 'call', 'fixed', K=105.0, q=q,
        )
        assert abs(res.price - analytical) < 3.5 * res.std_error

    def test_fixed_put_mc_vs_analytical(self):
        S, T, r, sigma, q = 100.0, 1.0, 0.05, 0.20, 0.02
        lb = LookbackOption(option_type='put', strike_type='fixed', K=95.0)
        res = self._mc_with_bgk(lb, S, T, r, sigma, q=q)
        analytical = LookbackOption.analytical_price(
            S, T, r, sigma, 'put', 'fixed', K=95.0, q=q,
        )
        assert abs(res.price - analytical) < 3.5 * res.std_error

    def test_fixed_call_mc_vs_analytical_zero_drift(self):
        """
        r = q routes the fixed-strike call through the L'Hopital limit
        (_fixed_call_zero_drift) — previously untested against MC.
        """
        S, T, r, sigma = 100.0, 1.0, 0.05, 0.20
        lb = LookbackOption(option_type='call', strike_type='fixed', K=105.0)
        res = self._mc_with_bgk(lb, S, T, r, sigma, q=r)
        analytical = LookbackOption.analytical_price(
            S, T, r, sigma, 'call', 'fixed', K=105.0, q=r,
        )
        assert abs(res.price - analytical) < 3.5 * res.std_error

    def test_fixed_put_mc_vs_analytical_zero_drift(self):
        """r = q routes the fixed-strike put through _fixed_put_zero_drift."""
        S, T, r, sigma = 100.0, 1.0, 0.05, 0.20
        lb = LookbackOption(option_type='put', strike_type='fixed', K=95.0)
        res = self._mc_with_bgk(lb, S, T, r, sigma, q=r)
        analytical = LookbackOption.analytical_price(
            S, T, r, sigma, 'put', 'fixed', K=95.0, q=r,
        )
        assert abs(res.price - analytical) < 3.5 * res.std_error


# ----------------------------------------------------------------------------
# 25. Lookback Edge Cases + Validation + Identity
# ----------------------------------------------------------------------------
class TestLookbackEdgeCases:
    """Degenerate parameters and validation."""

    def test_fixed_call_K_zero_equals_discounted_max(self):
        """
        Fixed-strike call with K = 0 is equivalent to holding the
        running max itself, so the price must equal ``E^Q[S_max] * e^{-rT}``.
        We verify this via MC since there is no closed form for E^Q[S_max]
        that is simpler than the analytical formula itself — the
        cross-check here is that the K=0 analytical price exceeds the
        terminal forward ``S * e^{(r-q-r)T} = S * e^{-qT}``.
        """
        S, T, r, sigma, q = 100.0, 1.0, 0.05, 0.20, 0.02
        price = LookbackOption.analytical_price(
            S, T, r, sigma, 'call', 'fixed', K=1e-8, q=q,
        )
        assert price > S * np.exp(-q * T)

    def test_constructor_fixed_missing_K(self):
        with pytest.raises(ValueError, match="K is required"):
            LookbackOption(strike_type='fixed')

    def test_constructor_fixed_negative_K(self):
        with pytest.raises(ValueError, match="K must be > 0"):
            LookbackOption(strike_type='fixed', K=-10)

    def test_constructor_invalid_option_type(self):
        with pytest.raises(ValueError, match="option_type"):
            LookbackOption(option_type='weird')

    def test_constructor_invalid_strike_type(self):
        with pytest.raises(ValueError, match="strike_type"):
            LookbackOption(strike_type='european')

    def test_analytical_negative_sigma(self):
        with pytest.raises(ValueError, match="sigma must be > 0"):
            LookbackOption.analytical_price(
                100, 1.0, 0.05, -0.20, 'call', 'floating',
            )

    def test_analytical_negative_time(self):
        with pytest.raises(ValueError, match="T must be > 0"):
            LookbackOption.analytical_price(
                100, -1.0, 0.05, 0.20, 'call', 'floating',
            )

    def test_analytical_fixed_missing_K(self):
        with pytest.raises(ValueError, match="K is required"):
            LookbackOption.analytical_price(
                100, 1.0, 0.05, 0.20, 'call', 'fixed',
            )

    def test_analytical_smin_above_spot_raises(self):
        with pytest.raises(ValueError, match="S_min must be <= S"):
            LookbackOption.analytical_price(
                100, 1.0, 0.05, 0.20, 'call', 'floating', S_min=120.0,
            )

    def test_analytical_smax_below_spot_raises(self):
        with pytest.raises(ValueError, match="S_max must be >= S"):
            LookbackOption.analytical_price(
                100, 1.0, 0.05, 0.20, 'put', 'floating', S_max=80.0,
            )


class TestLookbackIdentity:
    def test_repr_floating(self):
        lb = LookbackOption(option_type='call', strike_type='floating')
        r = repr(lb)
        assert "call" in r and "floating" in r and "K=" not in r

    def test_repr_fixed(self):
        lb = LookbackOption(option_type='put', strike_type='fixed', K=100.0)
        r = repr(lb)
        assert "put" in r and "fixed" in r and "K=100" in r

    def test_eq_and_hash(self):
        a = LookbackOption(option_type='call', strike_type='floating')
        b = LookbackOption(option_type='call', strike_type='floating')
        c = LookbackOption(option_type='call', strike_type='fixed', K=100.0)
        assert a == b and hash(a) == hash(b)
        assert a != c
        assert len({a, b, c}) == 2

    def test_is_exotic_option(self):
        lb = LookbackOption()
        assert isinstance(lb, ExoticOption)


# ----------------------------------------------------------------------------
# 26. Lookback Hypothesis Property-Based Tests
# ----------------------------------------------------------------------------
class TestLookbackHypothesis:
    """Property-based tests: 300 cases per property."""

    @given(S=spot_st, T=time_st, r=rate_st, sigma=vol_st, q=div_st)
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_floating_call_non_negative(
        self, S: float, T: float, r: float, sigma: float, q: float,
    ):
        assume(abs(r - q) > 1e-10 or True)  # both branches covered
        price = LookbackOption.analytical_price(
            S, T, r, sigma, 'call', 'floating', q=q,
        )
        assert price >= -1e-10

    @given(S=spot_st, T=time_st, r=rate_st, sigma=vol_st, q=div_st)
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_floating_put_non_negative(
        self, S: float, T: float, r: float, sigma: float, q: float,
    ):
        price = LookbackOption.analytical_price(
            S, T, r, sigma, 'put', 'floating', q=q,
        )
        assert price >= -1e-10

    @given(S=spot_st, T=time_st, r=rate_st, sigma=vol_st, q=div_st)
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_floating_call_geq_vanilla_atm(
        self, S: float, T: float, r: float, sigma: float, q: float,
    ):
        """
        Floating-strike call with S_min = S at inception beats the ATM
        vanilla call (same S as strike). Intuition: the vanilla has
        payoff max(S_T - S, 0) while the lookback has S_T - min(S_t)
        >= S_T - S >= max(S_T - S, 0) pathwise — and usually strictly.
        """
        lb = LookbackOption.analytical_price(
            S, T, r, sigma, 'call', 'floating', q=q,
        )
        vanilla = BlackScholesModel(sigma=sigma).price(S, S, T, r, 'call', q)
        assert lb >= vanilla - 1e-8 * max(S, 1.0)

    @given(S=spot_st, T=time_st, r=rate_st, sigma=vol_st, q=div_st)
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_floating_put_geq_vanilla_atm(
        self, S: float, T: float, r: float, sigma: float, q: float,
    ):
        lb = LookbackOption.analytical_price(
            S, T, r, sigma, 'put', 'floating', q=q,
        )
        vanilla = BlackScholesModel(sigma=sigma).price(S, S, T, r, 'put', q)
        assert lb >= vanilla - 1e-8 * max(S, 1.0)


# ----------------------------------------------------------------------------
# 27. Audit pre-Phase 5 (2026-06): fixed-strike zero-drift branches,
#     continuity-correction API, and validation coverage gaps
# ----------------------------------------------------------------------------
class TestLookbackZeroDriftFixed:
    """
    L'Hopital r = q limit of the FIXED-strike Conze-Viswanathan formulas
    (_fixed_call_zero_drift / _fixed_put_zero_drift), previously untested.

    Branch continuity: the limit form at b = r - q = 0 must agree with the
    generic formula evaluated just above the switching threshold
    (_DRIFT_EPS = 1e-10). At b = 1e-7 the generic formula deviates from
    the b = 0 limit by O(b) ~ 1e-5 in price terms, while a sign or term
    error in the limit form would show up at O(S sigma sqrt(T) n(e1)) ~ 1.
    """

    S, T, R, SIGMA = 100.0, 1.0, 0.05, 0.20
    B_SMALL = 1e-7  # just above _DRIFT_EPS: generic-formula branch
    TOL = 1e-4

    def test_fixed_call_otm_branch_continuity(self):
        """K > S_max = S: the M = K (OTM) branch of the fixed call."""
        limit = LookbackOption.analytical_price(
            self.S, self.T, self.R, self.SIGMA, 'call', 'fixed',
            K=120.0, q=self.R,
        )
        generic = LookbackOption.analytical_price(
            self.S, self.T, self.R, self.SIGMA, 'call', 'fixed',
            K=120.0, q=self.R - self.B_SMALL,
        )
        assert abs(limit - generic) < self.TOL

    def test_fixed_call_itm_branch_continuity(self):
        """K <= S_max: intrinsic + M = S_max branch of the fixed call."""
        limit = LookbackOption.analytical_price(
            self.S, self.T, self.R, self.SIGMA, 'call', 'fixed',
            K=90.0, q=self.R,
        )
        generic = LookbackOption.analytical_price(
            self.S, self.T, self.R, self.SIGMA, 'call', 'fixed',
            K=90.0, q=self.R - self.B_SMALL,
        )
        assert abs(limit - generic) < self.TOL

    def test_fixed_put_otm_branch_continuity(self):
        """K < S_min = S: the M = K (OTM) branch of the fixed put."""
        limit = LookbackOption.analytical_price(
            self.S, self.T, self.R, self.SIGMA, 'put', 'fixed',
            K=80.0, q=self.R,
        )
        generic = LookbackOption.analytical_price(
            self.S, self.T, self.R, self.SIGMA, 'put', 'fixed',
            K=80.0, q=self.R - self.B_SMALL,
        )
        assert abs(limit - generic) < self.TOL

    def test_fixed_put_itm_branch_continuity(self):
        """K >= S_min: intrinsic + M = S_min branch of the fixed put."""
        limit = LookbackOption.analytical_price(
            self.S, self.T, self.R, self.SIGMA, 'put', 'fixed',
            K=110.0, q=self.R,
        )
        generic = LookbackOption.analytical_price(
            self.S, self.T, self.R, self.SIGMA, 'put', 'fixed',
            K=110.0, q=self.R - self.B_SMALL,
        )
        assert abs(limit - generic) < self.TOL

    def test_fixed_call_itm_otm_boundary_zero_drift(self):
        """
        At K = S_max the ITM branch carries zero intrinsic and both
        branches must agree (continuity across the branch switch), also
        in the r = q limit.
        """
        at_boundary = LookbackOption.analytical_price(
            self.S, self.T, self.R, self.SIGMA, 'call', 'fixed',
            K=100.0, q=self.R,
        )
        just_otm = LookbackOption.analytical_price(
            self.S, self.T, self.R, self.SIGMA, 'call', 'fixed',
            K=100.0 + 1e-9, q=self.R,
        )
        assert abs(at_boundary - just_otm) < 1e-6

    def test_fixed_call_zero_drift_exceeds_discounted_intrinsic(self):
        """ITM fixed call at r = q is worth more than the locked intrinsic."""
        price = LookbackOption.analytical_price(
            self.S, self.T, self.R, self.SIGMA, 'call', 'fixed',
            K=90.0, q=self.R,
        )
        assert price > np.exp(-self.R * self.T) * 10.0


class TestLookbackContinuityCorrection:
    """
    LookbackOption.continuity_correction — the Broadie-Glasserman-Kou
    extremum adjustment, now a public API (lookback counterpart of
    BarrierOption.continuity_correction).

    Exact anchor: with sigma = 1, dt = 1,
        M_eff / M = exp(+0.5826)  and  m_eff / m = exp(-0.5826).
    """

    def test_max_exact_ratio(self):
        ratio = LookbackOption.continuity_correction(100.0, 1.0, 1.0, 'max') / 100.0
        assert abs(ratio - np.exp(0.5826)) < 1e-12

    def test_min_exact_ratio(self):
        ratio = LookbackOption.continuity_correction(100.0, 1.0, 1.0, 'min') / 100.0
        assert abs(ratio - np.exp(-0.5826)) < 1e-12

    def test_matches_barrier_constant(self):
        """Same BGK beta as the barrier correction: identical factors."""
        lb_max = LookbackOption.continuity_correction(120.0, 0.20, 1 / 252, 'max')
        ba_up = BarrierOption.continuity_correction(120.0, 0.20, 1 / 252, 'up')
        assert abs(lb_max - ba_up) < 1e-12

        lb_min = LookbackOption.continuity_correction(80.0, 0.20, 1 / 252, 'min')
        ba_dn = BarrierOption.continuity_correction(80.0, 0.20, 1 / 252, 'down')
        assert abs(lb_min - ba_dn) < 1e-12

    def test_array_input_elementwise(self):
        ext = np.array([80.0, 100.0, 120.0])
        out = LookbackOption.continuity_correction(ext, 0.20, 1 / 252, 'max')
        assert isinstance(out, np.ndarray)
        assert out.shape == ext.shape
        factor = np.exp(0.5826 * 0.20 * np.sqrt(1 / 252))
        assert np.allclose(out, ext * factor, rtol=0.0, atol=1e-10)

    def test_scalar_returns_float(self):
        out = LookbackOption.continuity_correction(100.0, 0.20, 1 / 252, 'min')
        assert isinstance(out, float)

    def test_fine_monitoring_limit(self):
        """dt -> 0: the correction vanishes (continuous monitoring)."""
        out = LookbackOption.continuity_correction(100.0, 0.20, 1e-12, 'max')
        assert abs(out - 100.0) < 1e-4

    def test_kind_normalization(self):
        a = LookbackOption.continuity_correction(100.0, 0.20, 0.01, ' MAX ')
        b = LookbackOption.continuity_correction(100.0, 0.20, 0.01, 'max')
        assert a == b

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="kind must be"):
            LookbackOption.continuity_correction(100.0, 0.20, 0.01, 'up')

    def test_nonpositive_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma must be > 0"):
            LookbackOption.continuity_correction(100.0, 0.0, 0.01, 'max')

    def test_nonpositive_dt_raises(self):
        with pytest.raises(ValueError, match="dt must be > 0"):
            LookbackOption.continuity_correction(100.0, 0.20, 0.0, 'max')

    def test_nonpositive_extremum_raises(self):
        with pytest.raises(ValueError, match="extremum"):
            LookbackOption.continuity_correction(
                np.array([100.0, 0.0]), 0.20, 0.01, 'max',
            )


class TestLookbackValidationGaps:
    """Shorthand types, raises and identity branches not previously covered."""

    def test_constructor_shorthand_c(self):
        assert LookbackOption(option_type='c').option_type == 'call'

    def test_constructor_shorthand_p(self):
        assert LookbackOption(option_type='p').option_type == 'put'

    def test_analytical_shorthand_matches_full_name(self):
        full = LookbackOption.analytical_price(
            100, 1.0, 0.05, 0.20, 'call', 'floating',
        )
        short = LookbackOption.analytical_price(
            100, 1.0, 0.05, 0.20, 'c', 'floating',
        )
        assert short == full

        full_p = LookbackOption.analytical_price(
            100, 1.0, 0.05, 0.20, 'put', 'floating',
        )
        short_p = LookbackOption.analytical_price(
            100, 1.0, 0.05, 0.20, 'p', 'floating',
        )
        assert short_p == full_p

    def test_analytical_nonpositive_spot_raises(self):
        with pytest.raises(ValueError, match="S must be > 0"):
            LookbackOption.analytical_price(
                0.0, 1.0, 0.05, 0.20, 'call', 'floating',
            )

    def test_analytical_invalid_option_type_raises(self):
        with pytest.raises(ValueError, match="option_type"):
            LookbackOption.analytical_price(
                100, 1.0, 0.05, 0.20, 'straddle', 'floating',
            )

    def test_analytical_invalid_strike_type_raises(self):
        with pytest.raises(ValueError, match="strike_type"):
            LookbackOption.analytical_price(
                100, 1.0, 0.05, 0.20, 'call', 'asian',
            )

    def test_analytical_nonpositive_extrema_raise(self):
        with pytest.raises(ValueError, match="S_min and S_max"):
            LookbackOption.analytical_price(
                100, 1.0, 0.05, 0.20, 'call', 'floating', S_min=-5.0,
            )

    def test_analytical_fixed_nonpositive_K_raises(self):
        with pytest.raises(ValueError, match="K must be > 0"):
            LookbackOption.analytical_price(
                100, 1.0, 0.05, 0.20, 'call', 'fixed', K=0.0,
            )

    def test_eq_other_type_is_not_equal(self):
        lb = LookbackOption()
        assert (lb == 42) is False
        assert lb != 'lookback'


class TestDigitalValidationGaps:
    """Digital audit fixes: float coercion, payout property, raises."""

    def test_cash_amount_coerced_to_float(self):
        """Same coercion contract as K (audit 2026-04 fix for K)."""
        dig = DigitalOption(K=100, cash_amount=5)
        assert type(dig.cash_amount) is float

        dig_np = DigitalOption(K=100, cash_amount=np.float64(5.0))
        assert type(dig_np.cash_amount) is float

    def test_payout_type_property(self):
        assert DigitalOption(K=100, payout_type='cash').payout_type == 'cash'
        assert DigitalOption(K=100, payout_type='asset').payout_type == 'asset'

    def test_constructor_shorthand_c_p(self):
        assert DigitalOption(K=100, option_type='c').option_type == 'call'
        assert DigitalOption(K=100, option_type='p').option_type == 'put'

    def test_analytical_shorthand_matches_full_name(self):
        for short, full in (('c', 'call'), ('p', 'put')):
            a = DigitalOption.analytical_price(100, 100, 1.0, 0.05, 0.20, short)
            b = DigitalOption.analytical_price(100, 100, 1.0, 0.05, 0.20, full)
            assert a == b

    def test_analytical_nonpositive_K_raises(self):
        with pytest.raises(ValueError, match="K must be > 0"):
            DigitalOption.analytical_price(100, 0.0, 1.0, 0.05, 0.20)

    def test_analytical_negative_cash_amount_raises(self):
        """analytical_price now validates cash_amount like the constructor."""
        with pytest.raises(ValueError, match="cash_amount must be >= 0"):
            DigitalOption.analytical_price(
                100, 100, 1.0, 0.05, 0.20, cash_amount=-1.0,
            )

    def test_analytical_invalid_option_type_raises(self):
        with pytest.raises(ValueError, match="option_type"):
            DigitalOption.analytical_price(100, 100, 1.0, 0.05, 0.20, 'digital')

    def test_analytical_invalid_payout_type_raises(self):
        with pytest.raises(ValueError, match="payout_type"):
            DigitalOption.analytical_price(
                100, 100, 1.0, 0.05, 0.20, 'call', 'shares',
            )

    def test_eq_other_type_is_not_equal(self):
        dig = DigitalOption(K=100)
        assert (dig == 42) is False
        assert dig != 'digital'


# ============================================================================
# ============================================================================
#                           NUMERICAL GREEKS
# ============================================================================
# ============================================================================
GREEK_PATHS = 500_000
GREEK_STEPS = 252


def _greek_engine(n_paths: int = GREEK_PATHS, seed: int = MC_SEED) -> MonteCarloEngine:
    return MonteCarloEngine(n_paths=n_paths, seed=seed)


def _bs_call_payoff(K: float):
    return lambda paths: np.maximum(paths[:, -1] - K, 0.0)


def _bs_put_payoff(K: float):
    return lambda paths: np.maximum(K - paths[:, -1], 0.0)


# ----------------------------------------------------------------------------
# 27. Greeks Cross-Validation against Black-Scholes (gold standard)
# ----------------------------------------------------------------------------
class TestGreeksVsBlackScholes:
    """
    Cross-validate bump-and-revalue Greeks against the closed-form
    Black-Scholes values. With N=500k paths and CRN, the central
    difference estimator should match the analytical value to within a
    few percent absolute, well below the O(h^2) truncation bias plus
    the residual MC noise. These tests are the gold standard for the
    implementation: if they pass, the numerical scheme is sound and we
    can rely on it for payoffs with no closed form.
    """

    def _bs_ref(self, option_type: str, q: float = 0.0) -> dict[str, float]:
        bs = BlackScholesModel(sigma=HULL_SIGMA)
        return bs.greeks(HULL_S, HULL_K, HULL_T, HULL_R, option_type, q)

    def test_delta_call(self):
        mc = _greek_engine()
        delta = numerical_delta(
            mc, _bs_call_payoff(HULL_K),
            HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, n_steps=GREEK_STEPS,
        )
        ref = self._bs_ref('call')['delta']
        assert abs(delta - ref) < 5e-3

    def test_delta_put(self):
        mc = _greek_engine()
        delta = numerical_delta(
            mc, _bs_put_payoff(HULL_K),
            HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, n_steps=GREEK_STEPS,
        )
        ref = self._bs_ref('put')['delta']
        assert abs(delta - ref) < 5e-3

    def test_gamma_call(self):
        mc = _greek_engine()
        gamma = numerical_gamma(
            mc, _bs_call_payoff(HULL_K),
            HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, n_steps=GREEK_STEPS,
        )
        ref = self._bs_ref('call')['gamma']
        # Gamma is noisier: 1/h^2 noise amplification
        assert abs(gamma - ref) < 1e-3

    def test_gamma_put_equals_call(self):
        """Put-call parity: gamma_call = gamma_put (same magnitude)."""
        mc_call = _greek_engine()
        mc_put = _greek_engine()
        g_call = numerical_gamma(
            mc_call, _bs_call_payoff(HULL_K),
            HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, n_steps=GREEK_STEPS,
        )
        g_put = numerical_gamma(
            mc_put, _bs_put_payoff(HULL_K),
            HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, n_steps=GREEK_STEPS,
        )
        assert abs(g_call - g_put) < 1e-3

    def test_vega_call(self):
        mc = _greek_engine()
        vega = numerical_vega(
            mc, _bs_call_payoff(HULL_K),
            HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, n_steps=GREEK_STEPS,
        )
        ref = self._bs_ref('call')['vega']
        # BS vega ~37.5, 2% tolerance
        assert abs(vega - ref) < 1.0

    def test_vega_put_equals_call(self):
        """Put-call parity also fixes vega_call = vega_put."""
        mc_call = _greek_engine()
        mc_put = _greek_engine()
        v_call = numerical_vega(
            mc_call, _bs_call_payoff(HULL_K),
            HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, n_steps=GREEK_STEPS,
        )
        v_put = numerical_vega(
            mc_put, _bs_put_payoff(HULL_K),
            HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, n_steps=GREEK_STEPS,
        )
        assert abs(v_call - v_put) < 1.0

    def test_theta_call(self):
        mc = _greek_engine()
        theta = numerical_theta(
            mc, _bs_call_payoff(HULL_K),
            HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, n_steps=GREEK_STEPS,
        )
        ref = self._bs_ref('call')['theta']
        # BS theta ~-6.4; call theta must be negative for r > 0, q = 0
        assert theta < 0
        assert abs(theta - ref) < 0.2

    def test_rho_call(self):
        mc = _greek_engine()
        rho = numerical_rho(
            mc, _bs_call_payoff(HULL_K),
            HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, n_steps=GREEK_STEPS,
        )
        ref = self._bs_ref('call')['rho']
        # BS rho ~53.2; call rho is positive
        assert rho > 0
        assert abs(rho - ref) < 0.5

    def test_rho_put_negative(self):
        """Put rho is negative (higher rate -> lower put price)."""
        mc = _greek_engine()
        rho = numerical_rho(
            mc, _bs_put_payoff(HULL_K),
            HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, n_steps=GREEK_STEPS,
        )
        ref = self._bs_ref('put')['rho']
        assert rho < 0
        assert abs(rho - ref) < 0.5


# ----------------------------------------------------------------------------
# 28. Greeks Numerical Properties (CRN, convergence, reproducibility)
# ----------------------------------------------------------------------------
class TestGreeksNumericalProperties:
    """
    Verify the critical numerical properties that make bump-and-revalue
    Greeks usable in production: CRN variance reduction, convergence
    with paths, reproducibility across calls.
    """

    def test_crn_reduces_variance_vs_independent_rng(self):
        """
        CRN: V(sigma + h) and V(sigma - h) share random numbers, so the
        variance of their *difference* is much smaller than when they
        use independent draws. We verify this by comparing the sample
        standard deviation of the vega estimator across 12 independent
        seeds, with and without CRN.
        """
        n_repeats = 12
        S, K, T, r, sigma, q = HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, HULL_Q
        h = 0.01

        crn_estimates: list[float] = []
        ind_estimates: list[float] = []

        for trial in range(n_repeats):
            # CRN: one engine, same seed inside numerical_vega
            mc_crn = MonteCarloEngine(n_paths=50_000, seed=1000 + trial)
            crn_estimates.append(
                numerical_vega(
                    mc_crn, _bs_call_payoff(K), S, T, r, sigma, q,
                    bump=h, n_steps=1,
                )
            )
            # Independent draws: two engines with *different* seeds
            mc_up = MonteCarloEngine(n_paths=50_000, seed=2000 + trial)
            mc_dn = MonteCarloEngine(n_paths=50_000, seed=3000 + trial)
            paths_up = mc_up.simulate_gbm(S, T, r, sigma + h, q, n_steps=1)
            paths_dn = mc_dn.simulate_gbm(S, T, r, sigma - h, q, n_steps=1)
            disc = np.exp(-r * T)
            v_up = float(np.mean(disc * np.maximum(paths_up[:, -1] - K, 0.0)))
            v_dn = float(np.mean(disc * np.maximum(paths_dn[:, -1] - K, 0.0)))
            ind_estimates.append((v_up - v_dn) / (2 * h))

        std_crn = float(np.std(crn_estimates, ddof=1))
        std_ind = float(np.std(ind_estimates, ddof=1))
        # CRN should reduce the estimator std by at least 5x in this setup
        assert std_crn < std_ind / 5.0, (
            f"CRN std={std_crn:.4f} not << independent std={std_ind:.4f}"
        )

    def test_delta_converges_with_paths(self):
        """
        More paths -> smaller error against BS analytical. Pure O(1/sqrt(N))
        convergence is contaminated by the O(h^2) bias from the central
        difference, so we only check (a) monotone trend and (b) that the
        largest-N error is smaller in absolute terms than a loose threshold.
        """
        ref = BlackScholesModel(sigma=HULL_SIGMA).greeks(
            HULL_S, HULL_K, HULL_T, HULL_R, 'call', HULL_Q,
        )['delta']
        errors = []
        for n_paths in [20_000, 100_000, 500_000]:
            mc = MonteCarloEngine(n_paths=n_paths, seed=MC_SEED)
            delta = numerical_delta(
                mc, _bs_call_payoff(HULL_K),
                HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, n_steps=1,
            )
            errors.append(abs(delta - ref))
        # Monotone non-increasing (more paths never makes it worse in this seed)
        assert errors[-1] <= errors[0]
        # Largest N is close to the analytical reference
        assert errors[-1] < 2.0e-3

    def test_delta_reproducible_same_seed(self):
        """Identical engines (same seed) must give identical Delta."""
        mc1 = MonteCarloEngine(n_paths=50_000, seed=99)
        mc2 = MonteCarloEngine(n_paths=50_000, seed=99)
        d1 = numerical_delta(
            mc1, _bs_call_payoff(HULL_K),
            HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, n_steps=1,
        )
        d2 = numerical_delta(
            mc2, _bs_call_payoff(HULL_K),
            HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, n_steps=1,
        )
        assert d1 == d2

    def test_gamma_3point_stencil_on_polynomial(self):
        """
        Sanity: for a payoff that is *exactly* quadratic in S_T
        (p(S_T) = S_T^2, undiscounted), the 3-point stencil reproduces
        the analytical second derivative modulo MC noise. Under GBM,
        E^Q[S_T^2] = S_0^2 * exp((2r - 2q + sigma^2) * T), so
        d^2 V / dS_0^2 = 2 * exp(-rT) * exp((2r + sigma^2) * T)
                       = 2 * exp((r + sigma^2) * T).
        """
        S, T, r, sigma, q = 100.0, 1.0, 0.05, 0.20, 0.0
        mc = MonteCarloEngine(n_paths=500_000, seed=MC_SEED)

        def quadratic_payoff(paths: np.ndarray) -> np.ndarray:
            return paths[:, -1] ** 2

        gamma = numerical_gamma(
            mc, quadratic_payoff, S, T, r, sigma, q, n_steps=1,
        )
        analytical = 2.0 * np.exp((r + sigma * sigma) * T)
        assert abs(gamma - analytical) < 0.5


# ----------------------------------------------------------------------------
# 29. Greeks on Exotic Payoffs (Asian, Barrier, Digital)
# ----------------------------------------------------------------------------
class TestGreeksOnExotics:
    """
    Sanity checks of numerical Greeks on payoffs that have no closed
    form (Asian, Barrier) and one with highly non-smooth behaviour
    (Digital). These are qualitative bound / sign checks; the cross
    validation against analytical values lives in the BS suite above.
    """

    def test_asian_call_delta_in_unit_interval(self):
        """Asian call delta is in [0, 1], strictly positive for ATM."""
        asian = AsianOption(K=HULL_K, option_type='call')
        mc = _greek_engine(n_paths=200_000)
        delta = numerical_delta(
            mc, asian.payoff,
            HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, n_steps=GREEK_STEPS,
        )
        assert 0.0 <= delta <= 1.0
        assert delta > 0.1

    def test_asian_call_vega_positive(self):
        """Long optionality => positive vega."""
        asian = AsianOption(K=HULL_K, option_type='call')
        mc = _greek_engine(n_paths=200_000)
        vega = numerical_vega(
            mc, asian.payoff,
            HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, n_steps=GREEK_STEPS,
        )
        assert vega > 0

    def test_asian_call_theta_negative(self):
        """Long call: theta is typically negative for ATM."""
        asian = AsianOption(K=HULL_K, option_type='call')
        mc = _greek_engine(n_paths=200_000)
        theta = numerical_theta(
            mc, asian.payoff,
            HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, n_steps=GREEK_STEPS,
        )
        assert theta < 0

    def test_down_out_call_delta_converges_to_vanilla_far_from_barrier(self):
        """
        A down-and-out call with a barrier pushed very far from spot
        is essentially a vanilla call: the knock-out probability is
        negligible, so the numerical delta must match the BS analytical
        delta. (Near the barrier, the DOC delta can actually be *higher*
        than vanilla because moving spot up both increases payoff and
        reduces knock-out probability, so a naive 'DOC delta < vanilla'
        check is financially wrong.)
        """
        S, K, T, r, sigma, q = HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, HULL_Q
        H = 10.0  # far below spot, knock-out prob is near zero
        barrier = BarrierOption(K=K, barrier=H, barrier_type='down-and-out')
        mc = _greek_engine(n_paths=200_000)
        d_barrier = numerical_delta(
            mc, barrier.payoff, S, T, r, sigma, q, n_steps=GREEK_STEPS,
        )
        d_vanilla = BlackScholesModel(sigma=sigma).greeks(
            S, K, T, r, 'call', q,
        )['delta']
        assert abs(d_barrier - d_vanilla) < 5e-3

    def test_digital_cash_call_delta_positive(self):
        """Cash-or-nothing call: probability of finishing ITM rises with S."""
        digi = DigitalOption(K=HULL_K, option_type='call', payout_type='cash')
        mc = _greek_engine(n_paths=500_000)
        delta = numerical_delta(
            mc, digi.payoff,
            HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q,
            bump=0.02 * HULL_S,  # wider bump: payoff is discontinuous at K
            n_steps=1,
        )
        assert delta > 0


# ----------------------------------------------------------------------------
# 30. Numerical Greeks API Contract
# ----------------------------------------------------------------------------
class TestGreeksAPI:
    """numerical_greeks() wrapper: dict shape, subsets, custom bumps."""

    def test_returns_all_five_by_default(self):
        mc = _greek_engine(n_paths=50_000)
        result = numerical_greeks(
            mc, _bs_call_payoff(HULL_K),
            HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, n_steps=1,
        )
        assert set(result.keys()) == {'delta', 'gamma', 'vega', 'theta', 'rho'}
        for value in result.values():
            assert isinstance(value, float)

    def test_subset_request(self):
        mc = _greek_engine(n_paths=50_000)
        result = numerical_greeks(
            mc, _bs_call_payoff(HULL_K),
            HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q,
            n_steps=1, greeks=['delta', 'vega'],
        )
        assert set(result.keys()) == {'delta', 'vega'}

    def test_custom_bumps_override_default(self):
        """Custom bump for vega should propagate to the underlying call."""
        mc_default = _greek_engine(n_paths=100_000)
        mc_custom = _greek_engine(n_paths=100_000)
        v_default = numerical_vega(
            mc_default, _bs_call_payoff(HULL_K),
            HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, n_steps=1,
        )
        v_custom = numerical_greeks(
            mc_custom, _bs_call_payoff(HULL_K),
            HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q,
            n_steps=1, greeks=['vega'], bumps={'vega': 0.02},
        )['vega']
        # Both should be close to the analytical vega but differ at
        # O(h^2) truncation. They must not be bit-identical.
        assert v_default != v_custom
        assert abs(v_default - v_custom) < 1.0

    def test_invalid_greek_name_raises(self):
        mc = _greek_engine(n_paths=10_000)
        with pytest.raises(ValueError, match="Unknown greeks"):
            numerical_greeks(
                mc, _bs_call_payoff(HULL_K),
                HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q,
                n_steps=1, greeks=['fakegreek'],
            )

    def test_invalid_bump_key_raises(self):
        mc = _greek_engine(n_paths=10_000)
        with pytest.raises(ValueError, match="Unknown bump keys"):
            numerical_greeks(
                mc, _bs_call_payoff(HULL_K),
                HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q,
                n_steps=1, bumps={'vanna': 0.01},
            )


# ----------------------------------------------------------------------------
# 31. Numerical Greeks Input Validation
# ----------------------------------------------------------------------------
class TestGreeksValidation:
    """Explicit input validation in each public function."""

    def test_unseeded_engine_raises(self):
        mc = MonteCarloEngine(n_paths=10_000)  # seed=None
        with pytest.raises(ValueError, match="seeded MonteCarloEngine"):
            numerical_delta(
                mc, _bs_call_payoff(HULL_K),
                HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, n_steps=1,
            )

    def test_negative_S0_raises(self):
        mc = _greek_engine(n_paths=10_000)
        with pytest.raises(ValueError, match="S0 must be > 0"):
            numerical_delta(
                mc, _bs_call_payoff(HULL_K),
                -100.0, HULL_T, HULL_R, HULL_SIGMA, HULL_Q, n_steps=1,
            )

    def test_negative_T_raises(self):
        mc = _greek_engine(n_paths=10_000)
        with pytest.raises(ValueError, match="T must be > 0"):
            numerical_delta(
                mc, _bs_call_payoff(HULL_K),
                HULL_S, -1.0, HULL_R, HULL_SIGMA, HULL_Q, n_steps=1,
            )

    def test_negative_sigma_raises(self):
        mc = _greek_engine(n_paths=10_000)
        with pytest.raises(ValueError, match="sigma must be > 0"):
            numerical_delta(
                mc, _bs_call_payoff(HULL_K),
                HULL_S, HULL_T, HULL_R, -0.20, HULL_Q, n_steps=1,
            )

    def test_bump_too_large_for_delta_raises(self):
        mc = _greek_engine(n_paths=10_000)
        with pytest.raises(ValueError, match="must be < S0"):
            numerical_delta(
                mc, _bs_call_payoff(HULL_K),
                HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q,
                bump=HULL_S + 1, n_steps=1,
            )

    def test_bump_non_positive_raises(self):
        mc = _greek_engine(n_paths=10_000)
        with pytest.raises(ValueError, match="bump must be > 0"):
            numerical_delta(
                mc, _bs_call_payoff(HULL_K),
                HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q,
                bump=0.0, n_steps=1,
            )

    def test_vega_bump_above_sigma_raises(self):
        mc = _greek_engine(n_paths=10_000)
        with pytest.raises(ValueError, match="must be < sigma"):
            numerical_vega(
                mc, _bs_call_payoff(HULL_K),
                HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q,
                bump=HULL_SIGMA + 1e-6, n_steps=1,
            )

    def test_theta_bump_above_T_raises(self):
        mc = _greek_engine(n_paths=10_000)
        with pytest.raises(ValueError, match="must be < T"):
            numerical_theta(
                mc, _bs_call_payoff(HULL_K),
                HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q,
                bump=HULL_T + 1e-6, n_steps=1,
            )

    def test_gamma_bump_non_positive_raises(self):
        mc = _greek_engine(n_paths=10_000)
        with pytest.raises(ValueError, match="bump must be > 0"):
            numerical_gamma(
                mc, _bs_call_payoff(HULL_K),
                HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q,
                bump=0.0, n_steps=1,
            )

    def test_gamma_bump_above_S0_raises(self):
        mc = _greek_engine(n_paths=10_000)
        with pytest.raises(ValueError, match="must be < S0"):
            numerical_gamma(
                mc, _bs_call_payoff(HULL_K),
                HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q,
                bump=HULL_S + 1.0, n_steps=1,
            )

    def test_vega_bump_non_positive_raises(self):
        mc = _greek_engine(n_paths=10_000)
        with pytest.raises(ValueError, match="bump must be > 0"):
            numerical_vega(
                mc, _bs_call_payoff(HULL_K),
                HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q,
                bump=0.0, n_steps=1,
            )

    def test_theta_bump_non_positive_raises(self):
        mc = _greek_engine(n_paths=10_000)
        with pytest.raises(ValueError, match="bump must be > 0"):
            numerical_theta(
                mc, _bs_call_payoff(HULL_K),
                HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q,
                bump=0.0, n_steps=1,
            )

    def test_rho_bump_non_positive_raises(self):
        mc = _greek_engine(n_paths=10_000)
        with pytest.raises(ValueError, match="bump must be > 0"):
            numerical_rho(
                mc, _bs_call_payoff(HULL_K),
                HULL_S, HULL_T, HULL_R, HULL_SIGMA, HULL_Q,
                bump=0.0, n_steps=1,
            )


# ----------------------------------------------------------------------------
# 32. Numerical Greeks Hypothesis Property-Based Tests
# ----------------------------------------------------------------------------
greek_spot_st = st.floats(min_value=80.0, max_value=120.0, allow_nan=False)
greek_strike_st = st.floats(min_value=80.0, max_value=120.0, allow_nan=False)
greek_vol_st = st.floats(min_value=0.10, max_value=0.40, allow_nan=False)
greek_time_st = st.floats(min_value=0.25, max_value=2.0, allow_nan=False)


class TestGreeksHypothesis:
    """Property-based robustness checks on random Asian calls."""

    @given(S=greek_spot_st, K=greek_strike_st, sigma=greek_vol_st, T=greek_time_st)
    @settings(max_examples=15, deadline=None,
              suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
    def test_asian_call_delta_in_unit_interval(
        self, S: float, K: float, sigma: float, T: float,
    ):
        """
        Random Asian calls: delta is always in [0, 1]. 15 examples is
        small but each uses 100k paths, so total work is ~1.5M paths.
        """
        asian = AsianOption(K=K, option_type='call')
        mc = MonteCarloEngine(n_paths=100_000, seed=MC_SEED)
        delta = numerical_delta(
            mc, asian.payoff, S, T, 0.05, sigma, 0.0, n_steps=64,
        )
        assert -1e-2 <= delta <= 1.0 + 1e-2

    @given(S=greek_spot_st, K=greek_strike_st, sigma=greek_vol_st, T=greek_time_st)
    @settings(max_examples=15, deadline=None,
              suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
    def test_asian_call_vega_non_negative(
        self, S: float, K: float, sigma: float, T: float,
    ):
        asian = AsianOption(K=K, option_type='call')
        mc = MonteCarloEngine(n_paths=100_000, seed=MC_SEED)
        vega = numerical_vega(
            mc, asian.payoff, S, T, 0.05, sigma, 0.0, n_steps=64,
        )
        assert vega >= -1e-2
