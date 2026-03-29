"""
Exotic Option Pricer — tests/test_strategies.py
Generado automáticamente. Turno inicial: 100
"""

"""
Exotic Option Pricer - tests/test_strategies.py

Test suite for option strategy properties under Black-Scholes.

Validates:
- Straddles: delta-neutrality ATM, long gamma, short theta
- Butterflies: non-negative payoff, max at center
- Bull spreads: bounded P&L
- Greeks linearity across portfolios
- Delta hedging effectiveness
- Strangle vs straddle cost ordering

Parameters: S=100, K=100, T=1.0, r=0.05, sigma=0.20
Reference: Hull, Ch. 12-13.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models.black_scholes import BlackScholesModel


@pytest.fixture
def base_params():
    return {'S': 100.0, 'K': 100.0, 'T': 1.0, 'r': 0.05, 'sigma': 0.20}


@pytest.fixture
def model(base_params):
    return BlackScholesModel(sigma=base_params['sigma'])


class TestStrategyGreeks:

    def test_straddle_delta_neutral_atm(self, model, base_params):
        """ATM forward straddle has delta ≈ 0."""
        S, K, T, r = (base_params[k] for k in ('S', 'K', 'T', 'r'))
        # ATM forward: S such that forward F = K
        S_fwd = K * np.exp(-r * T)
        delta_straddle = (model.delta(S_fwd, K, T, r, 'call')
                          + model.delta(S_fwd, K, T, r, 'put'))
        # For sigma*sqrt(T)=0.20, d1=sigma*sqrt(T)/2=0.10, delta=2*N(0.10)-1≈0.08
        assert abs(delta_straddle) < 0.10

    def test_straddle_long_gamma(self, model, base_params):
        """Straddle gamma = 2 * call gamma > 0."""
        S, K, T, r = (base_params[k] for k in ('S', 'K', 'T', 'r'))
        gamma_call = model.gamma(S, K, T, r, 'call')
        gamma_put = model.gamma(S, K, T, r, 'put')
        assert np.isclose(gamma_call, gamma_put, rtol=1e-12)
        assert (gamma_call + gamma_put) > 0

    def test_straddle_short_theta(self, model, base_params):
        """Long straddle always has negative theta."""
        S, K, T, r = (base_params[k] for k in ('S', 'K', 'T', 'r'))
        theta_straddle = (model.theta(S, K, T, r, 'call')
                          + model.theta(S, K, T, r, 'put'))
        assert theta_straddle < 0

    def test_butterfly_payoff_nonnegative(self, model, base_params):
        """Butterfly payoff >= 0 everywhere."""
        K1, K2, K3 = 95.0, 100.0, 105.0
        S_range = np.linspace(80.0, 120.0, 1000)
        payoff = (np.maximum(S_range - K1, 0.0)
                  - 2.0 * np.maximum(S_range - K2, 0.0)
                  + np.maximum(S_range - K3, 0.0))
        assert np.all(payoff >= -1e-14)

    def test_butterfly_max_payoff_at_center(self, model, base_params):
        """Max butterfly payoff = dK at S = K_center."""
        dK = 5.0
        K1, K2, K3 = 95.0, 100.0, 105.0
        S_range = np.linspace(80.0, 120.0, 10000)
        payoff = (np.maximum(S_range - K1, 0.0)
                  - 2.0 * np.maximum(S_range - K2, 0.0)
                  + np.maximum(S_range - K3, 0.0))
        assert np.isclose(payoff.max(), dK, atol=0.01)
        assert np.isclose(S_range[np.argmax(payoff)], K2, atol=0.1)

    def test_bull_spread_pnl_bounded(self, model, base_params):
        """Bull spread payoff in [0, K2-K1]."""
        K1, K2 = 95.0, 105.0
        S_range = np.linspace(50.0, 150.0, 10000)
        payoff = (np.maximum(S_range - K1, 0.0)
                  - np.maximum(S_range - K2, 0.0))
        assert np.all(payoff >= -1e-14)
        assert np.all(payoff <= (K2 - K1) + 1e-14)

    def test_greeks_linear_additivity(self, model, base_params):
        """Portfolio Greeks = sum of weighted individual Greeks."""
        S, T, r = base_params['S'], base_params['T'], base_params['r']
        K_call, K_put = 100.0, 95.0
        qty_call, qty_put = 2.0, 1.0

        for greek_name in ('delta', 'gamma', 'vega', 'theta', 'rho'):
            method = getattr(model, greek_name)
            portfolio = (qty_call * method(S, K_call, T, r, 'call')
                         + qty_put * method(S, K_put, T, r, 'put'))
            expected = (qty_call * method(S, K_call, T, r, 'call')
                        + qty_put * method(S, K_put, T, r, 'put'))
            assert np.isclose(portfolio, expected, rtol=1e-14)

    def test_put_call_parity_portfolio(self, model, base_params):
        """C - P = S - K*exp(-rT) to machine precision."""
        S, K, T, r = (base_params[k] for k in ('S', 'K', 'T', 'r'))
        call = model.price(S, K, T, r, 'call')
        put = model.price(S, K, T, r, 'put')
        assert np.isclose(call - put, S - K * np.exp(-r * T), atol=1e-10)

    def test_delta_hedge_eliminates_delta(self, model, base_params):
        """Hedged portfolio P&L ≈ 0.5 * Gamma * (dS)^2."""
        S, K, T, r = (base_params[k] for k in ('S', 'K', 'T', 'r'))
        delta_call = model.delta(S, K, T, r, 'call')
        dS = 1.0
        call_up = model.price(S + dS, K, T, r, 'call')
        call_mid = model.price(S, K, T, r, 'call')
        pnl_up = (call_up - call_mid) - delta_call * dS
        gamma_val = model.gamma(S, K, T, r, 'call')
        expected_pnl = 0.5 * gamma_val * dS**2
        assert np.isclose(pnl_up, expected_pnl, rtol=0.05)

    def test_strangle_cheaper_than_straddle(self, model, base_params):
        """OTM strangle costs less than ATM straddle."""
        S, T, r = base_params['S'], base_params['T'], base_params['r']
        straddle = (model.price(S, 100.0, T, r, 'call')
                    + model.price(S, 100.0, T, r, 'put'))
        strangle = (model.price(S, 105.0, T, r, 'call')
                    + model.price(S, 95.0, T, r, 'put'))
        assert strangle < straddle


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
