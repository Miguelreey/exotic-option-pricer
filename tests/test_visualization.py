"""
Exotic Option Pricer — tests/test_visualization.py

Smoke tests for the visualization module.
Verifies all functions run without errors and return correct types.
"""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib

matplotlib.use('Agg')
import matplotlib.figure
import matplotlib.pyplot as plt

from src.models.black_scholes import BlackScholesModel
from src.utils.visualization import (
    plot_all_greeks,
    plot_bs_price_and_intrinsic,
    plot_gamma_theta_tradeoff,
    plot_greek_vs_time,
    plot_greeks_heatmap,
    plot_implied_vol_smile,
    plot_payoff_diagram,
    plot_pnl_attribution,
    plot_portfolio_greeks,
    plot_vol_sensitivity,
    save_figure,
)


@pytest.fixture
def model():
    return BlackScholesModel(sigma=0.20)


@pytest.fixture
def S_range():
    return np.linspace(60, 140, 100)


@pytest.fixture
def T_range():
    return np.linspace(0.05, 2.0, 50)


class TestVisualizationOutputs:

    def test_plot_bs_price_and_intrinsic(self, model, S_range):
        fig, ax = plot_bs_price_and_intrinsic(model, S_range, K=100, T=1.0, r=0.05)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close(fig)

    def test_plot_bs_price_put(self, model, S_range):
        fig, ax = plot_bs_price_and_intrinsic(
            model, S_range, K=100, T=1.0, r=0.05, option_type='put')
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_all_greeks(self, model, S_range):
        fig, axes = plot_all_greeks(model, S_range, K=100, T=1.0, r=0.05)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert axes.shape == (2, 3)
        plt.close(fig)

    def test_plot_greek_vs_time(self, model, T_range):
        for greek_name in ('price', 'delta', 'gamma', 'vega', 'theta', 'rho'):
            fig, ax = plot_greek_vs_time(model, greek_name, S=100, K=100,
                                          r=0.05, T_range=T_range)
            assert isinstance(fig, matplotlib.figure.Figure)
            plt.close(fig)

    def test_plot_greek_vs_time_invalid(self, model, T_range):
        with pytest.raises(ValueError):
            plot_greek_vs_time(model, 'invalid', S=100, K=100, r=0.05, T_range=T_range)

    def test_plot_payoff_diagram(self, S_range):
        legs = [
            {'type': 'call', 'K': 100, 'qty': 1, 'premium': 10.45},
            {'type': 'put', 'K': 100, 'qty': 1, 'premium': 5.57},
        ]
        fig, ax = plot_payoff_diagram(legs, S_range)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_portfolio_greeks(self, model, S_range):
        legs = [
            {'type': 'call', 'K': 100, 'qty': 1},
            {'type': 'put', 'K': 100, 'qty': 1},
        ]
        fig, axes = plot_portfolio_greeks(legs, S_range, model, T=1.0, r=0.05)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert axes.shape == (2, 2)
        plt.close(fig)

    def test_plot_pnl_attribution(self):
        pnl = {'delta': 1.50, 'gamma': 0.30, 'theta': -0.80, 'vega': -0.20}
        fig, ax = plot_pnl_attribution(pnl)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_gamma_theta_tradeoff(self, model):
        fig, ax = plot_gamma_theta_tradeoff(
            model, S=100, r=0.05,
            K_values=np.linspace(80, 120, 5),
            T_values=np.linspace(0.1, 2.0, 5))
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_vol_sensitivity(self):
        fig, ax = plot_vol_sensitivity(
            BlackScholesModel, S=100, K=100, T=1.0, r=0.05,
            sigma_range=np.linspace(0.05, 0.80, 50))
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_vol_sensitivity_put(self):
        fig, ax = plot_vol_sensitivity(
            BlackScholesModel, S=100, K=100, T=1.0, r=0.05,
            sigma_range=np.linspace(0.05, 0.80, 50), option_type='put')
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


class TestSaveFigure:

    def test_save_creates_file(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'subdir', 'test.png')
            save_figure(fig, filepath, dpi=72)
            assert os.path.exists(filepath)

    def test_save_closes_figure(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test.png')
            fig_num = fig.number
            save_figure(fig, filepath)
            assert fig_num not in plt.get_fignums()

    def test_no_leftover_figures(self):
        plt.close('all')
        assert len(plt.get_fignums()) == 0


class TestNewVisualizations:

    def test_plot_implied_vol_smile(self, model):
        """Generate synthetic market prices and plot smile."""
        strikes = [85, 90, 95, 100, 105, 110, 115]
        market_prices = [
            {'K': k, 'price': model.price(100, k, 1.0, 0.05, 'call')}
            for k in strikes
        ]
        fig, ax = plot_implied_vol_smile(market_prices, S=100, T=1.0, r=0.05)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_greeks_heatmap_delta(self, model):
        S_range = np.linspace(70, 130, 30)
        T_range = np.linspace(0.05, 2.0, 20)
        fig, ax = plot_greeks_heatmap(model, 'delta', S_range, T_range,
                                       K=100, r=0.05)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_greeks_heatmap_gamma(self, model):
        S_range = np.linspace(70, 130, 30)
        T_range = np.linspace(0.05, 2.0, 20)
        fig, ax = plot_greeks_heatmap(model, 'gamma', S_range, T_range,
                                       K=100, r=0.05)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_greeks_heatmap_vanna(self, model):
        S_range = np.linspace(70, 130, 30)
        T_range = np.linspace(0.05, 2.0, 20)
        fig, ax = plot_greeks_heatmap(model, 'vanna', S_range, T_range,
                                       K=100, r=0.05)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
