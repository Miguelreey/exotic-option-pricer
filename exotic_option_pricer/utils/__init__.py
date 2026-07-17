"""
Exotic Option Pricer — exotic_option_pricer.utils

Visualization, Greeks and analysis utilities.
"""

from .greeks import (
    numerical_delta,
    numerical_gamma,
    numerical_greeks,
    numerical_rho,
    numerical_theta,
    numerical_vega,
)
from .visualization import (
    plot_all_greeks,
    plot_bs_price_and_intrinsic,
    plot_exotic_comparison,
    plot_exotic_payoff,
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

__all__ = [
    'numerical_delta',
    'numerical_gamma',
    'numerical_greeks',
    'numerical_rho',
    'numerical_theta',
    'numerical_vega',
    'plot_bs_price_and_intrinsic',
    'plot_all_greeks',
    'plot_greek_vs_time',
    'plot_payoff_diagram',
    'plot_portfolio_greeks',
    'plot_pnl_attribution',
    'plot_gamma_theta_tradeoff',
    'plot_vol_sensitivity',
    'plot_implied_vol_smile',
    'plot_greeks_heatmap',
    'plot_exotic_payoff',
    'plot_exotic_comparison',
    'save_figure',
]
