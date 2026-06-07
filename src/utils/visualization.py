"""
Exotic Option Pricer — src/utils/visualization.py

Professional visualization module for the Black-Scholes model.

Generates:
- Price and intrinsic value plots
- Individual and portfolio Greeks panels
- Strategy payoff/P&L diagrams
- Sensitivity analysis (vol, time)
- P&L attribution by Greek component
- Gamma-theta tradeoff scatter

Each function returns (fig, ax) or (fig, axes) for customization.

Color conventions (consistent throughout):
    calls   = #2196F3 (blue)
    puts    = #F44336 (red)
    delta   = #4CAF50 (green)
    gamma   = #FF9800 (orange)
    theta   = #9C27B0 (purple)
    vega    = #00BCD4 (cyan)
    rho     = #795548 (brown)
"""

from __future__ import annotations

import matplotlib

matplotlib.use('Agg')  # Non-interactive backend: safe for servers and CI/CD

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from src.models.black_scholes import BlackScholesModel

if TYPE_CHECKING:
    from src.engines.monte_carlo import MCResult
    from src.instruments.base import ExoticOption

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        pass

# ──────────────────────────────────────────────
# Color palette
# ──────────────────────────────────────────────
COLORS = {
    'call':      '#2196F3',
    'put':       '#F44336',
    'delta':     '#4CAF50',
    'gamma':     '#FF9800',
    'theta':     '#9C27B0',
    'vega':      '#00BCD4',
    'rho':       '#795548',
    'price':     '#2196F3',
    'intrinsic': '#9E9E9E',
    'payoff':    '#FF5722',
    'pnl_pos':   '#4CAF50',
    'pnl_neg':   '#F44336',
    'atm_line':  '#607D8B',
}

Numeric = Union[float, np.ndarray]


def save_figure(fig: plt.Figure, filepath: str, dpi: int = 150) -> None:
    """
    Save a matplotlib figure and close it to free memory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    filepath : str
        Output path (e.g., 'plots/greeks.png').
        Intermediate directories are created automatically.
    dpi : int, default 150
        Resolution in dots per inch.
    """
    try:
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    finally:
        plt.close(fig)


def plot_bs_price_and_intrinsic(model, S_range: np.ndarray, K: float,
                                 T: float, r: float,
                                 option_type: str = 'call'
                                 ) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot BS price, intrinsic value and time value shading.

    Parameters
    ----------
    model : BlackScholesModel
        Model instance with sigma configured.
    S_range : np.ndarray
        Spot prices to evaluate.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Risk-free rate.
    option_type : str, default 'call'
        'call' or 'put'.

    Returns
    -------
    tuple[Figure, Axes]
    """
    S_range = np.asarray(S_range, dtype=float)
    option_type_lower = option_type.lower().strip()

    bs_price = model.price(S_range, K, T, r, option_type=option_type_lower)

    if option_type_lower == 'call':
        intrinsic = np.maximum(S_range - K, 0.0)
    else:
        intrinsic = np.maximum(K - S_range, 0.0)

    color_main = COLORS['call'] if option_type_lower == 'call' else COLORS['put']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(S_range, bs_price, color=color_main, linewidth=2.2,
            label='BS Price', zorder=3)
    ax.plot(S_range, intrinsic, color=COLORS['intrinsic'], linewidth=1.5,
            linestyle='--', label='Intrinsic Value', zorder=2)
    ax.fill_between(S_range, intrinsic, bs_price, alpha=0.12,
                    color=color_main, label='Time Value')
    ax.axvline(x=K, color=COLORS['atm_line'], linestyle='-.',
               linewidth=0.9, alpha=0.6, label=f'ATM (K={K})')
    ax.set_xlabel('Spot Price ($S$)', fontsize=12)
    ax.set_ylabel('Option Value', fontsize=12)
    ax.set_title(
        f'Black-Scholes {option_type_lower.capitalize()} Price '
        f'($K={K}$, $T={T}$y, $r={r*100:.1f}\\%$, $\\sigma={model.sigma*100:.0f}\\%$)',
        fontsize=13
    )
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_xlim(S_range[0], S_range[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_all_greeks(model, S_range: np.ndarray, K: float,
                    T: float, r: float
                    ) -> Tuple[plt.Figure, np.ndarray]:
    """
    2x3 panel with price and five Greeks vs spot for call and put.

    Parameters
    ----------
    model : BlackScholesModel
    S_range : np.ndarray
    K, T, r : float

    Returns
    -------
    tuple[Figure, ndarray of Axes (2x3)]
    """
    S_range = np.asarray(S_range, dtype=float)

    greeks_data = {}
    for otype in ('call', 'put'):
        greeks_data[otype] = {
            'price': model.price(S_range, K, T, r, option_type=otype),
            'delta': model.delta(S_range, K, T, r, option_type=otype),
            'gamma': model.gamma(S_range, K, T, r, option_type=otype),
            'vega':  model.vega(S_range, K, T, r, option_type=otype),
            'theta': model.theta(S_range, K, T, r, option_type=otype),
            'rho':   model.rho(S_range, K, T, r, option_type=otype),
        }

    panel_config = [
        ('price', r'Price: $C = S\,\Phi(d_1) - Ke^{-rT}\Phi(d_2)$'),
        ('delta', r'Delta: $\Delta_C = \Phi(d_1)$'),
        ('gamma', r'Gamma: $\Gamma = \frac{\phi(d_1)}{S\sigma\sqrt{T}}$'),
        ('vega',  r'Vega: $\mathcal{V} = S\sqrt{T}\,\phi(d_1)$'),
        ('theta', r'Theta: $\Theta$'),
        ('rho',   r'Rho: $\rho$'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for idx, (greek_name, title_str) in enumerate(panel_config):
        ax = axes_flat[idx]
        ax.plot(S_range, greeks_data['call'][greek_name], color=COLORS['call'],
                linewidth=1.8, label='Call')
        ax.plot(S_range, greeks_data['put'][greek_name], color=COLORS['put'],
                linewidth=1.8, label='Put', linestyle='--')
        ax.axvline(x=K, color=COLORS['atm_line'], linestyle='-.',
                   linewidth=0.8, alpha=0.5)
        if greek_name in ('delta', 'theta', 'rho'):
            ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        ax.set_title(title_str, fontsize=10)
        ax.set_xlabel('$S$', fontsize=10)
        ax.legend(fontsize=9, loc='best', framealpha=0.8)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        f'Black-Scholes Greeks ($K={K}$, $T={T}$y, '
        f'$r={r*100:.1f}\\%$, $\\sigma={model.sigma*100:.0f}\\%$)',
        fontsize=14, y=1.01
    )
    fig.tight_layout()
    return fig, axes


def plot_greek_vs_time(model, greek_name: str, S: float, K: float,
                       r: float, T_range: np.ndarray
                       ) -> Tuple[plt.Figure, plt.Axes]:
    """
    Greek as function of time to expiry.

    Parameters
    ----------
    model : BlackScholesModel
    greek_name : str
        'price', 'delta', 'gamma', 'vega', 'theta', or 'rho'.
    S, K, r : float
    T_range : np.ndarray
        Array of times to expiry (T > 0).

    Returns
    -------
    tuple[Figure, Axes]
    """
    T_range = np.asarray(T_range, dtype=float)
    greek_methods = {
        'price': model.price, 'delta': model.delta, 'gamma': model.gamma,
        'vega': model.vega, 'theta': model.theta, 'rho': model.rho,
    }

    greek_name_lower = greek_name.lower().strip()
    if greek_name_lower not in greek_methods:
        raise ValueError(
            f"greek_name must be one of {list(greek_methods.keys())}. "
            f"Got: '{greek_name}'"
        )

    method = greek_methods[greek_name_lower]
    fig, ax = plt.subplots(figsize=(10, 6))

    if greek_name_lower in ('gamma', 'vega'):
        vals = method(S, K, T_range, r, option_type='call')
        color = COLORS.get(greek_name_lower, COLORS['price'])
        ax.plot(T_range, vals, color=color, linewidth=2.0,
                label=f'{greek_name_lower.capitalize()} (call = put)')
    else:
        call_vals = method(S, K, T_range, r, option_type='call')
        put_vals = method(S, K, T_range, r, option_type='put')
        ax.plot(T_range, call_vals, color=COLORS['call'], linewidth=2.0,
                label=f'{greek_name_lower.capitalize()} (Call)')
        ax.plot(T_range, put_vals, color=COLORS['put'], linewidth=2.0,
                linestyle='--', label=f'{greek_name_lower.capitalize()} (Put)')

    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Time to Expiry $T$ (years)', fontsize=12)
    ax.set_ylabel(greek_name_lower.capitalize(), fontsize=12)
    moneyness = 'ATM' if np.isclose(S, K) else ('ITM' if S > K else 'OTM')
    ax.set_title(
        f'{greek_name_lower.capitalize()} vs Time ({moneyness}, '
        f'$\\sigma={model.sigma*100:.0f}\\%$)', fontsize=13
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_payoff_diagram(legs: List[Dict], S_range: np.ndarray
                        ) -> Tuple[plt.Figure, plt.Axes]:
    """
    Payoff and P&L diagram for a multi-leg option portfolio.

    Parameters
    ----------
    legs : list of dict
        Each dict: {'type': 'call'|'put', 'K': float, 'qty': float, 'premium': float}
    S_range : np.ndarray
        Terminal spot range.

    Returns
    -------
    tuple[Figure, Axes]
    """
    S_range = np.asarray(S_range, dtype=float)
    total_payoff = np.zeros_like(S_range)
    total_premium = 0.0

    fig, ax = plt.subplots(figsize=(10, 6))

    for leg in legs:
        K, qty, premium = leg['K'], leg['qty'], leg['premium']
        otype = leg['type'].lower().strip()
        if otype == 'call':
            payoff = qty * np.maximum(S_range - K, 0.0)
        else:
            payoff = qty * np.maximum(K - S_range, 0.0)
        total_payoff += payoff
        total_premium += qty * premium
        ax.axvline(x=K, color=COLORS['atm_line'], linestyle=':', linewidth=0.7, alpha=0.5)

    pnl = total_payoff - total_premium

    ax.plot(S_range, total_payoff, color=COLORS['call'], linewidth=2.0,
            label='Payoff at Expiry', zorder=3)
    ax.plot(S_range, pnl, color=COLORS['pnl_pos'], linewidth=2.0,
            linestyle='--', label='Net P&L', zorder=3)
    ax.fill_between(S_range, pnl, 0, where=(pnl >= 0), alpha=0.15,  # type: ignore[arg-type]
                    color=COLORS['pnl_pos'], interpolate=True)
    ax.fill_between(S_range, pnl, 0, where=(pnl < 0), alpha=0.15,  # type: ignore[arg-type]
                    color=COLORS['pnl_neg'], interpolate=True)
    ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.4)
    ax.set_xlabel('Spot at Expiry ($S_T$)', fontsize=12)
    ax.set_ylabel('Value ($)', fontsize=12)
    ax.set_title('Payoff Diagram', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_portfolio_greeks(legs: List[Dict], S_range: np.ndarray,
                          model, T: float, r: float
                          ) -> Tuple[plt.Figure, np.ndarray]:
    """
    2x2 panel of aggregate portfolio Greeks vs spot.

    Parameters
    ----------
    legs : list of dict
        Each dict: {'type': 'call'|'put', 'K': float, 'qty': float}
    S_range, model, T, r : as before.

    Returns
    -------
    tuple[Figure, ndarray of Axes (2x2)]
    """
    S_range = np.asarray(S_range, dtype=float)
    greek_names = ['delta', 'gamma', 'theta', 'vega']
    greek_methods = {
        'delta': model.delta, 'gamma': model.gamma,
        'theta': model.theta, 'vega': model.vega,
    }

    portfolio_greeks = {g: np.zeros_like(S_range) for g in greek_names}
    for leg in legs:
        K, qty, otype = leg['K'], leg['qty'], leg['type'].lower().strip()
        for g_name in greek_names:
            portfolio_greeks[g_name] += qty * greek_methods[g_name](
                S_range, K, T, r, option_type=otype
            )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for idx, g_name in enumerate(greek_names):
        ax = axes.flatten()[idx]
        color = COLORS.get(g_name, COLORS['price'])
        ax.plot(S_range, portfolio_greeks[g_name], color=color, linewidth=2.0,
                label=f'Portfolio {g_name.capitalize()}')
        ax.axhline(y=0, color='black', linewidth=0.6, alpha=0.4)
        ax.set_title(g_name.capitalize(), fontsize=12)
        ax.set_xlabel('$S$', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    fig.suptitle(f'Portfolio Greeks ($T={T}$y, $\\sigma={model.sigma*100:.0f}\\%$)',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    return fig, axes


def plot_pnl_attribution(pnl_components: Dict[str, float],
                          labels: Optional[List[str]] = None
                          ) -> Tuple[plt.Figure, plt.Axes]:
    """
    Horizontal bar chart of P&L attribution by Greek component.

    Parameters
    ----------
    pnl_components : dict of {str: float}
        E.g., {'delta': 1.50, 'gamma': 0.30, 'theta': -0.80, 'vega': -0.20}
    labels : list of str, optional

    Returns
    -------
    tuple[Figure, Axes]
    """
    if labels is None:
        labels = list(pnl_components.keys())
    values = list(pnl_components.values())
    bar_colors = [COLORS['pnl_pos'] if v >= 0 else COLORS['pnl_neg'] for v in values]

    fig, ax = plt.subplots(figsize=(10, max(4, len(values) * 0.8 + 1)))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=bar_colors, edgecolor='white', height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([label.capitalize() for label in labels], fontsize=11)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('P&L Contribution ($)', fontsize=12)
    ax.set_title('P&L Attribution by Greek Component', fontsize=13)

    total_pnl = sum(values)
    ax.text(0.98, 0.02, f'Total P&L: ${total_pnl:+.4f}',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_gamma_theta_tradeoff(model, S: float, r: float,
                               K_values: np.ndarray,
                               T_values: np.ndarray
                               ) -> Tuple[plt.Figure, plt.Axes]:
    """
    Scatter of gamma vs theta for (K, T) combinations, colored by T.

    Parameters
    ----------
    model : BlackScholesModel
    S, r : float
    K_values, T_values : np.ndarray

    Returns
    -------
    tuple[Figure, Axes]
    """
    K_grid, T_grid = np.meshgrid(K_values, T_values, indexing='ij')
    K_flat, T_flat = K_grid.flatten(), T_grid.flatten()

    gammas = model.gamma(S, K_flat, T_flat, r, option_type='call')
    thetas = model.theta(S, K_flat, T_flat, r, option_type='call')

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(gammas, thetas, c=T_flat, cmap='viridis',
                         s=60, edgecolors='white', linewidth=0.5, alpha=0.85)
    fig.colorbar(scatter, ax=ax, label='Time to Expiry $T$ (years)')
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='black', linewidth=0.5, alpha=0.3)
    ax.set_xlabel(r'Gamma ($\Gamma$)', fontsize=12)
    ax.set_ylabel(r'Theta ($\Theta$)', fontsize=12)
    ax.set_title(f'Gamma-Theta Tradeoff ($S={S}$, $\\sigma={model.sigma*100:.0f}\\%$)',
                 fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_vol_sensitivity(model_class, S: float, K: float,
                         T: float, r: float,
                         sigma_range: np.ndarray,
                         option_type: str = 'call'
                         ) -> Tuple[plt.Figure, plt.Axes]:
    """
    Option price vs implied volatility.

    Parameters
    ----------
    model_class : type
        BlackScholesModel class (instantiated per sigma).
    S, K, T, r : float
    sigma_range : np.ndarray
        Volatilities to evaluate.
    option_type : str

    Returns
    -------
    tuple[Figure, Axes]
    """
    sigma_range = np.asarray(sigma_range, dtype=float)
    option_type_lower = option_type.lower().strip()

    prices = np.array([
        model_class(sigma=sig).price(S, K, T, r, option_type=option_type_lower)
        for sig in sigma_range
    ])

    fig, ax = plt.subplots(figsize=(10, 6))
    color_main = COLORS['call'] if option_type_lower == 'call' else COLORS['put']
    ax.plot(sigma_range * 100, prices, color=color_main, linewidth=2.2,
            label=f'{option_type_lower.capitalize()} Price')

    if option_type_lower == 'call':
        intrinsic = max(S - K * np.exp(-r * T), 0.0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S, 0.0)
    ax.axhline(y=intrinsic, color=COLORS['intrinsic'], linestyle=':',
               linewidth=1.0, alpha=0.6, label=f'Intrinsic = {intrinsic:.2f}')

    ax.set_xlabel('Implied Volatility (%)', fontsize=12)
    ax.set_ylabel('Option Price ($)', fontsize=12)
    ax.set_title(f'Price vs Vol ($S={S}$, $K={K}$, $T={T}$y)', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_implied_vol_smile(market_prices: List[Dict], S: float,
                            T: float, r: float,
                            option_type: str = 'call',
                            q: float = 0.0
                            ) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot implied volatility smile from market prices.

    Parameters
    ----------
    market_prices : list of dict
        Each dict: {'K': float, 'price': float}
    S, T, r : float
    option_type : str
    q : float

    Returns
    -------
    tuple[Figure, Axes]
    """
    strikes = np.array([mp['K'] for mp in market_prices])
    prices = np.array([mp['price'] for mp in market_prices])
    ivs = np.array([
        BlackScholesModel.implied_vol(p, S, k, T, r, option_type, q=q)
        for p, k in zip(prices, strikes)
    ])

    moneyness = strikes / S

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(moneyness, ivs * 100, color=COLORS['call'], linewidth=2.2,
            marker='o', markersize=5, label='Implied Vol')
    ax.axvline(x=1.0, color=COLORS['atm_line'], linestyle='-.',
               linewidth=0.9, alpha=0.6, label='ATM')
    ax.set_xlabel('Moneyness ($K/S$)', fontsize=12)
    ax.set_ylabel('Implied Volatility (%)', fontsize=12)
    ax.set_title(f'Volatility Smile ($S={S}$, $T={T}$y)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_greeks_heatmap(model, greek_name: str,
                         S_range: np.ndarray,
                         T_range: np.ndarray,
                         K: float, r: float,
                         option_type: str = 'call',
                         q: float = 0.0
                         ) -> Tuple[plt.Figure, plt.Axes]:
    """
    2D heatmap of a Greek over (Spot, Time-to-expiry) grid.

    Parameters
    ----------
    model : BlackScholesModel
    greek_name : str
        Any Greek method name ('delta', 'gamma', 'vanna', etc.).
    S_range, T_range : np.ndarray
    K, r : float
    option_type : str
    q : float

    Returns
    -------
    tuple[Figure, Axes]
    """
    greek_method = getattr(model, greek_name)
    S_grid, T_grid = np.meshgrid(S_range, T_range, indexing='ij')

    Z = greek_method(S_grid, K, T_grid, r, option_type, q=q)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.pcolormesh(T_range, S_range, Z, cmap='RdBu_r', shading='auto')
    fig.colorbar(im, ax=ax, label=greek_name.capitalize())
    ax.set_xlabel('Time to Expiry $T$ (years)', fontsize=12)
    ax.set_ylabel('Spot Price $S$', fontsize=12)
    ax.set_title(
        f'{greek_name.capitalize()} Heatmap ($K={K}$, $\\sigma={model.sigma*100:.0f}\\%$)',
        fontsize=13
    )
    ax.axhline(y=K, color='white', linestyle='--', linewidth=1.0, alpha=0.7,
               label=f'ATM ($K={K}$)')
    ax.legend(fontsize=9, loc='upper right')
    fig.tight_layout()
    return fig, ax


# ──────────────────────────────────────────────
# Monte Carlo visualizations (Phase 2)
# ──────────────────────────────────────────────

COLORS_MC = {
    'path':         '#90CAF9',
    'mean':         '#1565C0',
    'ci_fill':      '#1565C0',
    'reference':    '#F44336',
    'plain':        '#9E9E9E',
    'antithetic':   '#4CAF50',
    'control':      '#FF9800',
    'combined':     '#9C27B0',
    'slope_ref':    '#F44336',
}


def plot_mc_paths(
    paths: np.ndarray,
    T: float,
    n_show: int = 50,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot simulated GBM paths with mean and confidence band.

    Parameters
    ----------
    paths : np.ndarray, shape (n_paths, n_steps + 1)
        Simulated price paths from MonteCarloEngine.simulate_gbm().
    T : float
        Time horizon in years (for x-axis scaling).
    n_show : int, default 50
        Number of individual paths to display. Capped at n_paths.
    title : str, optional
        Custom plot title. If None, a default is generated.

    Returns
    -------
    tuple[Figure, Axes]
    """
    n_paths, n_points = paths.shape
    t_grid = np.linspace(0, T, n_points)
    n_show = min(n_show, n_paths)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Individual paths (subsample for visual clarity)
    indices = np.linspace(0, n_paths - 1, n_show, dtype=int)
    for i in indices:
        ax.plot(t_grid, paths[i], color=COLORS_MC['path'], linewidth=0.4, alpha=0.5)

    # Mean path
    mean_path = np.mean(paths, axis=0)
    ax.plot(t_grid, mean_path, color=COLORS_MC['mean'], linewidth=2.2,
            label='Mean path', zorder=5)

    # 95% pointwise distribution band: mean +/- 1.96 * std(S_t)
    # This shows the spread of individual paths, NOT the CI of the mean.
    # For the CI of the mean, divide std by sqrt(n_paths).
    std_path = np.std(paths, axis=0)
    lower = mean_path - 1.96 * std_path
    upper = mean_path + 1.96 * std_path
    ax.fill_between(t_grid, lower, upper, color=COLORS_MC['ci_fill'],
                    alpha=0.12, label=r'95% distribution band')

    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_ylabel('Price ($S$)', fontsize=12)
    ax.set_title(
        title or f'Monte Carlo GBM Paths ($N={n_paths:,}$, $S_0={paths[0, 0]:.0f}$)',
        fontsize=13
    )
    ax.legend(fontsize=10, loc='upper left')
    ax.set_xlim(0, T)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_mc_convergence(
    path_counts: List[int],
    std_errors: List[float],
    errors: Optional[List[float]] = None,
    reference_price: Optional[float] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Log-log plot of MC convergence: std_error and absolute error vs N.

    Overlays the theoretical O(1/sqrt(N)) reference line.

    Parameters
    ----------
    path_counts : list of int
        Number of paths for each data point.
    std_errors : list of float
        Standard error at each path count.
    errors : list of float, optional
        Absolute error vs reference price at each path count.
    reference_price : float, optional
        If provided, annotated on the plot.

    Returns
    -------
    tuple[Figure, Axes]
    """
    path_counts_arr = np.array(path_counts, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Standard error
    ax.loglog(path_counts_arr, std_errors, 'o-', color=COLORS_MC['mean'],
              linewidth=2.0, markersize=6, label='Std Error', zorder=3)

    # Absolute error (if provided)
    if errors is not None:
        ax.loglog(path_counts_arr, errors, 's--', color=COLORS_MC['antithetic'],
                  linewidth=1.5, markersize=5, label='|Error|', zorder=3)

    # O(1/sqrt(N)) reference line
    ref_y = std_errors[0] * np.sqrt(path_counts_arr[0])
    theoretical = ref_y / np.sqrt(path_counts_arr)
    ax.loglog(path_counts_arr, theoretical, ':', color=COLORS_MC['slope_ref'],
              linewidth=1.5, alpha=0.7, label=r'$O(1/\sqrt{N})$')

    if reference_price is not None:
        ax.text(0.02, 0.02, f'Reference: ${reference_price:.4f}$',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax.set_xlabel('Number of Paths ($N$)', fontsize=12)
    ax.set_ylabel('Error / Std Error', fontsize=12)
    ax.set_title('Monte Carlo Convergence', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    return fig, ax


def plot_variance_reduction_comparison(
    results_dict: dict[str, MCResult],
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Horizontal bar chart comparing std_error across variance reduction methods.

    Parameters
    ----------
    results_dict : dict of {str: MCResult}
        Keys are method labels (e.g., 'Plain', 'Antithetic', 'Control',
        'Antithetic + Control'). Values are MCResult instances.

    Returns
    -------
    tuple[Figure, Axes]
    """
    labels = list(results_dict.keys())
    se_values = [r.std_error for r in results_dict.values()]

    color_map = {
        'plain': COLORS_MC['plain'],
        'antithetic': COLORS_MC['antithetic'],
        'control': COLORS_MC['control'],
        'antithetic+control': COLORS_MC['combined'],
        'antithetic + control': COLORS_MC['combined'],
    }
    bar_colors = [
        color_map.get(label.lower(), COLORS_MC['plain']) for label in labels
    ]

    fig, ax = plt.subplots(figsize=(10, max(3, len(labels) * 1.0 + 1)))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, se_values, color=bar_colors, edgecolor='white', height=0.5)

    # Annotate with variance ratio relative to first entry (plain)
    if len(se_values) > 1:
        base_var = se_values[0] ** 2
        for i, (bar, se) in enumerate(zip(bars, se_values)):
            ratio = se ** 2 / base_var if base_var > 0 else 0
            if i > 0:
                ax.text(bar.get_width() + max(se_values) * 0.02,
                        bar.get_y() + bar.get_height() / 2,
                        f'VR={ratio:.2%}', va='center', fontsize=9,
                        fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Standard Error', fontsize=12)
    ax.set_title('Variance Reduction Comparison', fontsize=13)
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ──────────────────────────────────────────────
# Exotic option visualizations (Phase 3)
# ──────────────────────────────────────────────

COLORS_EXOTIC = {
    'path':        '#90CAF9',
    'path_itm':    '#4CAF50',
    'path_otm':    '#B0BEC5',
    'path_ko':     '#F44336',
    'strike':      '#FF9800',
    'barrier':     '#F44336',
    'max':         '#9C27B0',
    'min':         '#00BCD4',
    'average':     '#6A1B9A',
    'payoff_fill': '#FF5722',
    'mean':        '#1565C0',
    'reference':   '#F44336',
}


def plot_exotic_payoff(
    exotic: 'ExoticOption',
    S0: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    n_paths_show: int = 40,
    n_steps: int = 252,
    seed: int = 42,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Two-panel diagram illustrating the path-dependence of an exotic payoff.

    Left panel shows simulated GBM paths with instrument-specific overlays
    (strike, barrier level, running max/min) so the reader can see WHICH
    feature of the path drives the payoff. Right panel is the empirical
    distribution of terminal payoffs over a larger MC sample.

    Parameters
    ----------
    exotic : ExoticOption
        Any subclass of ``src.instruments.base.ExoticOption`` — must expose
        ``payoff(paths: np.ndarray) -> np.ndarray``. The function discovers
        contract attributes (``K``, ``barrier``, ``lookback_type``,
        ``option_type``) reflectively, so it works uniformly for Asian,
        Barrier, Lookback and Digital.
    S0, T, r, sigma, q : float
        Market parameters for the underlying GBM.
    n_paths_show : int, default 40
        Number of paths to display in the left panel.
    n_steps : int, default 252
        Time discretization (daily monitoring by default).
    seed : int, default 42
        RNG seed — reproducible across calls.
    title : str, optional
        Overrides the auto-generated title.

    Returns
    -------
    tuple[Figure, np.ndarray[Axes]]
        A figure with two axes (left: paths, right: payoff histogram).

    Notes
    -----
    The histogram uses a **larger** path count (``max(5 * n_paths_show,
    5000)``) than the panel display, so the distribution is smooth even
    when only 40 paths are plotted.
    """
    from src.engines.monte_carlo import MonteCarloEngine  # lazy: avoid cycles

    if S0 <= 0 or T <= 0 or sigma <= 0:
        raise ValueError(
            f'Invalid market parameters: S0={S0}, T={T}, sigma={sigma} '
            '(all must be strictly positive).'
        )

    n_paths_hist = max(5 * n_paths_show, 5_000)
    engine = MonteCarloEngine(n_paths=n_paths_hist, seed=seed)
    paths = engine.simulate_gbm(
        S0=S0, T=T, r=r, sigma=sigma, q=q, n_steps=n_steps,
    )
    payoffs = exotic.payoff(paths)

    t_grid = np.linspace(0.0, T, paths.shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_paths, ax_hist = axes

    K = getattr(exotic, 'K', None)
    barrier = getattr(exotic, 'barrier', None)
    barrier_type = getattr(exotic, 'barrier_type', None)
    option_type = getattr(exotic, 'option_type', 'call')

    display_idx = np.linspace(0, n_paths_hist - 1, n_paths_show, dtype=int)

    for idx in display_idx:
        path = paths[idx]
        color = COLORS_EXOTIC['path']

        if barrier is not None:
            knocked = False
            if barrier_type and 'up' in barrier_type:
                knocked = np.max(path) >= barrier
            elif barrier_type and 'down' in barrier_type:
                knocked = np.min(path) <= barrier
            knock_out = barrier_type and 'out' in barrier_type
            if knocked and knock_out:
                color = COLORS_EXOTIC['path_ko']
            elif knocked and not knock_out:
                color = COLORS_EXOTIC['path_itm']
            else:
                color = COLORS_EXOTIC['path_otm']
        elif K is not None and option_type in ('call', 'put'):
            terminal = path[-1]
            itm = (terminal > K) if option_type == 'call' else (terminal < K)
            color = COLORS_EXOTIC['path_itm'] if itm else COLORS_EXOTIC['path_otm']

        ax_paths.plot(t_grid, path, color=color, linewidth=0.9, alpha=0.7)

    ax_paths.plot(
        t_grid, np.mean(paths[display_idx], axis=0),
        color=COLORS_EXOTIC['mean'], linewidth=2.2, label='Displayed mean',
        zorder=5,
    )

    if K is not None:
        ax_paths.axhline(
            K, color=COLORS_EXOTIC['strike'], linewidth=1.6, linestyle='--',
            label=f'Strike $K={K:g}$', zorder=4,
        )
    if barrier is not None:
        ax_paths.axhline(
            barrier, color=COLORS_EXOTIC['barrier'], linewidth=1.8,
            linestyle='-.', label=f'Barrier $H={barrier:g}$', zorder=4,
        )
    from src.instruments.asian import AsianOption  # lazy: avoid cycles
    from src.instruments.lookback import LookbackOption  # lazy: avoid cycles

    sample_path = paths[display_idx[0]]
    if isinstance(exotic, LookbackOption):
        running_max = np.maximum.accumulate(sample_path)
        running_min = np.minimum.accumulate(sample_path)
        ax_paths.plot(
            t_grid, running_max, color=COLORS_EXOTIC['max'], linewidth=1.4,
            linestyle=':', alpha=0.85, label='Running max (sample)',
        )
        ax_paths.plot(
            t_grid, running_min, color=COLORS_EXOTIC['min'], linewidth=1.4,
            linestyle=':', alpha=0.85, label='Running min (sample)',
        )
    elif isinstance(exotic, AsianOption):
        fixings = sample_path[1:]  # exclude S_0 from the average, as the payoff does
        n_fixings = np.arange(1, fixings.size + 1)
        if exotic.avg_type == 'geometric':
            running_avg = np.exp(np.cumsum(np.log(fixings)) / n_fixings)
            avg_label = 'Running geo. avg (sample)'
        else:
            running_avg = np.cumsum(fixings) / n_fixings
            avg_label = 'Running avg (sample)'
        ax_paths.plot(
            t_grid[1:], running_avg, color=COLORS_EXOTIC['average'], linewidth=1.4,
            linestyle=':', alpha=0.85, label=avg_label,
        )

    ax_paths.set_xlabel('Time (years)', fontsize=12)
    ax_paths.set_ylabel('Underlying price $S_t$', fontsize=12)
    ax_paths.set_xlim(0.0, T)
    ax_paths.grid(True, alpha=0.3)
    ax_paths.legend(fontsize=9, loc='best')

    positive = payoffs[payoffs > 0]
    zero_frac = float(np.mean(payoffs <= 0.0))

    if positive.size > 0:
        ax_hist.hist(
            positive, bins=40, color=COLORS_EXOTIC['payoff_fill'],
            edgecolor='white', alpha=0.85,
        )
        mean_payoff = float(np.mean(payoffs))
        ax_hist.axvline(
            mean_payoff, color=COLORS_EXOTIC['mean'], linewidth=2.0,
            linestyle='--',
            label=f'Mean payoff $={mean_payoff:.4f}$',
        )
        ax_hist.legend(fontsize=10, loc='upper right')
    else:
        ax_hist.text(
            0.5, 0.5, 'All payoffs are zero',
            ha='center', va='center', transform=ax_hist.transAxes,
            fontsize=12, color='#888',
        )

    ax_hist.set_xlabel('Payoff at maturity', fontsize=12)
    ax_hist.set_ylabel('Frequency (positive payoffs)', fontsize=12)
    ax_hist.set_title(
        f'Terminal payoff distribution '
        f'(P[payoff=0] = {zero_frac:.1%}, N = {n_paths_hist:,})',
        fontsize=12,
    )
    ax_hist.grid(True, alpha=0.3)

    exotic_name = type(exotic).__name__
    fig.suptitle(
        title or f'{exotic_name} — path-dependence diagnostic',
        fontsize=14, fontweight='bold',
    )
    fig.tight_layout()
    return fig, axes


def plot_exotic_comparison(
    results: Dict[str, 'MCResult'],
    reference: Optional[float] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Four-panel side-by-side comparison of multiple exotic pricings.

    Useful as a README asset and as a diagnostic when sweeping methods
    (plain MC, antithetic, control variate, analytical) across several
    exotic instruments.

    Parameters
    ----------
    results : dict[str, MCResult]
        Ordered mapping of label -> MCResult. The first entry is used as
        the variance-ratio baseline. Labels are shown verbatim on every
        axis.
    reference : float, optional
        If provided, drawn as a horizontal line on the price panel (useful
        to cross-check MC against an analytical value).
    title : str, optional
        Overrides the auto-generated figure suptitle.

    Returns
    -------
    tuple[Figure, np.ndarray[Axes]]
        2x2 grid of axes: [price, std_error, variance ratio, 95% CI width].

    Notes
    -----
    ``MCResult`` is expected to expose at least ``price`` and
    ``std_error``. The variance ratio compares ``std_error**2`` of each
    entry to the first one (baseline). The CI width panel uses
    ``1.96 * std_error`` as a two-sided 95% half-width.
    """
    if not results:
        raise ValueError('results must contain at least one entry.')

    labels = list(results.keys())
    prices = np.array([r.price for r in results.values()], dtype=float)
    se = np.array([r.std_error for r in results.values()], dtype=float)
    base_var = se[0] ** 2
    var_ratio = (se ** 2) / base_var if base_var > 0 else np.zeros_like(se)
    ci95_half = 1.96 * se

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    x_pos = np.arange(len(labels))
    palette = [
        '#2196F3', '#4CAF50', '#FF9800', '#9C27B0',
        '#00BCD4', '#F44336', '#795548', '#607D8B',
    ]
    colors = [palette[i % len(palette)] for i in range(len(labels))]

    ax = axes[0, 0]
    bars = ax.bar(
        x_pos, prices, yerr=ci95_half, color=colors, edgecolor='white',
        capsize=5, error_kw={'elinewidth': 1.2, 'ecolor': '#333'},
    )
    for bar, price in zip(bars, prices):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(), f'{price:.4f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold',
        )
    if reference is not None:
        ax.axhline(
            reference, color=COLORS_EXOTIC['reference'], linewidth=1.6,
            linestyle='--', label=f'Reference $={reference:.4f}$',
        )
        ax.legend(fontsize=9, loc='best')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=10)
    ax.set_ylabel('Price (with 95% CI)', fontsize=11)
    ax.set_title('Exotic price', fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    ax = axes[0, 1]
    ax.bar(x_pos, se, color=colors, edgecolor='white')
    for i, value in enumerate(se):
        ax.text(
            i, value, f'{value:.2e}',
            ha='center', va='bottom', fontsize=9,
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=10)
    ax.set_ylabel('Standard error', fontsize=11)
    ax.set_title('Monte Carlo noise', fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    ax = axes[1, 0]
    ax.bar(x_pos, var_ratio, color=colors, edgecolor='white')
    for i, value in enumerate(var_ratio):
        ax.text(
            i, value, f'{value:.2%}',
            ha='center', va='bottom', fontsize=9,
        )
    ax.axhline(1.0, color='#888', linewidth=1.0, linestyle=':')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=10)
    ax.set_ylabel(f'Var ratio vs "{labels[0]}"', fontsize=11)
    ax.set_title('Variance reduction', fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    ax = axes[1, 1]
    ax.barh(x_pos, ci95_half, color=colors, edgecolor='white')
    for i, value in enumerate(ci95_half):
        ax.text(
            value, i, f' {value:.2e}',
            va='center', fontsize=9,
        )
    ax.set_yticks(x_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('95% CI half-width (1.96 x SE)', fontsize=11)
    ax.set_title('Interval precision', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)

    fig.suptitle(
        title or 'Exotic pricing comparison',
        fontsize=14, fontweight='bold',
    )
    fig.tight_layout()
    return fig, axes
