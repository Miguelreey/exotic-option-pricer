# Exotic Option Pricer

[![CI](https://github.com/Miguelreey/exotic-option-pricer/actions/workflows/ci.yml/badge.svg)](https://github.com/Miguelreey/exotic-option-pricer/actions/workflows/ci.yml)
[![coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)](https://github.com/Miguelreey/exotic-option-pricer/actions/workflows/ci.yml)
[![Python 3.10-3.13](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![ruff](https://img.shields.io/badge/linting-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/type%20check-mypy-blue.svg)](https://mypy-lang.org/)

Production-grade derivatives pricing library implementing analytical and numerical models used in institutional quantitative finance.

## Phase 1: Black-Scholes-Merton Analytical Engine

Full implementation of the Black-Scholes-Merton (1973) model with:

- **16 Greeks** — first-order (delta, gamma, vega, theta, rho), second-order (vanna, volga, charm, speed, zomma, color), dual (dual_delta, dual_gamma), and utilities (probability ITM, elasticity)
- **Implied volatility solver** — Halley's method with cubic convergence (2-4 iterations), Brenner-Subrahmanyam initial guess, Brent fallback
- **Continuous dividend yield** — Merton (1973) extension throughout all methods
- **Vectorized computation** — scalar and NumPy array inputs, batch Greeks in single pass

## Phase 2: Monte Carlo Engine

Production Monte Carlo engine for derivative pricing under GBM dynamics:

- **3 SDE schemes** — exact (log-space, zero discretization error), Euler-Maruyama, Milstein (vectorized via cumsum)
- **Variance reduction** — antithetic variates, control variates, importance sampling, combined (up to 98% variance reduction)
- **Quasi-Monte Carlo** — scrambled Sobol sequences with O(1/N) convergence vs O(1/√N) for standard MC
- **Importance sampling** — optimal drift shift for deep OTM options, likelihood ratio correction (Glasserman §4.6)
- **Euler absorption** — clamp paths at zero for non-negative processes (prepares Heston/rBergomi)
- **Batch pricing** — `price_batch()` reuses simulated paths across multiple payoffs/strikes
- **Generic payoff API** — `price(payoff_fn, paths)` accepts any path-dependent payoff for exotic derivatives
- **Convergence analysis** — std_error vs N diagnostics with optional VR
- **Strong convergence verified** — Milstein O(dt), Euler O(sqrt(dt)), tested via shared Brownian motion
- **13 visualization functions** — price surfaces, Greeks panels, MC paths, convergence plots, VR comparison

### Roadmap

| Phase | Model | Status |
|-------|-------|--------|
| 1 | Black-Scholes analytical | **Complete** |
| 2 | Monte Carlo engine | **Complete** |
| 3 | Exotic options (Asian, Barrier, Lookback, Digital) | Planned |
| 4 | Heston stochastic volatility | Planned |
| 5 | Rough Bergomi (fractional Brownian motion) | Planned |

## Quick Start

```python
import numpy as np
from src.models import BlackScholesModel
from src.engines import MonteCarloEngine

# Analytical pricing
bs = BlackScholesModel(sigma=0.20)
price = bs.price(S=100, K=100, T=1.0, r=0.05, option_type='call')
# 10.4506

# All 16 Greeks in one pass
greeks = bs.greeks(S=100, K=100, T=1.0, r=0.05, option_type='call')
# {'price': 10.45, 'delta': 0.637, 'gamma': 0.019, 'vega': 37.52, ...}

# Implied volatility from market price
iv = BlackScholesModel.implied_vol(
    price_market=10.45, S=100, K=100, T=1.0, r=0.05, option_type='call'
)
# 0.2000

# Monte Carlo pricing with variance reduction
mc = MonteCarloEngine(n_paths=500_000, seed=42)
result = mc.price_european(100, 100, 1.0, 0.05, 0.20, 'call',
                           antithetic=True, control_variate=True)
# MCResult(price=10.4494, std_error=0.0029, CI=[10.4437, 10.4551], vr='antithetic+control')

# Generic payoff for path-dependent exotics
paths = mc.simulate_gbm(100, 1.0, 0.05, 0.20, n_steps=252)
asian = mc.price(lambda p: np.maximum(np.mean(p[:, 1:], axis=1) - 100, 0),
                 paths, 0.05, 1.0)
```

## Installation

```bash
git clone https://github.com/Miguelreey/exotic-option-pricer.git
cd exotic-option-pricer
pip install -e ".[dev]"
```

## Testing

```bash
pytest
```

337 tests validate correctness through multiple independent methods:

| Suite | What it validates |
|-------|-------------------|
| `test_pricing.py` | Hull benchmarks, put-call parity, boundary conditions, finite-difference Greeks, BS PDE satisfaction, homogeneity, no-arbitrage bounds |
| `test_properties.py` | 8 mathematical invariants across ~3,000 random parameter sets (Hypothesis) |
| `test_benchmark.py` | 7,500-point grid vs independent reference, BS throughput (>100k opts/sec), MC throughput (paths/sec, VR overhead) |
| `test_monte_carlo.py` | GBM distributions, Euler/Milstein weak+strong convergence, MC vs BS cross-validation (20-point grid), antithetic+control+IS variance reduction, QMC Sobol convergence, Euler absorption, batch pricing, 95% CI coverage, PCP path-by-path, Q-martingale, Hypothesis properties |
| `test_strategies.py` | Straddle delta-neutrality, butterfly bounds, Greeks linearity, delta-hedge P&L |
| `test_visualization.py` | All 13 visualization functions, figure cleanup, edge cases |

## Architecture

```
src/
├── models/
│   ├── base.py              # ABC PricingModel interface
│   └── black_scholes.py     # BS-Merton analytical engine (16 Greeks)
├── engines/
│   ├── monte_carlo.py       # MC engine: GBM simulation, European + generic pricing
│   └── variance_reduction.py # Antithetic, control variates, importance sampling
├── utils/
│   └── visualization.py     # 13 visualization functions
├── instruments/              # Exotic payoffs (Phase 3)
└── calibration/              # Vol surface calibration (Phase 4)
```

All pricing models inherit from `PricingModel` (abstract base class), ensuring interchangeability in portfolio valuation, calibration, and risk management.

## Greeks Reference

| Greek | Order | Formula | Description |
|-------|-------|---------|-------------|
| Delta | 1st | dV/dS | Spot sensitivity |
| Gamma | 1st | d²V/dS² | Delta convexity |
| Vega | 1st | dV/dσ | Volatility sensitivity |
| Theta | 1st | dV/dt | Time decay |
| Rho | 1st | dV/dr | Rate sensitivity |
| Vanna | 2nd | d²V/(dS·dσ) | Delta-vol cross |
| Volga | 2nd | d²V/dσ² | Vega convexity |
| Charm | 2nd | dΔ/dT | Delta decay |
| Speed | 2nd | dΓ/dS | Gamma sensitivity to spot |
| Zomma | 2nd | dΓ/dσ | Gamma sensitivity to vol |
| Color | 2nd | dΓ/dt | Gamma decay |
| Dual Delta | Dual | dV/dK | Strike sensitivity |
| Dual Gamma | Dual | d²V/dK² | Breeden-Litzenberger density |

## References

- Black & Scholes (1973). *The Pricing of Options and Corporate Liabilities.* JPE 81(3).
- Merton (1973). *Theory of Rational Option Pricing.* Bell J. Econ. 4(1).
- Hull (2018). *Options, Futures, and Other Derivatives.* 10th ed.
- Haug (2007). *The Complete Guide to Option Pricing Formulas.* 2nd ed.
- Jaeckel (2017). *Let's Be Rational.* Wilmott.
- Glasserman (2003). *Monte Carlo Methods in Financial Engineering.* Springer.
- Kloeden & Platen (1992). *Numerical Solution of Stochastic Differential Equations.* Springer.

## License

MIT
