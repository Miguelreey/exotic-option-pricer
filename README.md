# Exotic Option Pricer

[![CI](https://github.com/Miguelreey/exotic-option-pricer/actions/workflows/ci.yml/badge.svg)](https://github.com/Miguelreey/exotic-option-pricer/actions/workflows/ci.yml)
[![coverage](https://img.shields.io/badge/coverage-92%25-brightgreen.svg)](https://github.com/Miguelreey/exotic-option-pricer/actions/workflows/ci.yml)
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
- **15 visualization functions** — price surfaces, Greeks panels, MC paths, convergence plots, VR comparison, exotic payoff diagrams

## Phase 3: Exotic Options

Four families of path-dependent exotic instruments with analytical closed forms and full Monte Carlo integration:

- **Asian options** — arithmetic and geometric averaging, fixed and floating strike. Kemna-Vorst (1990) closed form for geometric. Geometric-as-control-variate for arithmetic (>95% variance reduction)
- **Barrier options** — 8 types (up/down × in/out × call/put) with optional rebate. Reiner-Rubinstein (1991) closed form (components A-F). Broadie-Glasserman-Yor (1997) continuity correction for discrete monitoring
- **Lookback options** — floating and fixed strike, call and put. Goldman-Sosin-Gatto (1979) / Conze-Viswanathan (1991) closed forms. Dedicated L'Hôpital branch for zero-drift case (r = q)
- **Digital options** — cash-or-nothing and asset-or-nothing (4 types). Black-Scholes closed form. Vanilla decomposition identity: C = AoN_call - K × CoN_call
- **Numerical Greeks** — bump-and-revalue for delta, gamma, vega, theta, rho on any exotic. GBM path rescaling for delta/gamma (single simulation). Common random numbers via seed reset for vega/theta/rho
- **ExoticOption ABC** — unified `payoff(paths)` interface plugging directly into `MonteCarloEngine.price()`

### Roadmap

| Phase | Model | Status |
|-------|-------|--------|
| 1 | Black-Scholes analytical | **Complete** |
| 2 | Monte Carlo engine | **Complete** |
| 3 | Exotic options (Asian, Barrier, Lookback, Digital) | **Complete** |
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

# Exotic option pricing with variance reduction
from src.instruments import AsianOption, BarrierOption, LookbackOption, DigitalOption

asian = AsianOption(K=100, option_type='call', avg_type='arithmetic')
mc = MonteCarloEngine(n_paths=500_000, seed=42)
paths = mc.simulate_gbm(100, 1.0, 0.05, 0.20, n_steps=252)
cv_fn = asian.geometric_control_fn(100, 1.0, 0.05, 0.20, n_obs=252)
result = mc.price(asian.payoff, paths, 0.05, 1.0, control_fn=cv_fn)
# MCResult(price=5.55, std_error=0.002, vr='control')

# Numerical Greeks for any exotic
from src.utils.greeks import numerical_greeks
greeks = numerical_greeks(mc, asian.payoff, S0=100, T=1.0, r=0.05, sigma=0.20)
# {'delta': 0.58, 'gamma': 0.025, 'vega': 23.1, 'theta': -3.8, 'rho': 32.4}
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

582 tests validate correctness through multiple independent methods:

| Suite | Tests | What it validates |
|-------|-------|-------------------|
| `test_pricing.py` | 178 | Hull benchmarks, put-call parity, boundary conditions, FD Greeks, BS PDE, homogeneity, no-arbitrage bounds |
| `test_properties.py` | ~3,000 | 8 mathematical invariants across random parameter sets (Hypothesis) |
| `test_benchmark.py` | 13 | 7,500-point grid vs independent reference, BS + MC throughput |
| `test_monte_carlo.py` | 127 | GBM distributions, Euler/Milstein strong convergence, MC vs BS cross-validation, VR (antithetic+control+IS), QMC, Euler absorption, batch pricing, Q-martingale |
| `test_exotics.py` | 240 | 4 exotic instruments: MC vs analytical cross-validation, in-out parity, AM≥GM, complementarity, vanilla decomposition, boundary conditions, Greeks vs BS, Hypothesis (6,000+ random cases) |
| `test_strategies.py` | 10 | Straddle delta-neutrality, butterfly bounds, Greeks linearity, delta-hedge P&L |
| `test_visualization.py` | 19 | All 15 visualization functions, exotic payoff diagrams, figure cleanup |

## Architecture

```
src/
├── models/
│   ├── base.py              # ABC PricingModel interface
│   └── black_scholes.py     # BS-Merton analytical engine (16 Greeks)
├── engines/
│   ├── monte_carlo.py       # MC engine: GBM simulation, European + generic pricing
│   └── variance_reduction.py # Antithetic, control variates, importance sampling
├── instruments/
│   ├── base.py              # ABC ExoticOption interface
│   ├── asian.py             # Asian options: Kemna-Vorst CV, arithmetic/geometric
│   ├── barrier.py           # Barrier options: Reiner-Rubinstein, BGY correction
│   ├── lookback.py          # Lookback options: Goldman-Sosin-Gatto, L'Hôpital
│   └── digital.py           # Digital options: cash/asset-or-nothing
├── utils/
│   ├── visualization.py     # 15 visualization functions
│   └── greeks.py            # Numerical Greeks: bump-and-revalue, CRN, path rescaling
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
- Kemna & Vorst (1990). *A Pricing Method for Options Based on Average Asset Values.* J. Banking & Finance 14.
- Reiner & Rubinstein (1991). *Breaking Down the Barriers.* Risk 4(8).
- Goldman, Sosin & Gatto (1979). *Path Dependent Options: Buy at the Low, Sell at the High.* J. Finance 34(5).
- Conze & Viswanathan (1991). *Path Dependent Options: The Case of Lookback Options.* J. Finance 46(5).
- Broadie, Glasserman & Kou (1997). *A Continuity Correction for Discrete Barrier Options.* Math. Finance 7(4).
- Kloeden & Platen (1992). *Numerical Solution of Stochastic Differential Equations.* Springer.

## License

MIT
