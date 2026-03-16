# Exotic Option Pricer

Production-grade derivatives pricing library implementing analytical and numerical models used in institutional quantitative finance.

## Phase 1: Black-Scholes-Merton Analytical Engine

Full implementation of the Black-Scholes-Merton (1973) model with:

- **16 Greeks** — first-order (delta, gamma, vega, theta, rho), second-order (vanna, volga, charm, speed, zomma, color), dual (dual_delta, dual_gamma), and utilities (probability ITM, elasticity)
- **Implied volatility solver** — Halley's method with cubic convergence (2-4 iterations), Brenner-Subrahmanyam initial guess, Brent fallback
- **Continuous dividend yield** — Merton (1973) extension throughout all methods
- **Vectorized computation** — scalar and NumPy array inputs, batch Greeks in single pass
- **10 visualization functions** — price surfaces, Greeks panels, payoff diagrams, IV smile, heatmaps

### Planned Phases

| Phase | Model | Status |
|-------|-------|--------|
| 1 | Black-Scholes analytical | **Complete** |
| 2 | Monte Carlo engine | Planned |
| 3 | Exotic options (Asian, Barrier, Lookback, Digital) | Planned |
| 4 | Heston stochastic volatility | Planned |
| 5 | Rough Bergomi (fractional Brownian motion) | Planned |

## Quick Start

```python
from src.models import BlackScholesModel

bs = BlackScholesModel(sigma=0.20)

# European call price
price = bs.price(S=100, K=100, T=1.0, r=0.05, option_type='call')
# 10.4506

# All 16 Greeks in one pass
greeks = bs.greeks(S=100, K=100, T=1.0, r=0.05, option_type='call')
# {'price': 10.45, 'delta': 0.637, 'gamma': 0.019, 'vega': 37.52, ...}

# Implied volatility from market price
iv = BlackScholesModel.implied_vol(
    market_price=10.45, S=100, K=100, T=1.0, r=0.05, option_type='call'
)
# 0.2000
```

## Installation

```bash
git clone https://github.com/MiguelReyy/exotic-option-pricer.git
cd exotic-option-pricer
pip install -e ".[dev]"
```

## Testing

```bash
pytest
```

The test suite validates correctness through multiple independent methods:

| Suite | What it validates |
|-------|-------------------|
| `test_pricing.py` | Hull benchmarks, put-call parity, boundary conditions, finite-difference Greeks, BS PDE satisfaction, homogeneity, no-arbitrage bounds, Monte Carlo cross-validation |
| `test_properties.py` | 8 mathematical invariants tested across ~3,000 randomly generated parameter sets (Hypothesis) |
| `test_benchmark.py` | 7,500-point parameter grid cross-validated against independent reference implementation, throughput benchmarks (>100k options/sec vectorized) |
| `test_strategies.py` | Straddle delta-neutrality, butterfly bounds, Greeks linearity, delta-hedge P&L |
| `test_visualization.py` | All 10 visualization functions, figure cleanup, edge cases |

## Architecture

```
src/
├── models/
│   ├── base.py              # ABC PricingModel interface
│   └── black_scholes.py     # BS-Merton analytical engine (638 lines, 16 Greeks)
├── utils/
│   └── visualization.py     # 10 professional visualization functions
├── engines/                  # Monte Carlo, PDE solvers (Phase 2)
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

## License

MIT
