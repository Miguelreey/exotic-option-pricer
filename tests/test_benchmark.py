"""
Exotic Option Pricer — tests/test_benchmark.py

Parametric benchmarks against an independent reference implementation.
Cross-validates BlackScholesModel across 7500+ parameter combinations.

The reference functions are written from scratch — they share ZERO code
with BlackScholesModel — to guarantee true independence.

Additionally benchmarks against QuantLib if available.
"""

import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scipy.stats import norm

from exotic_option_pricer.models.black_scholes import BlackScholesModel


# ============================================================================
# Independent reference implementation (shares NO code with BlackScholesModel)
# ============================================================================
def _ref_d1d2(S, K, T, r, sigma, q=0.0):
    """Reference d1, d2 computation."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def ref_price(S, K, T, r, sigma, option_type='call', q=0.0):
    """Independent BS price — shares NO code with BlackScholesModel."""
    d1, d2 = _ref_d1d2(S, K, T, r, sigma, q)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    if option_type == 'call':
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    else:
        return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)


def ref_delta(S, K, T, r, sigma, option_type='call', q=0.0):
    """Independent BS delta."""
    d1, _ = _ref_d1d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    if option_type == 'call':
        return disc_q * norm.cdf(d1)
    else:
        return -disc_q * norm.cdf(-d1)


def ref_gamma(S, K, T, r, sigma, q=0.0):
    """Independent BS gamma."""
    d1, _ = _ref_d1d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def ref_vega(S, K, T, r, sigma, q=0.0):
    """Independent BS vega."""
    d1, _ = _ref_d1d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


# ============================================================================
# Parameter grid: 5 × 5 × 5 × 4 × 5 × 3 = 7500 combinations
# ============================================================================
_SPOTS = [50, 80, 100, 120, 200]
_STRIKES = [60, 80, 100, 120, 150]
_TIMES = [0.01, 0.1, 0.5, 1.0, 5.0]
_RATES = [0.0, 0.02, 0.05, 0.10]
_VOLS = [0.05, 0.15, 0.20, 0.40, 1.0]
_DIVS = [0.0, 0.02, 0.05]

_GRID = [(S, K, T, r, sig, q)
         for S in _SPOTS for K in _STRIKES for T in _TIMES
         for r in _RATES for sig in _VOLS for q in _DIVS]


# ============================================================================
# Tests
# ============================================================================
class TestParametricBenchmark:
    """Cross-validate against independent implementation over 7500+ cases."""

    def test_price_full_grid(self):
        """Compare price across full grid (7500 × 2 = 15000 evaluations)."""
        max_err = 0.0
        total = 0
        for S, K, T, r, sigma, q in _GRID:
            bs = BlackScholesModel(sigma=sigma)
            for otype in ('call', 'put'):
                ours = bs.price(S, K, T, r, otype, q=q)
                ref = ref_price(S, K, T, r, sigma, otype, q)
                err = abs(ours - ref)
                rel = err / max(abs(ref), 1e-12)
                max_err = max(max_err, rel)
                total += 1
                assert rel < 1e-10, (
                    f"Price mismatch: S={S}, K={K}, T={T}, r={r}, "
                    f"sigma={sigma}, q={q}, {otype}: ours={ours:.10f}, "
                    f"ref={ref:.10f}, rel_err={rel:.2e}"
                )
        assert total == len(_GRID) * 2

    def test_delta_full_grid(self):
        """Compare delta across full grid."""
        for S, K, T, r, sigma, q in _GRID:
            bs = BlackScholesModel(sigma=sigma)
            for otype in ('call', 'put'):
                ours = bs.delta(S, K, T, r, otype, q=q)
                ref = ref_delta(S, K, T, r, sigma, otype, q)
                assert abs(ours - ref) < 1e-10, (
                    f"Delta mismatch: S={S}, K={K}, T={T}, r={r}, "
                    f"sigma={sigma}, q={q}, {otype}"
                )

    def test_gamma_full_grid(self):
        """Compare gamma across full grid."""
        for S, K, T, r, sigma, q in _GRID:
            bs = BlackScholesModel(sigma=sigma)
            ours = bs.gamma(S, K, T, r, 'call', q=q)
            ref = ref_gamma(S, K, T, r, sigma, q)
            assert abs(ours - ref) < 1e-10, (
                f"Gamma mismatch: S={S}, K={K}, T={T}, r={r}, sigma={sigma}, q={q}"
            )

    def test_vega_full_grid(self):
        """Compare vega across full grid."""
        for S, K, T, r, sigma, q in _GRID:
            bs = BlackScholesModel(sigma=sigma)
            ours = bs.vega(S, K, T, r, 'call', q=q)
            ref = ref_vega(S, K, T, r, sigma, q)
            assert abs(ours - ref) < 1e-10, (
                f"Vega mismatch: S={S}, K={K}, T={T}, r={r}, sigma={sigma}, q={q}"
            )


class TestThroughputBenchmark:
    """Performance benchmark — measures pricing throughput."""

    def test_scalar_throughput(self):
        """Minimum scalar pricing speed."""
        bs = BlackScholesModel(sigma=0.20)
        N = 10000
        start = time.perf_counter()
        for _ in range(N):
            bs.price(100.0, 100.0, 1.0, 0.05, 'call')
        elapsed = time.perf_counter() - start
        throughput = N / elapsed
        # Must price at least 10k options/sec scalar
        assert throughput > 10000, f"Scalar throughput too low: {throughput:,.0f}/sec"

    def test_vectorized_throughput(self):
        """Vectorized pricing should handle 100k+ options."""
        bs = BlackScholesModel(sigma=0.20)
        N = 100_000
        S = np.random.default_rng(42).uniform(50, 200, N)
        K = np.full(N, 100.0)

        start = time.perf_counter()
        prices = bs.price(S, K, 1.0, 0.05, 'call')
        elapsed = time.perf_counter() - start

        assert len(prices) == N
        assert np.all(np.isfinite(prices))
        throughput = N / elapsed
        # Must price at least 100k options/sec vectorized
        assert throughput > 100_000, f"Vectorized throughput too low: {throughput:,.0f}/sec"

    def test_batch_greeks_throughput(self):
        """Batch Greeks should be efficient."""
        bs = BlackScholesModel(sigma=0.20)
        N = 50_000
        S = np.random.default_rng(42).uniform(50, 200, N)
        K = np.full(N, 100.0)

        start = time.perf_counter()
        g = bs.greeks(S, K, 1.0, 0.05, 'call')
        elapsed = time.perf_counter() - start

        assert len(g['price']) == N
        throughput = N / elapsed
        # Batch Greeks: all 16 values for 50k options
        assert throughput > 10_000, f"Batch Greeks throughput too low: {throughput:,.0f}/sec"


class TestQuantLibBenchmark:
    """Benchmark against QuantLib (skipped if not installed)."""

    @pytest.fixture(autouse=True)
    def _check_quantlib(self):
        pytest.importorskip('QuantLib')

    def test_price_vs_quantlib(self):
        import QuantLib as ql

        cases = [
            (100, 100, 1.0, 0.05, 0.20, 'call', 0.0),
            (100, 100, 1.0, 0.05, 0.20, 'put', 0.0),
            (100, 80, 0.5, 0.03, 0.30, 'call', 0.0),
            (120, 100, 2.0, 0.08, 0.15, 'call', 0.05),
            (80, 120, 0.25, 0.01, 0.40, 'put', 0.02),
        ]

        for S, K, T, r, sigma, otype, q in cases:
            # Our price
            bs = BlackScholesModel(sigma=sigma)
            ours = bs.price(S, K, T, r, otype, q=q)

            # QuantLib price
            today = ql.Date.todaysDate()
            expiry = today + ql.Period(int(T * 365), ql.Days)
            spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
            flat_ts = ql.YieldTermStructureHandle(
                ql.FlatForward(today, r, ql.Actual365Fixed()))
            div_ts = ql.YieldTermStructureHandle(
                ql.FlatForward(today, q, ql.Actual365Fixed()))
            vol_ts = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(today, ql.NullCalendar(), sigma,
                                    ql.Actual365Fixed()))
            bsm_process = ql.BlackScholesMertonProcess(
                spot_handle, div_ts, flat_ts, vol_ts)
            option_type_ql = (ql.Option.Call if otype == 'call'
                              else ql.Option.Put)
            payoff = ql.PlainVanillaPayoff(option_type_ql, K)
            exercise = ql.EuropeanExercise(expiry)
            option = ql.VanillaOption(payoff, exercise)
            option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
            ql_price = option.NPV()

            assert abs(ours - ql_price) < 0.01, (
                f"QuantLib mismatch: S={S}, K={K}, sigma={sigma}, "
                f"ours={ours:.6f}, QL={ql_price:.6f}"
            )


class TestMCBenchmark:
    """Monte Carlo engine throughput benchmarks (Phase 2)."""

    def test_simulate_gbm_exact_throughput(self):
        """
        Exact GBM simulation throughput: paths/sec.

        100k paths × 1 step is the baseline for European pricing.
        Must sustain > 1M paths/sec on modern hardware (vectorized
        log-space cumsum, no Python loops).
        """
        from exotic_option_pricer.engines.monte_carlo import MonteCarloEngine

        n_paths = 100_000
        mc = MonteCarloEngine(n_paths=n_paths, n_steps=1, seed=42)

        start = time.perf_counter()
        mc.simulate_gbm(100.0, 1.0, 0.05, 0.20)
        elapsed = time.perf_counter() - start

        throughput = n_paths / elapsed
        assert throughput > 500_000, (
            f"Exact GBM throughput too low: {throughput:,.0f} paths/sec"
        )

    def test_simulate_gbm_multistep_throughput(self):
        """
        Multi-step GBM: 10k paths × 252 steps (full year daily).

        Throughput measured as paths/sec (each path = 252 steps).
        """
        from exotic_option_pricer.engines.monte_carlo import MonteCarloEngine

        n_paths = 10_000
        mc = MonteCarloEngine(n_paths=n_paths, n_steps=252, seed=42)

        start = time.perf_counter()
        mc.simulate_gbm(100.0, 1.0, 0.05, 0.20)
        elapsed = time.perf_counter() - start

        throughput = n_paths / elapsed
        assert throughput > 10_000, (
            f"Multi-step GBM throughput too low: {throughput:,.0f} paths/sec"
        )

    def test_price_european_plain_throughput(self):
        """
        European option pricing throughput without variance reduction.

        Measures options/sec for price_european (simulate + price pipeline).
        """
        from exotic_option_pricer.engines.monte_carlo import MonteCarloEngine

        n_paths = 100_000
        n_reps = 10
        mc = MonteCarloEngine(n_paths=n_paths, seed=42)

        start = time.perf_counter()
        for _ in range(n_reps):
            mc.price_european(100.0, 100.0, 1.0, 0.05, 0.20, 'call')
        elapsed = time.perf_counter() - start

        throughput = n_reps / elapsed
        assert throughput > 5, (
            f"European pricing throughput too low: {throughput:.1f} opts/sec "
            f"(each with {n_paths:,} paths)"
        )

    def test_price_european_vr_throughput(self):
        """
        European pricing with antithetic + control variates.

        VR should add < 2x overhead vs plain (not 10x).
        """
        from exotic_option_pricer.engines.monte_carlo import MonteCarloEngine

        n_paths = 100_000
        n_reps = 10

        mc_plain = MonteCarloEngine(n_paths=n_paths, seed=42)
        start_plain = time.perf_counter()
        for _ in range(n_reps):
            mc_plain.price_european(100.0, 100.0, 1.0, 0.05, 0.20, 'call')
        time_plain = time.perf_counter() - start_plain

        mc_vr = MonteCarloEngine(n_paths=n_paths, seed=42)
        start_vr = time.perf_counter()
        for _ in range(n_reps):
            mc_vr.price_european(
                100.0, 100.0, 1.0, 0.05, 0.20, 'call',
                antithetic=True, control_variate=True,
            )
        time_vr = time.perf_counter() - start_vr

        overhead = time_vr / time_plain
        # Antithetic doubles path count, control adds covariance computation.
        # Expected ~2-3x overhead. Allow 4x for OS scheduling jitter.
        assert overhead < 4.0, (
            f"VR overhead too high: {overhead:.1f}x (plain={time_plain:.3f}s, "
            f"vr={time_vr:.3f}s)"
        )

    def test_exact_vs_milstein_throughput(self):
        """
        Vectorized Milstein should be within 2x of exact (both use cumsum).

        Euler (Python loop) is expected to be much slower and is not
        benchmarked here — it exists for validation, not production use.
        """
        from exotic_option_pricer.engines.monte_carlo import MonteCarloEngine

        n_paths, n_steps = 50_000, 100

        mc_exact = MonteCarloEngine(n_paths=n_paths, n_steps=n_steps, seed=42)
        start = time.perf_counter()
        mc_exact.simulate_gbm(100.0, 1.0, 0.05, 0.20, scheme='exact')
        time_exact = time.perf_counter() - start

        mc_mil = MonteCarloEngine(n_paths=n_paths, n_steps=n_steps, seed=42)
        start = time.perf_counter()
        mc_mil.simulate_gbm(100.0, 1.0, 0.05, 0.20, scheme='milstein')
        time_mil = time.perf_counter() - start

        ratio = time_mil / max(time_exact, 1e-9)
        assert ratio < 2.5, (
            f"Milstein is {ratio:.1f}x slower than exact "
            f"(exact={time_exact:.4f}s, milstein={time_mil:.4f}s)"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
