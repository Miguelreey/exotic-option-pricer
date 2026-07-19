"""
Exotic Option Pricer — tests/test_monte_carlo.py

Comprehensive test suite for the Monte Carlo engine (Phase 2).

Covers:
1.  TestGBMExactSimulation     — E[S_T], Var[log S_T], KS test, S_0 anchor
2.  TestEulerMaruyamaConvergence — weak O(dt), convergence to exact
3.  TestMilsteinScheme         — equivalence to exact for GBM, strong O(dt)
4.  TestEuropeanPricing        — cross-validation vs BS analytical, grid
5.  TestAntitheticVariates     — unbiased, variance reduction for calls/puts
6.  TestControlVariates        — unbiased, variance reduction, beta sign
7.  TestConvergenceRate        — log-log slope ~ -0.5
8.  TestConfidenceIntervals    — 95% CI coverage >= 90% over 100 reps
9.  TestReproducibility        — same seed → same result, diff seed → diff
10. TestEdgeCases              — T→0, sigma→0, deep ITM/OTM
11. TestPropertyBased          — Hypothesis: non-negativity, PCP, antithetic

Reference: Hull (2018), Glasserman (2003).
"""

import os
import sys

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exotic_option_pricer.engines.monte_carlo import MCResult, MonteCarloEngine
from exotic_option_pricer.engines.variance_reduction import (
    control_variate_adjust,
    generate_antithetic_normals,
    importance_sampling_likelihood,
    importance_sampling_shift,
    optimal_beta,
)
from exotic_option_pricer.models.black_scholes import BlackScholesModel

# ============================================================================
# Standard test parameters (same as Phase 1 for cross-validation)
# ============================================================================
HULL_S = 100.0
HULL_K = 100.0
HULL_T = 1.0
HULL_R = 0.05
HULL_SIGMA = 0.20
HULL_CALL = 10.4506
HULL_PUT = 5.5735

MC_SEED = 42
MC_PATHS = 200_000  # enough for < 0.05 error without VR


# ============================================================================
# 1. GBM Exact Simulation
# ============================================================================
class TestGBMExactSimulation:
    """Verify the exact (log-space) GBM simulation produces correct distributions."""

    def test_paths_start_at_S0(self):
        """All paths must start at S0 — basic sanity."""
        mc = MonteCarloEngine(n_paths=1_000, n_steps=50, seed=MC_SEED)
        paths = mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA)
        np.testing.assert_array_equal(paths[:, 0], HULL_S)

    def test_paths_shape(self):
        """Output shape must be (n_paths, n_steps + 1)."""
        n_paths, n_steps = 500, 100
        mc = MonteCarloEngine(n_paths=n_paths, n_steps=n_steps, seed=MC_SEED)
        paths = mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA)
        assert paths.shape == (n_paths, n_steps + 1)

    def test_mean_terminal_value(self):
        """
        E^Q[S_T] = S_0 * exp((r - q) * T).

        Under Q, the drift is (r - q). The sample mean should be within
        3 standard errors of the theoretical value.
        """
        mc = MonteCarloEngine(n_paths=500_000, n_steps=1, seed=MC_SEED)
        paths = mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA)
        ST = paths[:, -1]

        expected_mean = HULL_S * np.exp((HULL_R - 0.0) * HULL_T)
        sample_mean = np.mean(ST)
        sample_se = np.std(ST, ddof=1) / np.sqrt(len(ST))

        assert abs(sample_mean - expected_mean) < 3 * sample_se, (
            f"E[S_T] = {expected_mean:.4f}, sample = {sample_mean:.4f}, "
            f"SE = {sample_se:.4f}, diff = {abs(sample_mean - expected_mean):.4f}"
        )

    def test_mean_terminal_with_dividends(self):
        """E^Q[S_T] = S_0 * exp((r - q) * T) with q > 0."""
        q = 0.03
        mc = MonteCarloEngine(n_paths=500_000, n_steps=1, seed=MC_SEED)
        paths = mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA, q=q)
        ST = paths[:, -1]

        expected_mean = HULL_S * np.exp((HULL_R - q) * HULL_T)
        sample_mean = np.mean(ST)
        sample_se = np.std(ST, ddof=1) / np.sqrt(len(ST))

        assert abs(sample_mean - expected_mean) < 3 * sample_se

    def test_variance_log_returns(self):
        """
        Var[log(S_T / S_0)] = sigma^2 * T.

        The log-return variance is fully determined by sigma and T,
        independent of drift.
        """
        mc = MonteCarloEngine(n_paths=500_000, n_steps=1, seed=MC_SEED)
        paths = mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA)
        log_returns = np.log(paths[:, -1] / HULL_S)

        expected_var = HULL_SIGMA**2 * HULL_T
        sample_var = np.var(log_returns, ddof=1)

        # Variance estimator has SE ~ sigma^2 * sqrt(2/(N-1))
        var_se = expected_var * np.sqrt(2.0 / (len(log_returns) - 1))
        assert abs(sample_var - expected_var) < 3 * var_se

    def test_log_returns_normality_ks(self):
        """
        KS test: log(S_T / S_0) ~ N(mu, sigma^2 * T)
        where mu = (r - q - sigma^2/2) * T.

        P-value > 0.01 (fail to reject normality at 1% level).
        """
        mc = MonteCarloEngine(n_paths=50_000, n_steps=1, seed=MC_SEED)
        paths = mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA)
        log_returns = np.log(paths[:, -1] / HULL_S)

        mu = (HULL_R - 0.0 - 0.5 * HULL_SIGMA**2) * HULL_T
        sigma_total = HULL_SIGMA * np.sqrt(HULL_T)

        # Standardize and test against N(0, 1): the KS statistic is
        # invariant under the affine map, so this is exactly equivalent to
        # kstest(log_returns, 'norm', args=(mu, sigma_total)) — whose
        # `args` fast path broke in scipy 1.18.0 (ndtr() TypeError).
        z = (log_returns - mu) / sigma_total
        stat, pvalue = stats.kstest(z, "norm")
        assert pvalue > 0.01, f"KS test failed: stat={stat:.4f}, p={pvalue:.4f}"

    def test_paths_strictly_positive(self):
        """Exact GBM paths are always > 0 (log-space guarantees this)."""
        mc = MonteCarloEngine(n_paths=10_000, n_steps=252, seed=MC_SEED)
        paths = mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA)
        assert np.all(paths > 0)

    def test_discounted_stock_is_q_martingale(self):
        """
        Fundamental Theorem of Asset Pricing: under Q, the discounted
        stock price e^{-rt} S_t is a martingale (for q=0).

        For q > 0: e^{-(r-q)t} S_t is a Q-martingale, or equivalently
        E^Q[e^{-rT} S_T] = e^{-qT} S_0.

        We test this at MULTIPLE intermediate times, not just at T.
        This catches bugs that only affect intermediate steps (e.g.,
        incorrect drift in multi-step simulation).
        """
        n_paths = 500_000
        n_steps = 252
        q = 0.03
        mc = MonteCarloEngine(n_paths=n_paths, n_steps=n_steps, seed=MC_SEED)
        paths = mc.simulate_gbm(
            HULL_S,
            HULL_T,
            HULL_R,
            HULL_SIGMA,
            q=q,
        )

        t_grid = np.linspace(0, HULL_T, n_steps + 1)

        for step_idx in [1, 63, 126, 189, 252]:
            t = t_grid[step_idx]
            S_t = paths[:, step_idx]
            discounted = np.exp(-HULL_R * t) * S_t
            expected = HULL_S * np.exp(-q * t)
            se = np.std(discounted, ddof=1) / np.sqrt(n_paths)
            assert abs(np.mean(discounted) - expected) < 3 * se, (
                f"Martingale condition violated at t={t:.3f}: "
                f"E[e^{{-rt}}S_t]={np.mean(discounted):.4f}, "
                f"S_0*e^{{-qt}}={expected:.4f}, SE={se:.6f}"
            )


# ============================================================================
# 2. Euler-Maruyama Convergence
# ============================================================================
class TestEulerMaruyamaConvergence:
    """Verify Euler-Maruyama weak convergence O(dt) to exact GBM."""

    def test_weak_convergence_order(self):
        """
        Euler weak error for E[S_T^2] should decrease as O(dt).

        E[S_T^2] = S_0^2 * exp((2(r-q) + sigma^2)*T) under Q.
        The Euler second-moment bias is O(sigma^4 * dt) per step, making
        it a more sensitive diagnostic than the first moment (where the
        multiplicative structure of the Euler scheme for GBM accidentally
        preserves E[S_T] to very high accuracy).

        We average over multiple seeds to suppress MC noise and isolate
        the systematic discretization bias.
        """
        step_counts = [5, 20, 100, 500]
        n_seeds = 20
        S0, T, r, sigma = 100.0, 1.0, 0.05, 0.50  # high vol amplifies bias
        expected_second_moment = S0**2 * np.exp((2 * r + sigma**2) * T)
        avg_biases = []

        for n_steps in step_counts:
            biases = []
            for seed in range(n_seeds):
                mc = MonteCarloEngine(n_paths=50_000, n_steps=n_steps, seed=seed * 100)
                paths = mc.simulate_gbm(S0, T, r, sigma, scheme="euler")
                ST = paths[:, -1]
                biases.append(np.mean(ST**2) - expected_second_moment)
            avg_biases.append(abs(np.mean(biases)))

        # Finest discretization should have clearly less bias than coarsest
        assert avg_biases[-1] < avg_biases[0] * 0.5, (
            f"Euler E[S_T^2] bias did not decrease sufficiently: "
            f"coarse={avg_biases[0]:.2f}, fine={avg_biases[-1]:.2f}"
        )

    def test_euler_converges_to_exact_pricing(self):
        """
        With small dt (many steps), Euler MC price should match exact MC price.
        """
        mc_exact = MonteCarloEngine(n_paths=100_000, n_steps=1, seed=MC_SEED)
        mc_euler = MonteCarloEngine(n_paths=100_000, n_steps=1000, seed=MC_SEED)

        paths_exact = mc_exact.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA, scheme="exact")
        paths_euler = mc_euler.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA, scheme="euler")

        payoff_exact = np.mean(np.maximum(paths_exact[:, -1] - HULL_K, 0))
        payoff_euler = np.mean(np.maximum(paths_euler[:, -1] - HULL_K, 0))

        # With 1000 steps, Euler should be very close to exact
        assert abs(payoff_exact - payoff_euler) < 0.5, (
            f"Euler (1000 steps) vs exact: {payoff_euler:.4f} vs {payoff_exact:.4f}"
        )

    def test_euler_paths_shape(self):
        mc = MonteCarloEngine(n_paths=100, n_steps=50, seed=MC_SEED)
        paths = mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA, scheme="euler")
        assert paths.shape == (100, 51)
        np.testing.assert_array_equal(paths[:, 0], HULL_S)


# ============================================================================
# 3. Milstein Scheme
# ============================================================================
class TestMilsteinScheme:
    """Verify Milstein scheme properties for GBM."""

    def test_milstein_equivalent_to_exact_for_gbm(self):
        """
        For GBM, the Milstein correction completes the Taylor expansion,
        making it numerically equivalent to the exact scheme.

        Compare terminal distributions: mean and variance should match.
        """
        n_paths, n_steps = 200_000, 100
        mc_exact = MonteCarloEngine(n_paths=n_paths, n_steps=n_steps, seed=MC_SEED)
        mc_milstein = MonteCarloEngine(n_paths=n_paths, n_steps=n_steps, seed=MC_SEED)

        paths_exact = mc_exact.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA, scheme="exact")
        paths_mil = mc_milstein.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA, scheme="milstein")

        ST_exact = paths_exact[:, -1]
        ST_mil = paths_mil[:, -1]

        # Means should be very close (both use same RNG seed)
        assert abs(np.mean(ST_exact) - np.mean(ST_mil)) < 0.5
        # Variances should be close
        assert abs(np.var(ST_exact) - np.var(ST_mil)) / np.var(ST_exact) < 0.05

    def test_milstein_paths_positive(self):
        """Milstein paths should stay positive for reasonable parameters."""
        mc = MonteCarloEngine(n_paths=10_000, n_steps=252, seed=MC_SEED)
        paths = mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA, scheme="milstein")
        # With sigma=0.20 and 252 steps, paths should stay positive
        assert np.all(paths[:, 0] > 0)
        # Terminal values should be mostly positive
        assert np.mean(paths[:, -1] > 0) > 0.99

    def test_milstein_mean_terminal(self):
        """E[S_T] under Milstein should match theoretical value."""
        mc = MonteCarloEngine(n_paths=300_000, n_steps=100, seed=MC_SEED)
        paths = mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA, scheme="milstein")
        ST = paths[:, -1]
        expected = HULL_S * np.exp(HULL_R * HULL_T)
        se = np.std(ST, ddof=1) / np.sqrt(len(ST))
        assert abs(np.mean(ST) - expected) < 3 * se

    def test_milstein_strong_convergence_order(self):
        """
        Milstein has strong convergence order 1: E[|S_T^mil - S_T^exact|] = O(dt).

        Method:
        1. Generate fine-grained Brownian increments dW at resolution dt_fine.
        2. For each coarser resolution dt = k * dt_fine, aggregate increments
           by summing groups of k (preserving the SAME Brownian motion).
        3. Run Milstein at each resolution using these aggregated increments.
        4. Compare against exact S_T (computed from the total W_T).

        The log-log slope of E[|error|] vs dt should be ~1.0 for Milstein
        and ~0.5 for Euler. We test Milstein only (Euler loop is too slow
        at fine resolutions for this to be practical in CI).

        Reference: Kloeden & Platen (1992), Ch. 10.
        """
        S0, T, r, sigma = 100.0, 1.0, 0.05, 0.40  # high vol amplifies error
        n_paths = 20_000
        n_fine = 1024  # finest resolution (power of 2 for clean aggregation)
        rng = np.random.default_rng(MC_SEED)

        dt_fine = T / n_fine
        # Generate all fine increments: (n_paths, n_fine) standard normals
        Z_fine = rng.standard_normal((n_paths, n_fine))
        dW_fine = np.sqrt(dt_fine) * Z_fine  # actual Brownian increments

        # Exact terminal value from total Brownian: W_T = sum(dW)
        W_T = np.sum(dW_fine, axis=1)
        ST_exact = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * W_T)

        # Test Milstein at different resolutions by aggregating increments
        step_counts = [16, 32, 64, 128, 256]
        mean_errors = []

        for n_steps in step_counts:
            assert n_fine % n_steps == 0
            k = n_fine // n_steps
            dt = T / n_steps
            sqrt_dt = np.sqrt(dt)

            # Aggregate fine increments into coarse dW
            dW_coarse = dW_fine.reshape(n_paths, n_steps, k).sum(axis=2)
            Z_coarse = dW_coarse / sqrt_dt  # back to standard normals

            # Milstein step-by-step (multiplicative)
            drift_coeff = (r) * dt
            diff_coeff = sigma * sqrt_dt
            mil_coeff = 0.5 * sigma**2 * dt

            S = np.full(n_paths, S0)
            for step in range(n_steps):
                z = Z_coarse[:, step]
                S = S * (1.0 + drift_coeff + diff_coeff * z + mil_coeff * (z * z - 1.0))

            mean_errors.append(np.mean(np.abs(S - ST_exact)))

        # Fit log-log slope: E[|error|] ~ C * dt^p, expect p ~ 1.0
        log_dt = np.log([T / n for n in step_counts])
        log_err = np.log(mean_errors)
        slope, _, _, _, _ = stats.linregress(log_dt, log_err)

        assert 0.7 < slope < 1.5, (
            f"Milstein strong convergence slope = {slope:.3f}, expected ~1.0. "
            f"Errors: {[f'{e:.4f}' for e in mean_errors]}"
        )

    def test_euler_strong_convergence_order(self):
        """
        Euler-Maruyama has strong convergence order 0.5:
        E[|S_T^euler - S_T^exact|] = O(sqrt(dt)).

        Same methodology as the Milstein test but with coarser resolutions
        (Euler's Python loop makes fine resolutions too slow for CI).
        The slope should be ~0.5, clearly distinguishable from Milstein's ~1.0.
        """
        S0, T, r, sigma = 100.0, 1.0, 0.05, 0.40
        n_paths = 20_000
        n_fine = 512
        rng = np.random.default_rng(MC_SEED)

        dt_fine = T / n_fine
        Z_fine = rng.standard_normal((n_paths, n_fine))
        dW_fine = np.sqrt(dt_fine) * Z_fine

        W_T = np.sum(dW_fine, axis=1)
        ST_exact = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * W_T)

        # Coarser resolutions than Milstein test (Euler loop is slower)
        step_counts = [8, 16, 32, 64, 128]
        mean_errors = []

        for n_steps in step_counts:
            assert n_fine % n_steps == 0
            k = n_fine // n_steps
            dt = T / n_steps
            sqrt_dt = np.sqrt(dt)

            dW_coarse = dW_fine.reshape(n_paths, n_steps, k).sum(axis=2)
            Z_coarse = dW_coarse / sqrt_dt

            # Euler step-by-step
            drift_coeff = r * dt
            diff_coeff = sigma * sqrt_dt

            S = np.full(n_paths, S0)
            for step in range(n_steps):
                S = S * (1.0 + drift_coeff + diff_coeff * Z_coarse[:, step])

            mean_errors.append(np.mean(np.abs(S - ST_exact)))

        log_dt = np.log([T / n for n in step_counts])
        log_err = np.log(mean_errors)
        slope, _, _, _, _ = stats.linregress(log_dt, log_err)

        # Euler: slope ~ 0.5 (clearly below Milstein's ~1.0)
        assert 0.3 < slope < 0.8, (
            f"Euler strong convergence slope = {slope:.3f}, expected ~0.5. "
            f"Errors: {[f'{e:.4f}' for e in mean_errors]}"
        )


# ============================================================================
# 4. European Pricing (Cross-validation with BS Analytical)
# ============================================================================
class TestEuropeanPricing:
    """Cross-validate MC European prices against BS analytical (Phase 1)."""

    @pytest.fixture
    def bs(self):
        return BlackScholesModel(sigma=HULL_SIGMA)

    def test_call_atm(self, bs):
        mc = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        result = mc.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")
        analytical = bs.price(HULL_S, HULL_K, HULL_T, HULL_R, "call")
        assert abs(result.price - analytical) < max(0.05, 3 * result.std_error)

    def test_put_atm(self, bs):
        mc = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        result = mc.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "put")
        analytical = bs.price(HULL_S, HULL_K, HULL_T, HULL_R, "put")
        assert abs(result.price - analytical) < max(0.05, 3 * result.std_error)

    def test_call_itm(self, bs):
        K = 90.0  # ITM call
        mc = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        result = mc.price_european(HULL_S, K, HULL_T, HULL_R, HULL_SIGMA, "call")
        analytical = bs.price(HULL_S, K, HULL_T, HULL_R, "call")
        assert abs(result.price - analytical) < max(0.05, 3 * result.std_error)

    def test_put_itm(self, bs):
        K = 110.0  # ITM put
        mc = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        result = mc.price_european(HULL_S, K, HULL_T, HULL_R, HULL_SIGMA, "put")
        analytical = bs.price(HULL_S, K, HULL_T, HULL_R, "put")
        assert abs(result.price - analytical) < max(0.05, 3 * result.std_error)

    def test_call_otm(self, bs):
        K = 120.0  # OTM call
        mc = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        result = mc.price_european(HULL_S, K, HULL_T, HULL_R, HULL_SIGMA, "call")
        analytical = bs.price(HULL_S, K, HULL_T, HULL_R, "call")
        assert abs(result.price - analytical) < max(0.05, 3 * result.std_error)

    def test_put_otm(self, bs):
        K = 80.0  # OTM put
        mc = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        result = mc.price_european(HULL_S, K, HULL_T, HULL_R, HULL_SIGMA, "put")
        analytical = bs.price(HULL_S, K, HULL_T, HULL_R, "put")
        assert abs(result.price - analytical) < max(0.05, 3 * result.std_error)

    def test_with_dividends(self, bs):
        q = 0.03
        bs_div = BlackScholesModel(sigma=HULL_SIGMA)
        mc = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        result = mc.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call", q=q)
        analytical = bs_div.price(HULL_S, HULL_K, HULL_T, HULL_R, "call", q=q)
        assert abs(result.price - analytical) < max(0.05, 3 * result.std_error)

    @pytest.mark.parametrize(
        "S,K,T,r,sigma,q",
        [
            (100, 100, 1.0, 0.05, 0.20, 0.0),
            (100, 100, 0.5, 0.05, 0.30, 0.0),
            (100, 110, 1.0, 0.05, 0.20, 0.0),
            (100, 90, 1.0, 0.05, 0.20, 0.0),
            (50, 50, 2.0, 0.03, 0.25, 0.0),
            (200, 200, 0.25, 0.08, 0.15, 0.0),
            (100, 100, 1.0, 0.05, 0.40, 0.0),
            (100, 100, 1.0, 0.05, 0.10, 0.0),
            (100, 100, 1.0, 0.05, 0.20, 0.02),
            (100, 100, 1.0, 0.05, 0.20, 0.05),
            (100, 80, 0.5, 0.03, 0.35, 0.01),
            (100, 120, 2.0, 0.07, 0.25, 0.03),
            (150, 140, 0.75, 0.04, 0.18, 0.0),
            (50, 55, 1.5, 0.06, 0.22, 0.0),
            (100, 100, 3.0, 0.05, 0.20, 0.0),
            (100, 100, 0.1, 0.05, 0.20, 0.0),
            (80, 100, 1.0, 0.05, 0.30, 0.0),
            (120, 100, 1.0, 0.05, 0.30, 0.0),
            (100, 100, 1.0, 0.0, 0.20, 0.0),
            (100, 100, 1.0, 0.10, 0.20, 0.0),
        ],
    )
    def test_grid_cross_validation(self, S, K, T, r, sigma, q):
        """Grid of 20 parameter combinations: MC vs BS analytical."""
        bs_model = BlackScholesModel(sigma=sigma)
        mc = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)

        for opt in ("call", "put"):
            result = mc.price_european(S, K, T, r, sigma, opt, q=q)
            analytical = bs_model.price(S, K, T, r, opt, q=q)
            tol = max(0.05, 3 * result.std_error)
            assert abs(result.price - analytical) < tol, (
                f"{opt} S={S} K={K} T={T} r={r} sig={sigma} q={q}: "
                f"MC={result.price:.4f} BS={analytical:.4f} tol={tol:.4f}"
            )


# ============================================================================
# 5. Antithetic Variates
# ============================================================================
class TestAntitheticVariates:
    """Verify antithetic variates reduce variance without introducing bias."""

    def test_antithetic_unbiased_call(self):
        """Antithetic call price should match BS analytical (unbiased)."""
        mc = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        anti = mc.price_european(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call", antithetic=True
        )

        bs = BlackScholesModel(sigma=HULL_SIGMA)
        analytical = bs.price(HULL_S, HULL_K, HULL_T, HULL_R, "call")

        assert abs(anti.price - analytical) < max(0.05, 3 * anti.std_error)

    def test_antithetic_unbiased_put(self):
        mc = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        anti = mc.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "put", antithetic=True)
        bs = BlackScholesModel(sigma=HULL_SIGMA)
        analytical = bs.price(HULL_S, HULL_K, HULL_T, HULL_R, "put")
        assert abs(anti.price - analytical) < max(0.05, 3 * anti.std_error)

    def test_antithetic_reduces_variance_call(self):
        """Antithetic std_error should be less than plain for calls."""
        mc_plain = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        mc_anti = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)

        plain = mc_plain.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")
        anti = mc_anti.price_european(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call", antithetic=True
        )

        assert anti.std_error < plain.std_error, (
            f"Antithetic did not reduce variance: "
            f"plain SE={plain.std_error:.6f}, anti SE={anti.std_error:.6f}"
        )

    def test_antithetic_reduces_variance_put(self):
        """Antithetic std_error should be less than plain for puts."""
        mc_plain = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        mc_anti = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)

        plain = mc_plain.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "put")
        anti = mc_anti.price_european(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "put", antithetic=True
        )

        assert anti.std_error < plain.std_error

    def test_antithetic_variance_ratio(self):
        """Variance ratio (antithetic / plain) should be < 0.70 for ATM call."""
        mc_plain = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        mc_anti = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)

        plain = mc_plain.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")
        anti = mc_anti.price_european(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call", antithetic=True
        )

        ratio = (anti.std_error / plain.std_error) ** 2
        assert ratio < 0.70, f"Antithetic variance ratio = {ratio:.4f}, expected < 0.70"

    def test_generate_antithetic_normals(self):
        """Verify generate_antithetic_normals returns (Z, -Z)."""
        Z = np.random.default_rng(42).standard_normal((100, 10))
        Z_pos, Z_neg = generate_antithetic_normals(Z)
        np.testing.assert_array_equal(Z_pos, Z)
        np.testing.assert_array_equal(Z_neg, -Z)


# ============================================================================
# 6. Control Variates
# ============================================================================
class TestControlVariates:
    """Verify control variates reduce variance without introducing bias."""

    def test_control_unbiased_call(self):
        mc = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        result = mc.price_european(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call", control_variate=True
        )
        bs = BlackScholesModel(sigma=HULL_SIGMA)
        analytical = bs.price(HULL_S, HULL_K, HULL_T, HULL_R, "call")
        assert abs(result.price - analytical) < max(0.05, 3 * result.std_error)

    def test_control_unbiased_put(self):
        mc = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        result = mc.price_european(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "put", control_variate=True
        )
        bs = BlackScholesModel(sigma=HULL_SIGMA)
        analytical = bs.price(HULL_S, HULL_K, HULL_T, HULL_R, "put")
        assert abs(result.price - analytical) < max(0.05, 3 * result.std_error)

    def test_control_reduces_variance(self):
        """Control variate should reduce std_error vs plain MC."""
        mc_plain = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        mc_cv = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)

        plain = mc_plain.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")
        cv = mc_cv.price_european(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call", control_variate=True
        )

        assert cv.std_error < plain.std_error, (
            f"Control variate did not reduce variance: "
            f"plain SE={plain.std_error:.6f}, CV SE={cv.std_error:.6f}"
        )

    def test_control_variance_ratio(self):
        """
        Control variate variance ratio should be < 0.20 for ATM call.

        With S_T as control and E[e^{-rT}S_T] = S_0 e^{-qT} as known
        expectation, the correlation rho ~ 0.90 for ATM options gives
        Var_ratio = 1 - rho^2 ~ 0.19.
        """
        mc_plain = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        mc_cv = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)

        plain = mc_plain.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")
        cv = mc_cv.price_european(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call", control_variate=True
        )

        ratio = (cv.std_error / plain.std_error) ** 2
        assert ratio < 0.20, f"Control variance ratio = {ratio:.4f}, expected < 0.20"

    def test_combined_antithetic_control(self):
        """Combined antithetic + control should be best of all."""
        mc = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        result = mc.price_european(
            HULL_S,
            HULL_K,
            HULL_T,
            HULL_R,
            HULL_SIGMA,
            "call",
            antithetic=True,
            control_variate=True,
        )
        assert result.variance_reduction == "antithetic+control"

        bs = BlackScholesModel(sigma=HULL_SIGMA)
        analytical = bs.price(HULL_S, HULL_K, HULL_T, HULL_R, "call")
        assert abs(result.price - analytical) < max(0.05, 3 * result.std_error)

    def test_optimal_beta_positive_for_calls(self):
        """
        beta* should be positive for calls: payoff is monotonically
        increasing in S_T, so Cov[payoff, S_T] > 0.
        """
        rng = np.random.default_rng(MC_SEED)
        Z = rng.standard_normal(100_000)
        ST = HULL_S * np.exp(
            (HULL_R - 0.5 * HULL_SIGMA**2) * HULL_T + HULL_SIGMA * np.sqrt(HULL_T) * Z
        )
        payoffs = np.exp(-HULL_R * HULL_T) * np.maximum(ST - HULL_K, 0)
        control = np.exp(-HULL_R * HULL_T) * ST

        beta = optimal_beta(payoffs, control)
        assert beta > 0, f"Expected positive beta for calls, got {beta:.4f}"

    def test_optimal_beta_validation(self):
        """optimal_beta should raise on mismatched arrays."""
        with pytest.raises(ValueError):
            optimal_beta(np.array([1, 2, 3]), np.array([1, 2]))

    def test_control_variate_adjust_preserves_mean(self):
        """
        Adjustment is unbiased: E[Y_cv] = E[Y] because E[C - E[C]] = 0.
        Verify empirically that mean doesn't change significantly.
        """
        rng = np.random.default_rng(MC_SEED)
        payoffs = rng.standard_normal(50_000) + 10.0
        control = rng.standard_normal(50_000) + 5.0
        adjusted, beta = control_variate_adjust(payoffs, control, 5.0)

        # Means should be very close (not exactly equal due to in-sample beta)
        assert abs(np.mean(adjusted) - np.mean(payoffs)) < 0.1


# ============================================================================
# 7. Convergence Rate
# ============================================================================
class TestConvergenceRate:
    """Verify MC error converges as O(1/sqrt(N))."""

    def test_log_log_slope(self):
        """
        In a log-log plot of std_error vs N, the slope should be ~ -0.5.

        std_error ~ C / sqrt(N)  =>  log(SE) ~ log(C) - 0.5 * log(N)
        """
        path_counts = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
        std_errors = []

        for n in path_counts:
            mc = MonteCarloEngine(n_paths=n, seed=MC_SEED)
            result = mc.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")
            std_errors.append(result.std_error)

        log_n = np.log(path_counts)
        log_se = np.log(std_errors)
        slope, _, _, _, _ = stats.linregress(log_n, log_se)

        assert -0.60 < slope < -0.40, f"Log-log slope = {slope:.4f}, expected ~ -0.50"

    def test_doubling_paths_halves_variance(self):
        """Doubling N should reduce std_error by factor ~ sqrt(2) ~ 1.41."""
        mc1 = MonteCarloEngine(n_paths=100_000, seed=MC_SEED)
        mc2 = MonteCarloEngine(n_paths=200_000, seed=MC_SEED)

        r1 = mc1.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")
        r2 = mc2.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")

        ratio = r1.std_error / r2.std_error
        # Should be ~ sqrt(2) = 1.414, allow 20% tolerance
        assert 1.1 < ratio < 1.8, f"SE ratio = {ratio:.4f}, expected ~ 1.41"


# ============================================================================
# 8. Confidence Intervals
# ============================================================================
class TestConfidenceIntervals:
    """Verify 95% CI has correct coverage."""

    def test_ci_coverage(self):
        """
        Run 100 independent MC simulations. The 95% CI should contain
        the true BS price in >= 90% of cases (allowing for finite-sample
        CLT approximation error).
        """
        bs = BlackScholesModel(sigma=HULL_SIGMA)
        true_price = bs.price(HULL_S, HULL_K, HULL_T, HULL_R, "call")
        n_reps = 100
        n_paths = 50_000
        covered = 0

        for i in range(n_reps):
            mc = MonteCarloEngine(n_paths=n_paths, seed=i * 1000 + 1)
            result = mc.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")
            if result.ci_lower <= true_price <= result.ci_upper:
                covered += 1

        coverage = covered / n_reps
        # With N=50_000 paths, CLT is excellent. Under H0: true coverage = 0.95,
        # P(coverage < 0.92 | n_reps=100) < 0.05 by binomial test.
        # Threshold 0.88 was too permissive for a production-grade engine.
        assert coverage >= 0.92, f"CI coverage = {coverage:.2%}, expected >= 92%"

    def test_ci_narrows_with_n(self):
        """CI width should decrease with N."""
        widths = []
        for n in [10_000, 50_000, 200_000]:
            mc = MonteCarloEngine(n_paths=n, seed=MC_SEED)
            result = mc.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")
            widths.append(result.ci_upper - result.ci_lower)

        assert widths[0] > widths[1] > widths[2], f"CI widths should decrease: {widths}"


# ============================================================================
# 9. Reproducibility
# ============================================================================
class TestReproducibility:
    """Verify seed-based reproducibility."""

    def test_same_seed_same_price(self):
        mc1 = MonteCarloEngine(n_paths=50_000, seed=42)
        mc2 = MonteCarloEngine(n_paths=50_000, seed=42)

        r1 = mc1.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")
        r2 = mc2.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")

        assert r1.price == r2.price, f"Same seed gave different prices: {r1.price} vs {r2.price}"
        assert r1.std_error == r2.std_error

    def test_different_seed_different_price(self):
        mc1 = MonteCarloEngine(n_paths=50_000, seed=42)
        mc2 = MonteCarloEngine(n_paths=50_000, seed=99)

        r1 = mc1.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")
        r2 = mc2.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")

        assert r1.price != r2.price, "Different seeds should give different prices"

    def test_same_seed_same_paths(self):
        mc1 = MonteCarloEngine(n_paths=1_000, n_steps=50, seed=42)
        mc2 = MonteCarloEngine(n_paths=1_000, n_steps=50, seed=42)

        paths1 = mc1.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA)
        paths2 = mc2.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA)

        np.testing.assert_array_equal(paths1, paths2)

    def test_reset_restores_state(self):
        """reset() restores RNG to initial state — same results after reset."""
        mc = MonteCarloEngine(n_paths=50_000, seed=42)
        r1 = mc.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")
        mc.reset()
        r2 = mc.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")
        assert r1.price == r2.price
        assert r1.std_error == r2.std_error

    def test_reset_with_new_seed(self):
        """reset(seed=new) changes the seed and produces different results."""
        mc = MonteCarloEngine(n_paths=50_000, seed=42)
        r1 = mc.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")
        mc.reset(seed=99)
        r2 = mc.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")
        assert r1.price != r2.price
        assert mc.seed == 99

    def test_reset_paths_identical(self):
        """reset() gives bit-identical paths, not just prices."""
        mc = MonteCarloEngine(n_paths=1_000, n_steps=50, seed=42)
        paths1 = mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA)
        mc.reset()
        paths2 = mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA)
        np.testing.assert_array_equal(paths1, paths2)


# ============================================================================
# 10. Edge Cases
# ============================================================================
class TestEdgeCases:
    """Boundary conditions and degenerate parameters."""

    def test_near_expiry(self):
        """
        T → 0: option price → intrinsic value.
        Call intrinsic = max(S - K, 0).
        """
        T = 0.001  # ~8 hours to expiry
        mc = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)

        # ITM call (S=100, K=95): intrinsic ~ 5
        result = mc.price_european(100, 95, T, HULL_R, HULL_SIGMA, "call")
        intrinsic = max(100 - 95 * np.exp(-HULL_R * T), 0)
        assert abs(result.price - intrinsic) < 0.5

        # OTM call (S=100, K=110): intrinsic = 0
        result = mc.price_european(100, 110, T, HULL_R, HULL_SIGMA, "call")
        assert result.price < 0.5

    def test_low_vol(self):
        """
        sigma → 0: paths are nearly deterministic.
        Price → discounted intrinsic = max(S*e^{(r-q)T} - K, 0) * e^{-rT}.
        """
        sigma = 0.001
        mc = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        result = mc.price_european(100, 95, HULL_T, HULL_R, sigma, "call")

        forward = 100 * np.exp(HULL_R * HULL_T)
        expected = np.exp(-HULL_R * HULL_T) * max(forward - 95, 0)
        assert abs(result.price - expected) < 0.1

    def test_deep_itm_call(self):
        """Deep ITM call: price ~ S - K*e^{-rT}."""
        mc = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        result = mc.price_european(200, 100, HULL_T, HULL_R, HULL_SIGMA, "call")
        lower_bound = 200 - 100 * np.exp(-HULL_R * HULL_T)
        assert result.price > lower_bound * 0.95  # within 5% of intrinsic

    def test_deep_otm_call(self):
        """Deep OTM call: price ~ 0 with std_error possibly > price."""
        mc = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        result = mc.price_european(100, 200, HULL_T, HULL_R, HULL_SIGMA, "call")
        assert result.price < 0.5  # essentially zero

    def test_deep_itm_put(self):
        """Deep ITM put: price ~ K*e^{-rT} - S."""
        mc = MonteCarloEngine(n_paths=MC_PATHS, seed=MC_SEED)
        result = mc.price_european(50, 100, HULL_T, HULL_R, HULL_SIGMA, "put")
        lower_bound = 100 * np.exp(-HULL_R * HULL_T) - 50
        assert result.price > lower_bound * 0.90

    def test_invalid_inputs(self):
        """Engine should reject invalid parameters."""
        mc = MonteCarloEngine(n_paths=100, seed=MC_SEED)
        with pytest.raises(ValueError):
            mc.simulate_gbm(-100, HULL_T, HULL_R, HULL_SIGMA)
        with pytest.raises(ValueError):
            mc.simulate_gbm(HULL_S, -1.0, HULL_R, HULL_SIGMA)
        with pytest.raises(ValueError):
            mc.simulate_gbm(HULL_S, HULL_T, HULL_R, -0.1)
        with pytest.raises(ValueError):
            mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA, scheme="invalid")

    def test_invalid_engine_params(self):
        with pytest.raises(ValueError, match="n_paths must be >= 2"):
            MonteCarloEngine(n_paths=1)
        with pytest.raises(ValueError, match="n_paths must be >= 2"):
            MonteCarloEngine(n_paths=0)
        with pytest.raises(ValueError):
            MonteCarloEngine(n_paths=100, n_steps=0)

    def test_price_european_input_validation(self):
        """price_european should reject invalid S0, K, T, sigma."""
        mc = MonteCarloEngine(n_paths=100, seed=MC_SEED)
        with pytest.raises(ValueError):
            mc.price_european(-100, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")
        with pytest.raises(ValueError):
            mc.price_european(HULL_S, -1, HULL_T, HULL_R, HULL_SIGMA, "call")
        with pytest.raises(ValueError):
            mc.price_european(HULL_S, HULL_K, -1, HULL_R, HULL_SIGMA, "call")
        with pytest.raises(ValueError):
            mc.price_european(HULL_S, HULL_K, HULL_T, HULL_R, -0.1, "call")
        with pytest.raises(ValueError):
            mc.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "straddle")

    def test_option_type_shorthand(self):
        """'c' and 'p' are accepted as shorthand for 'call' and 'put'."""
        mc1 = MonteCarloEngine(n_paths=10_000, seed=MC_SEED)
        mc2 = MonteCarloEngine(n_paths=10_000, seed=MC_SEED)
        mc3 = MonteCarloEngine(n_paths=10_000, seed=MC_SEED)
        mc4 = MonteCarloEngine(n_paths=10_000, seed=MC_SEED)

        call_full = mc1.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")
        call_short = mc2.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "c")
        put_full = mc3.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "put")
        put_short = mc4.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "p")

        assert call_full.price == call_short.price
        assert put_full.price == put_short.price

    def test_n_steps_override_invalid(self):
        """n_steps=0 in simulate_gbm override should raise ValueError."""
        mc = MonteCarloEngine(n_paths=100, n_steps=252, seed=MC_SEED)
        with pytest.raises(ValueError, match="n_steps must be >= 1"):
            mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA, n_steps=0)

    def test_repr(self):
        """Engine repr should show n_paths, n_steps, seed."""
        mc = MonteCarloEngine(n_paths=50_000, n_steps=252, seed=42)
        r = repr(mc)
        assert "50,000" in r
        assert "252" in r
        assert "42" in r


# ============================================================================
# 11. Property-Based Tests (Hypothesis)
# ============================================================================
class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        S=st.floats(min_value=10.0, max_value=500.0),
        K=st.floats(min_value=10.0, max_value=500.0),
        T=st.floats(min_value=0.05, max_value=5.0),
        r=st.floats(min_value=-0.02, max_value=0.15),
        sigma=st.floats(min_value=0.05, max_value=1.0),
    )
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_price_non_negative(self, S, K, T, r, sigma):
        """MC price must be >= 0 for all valid inputs."""
        mc = MonteCarloEngine(n_paths=5_000, seed=MC_SEED)
        for opt in ("call", "put"):
            result = mc.price_european(S, K, T, r, sigma, opt)
            assert result.price >= -result.std_error * 3, (
                f"Negative price: {result.price:.6f} for {opt} "
                f"S={S:.2f} K={K:.2f} T={T:.4f} r={r:.4f} sigma={sigma:.4f}"
            )

    @given(
        S=st.floats(min_value=50.0, max_value=200.0),
        K=st.floats(min_value=50.0, max_value=200.0),
        T=st.floats(min_value=0.1, max_value=3.0),
        r=st.floats(min_value=0.0, max_value=0.10),
        sigma=st.floats(min_value=0.10, max_value=0.60),
    )
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_put_call_parity(self, S, K, T, r, sigma):
        """
        Put-Call Parity: C - P = S - K*e^{-rT} (for q=0).

        Uses a SINGLE simulation and computes both payoffs on the same
        terminal values. PCP holds path-by-path:
            max(S_T - K, 0) - max(K - S_T, 0) = S_T - K

        so the MC estimator satisfies PCP to machine precision — the only
        error is from discounting, which is deterministic.
        """
        mc = MonteCarloEngine(n_paths=50_000, n_steps=1, seed=MC_SEED)
        paths = mc.simulate_gbm(S, T, r, sigma, scheme="exact")
        ST = paths[:, -1]

        discount = np.exp(-r * T)
        call_payoffs = discount * np.maximum(ST - K, 0.0)
        put_payoffs = discount * np.maximum(K - ST, 0.0)

        # PCP path-by-path: call - put = discount * (S_T - K)
        parity_lhs = np.mean(call_payoffs) - np.mean(put_payoffs)
        parity_rhs = S - K * np.exp(-r * T)

        # With same S_T for both, the only noise is from E[S_T] vs S*e^{rT}.
        # Tolerance: based on SE of discounted S_T estimator.
        se_ST = np.std(discount * ST, ddof=1) / np.sqrt(len(ST))
        tol = max(0.1, 3 * se_ST)

        assert abs(parity_lhs - parity_rhs) < tol, (
            f"PCP violated: C-P={parity_lhs:.4f}, S-Ke^(-rT)={parity_rhs:.4f}, "
            f"diff={abs(parity_lhs - parity_rhs):.4f}, tol={tol:.4f}"
        )

    @given(
        S=st.floats(min_value=50.0, max_value=200.0),
        K=st.floats(min_value=50.0, max_value=200.0),
        T=st.floats(min_value=0.1, max_value=3.0),
        r=st.floats(min_value=0.0, max_value=0.10),
        sigma=st.floats(min_value=0.10, max_value=0.60),
    )
    @settings(
        max_examples=500,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
        deadline=None,
    )
    def test_antithetic_never_increases_variance_calls(self, S, K, T, r, sigma):
        """
        For monotone payoffs (calls), antithetic variates should never
        increase variance. Cov[f(Z), f(-Z)] <= 0 for monotone f.

        Deep OTM options (P(ITM) < 1%) are excluded: nearly all payoffs are
        zero, making the variance estimator itself extremely noisy.  The
        theoretical guarantee still holds, but the empirical comparison is
        uninformative when both SE estimates are dominated by estimation
        noise of the variance of a near-degenerate distribution.
        """
        # Skip options that are too far OTM in normalized terms.
        # Raw moneyness (forward/K > 0.70) misses cases where low vol +
        # short maturity make an 85%-moneyness option 4+ sigma OTM.
        forward = S * np.exp(r * T)
        sigma_T = sigma * np.sqrt(T)
        normalized_moneyness = np.log(forward / K) / sigma_T
        assume(normalized_moneyness > -1.5)  # within 1.5 sigma of ATM

        mc_plain = MonteCarloEngine(n_paths=20_000, seed=MC_SEED)
        mc_anti = MonteCarloEngine(n_paths=20_000, seed=MC_SEED)

        plain = mc_plain.price_european(S, K, T, r, sigma, "call")
        anti = mc_anti.price_european(S, K, T, r, sigma, "call", antithetic=True)

        # Allow 15% margin for sampling noise in variance estimator
        assert anti.std_error <= plain.std_error * 1.15, (
            f"Antithetic increased variance: plain SE={plain.std_error:.6f}, "
            f"anti SE={anti.std_error:.6f}, "
            f"normalized_moneyness={normalized_moneyness:.2f}"
        )


# ============================================================================
# Variance Reduction Module Unit Tests
# ============================================================================
class TestVarianceReductionModule:
    """Direct tests for variance_reduction.py functions."""

    def test_antithetic_normals_shapes(self):
        Z = np.ones((100, 50))
        Z_pos, Z_neg = generate_antithetic_normals(Z)
        assert Z_pos.shape == (100, 50)
        assert Z_neg.shape == (100, 50)

    def test_antithetic_normals_values(self):
        Z = np.array([[1.0, -2.0], [3.0, 0.5]])
        Z_pos, Z_neg = generate_antithetic_normals(Z)
        np.testing.assert_array_equal(Z_pos, Z)
        np.testing.assert_array_equal(Z_neg, -Z)

    def test_optimal_beta_known(self):
        """For perfectly correlated Y = 2C + noise, beta ~ 2."""
        rng = np.random.default_rng(42)
        C = rng.standard_normal(100_000)
        Y = 2 * C + 0.1 * rng.standard_normal(100_000)
        beta = optimal_beta(Y, C)
        assert abs(beta - 2.0) < 0.05

    def test_control_adjust_reduces_variance(self):
        """After adjustment, Var[Y_cv] < Var[Y] when rho is high."""
        rng = np.random.default_rng(42)
        C = rng.standard_normal(50_000)
        Y = C + 0.1 * rng.standard_normal(50_000) + 5.0
        adjusted, beta = control_variate_adjust(Y, C, 0.0)
        assert np.var(adjusted) < np.var(Y)

    def test_control_adjust_with_fixed_beta(self):
        payoffs = np.array([10.0, 11.0, 12.0, 9.0, 10.5])
        control = np.array([100.0, 101.0, 102.0, 99.0, 100.5])
        adjusted, beta_used = control_variate_adjust(payoffs, control, 100.0, beta=1.0)
        expected = payoffs - 1.0 * (control - 100.0)
        np.testing.assert_array_almost_equal(adjusted, expected)
        assert beta_used == 1.0

    def test_optimal_beta_too_few_samples(self):
        """optimal_beta requires at least 2 samples."""
        with pytest.raises(ValueError, match="at least 2 samples"):
            optimal_beta(np.array([1.0]), np.array([2.0]))

    def test_optimal_beta_zero_variance_control(self):
        """When control has zero variance, beta should be 0 (fallback)."""
        payoffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        control = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # constant
        beta = optimal_beta(payoffs, control)
        assert beta == 0.0


# ============================================================================
# MCResult Tests
# ============================================================================
class TestMCResult:
    """Test MCResult dataclass."""

    def test_frozen(self):
        """MCResult should be immutable."""
        result = MCResult(10.0, 0.01, 9.98, 10.02, 100_000, "none")
        with pytest.raises(AttributeError):
            result.price = 20.0  # type: ignore[misc]

    def test_repr(self):
        result = MCResult(10.4506, 0.0234, 10.4047, 10.4965, 100_000, "antithetic")
        r = repr(result)
        assert "MCResult" in r
        assert "10.4506" in r
        assert "antithetic" in r


# ============================================================================
# Generic price() Method Tests
# ============================================================================
class TestGenericPrice:
    """Test the generic price() method with custom payoff functions."""

    def test_european_call_via_generic(self):
        """price() with European call payoff should match price_european()."""
        mc1 = MonteCarloEngine(n_paths=100_000, n_steps=1, seed=MC_SEED)
        mc2 = MonteCarloEngine(n_paths=100_000, n_steps=1, seed=MC_SEED)

        direct = mc1.price_european(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")

        paths = mc2.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA, scheme="exact")
        generic = mc2.price(lambda p: np.maximum(p[:, -1] - HULL_K, 0), paths, HULL_R, HULL_T)

        assert abs(direct.price - generic.price) < 0.1

    def test_payoff_fn_shape_validation(self):
        mc = MonteCarloEngine(n_paths=100, n_steps=10, seed=MC_SEED)
        paths = mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA)

        with pytest.raises(ValueError, match="payoff_fn returned"):
            mc.price(lambda p: np.array([1.0, 2.0]), paths, HULL_R, HULL_T)

    def test_convergence_analysis(self):
        mc = MonteCarloEngine(n_paths=10_000, seed=MC_SEED)
        bs = BlackScholesModel(sigma=HULL_SIGMA)
        ref = bs.price(HULL_S, HULL_K, HULL_T, HULL_R, "call")

        result = mc.convergence_analysis(
            HULL_S,
            HULL_K,
            HULL_T,
            HULL_R,
            HULL_SIGMA,
            path_counts=[1_000, 5_000, 10_000],
            reference_price=ref,
        )

        assert len(result["prices"]) == 3
        assert len(result["std_errors"]) == 3
        assert len(result["errors"]) == 3
        assert result["reference_price"] == ref

    def test_convergence_analysis_with_vr(self):
        """convergence_analysis with VR should yield lower std_errors."""
        mc_plain = MonteCarloEngine(n_paths=10_000, seed=MC_SEED)
        mc_vr = MonteCarloEngine(n_paths=10_000, seed=MC_SEED)
        counts = [5_000, 10_000, 50_000]

        plain = mc_plain.convergence_analysis(
            HULL_S,
            HULL_K,
            HULL_T,
            HULL_R,
            HULL_SIGMA,
            path_counts=counts,
        )
        with_vr = mc_vr.convergence_analysis(
            HULL_S,
            HULL_K,
            HULL_T,
            HULL_R,
            HULL_SIGMA,
            path_counts=counts,
            antithetic=True,
            control_variate=True,
        )

        assert with_vr["variance_reduction"] == "antithetic+control"
        # VR should reduce std_error at every path count
        for se_plain, se_vr in zip(plain["std_errors"], with_vr["std_errors"]):
            assert se_vr < se_plain

    def test_convergence_analysis_default_path_counts(self):
        """convergence_analysis uses sensible defaults when path_counts is None."""
        mc = MonteCarloEngine(n_paths=10_000, seed=MC_SEED)
        result = mc.convergence_analysis(
            HULL_S,
            HULL_K,
            HULL_T,
            HULL_R,
            HULL_SIGMA,
        )
        assert result["path_counts"] == [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
        assert len(result["prices"]) == 6

    def test_convergence_analysis_empty_path_counts(self):
        """convergence_analysis rejects empty path_counts."""
        mc = MonteCarloEngine(n_paths=10_000, seed=MC_SEED)
        with pytest.raises(ValueError, match="must not be empty"):
            mc.convergence_analysis(
                HULL_S,
                HULL_K,
                HULL_T,
                HULL_R,
                HULL_SIGMA,
                path_counts=[],
            )


# ============================================================================
# Generic price() with Variance Reduction
# ============================================================================
class TestGenericPriceVR:
    """Test price() with antithetic and control variate support (Phase 3 API)."""

    def test_simulate_gbm_antithetic_shapes(self):
        """simulate_gbm(antithetic=True) returns tuple of correct shapes."""
        mc = MonteCarloEngine(n_paths=1_000, n_steps=50, seed=MC_SEED)
        paths_pos, paths_neg = mc.simulate_gbm(
            HULL_S,
            HULL_T,
            HULL_R,
            HULL_SIGMA,
            antithetic=True,
        )
        assert paths_pos.shape == (1_000, 51)
        assert paths_neg.shape == (1_000, 51)
        np.testing.assert_array_equal(paths_pos[:, 0], HULL_S)
        np.testing.assert_array_equal(paths_neg[:, 0], HULL_S)

    def test_simulate_gbm_antithetic_mirror(self):
        """
        Antithetic paths use negated normals: log-returns sum to 2*drift.

        For single-step exact GBM:
            log(S_T^+/S_0) + log(S_T^-/S_0) = 2*(r - sigma^2/2)*T
        because the Z and -Z terms cancel exactly.
        """
        mc = MonteCarloEngine(n_paths=1_000, n_steps=1, seed=MC_SEED)
        paths_pos, paths_neg = mc.simulate_gbm(
            HULL_S,
            HULL_T,
            HULL_R,
            HULL_SIGMA,
            antithetic=True,
        )
        log_pos = np.log(paths_pos[:, -1] / HULL_S)
        log_neg = np.log(paths_neg[:, -1] / HULL_S)
        expected_sum = 2 * (HULL_R - 0.5 * HULL_SIGMA**2) * HULL_T
        np.testing.assert_allclose(log_pos + log_neg, expected_sum, atol=1e-12)

    def test_simulate_gbm_n_steps_override(self):
        """n_steps keyword overrides engine's default n_steps."""
        mc = MonteCarloEngine(n_paths=100, n_steps=252, seed=MC_SEED)
        paths = mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA, n_steps=1)
        assert paths.shape == (100, 2)

    def test_simulate_gbm_antithetic_multistep(self):
        """Antithetic paths work correctly with multiple time steps."""
        mc = MonteCarloEngine(n_paths=5_000, n_steps=100, seed=MC_SEED)
        paths_pos, paths_neg = mc.simulate_gbm(
            HULL_S,
            HULL_T,
            HULL_R,
            HULL_SIGMA,
            antithetic=True,
        )
        # Both should have correct shape
        assert paths_pos.shape == (5_000, 101)
        assert paths_neg.shape == (5_000, 101)
        # Both start at S0
        np.testing.assert_array_equal(paths_pos[:, 0], HULL_S)
        np.testing.assert_array_equal(paths_neg[:, 0], HULL_S)
        # Both should be strictly positive (exact scheme)
        assert np.all(paths_pos > 0)
        assert np.all(paths_neg > 0)

    def test_price_antithetic_unbiased(self):
        """price() with paths_anti is unbiased for call payoff."""
        mc = MonteCarloEngine(n_paths=MC_PATHS, n_steps=1, seed=MC_SEED)
        paths, paths_anti = mc.simulate_gbm(
            HULL_S,
            HULL_T,
            HULL_R,
            HULL_SIGMA,
            antithetic=True,
        )
        result = mc.price(
            lambda p: np.maximum(p[:, -1] - HULL_K, 0),
            paths,
            HULL_R,
            HULL_T,
            paths_anti=paths_anti,
        )
        bs = BlackScholesModel(sigma=HULL_SIGMA)
        analytical = bs.price(HULL_S, HULL_K, HULL_T, HULL_R, "call")
        assert abs(result.price - analytical) < max(0.05, 3 * result.std_error)
        assert result.variance_reduction == "antithetic"

    def test_price_control_fn_reduces_variance(self):
        """price() with control_fn reduces std_error vs plain MC."""
        mc_plain = MonteCarloEngine(n_paths=MC_PATHS, n_steps=1, seed=MC_SEED)
        mc_cv = MonteCarloEngine(n_paths=MC_PATHS, n_steps=1, seed=MC_SEED)

        paths_plain = mc_plain.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA)
        paths_cv = mc_cv.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA)

        plain = mc_plain.price(
            lambda p: np.maximum(p[:, -1] - HULL_K, 0),
            paths_plain,
            HULL_R,
            HULL_T,
        )

        discount = np.exp(-HULL_R * HULL_T)

        def ctrl(p: np.ndarray) -> tuple[np.ndarray, float]:
            return discount * p[:, -1], HULL_S

        cv = mc_cv.price(
            lambda p: np.maximum(p[:, -1] - HULL_K, 0),
            paths_cv,
            HULL_R,
            HULL_T,
            control_fn=ctrl,
        )

        assert cv.std_error < plain.std_error
        assert cv.variance_reduction == "control"

    def test_price_combined_vr(self):
        """price() with antithetic + control gives best variance reduction."""
        mc = MonteCarloEngine(n_paths=MC_PATHS, n_steps=1, seed=MC_SEED)
        paths, paths_anti = mc.simulate_gbm(
            HULL_S,
            HULL_T,
            HULL_R,
            HULL_SIGMA,
            antithetic=True,
        )

        discount = np.exp(-HULL_R * HULL_T)
        cv_expected = HULL_S  # S_0 * exp(-q*T) with q=0

        def ctrl(p: np.ndarray) -> tuple[np.ndarray, float]:
            return discount * p[:, -1], cv_expected

        result = mc.price(
            lambda p: np.maximum(p[:, -1] - HULL_K, 0),
            paths,
            HULL_R,
            HULL_T,
            paths_anti=paths_anti,
            control_fn=ctrl,
        )

        assert result.variance_reduction == "antithetic+control"
        bs = BlackScholesModel(sigma=HULL_SIGMA)
        analytical = bs.price(HULL_S, HULL_K, HULL_T, HULL_R, "call")
        assert abs(result.price - analytical) < max(0.05, 3 * result.std_error)

    def test_price_european_delegates_to_price(self):
        """
        price_european and simulate_gbm+price give bit-for-bit identical
        results because price_european now delegates internally.
        """
        mc1 = MonteCarloEngine(n_paths=50_000, seed=MC_SEED)
        mc2 = MonteCarloEngine(n_paths=50_000, n_steps=1, seed=MC_SEED)

        direct = mc1.price_european(
            HULL_S,
            HULL_K,
            HULL_T,
            HULL_R,
            HULL_SIGMA,
            "call",
        )
        paths = mc2.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA)
        generic = mc2.price(
            lambda p: np.maximum(p[:, -1] - HULL_K, 0),
            paths,
            HULL_R,
            HULL_T,
        )

        assert direct.price == generic.price
        assert direct.std_error == generic.std_error

    def test_price_path_dependent_payoff(self):
        """
        price() works with a path-dependent payoff (arithmetic Asian).

        The arithmetic Asian call price is bounded:
            Asian_call <= European_call
        because the average is less volatile than the terminal value.
        """
        mc = MonteCarloEngine(n_paths=MC_PATHS, n_steps=252, seed=MC_SEED)
        paths = mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA)

        # Arithmetic average (monitoring daily over the full path)
        def asian_call_payoff(p: np.ndarray) -> np.ndarray:
            avg = np.mean(p[:, 1:], axis=1)
            return np.maximum(avg - HULL_K, 0.0)

        asian = mc.price(asian_call_payoff, paths, HULL_R, HULL_T)

        bs = BlackScholesModel(sigma=HULL_SIGMA)
        european = bs.price(HULL_S, HULL_K, HULL_T, HULL_R, "call")

        assert asian.price > 0
        assert asian.price < european + 3 * asian.std_error


# ============================================================================
# 16. Quasi-Monte Carlo (Sobol)
# ============================================================================
class TestQuasiMonteCarlo:
    """QMC with scrambled Sobol sequences."""

    def test_qmc_price_unbiased(self):
        """QMC European call price is consistent with BS analytical."""
        bs = BlackScholesModel(sigma=HULL_SIGMA)
        bs_price = bs.price(HULL_S, HULL_K, HULL_T, HULL_R, "call")

        mc = MonteCarloEngine(n_paths=50_000, seed=42)
        paths = mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA, quasi=True, n_steps=1)
        result = mc.price(
            lambda p: np.maximum(p[:, -1] - HULL_K, 0.0),
            paths,
            HULL_R,
            HULL_T,
        )
        assert abs(result.price - bs_price) < 0.10

    def test_qmc_lower_std_error_than_mc(self):
        """QMC should produce lower std_error than pseudo-random MC."""
        n_paths = 16_384  # power of 2 for optimal Sobol

        mc_pseudo = MonteCarloEngine(n_paths=n_paths, seed=42)
        paths_pseudo = mc_pseudo.simulate_gbm(
            HULL_S,
            HULL_T,
            HULL_R,
            HULL_SIGMA,
            n_steps=1,
        )
        result_pseudo = mc_pseudo.price(
            lambda p: np.maximum(p[:, -1] - HULL_K, 0.0),
            paths_pseudo,
            HULL_R,
            HULL_T,
        )

        mc_qmc = MonteCarloEngine(n_paths=n_paths, seed=42)
        paths_qmc = mc_qmc.simulate_gbm(
            HULL_S,
            HULL_T,
            HULL_R,
            HULL_SIGMA,
            quasi=True,
            n_steps=1,
        )
        result_qmc = mc_qmc.price(
            lambda p: np.maximum(p[:, -1] - HULL_K, 0.0),
            paths_qmc,
            HULL_R,
            HULL_T,
        )

        # QMC std_error should be noticeably lower
        assert result_qmc.std_error < result_pseudo.std_error

    def test_qmc_paths_shape(self):
        """QMC paths have correct shape."""
        mc = MonteCarloEngine(n_paths=100, n_steps=10, seed=42)
        paths = mc.simulate_gbm(100, 1.0, 0.05, 0.20, quasi=True)
        assert paths.shape == (100, 11)
        assert np.all(paths[:, 0] == 100.0)
        assert np.all(paths > 0)

    def test_qmc_with_antithetic(self):
        """QMC + antithetic produces valid paired paths."""
        mc = MonteCarloEngine(n_paths=100, seed=42)
        paths, paths_anti = mc.simulate_gbm(
            100,
            1.0,
            0.05,
            0.20,
            quasi=True,
            n_steps=1,
            antithetic=True,
        )
        assert paths.shape == paths_anti.shape
        assert paths.shape == (100, 2)

    def test_qmc_multistep_finite(self):
        """QMC with multiple steps produces finite paths."""
        mc = MonteCarloEngine(n_paths=256, n_steps=50, seed=42)
        paths = mc.simulate_gbm(100, 1.0, 0.05, 0.20, quasi=True)
        assert np.all(np.isfinite(paths))
        assert np.all(paths > 0)


# ============================================================================
# 17. Importance Sampling
# ============================================================================
class TestImportanceSampling:
    """Importance sampling for deep OTM options."""

    def test_is_shift_atm(self):
        """ATM option: optimal shift is near zero."""
        theta = importance_sampling_shift(100, 100, 1.0, 0.05, 0.20)
        assert abs(theta) < 0.5

    def test_is_shift_deep_otm_call(self):
        """Deep OTM call (K >> S0): shift is positive (move right)."""
        theta = importance_sampling_shift(100, 200, 1.0, 0.05, 0.20)
        assert theta > 0

    def test_is_shift_deep_otm_put(self):
        """Deep OTM put (K << S0): shift is negative (move left)."""
        theta = importance_sampling_shift(100, 50, 1.0, 0.05, 0.20)
        assert theta < 0

    def test_is_shift_clamped(self):
        """Extreme shifts are clamped to [-5, 5]."""
        theta = importance_sampling_shift(100, 1000, 0.01, 0.05, 0.20)
        assert abs(theta) <= 5.0

    def test_is_likelihood_ratio_mean(self):
        """E[L(Z)] should be approximately 1 (unbiasedness condition)."""
        rng = np.random.default_rng(42)
        Z = rng.standard_normal(500_000)
        theta = 2.0
        lr = importance_sampling_likelihood(Z, theta)
        # E[exp(-theta*(Z+theta) + theta^2/2)] = E[exp(-theta*Z - theta^2/2)]
        # = exp(-theta^2/2) * E[exp(-theta*Z)]
        # = exp(-theta^2/2) * exp(theta^2/2) = 1
        assert abs(np.mean(lr) - 1.0) < 0.01

    def test_is_price_unbiased_atm(self):
        """IS should give correct price for ATM options."""
        bs = BlackScholesModel(sigma=HULL_SIGMA)
        bs_price = bs.price(HULL_S, HULL_K, HULL_T, HULL_R, "call")

        mc = MonteCarloEngine(n_paths=200_000, seed=42)
        result = mc.price_european(
            HULL_S,
            HULL_K,
            HULL_T,
            HULL_R,
            HULL_SIGMA,
            "call",
            importance_sampling=True,
        )
        assert abs(result.price - bs_price) < 4 * result.std_error

    def test_is_deep_otm_variance_reduction(self):
        """IS should dramatically reduce variance for deep OTM options."""
        S, K, T, r, sigma = 100, 150, 0.5, 0.05, 0.20
        n_paths = 100_000

        mc_plain = MonteCarloEngine(n_paths=n_paths, seed=42)
        result_plain = mc_plain.price_european(S, K, T, r, sigma, "call")

        mc_is = MonteCarloEngine(n_paths=n_paths, seed=42)
        result_is = mc_is.price_european(
            S,
            K,
            T,
            r,
            sigma,
            "call",
            importance_sampling=True,
        )

        # IS should have much lower std_error for deep OTM
        assert result_is.std_error < result_plain.std_error

    def test_is_deep_otm_put_unbiased(self):
        """IS for deep OTM put should match BS price."""
        S, K, T, r, sigma = 100, 60, 1.0, 0.05, 0.20
        bs = BlackScholesModel(sigma=sigma)
        bs_price = bs.price(S, K, T, r, "put")

        mc = MonteCarloEngine(n_paths=200_000, seed=42)
        result = mc.price_european(
            S,
            K,
            T,
            r,
            sigma,
            "put",
            importance_sampling=True,
        )
        assert abs(result.price - bs_price) < 4 * result.std_error

    def test_is_incompatible_with_antithetic(self):
        """IS and antithetic should raise ValueError."""
        mc = MonteCarloEngine(n_paths=1000, seed=42)
        with pytest.raises(ValueError, match="importance_sampling and antithetic"):
            mc.price_european(
                100,
                100,
                1.0,
                0.05,
                0.20,
                "call",
                importance_sampling=True,
                antithetic=True,
            )

    def test_is_with_control_variate(self):
        """IS + control variate should produce valid results."""
        bs = BlackScholesModel(sigma=HULL_SIGMA)
        bs_price = bs.price(HULL_S, HULL_K, HULL_T, HULL_R, "call")

        mc = MonteCarloEngine(n_paths=100_000, seed=42)
        result = mc.price_european(
            HULL_S,
            HULL_K,
            HULL_T,
            HULL_R,
            HULL_SIGMA,
            "call",
            importance_sampling=True,
            control_variate=True,
        )
        assert abs(result.price - bs_price) < 4 * result.std_error
        assert result.variance_reduction == "importance+control"

    def test_is_vr_label(self):
        """IS variance reduction label is correct."""
        mc = MonteCarloEngine(n_paths=1000, seed=42)
        result = mc.price_european(
            100,
            100,
            1.0,
            0.05,
            0.20,
            "call",
            importance_sampling=True,
        )
        assert result.variance_reduction == "importance"


# ============================================================================
# 18. Euler Absorption at Zero
# ============================================================================
class TestEulerAbsorption:
    """Euler scheme with absorption at zero."""

    def test_absorb_prevents_negative_prices(self):
        """With high vol and large dt, Euler without absorb can go negative."""
        # High vol + few steps maximizes chance of negative S
        mc = MonteCarloEngine(n_paths=10_000, n_steps=10, seed=42)
        paths = mc.simulate_gbm(
            100,
            1.0,
            0.05,
            2.0,
            scheme="euler",
            absorb=True,
        )
        assert np.all(paths >= 0.0)

    def test_absorb_no_effect_on_exact_scheme(self):
        """Exact scheme always produces positive paths; absorb is irrelevant."""
        mc = MonteCarloEngine(n_paths=1000, n_steps=10, seed=42)
        paths1 = mc.simulate_gbm(100, 1.0, 0.05, 0.20, scheme="exact")
        mc.reset()
        paths2 = mc.simulate_gbm(100, 1.0, 0.05, 0.20, scheme="exact", absorb=True)
        np.testing.assert_array_equal(paths1, paths2)

    def test_absorb_false_can_produce_negative(self):
        """Without absorb, high-vol Euler may produce negative prices."""
        mc = MonteCarloEngine(n_paths=50_000, n_steps=5, seed=42)
        paths = mc.simulate_gbm(
            100,
            1.0,
            0.05,
            3.0,
            scheme="euler",
            absorb=False,
        )
        # With sigma=3.0 and 5 steps, very likely to get some negatives
        assert np.any(paths < 0)

    def test_absorb_preserves_initial_price(self):
        """S_0 column is unchanged with absorb."""
        mc = MonteCarloEngine(n_paths=100, n_steps=10, seed=42)
        paths = mc.simulate_gbm(
            100,
            1.0,
            0.05,
            1.0,
            scheme="euler",
            absorb=True,
        )
        assert np.all(paths[:, 0] == 100.0)


# ============================================================================
# 19. Batch Pricing
# ============================================================================
class TestBatchPricing:
    """Batch pricing of multiple payoffs on shared paths."""

    def test_batch_matches_individual(self):
        """Batch results must match individual pricing exactly."""
        mc = MonteCarloEngine(n_paths=50_000, seed=42)
        paths = mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA, n_steps=1)

        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        payoff_fns = [lambda p, K=K: np.maximum(p[:, -1] - K, 0.0) for K in strikes]

        # Batch
        batch_results = mc.price_batch(payoff_fns, paths, HULL_R, HULL_T)

        # Individual (on SAME paths — no re-simulation)
        individual_results = [mc.price(pf, paths, HULL_R, HULL_T) for pf in payoff_fns]

        assert len(batch_results) == len(strikes)
        for br, ir in zip(batch_results, individual_results):
            assert br.price == ir.price
            assert br.std_error == ir.std_error

    def test_batch_with_antithetic(self):
        """Batch pricing with shared antithetic paths."""
        mc = MonteCarloEngine(n_paths=50_000, seed=42)
        paths, paths_anti = mc.simulate_gbm(
            HULL_S,
            HULL_T,
            HULL_R,
            HULL_SIGMA,
            n_steps=1,
            antithetic=True,
        )

        payoff_fns = [
            lambda p: np.maximum(p[:, -1] - 95, 0.0),
            lambda p: np.maximum(p[:, -1] - 100, 0.0),
            lambda p: np.maximum(p[:, -1] - 105, 0.0),
        ]

        results = mc.price_batch(
            payoff_fns,
            paths,
            HULL_R,
            HULL_T,
            paths_anti=paths_anti,
        )
        assert len(results) == 3
        assert all(r.variance_reduction == "antithetic" for r in results)
        # Lower strike -> higher price
        assert results[0].price > results[1].price > results[2].price

    def test_batch_prices_correct_vs_bs(self):
        """Batch prices are consistent with BS analytical."""
        bs = BlackScholesModel(sigma=HULL_SIGMA)
        mc = MonteCarloEngine(n_paths=200_000, seed=42)
        paths = mc.simulate_gbm(HULL_S, HULL_T, HULL_R, HULL_SIGMA, n_steps=1)

        strikes = [90.0, 100.0, 110.0]
        payoff_fns = [lambda p, K=K: np.maximum(p[:, -1] - K, 0.0) for K in strikes]

        results = mc.price_batch(payoff_fns, paths, HULL_R, HULL_T)

        for K, result in zip(strikes, results):
            bs_price = bs.price(HULL_S, K, HULL_T, HULL_R, "call")
            assert abs(result.price - bs_price) < 4 * result.std_error

    def test_batch_empty_list(self):
        """Empty payoff list returns empty results."""
        mc = MonteCarloEngine(n_paths=100, seed=42)
        paths = mc.simulate_gbm(100, 1.0, 0.05, 0.20, n_steps=1)
        results = mc.price_batch([], paths, 0.05, 1.0)
        assert results == []
