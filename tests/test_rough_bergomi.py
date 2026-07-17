"""
Exotic Option Pricer — tests/test_rough_bergomi.py

Phase 5 test suite: rough Bergomi (Bayer-Friz-Gatheral 2016).

rBergomi has no external benchmark (no QuantLib support): validation is by
two mathematically independent implementations that must agree (exact
Cholesky vs hybrid scheme), exact analytical anchors, and scaling laws —
the same standard used in the literature. Covers:

- Parameter validation, dunder methods (model parameters only — MC
  resolution is a numerical setting, not part of model identity)
- Volterra covariance quadrature vs exact anchors: E[V_t^2] = t^(2H),
  H = 1/2 -> Brownian covariance min(s, t)
- Exact Cholesky simulation: sample moments vs closed forms
- Hybrid scheme: exact first-interval variance, internal variance formula,
  H = 1/2 special branch (V = B exactly)
- Hybrid vs Cholesky cross-validation on PRICES (the central Phase 5 test:
  an off-by-one in the hybrid convolution produces plausible prices with a
  wrong skew, and only the exact reference catches it)
- Variance process: E[v_t] = xi0 and E[v_t^2] = xi0^2 exp(eta^2 t^(2H))
- Exact left-point martingale (small and large n — the property is exact
  in n, so failure at small n means the scheme is wrong)
- Black-Scholes limit (eta -> 0): deterministic collapse of the
  conditional estimator at rho = 0 (tolerance 1e-8), Greeks included
- Conditional estimator: agreement with plain payoff MC, variance
  reduction vs plain MC, exact put-call parity, exact strike monotonicity
  on shared draws, no-arbitrage bounds
- ABC Greeks and model_greeks: parity relations, chain rule
  vega = dV/dxi0 * 2 sqrt(xi0), CRN consistency vs brute-force bumping
- The defining property: ATM skew power law psi(T) ~ T^(H - 1/2), and the
  structural contrast with Heston (Phase 4 SPX calibration): Heston's
  short-maturity skew SATURATES while rBergomi's keeps the power law
- Phase 3 exotics priced on rBergomi paths without modification
- Hurst roundtrip: H re-estimated from simulated log-variance increments
- Property-based tests (Hypothesis)

References
----------
.. [1] Bayer, Friz, Gatheral (2016). Quantitative Finance 16(6), 887-904.
.. [2] Bennedsen, Lunde, Pakkanen (2017). Finance & Stochastics 21(4).
.. [3] McCrickerd, Pakkanen (2018). Quantitative Finance 18(11), 1877-1886.
.. [4] Gatheral, Jaisson, Rosenbaum (2018). Quantitative Finance 18(6).
"""

import time
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from exotic_option_pricer.engines.monte_carlo import MonteCarloEngine
from exotic_option_pricer.instruments.asian import AsianOption
from exotic_option_pricer.instruments.barrier import BarrierOption
from exotic_option_pricer.models.black_scholes import BlackScholesModel
from exotic_option_pricer.models.heston import HestonModel
from exotic_option_pricer.models.rough_bergomi import RoughBergomiModel

# ──────────────────────────────────────────────
# Shared parameters
# ──────────────────────────────────────────────

S0 = 100.0

# Canonical equity-style rough-vol set (Bayer-Friz-Gatheral 2016 ballpark)
CANONICAL = dict(xi0=0.04, eta=1.9, H=0.1, rho=-0.9)

# Phase 4 SPX calibration result (2026-06-11): the Heston parameters that
# chased the short-dated SPX skew (kappa, xi extreme) and still missed the
# short wings — the empirical motivation for Phase 5.
HESTON_SPX = dict(v0=0.033, kappa=13.0, theta=0.048, xi=3.0, rho=-0.64)


def make_canonical(**overrides) -> RoughBergomiModel:
    params = {**CANONICAL, **overrides}
    return RoughBergomiModel(**params)


def make_heston_spx() -> HestonModel:
    """SPX-calibrated Heston, suppressing the expected Feller warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return HestonModel(**HESTON_SPX)


# ──────────────────────────────────────────────
# Validation and dunder methods
# ──────────────────────────────────────────────


class TestValidation:
    def test_valid_construction(self):
        m = make_canonical()
        assert m.xi0 == 0.04 and m.eta == 1.9
        assert m.H == 0.1 and m.rho_sv == -0.9

    @pytest.mark.parametrize("bad", [0.0, -0.04, np.nan, np.inf])
    def test_invalid_xi0(self, bad):
        with pytest.raises(ValueError):
            RoughBergomiModel(xi0=bad, eta=1.9, H=0.1, rho=-0.9)

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
    def test_invalid_eta(self, bad):
        with pytest.raises(ValueError):
            RoughBergomiModel(xi0=0.04, eta=bad, H=0.1, rho=-0.9)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 0.51, 0.7, np.nan])
    def test_invalid_H(self, bad):
        with pytest.raises(ValueError):
            RoughBergomiModel(xi0=0.04, eta=1.9, H=bad, rho=-0.9)

    def test_H_boundary_half_is_valid(self):
        RoughBergomiModel(xi0=0.04, eta=1.9, H=0.5, rho=-0.9)

    @pytest.mark.parametrize("bad", [-1.0, 1.0, -1.5, np.nan])
    def test_invalid_rho(self, bad):
        with pytest.raises(ValueError):
            RoughBergomiModel(xi0=0.04, eta=1.9, H=0.1, rho=bad)

    def test_invalid_mc_settings(self):
        with pytest.raises(ValueError):
            make_canonical(mc_paths=1)
        with pytest.raises(ValueError):
            make_canonical(mc_steps=0)

    def test_market_input_validation_inherited_from_abc(self):
        m = make_canonical(mc_paths=100, mc_steps=4)
        with pytest.raises(ValueError):
            m.price(-100, 100, 1.0, 0.05)
        with pytest.raises(ValueError):
            m.price(100, -1.0, 1.0, 0.05)
        with pytest.raises(ValueError):
            m.price(100, 100, 0.0, 0.05)
        with pytest.raises(ValueError):
            m.price(100, 100, 1.0, 0.05, option_type="straddle")

    def test_option_type_normalization(self):
        m = make_canonical(mc_paths=2_000, mc_steps=8)
        assert m.price(100, 100, 1.0, 0.05, "C") == m.price(100, 100, 1.0, 0.05, "call")
        assert m.price(100, 100, 1.0, 0.05, " Put ") == m.price(100, 100, 1.0, 0.05, "put")

    def test_simulate_validation(self):
        m = make_canonical(mc_paths=100, mc_steps=4)
        with pytest.raises(ValueError):
            m.simulate(-1.0, 1.0, 0.05)
        with pytest.raises(ValueError):
            m.simulate(100, 0.0, 0.05)
        with pytest.raises(ValueError):
            m.simulate(100, 1.0, np.nan)
        with pytest.raises(ValueError):
            m.simulate(100, 1.0, 0.05, n_paths=1)
        with pytest.raises(ValueError):
            m.simulate(100, 1.0, 0.05, n_steps=0)
        with pytest.raises(ValueError):
            m.simulate(100, 1.0, 0.05, method="euler")

    def test_repr_eq_hash_use_model_parameters_only(self):
        # D1: two instances with different MC resolution are the SAME model
        a = make_canonical(mc_paths=1000, mc_steps=16, seed=1)
        b = make_canonical(mc_paths=2000, mc_steps=32, seed=99)
        c = make_canonical(H=0.3)
        assert a == b and hash(a) == hash(b)
        assert a != c
        assert a != "not a model"  # NotImplemented -> False
        rep = repr(a)
        assert "mc_paths" not in rep and "seed" not in rep
        assert "xi0=0.04" in rep and "H=0.1" in rep


# ──────────────────────────────────────────────
# Volterra covariance: quadrature vs exact anchors
# ──────────────────────────────────────────────


class TestVolterraCovariance:
    @pytest.mark.parametrize("H", [0.05, 0.1, 0.3, 0.45])
    @pytest.mark.parametrize("t", [0.05, 0.5, 1.0, 2.0])
    def test_diagonal_exact(self, H, t):
        """E[V_t^2] = 2H int_0^t (t-u)^(2H-1) du = t^(2H), exact."""
        m = make_canonical(H=H)
        assert m._volterra_covariance(t, t) == pytest.approx(t ** (2 * H), abs=1e-10)

    @pytest.mark.parametrize("s,t", [(0.3, 0.7), (0.5, 0.5), (1.0, 0.25)])
    def test_H_half_is_brownian(self, s, t):
        """H = 1/2: kernel identically 1 -> E[V_s V_t] = min(s, t)."""
        m = make_canonical(H=0.5)
        assert m._volterra_covariance(s, t) == pytest.approx(min(s, t), abs=1e-10)

    def test_symmetry(self):
        m = make_canonical()
        assert m._volterra_covariance(0.3, 0.9) == pytest.approx(
            m._volterra_covariance(0.9, 0.3), abs=1e-14
        )

    def test_zero_time(self):
        """V_0 = 0 a.s. -> zero covariance with anything."""
        m = make_canonical()
        assert m._volterra_covariance(0.0, 1.0) == 0.0
        assert m._volterra_covariance(1.0, 0.0) == 0.0

    def test_cauchy_schwarz(self):
        m = make_canonical()
        c = m._volterra_covariance(0.4, 1.2)
        bound = np.sqrt(0.4 ** (2 * m.H) * 1.2 ** (2 * m.H))
        assert 0 < c < bound


# ──────────────────────────────────────────────
# Exact Cholesky simulation: sample moments vs closed forms
# ──────────────────────────────────────────────


class TestCholeskyMoments:
    N_PATHS = 400_000
    N_STEPS = 8
    T = 1.0

    @pytest.fixture(scope="class")
    def draws(self):
        m = make_canonical()
        rng = np.random.default_rng(123)
        V, dB = m._simulate_volterra_cholesky(self.T, self.N_STEPS, self.N_PATHS, rng)
        return m, V, dB.cumsum(axis=1)  # (model, V, B)

    def test_volterra_variance(self, draws):
        """Var[V_{t_i}] = t_i^(2H). SE of a Gaussian variance estimator is
        sqrt(2/N) * true variance."""
        m, V, _ = draws
        t = self.T / self.N_STEPS * np.arange(1, self.N_STEPS + 1)
        sample = V.var(axis=0, ddof=1)
        exact = t ** (2 * m.H)
        se = np.sqrt(2.0 / self.N_PATHS) * exact
        assert np.all(np.abs(sample - exact) < 3 * se)

    def test_brownian_variance(self, draws):
        _, _, B = draws
        t = self.T / self.N_STEPS * np.arange(1, self.N_STEPS + 1)
        sample = B.var(axis=0, ddof=1)
        se = np.sqrt(2.0 / self.N_PATHS) * t
        assert np.all(np.abs(sample - t) < 3 * se)

    def test_zero_means(self, draws):
        m, V, B = draws
        t = self.T / self.N_STEPS * np.arange(1, self.N_STEPS + 1)
        se_v = t**m.H / np.sqrt(self.N_PATHS)
        se_b = np.sqrt(t / self.N_PATHS)
        assert np.all(np.abs(V.mean(axis=0)) < 3 * se_v)
        assert np.all(np.abs(B.mean(axis=0)) < 3 * se_b)

    def test_volterra_brownian_covariance(self, draws):
        """E[V_t B_s] = sqrt(2H)/(H+1/2) (t^(H+1/2) - (t-min(s,t))^(H+1/2)),
        tested across the (t, s) grid including s > t, s = t, s < t."""
        m, V, B = draws
        t = self.T / self.N_STEPS * np.arange(1, self.N_STEPS + 1)
        Hp = m.H + 0.5
        for i in [0, 3, 7]:  # t index
            for j in [0, 3, 7]:  # s index
                prod = V[:, i] * B[:, j]
                exact = (np.sqrt(2 * m.H) / Hp) * (
                    t[i] ** Hp - max(t[i] - min(t[i], t[j]), 0.0) ** Hp
                )
                se = prod.std(ddof=1) / np.sqrt(self.N_PATHS)
                assert abs(prod.mean() - exact) < 3 * se, (i, j)


# ──────────────────────────────────────────────
# Hybrid scheme: exact local variance and internal consistency
# ──────────────────────────────────────────────


def _hybrid_theoretical_variance(H: float, T: float, n: int) -> np.ndarray:
    """
    The hybrid scheme's OWN variance of V_{t_i}: the exact W1 term plus the
    kernel-evaluation sum (W1_i is independent of dB_{i-k}, k >= 2, and the
    dB's are i.i.d.):

        Var[V_{t_i}] = 2H (dt^(2H)/(2H) + sum_{k=2}^{i} G_k^2 dt)
    """
    dt = T / n
    gamma = H - 0.5
    k = np.arange(2, n + 1, dtype=np.float64)
    b = ((k ** (gamma + 1) - (k - 1) ** (gamma + 1)) / (gamma + 1)) ** (1 / gamma)
    G = (b * dt) ** gamma
    return 2 * H * (dt ** (2 * H) / (2 * H) + np.concatenate(([0.0], np.cumsum(G * G * dt))))


class TestHybridScheme:
    N_PATHS = 400_000
    N_STEPS = 16
    T = 1.0

    @pytest.mark.parametrize("H", [0.1, 0.3])
    def test_sample_variance_matches_scheme_variance(self, H):
        """Sample variance vs the scheme's own closed-form variance (3 SE):
        catches wrong kernel weights, missing dt in G_k, missing sqrt(2H)."""
        m = make_canonical(H=H)
        rng = np.random.default_rng(7)
        V, _ = m._simulate_volterra_hybrid(self.T, self.N_STEPS, self.N_PATHS, rng)
        theo = _hybrid_theoretical_variance(H, self.T, self.N_STEPS)
        sample = V.var(axis=0, ddof=1)
        se = np.sqrt(2.0 / self.N_PATHS) * theo
        assert np.all(np.abs(sample - theo) < 3 * se)

    @pytest.mark.parametrize("H", [0.1, 0.3])
    def test_scheme_variance_close_to_exact(self, H):
        """The hybrid variance approximates the exact t^(2H) to a few
        permille at n = 16 (measured: |rel| < 1e-3 for H in {0.1, 0.3}) —
        the accuracy claim of BLP Prop. 2.8's optimal evaluation points."""
        theo = _hybrid_theoretical_variance(H, self.T, self.N_STEPS)
        t = self.T / self.N_STEPS * np.arange(1, self.N_STEPS + 1)
        rel = theo / t ** (2 * H) - 1.0
        assert np.all(np.abs(rel) < 5e-3)

    def test_first_step_variance_is_exact(self):
        """V_{t_1} is the bare W1 term — exact: Var = dt^(2H)."""
        m = make_canonical()
        rng = np.random.default_rng(11)
        V, _ = m._simulate_volterra_hybrid(self.T, self.N_STEPS, self.N_PATHS, rng)
        dt = self.T / self.N_STEPS
        exact = dt ** (2 * m.H)
        se = np.sqrt(2.0 / self.N_PATHS) * exact
        assert abs(V[:, 0].var(ddof=1) - exact) < 3 * se

    def test_local_pair_correlation(self):
        """Corr[dB_j, W1_{j+1}] = Cov / sqrt(Var dB * Var W1) =
        sqrt(2H) / (H + 1/2) — forgetting to correlate the local pair is
        classic error #3 and shows up exactly here."""
        m = make_canonical()
        rng = np.random.default_rng(13)
        V, dB = m._simulate_volterra_hybrid(self.T, 1, self.N_PATHS, rng)
        # With n = 1, V = sqrt(2H) * W1 exactly
        W1 = V[:, 0] / np.sqrt(2 * m.H)
        expected = np.sqrt(2 * m.H) / (m.H + 0.5)
        sample = np.corrcoef(dB[:, 0], W1)[0, 1]
        assert sample == pytest.approx(expected, abs=3.0 / np.sqrt(self.N_PATHS))

    def test_H_half_special_branch_exact(self):
        """H = 1/2: kernel = 1 -> V = B = cumsum(dB), exact array equality
        (the general formula hits b_k^(1/0) = 1^inf, hence the branch)."""
        m = make_canonical(H=0.5)
        rng = np.random.default_rng(17)
        V, dB = m._simulate_volterra_hybrid(1.0, 32, 1000, rng)
        np.testing.assert_array_equal(V, np.cumsum(dB, axis=1))

    def test_H_half_cholesky_degenerate_covariance(self):
        """At H = 1/2 the joint (V, B) covariance is exactly singular
        (V = B): the documented 1e-12 jitter must keep the factorization
        alive and return V ~= B up to the jitter noise."""
        m = make_canonical(H=0.5)
        rng = np.random.default_rng(19)
        V, dB = m._simulate_volterra_cholesky(1.0, 16, 2000, rng)
        assert np.max(np.abs(V - np.cumsum(dB, axis=1))) < 1e-4


# ──────────────────────────────────────────────
# Hybrid vs exact Cholesky on prices — the central Phase 5 cross-validation
# ──────────────────────────────────────────────


class TestHybridVsCholesky:
    """
    D5: with no external benchmark, the two mathematically independent
    Volterra implementations must produce the same prices. This is the only
    test that catches an off-by-one in the hybrid convolution (error #2):
    the covariance V-B shifts, the skew is wrong, but prices stay plausible.

    Independent seeds so the standard errors combine as independent.
    """

    N_PATHS = 65_536
    N_STEPS = 32

    @pytest.mark.parametrize("H", [0.1, 0.3])
    def test_prices_agree(self, H):
        m = make_canonical(H=H)
        for K in [80.0, 90.0, 100.0, 110.0, 120.0]:
            p_h, se_h = m.price_european_conditional(
                S0,
                K,
                1.0,
                0.0,
                n_paths=self.N_PATHS,
                n_steps=self.N_STEPS,
                method="hybrid",
                seed=101,
            )
            p_c, se_c = m.price_european_conditional(
                S0,
                K,
                1.0,
                0.0,
                n_paths=self.N_PATHS,
                n_steps=self.N_STEPS,
                method="cholesky",
                seed=202,
            )
            z = (p_h - p_c) / np.hypot(se_h, se_c)
            assert abs(z) < 3.0, f"K={K}: hybrid={p_h:.4f} chol={p_c:.4f} z={z:.2f}"

    def test_martingale_cholesky(self):
        """The exact method must satisfy the exact left-point martingale."""
        m = make_canonical()
        paths = m.simulate(
            S0, 1.0, 0.03, 0.01, n_paths=100_000, n_steps=32, method="cholesky", seed=31
        )
        disc = np.exp(-(0.03 - 0.01) * 1.0) * paths[:, -1]
        se = disc.std(ddof=1) / np.sqrt(len(disc))
        assert abs(disc.mean() - S0) < 3 * se


# ──────────────────────────────────────────────
# Variance process moments
# ──────────────────────────────────────────────


class TestVarianceProcess:
    def test_mean_is_xi0(self):
        """E[v_t] = xi0 for every t (exact lognormal mean correction) —
        catches a missing sqrt(2H) or a missing -eta^2/2 t^(2H)."""
        m = make_canonical(seed=3)
        _, v = m.simulate(S0, 1.0, 0.0, n_paths=200_000, n_steps=64, return_variance=True)
        for i in [1, 16, 32, 64]:
            col = v[:, i]
            se = col.std(ddof=1) / np.sqrt(len(col))
            assert abs(col.mean() - m.xi0) < 3 * se, f"step {i}"

    def test_second_moment(self):
        """E[v_t^2] = xi0^2 exp(eta^2 t^(2H)). Tested at eta = 1.0 to keep
        the kurtosis of v^2 (hence the SE of its sample mean) sane."""
        m = make_canonical(eta=1.0, seed=5)
        _, v = m.simulate(S0, 1.0, 0.0, n_paths=400_000, n_steps=16, return_variance=True)
        t_grid = np.arange(1, 17) / 16.0
        for i in [4, 8, 16]:
            sq = v[:, i] ** 2
            exact = m.xi0**2 * np.exp(m.eta**2 * t_grid[i - 1] ** (2 * m.H))
            se = sq.std(ddof=1) / np.sqrt(len(sq))
            assert abs(sq.mean() - exact) < 3 * se, f"step {i}"

    def test_variance_strictly_positive(self):
        """v > 0 by construction (exponential) — no absorption branches."""
        m = make_canonical(seed=7)
        _, v = m.simulate(S0, 1.0, 0.0, n_paths=50_000, n_steps=64, return_variance=True)
        assert np.all(v > 0)
        assert np.all(v[:, 0] == m.xi0)


# ──────────────────────────────────────────────
# Exact left-point martingale
# ──────────────────────────────────────────────


class TestMartingale:
    """
    E[e^{-(r-q)T} S_T] = S0 exactly for ANY n (§3.3 of the spec): the
    left-point exponent is conditionally Gaussian with mean -v dt/2 and
    variance v dt. Tested at n = 50 AND n = 500 — if it fails at small n,
    the scheme (not the resolution) is wrong. No Andersen-style correction
    exists or is needed.
    """

    @pytest.mark.parametrize("n_steps,n_paths", [(50, 200_000), (500, 50_000)])
    def test_martingale(self, n_steps, n_paths):
        m = make_canonical(seed=7)
        paths = m.simulate(S0, 1.0, 0.03, 0.01, n_paths=n_paths, n_steps=n_steps)
        disc = np.exp(-(0.03 - 0.01) * 1.0) * paths[:, -1]
        se = disc.std(ddof=1) / np.sqrt(len(disc))
        assert abs(disc.mean() - S0) < 3 * se

    def test_martingale_H_half(self):
        """H = 1/2 branch: classical lognormal-vol pipeline check."""
        m = make_canonical(H=0.5, seed=9)
        paths = m.simulate(S0, 1.0, 0.05, 0.0, n_paths=200_000, n_steps=50)
        disc = np.exp(-0.05) * paths[:, -1]
        se = disc.std(ddof=1) / np.sqrt(len(disc))
        assert abs(disc.mean() - S0) < 3 * se

    def test_conditional_estimator_preserves_forward(self):
        """E[e^{-(r-q)T} S_cond] = S0 exactly (tower over the same
        left-point steps) — this is what makes put-by-parity unbiased."""
        m = make_canonical(seed=11)
        A, int_var = m._conditional_draws(1.0, n_paths=200_000, n_steps=50)
        rho = m.rho_sv
        s_cond = S0 * np.exp(rho * A - 0.5 * rho * rho * int_var)  # (r-q) discounted
        se = s_cond.std(ddof=1) / np.sqrt(len(s_cond))
        assert abs(s_cond.mean() - S0) < 3 * se


# ──────────────────────────────────────────────
# Black-Scholes limit (eta -> 0) — cross-validation with Phase 1
# ──────────────────────────────────────────────


class TestBlackScholesLimit:
    """
    eta -> 0 makes v = xi0 deterministic. At rho = 0 the conditional
    estimator is then DETERMINISTIC (S_cond and Sigma are path-independent)
    and must return the Black-Scholes price with sigma = sqrt(xi0) to 1e-8
    with ~zero standard error. At rho != 0 the estimator is a genuine
    mixture (S_cond carries rho * A noise) — unbiased, tested at 3 SE.
    """

    BS = BlackScholesModel(0.2)

    def make_limit(self, rho=0.0):
        return RoughBergomiModel(xi0=0.04, eta=1e-8, H=0.1, rho=rho, mc_paths=50_000, mc_steps=64)

    @pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
    @pytest.mark.parametrize("opt", ["call", "put"])
    def test_price_deterministic_at_rho_zero(self, K, opt):
        m = self.make_limit()
        p, se = m.price_european_conditional(S0, K, 1.0, 0.05, opt, 0.01)
        assert p == pytest.approx(self.BS.price(S0, K, 1.0, 0.05, opt, 0.01), abs=1e-8)
        assert se < 1e-8

    def test_price_with_correlation_matches_bs(self):
        """rho = -0.9, eta ~ 0: mixture over lognormal S_cond still
        integrates to the BS price."""
        m = self.make_limit(rho=-0.9)
        p, se = m.price_european_conditional(S0, 100.0, 1.0, 0.05, n_paths=200_000, n_steps=64)
        assert abs(p - self.BS.price(S0, 100.0, 1.0, 0.05)) < 3 * max(se, 1e-12)

    def test_plain_mc_matches_bs(self):
        m = self.make_limit(rho=-0.9)
        res = m.price_mc(S0, 100.0, 1.0, 0.05, n_paths=200_000, n_steps=64)
        assert abs(res.price - self.BS.price(S0, 100.0, 1.0, 0.05)) < 3 * res.std_error

    @pytest.mark.parametrize(
        "greek,rtol",
        [
            ("delta", 1e-4),
            ("gamma", 1e-4),
            ("vega", 1e-5),
            ("theta", 1e-3),
            ("rho", 1e-5),
        ],
    )
    def test_greeks_collapse_to_bs(self, greek, rtol):
        """All ABC Greeks vs Phase 1 closed forms (measured relative errors
        ~1e-6 to 9e-6; tolerances cover the FD truncation). Note vega:
        dV/d(sqrt(xi0)) IS the entire BS vega here, because xi0 is the
        whole (flat) forward-variance curve."""
        m = self.make_limit()
        got = getattr(m, greek)(S0, 100.0, 1.0, 0.05, "call", 0.01)
        want = getattr(self.BS, greek)(S0, 100.0, 1.0, 0.05, "call", 0.01)
        assert got == pytest.approx(want, rel=rtol)


# ──────────────────────────────────────────────
# Conditional estimator vs plain Monte Carlo
# ──────────────────────────────────────────────


class TestConditionalEstimator:
    def test_agrees_with_plain_mc(self):
        """Same discretization, independent draws: |z| < 3."""
        m = make_canonical()
        p_c, se_c = m.price_european_conditional(
            S0, 100.0, 1.0, 0.0, n_paths=100_000, n_steps=100, seed=5
        )
        res = m.price_mc(S0, 100.0, 1.0, 0.0, n_paths=100_000, n_steps=100, seed=9)
        z = (p_c - res.price) / np.hypot(se_c, res.std_error)
        assert abs(z) < 3.0

    def test_put_plain_mc_antithetic_agrees(self):
        """price_mc put branch with antithetic pairing vs the conditional
        put on an independent seed."""
        m = make_canonical()
        res = m.price_mc(
            S0, 110.0, 1.0, 0.02, "put", 0.01, n_paths=50_000, n_steps=64, antithetic=True, seed=13
        )
        p_c, se_c = m.price_european_conditional(
            S0, 110.0, 1.0, 0.02, "put", 0.01, n_paths=100_000, n_steps=64, seed=21
        )
        assert res.variance_reduction == "antithetic"
        z = (res.price - p_c) / np.hypot(res.std_error, se_c)
        assert abs(z) < 3.0

    @pytest.mark.parametrize("K,max_ratio", [(80.0, 0.40), (100.0, 0.55)])
    def test_variance_reduction_vs_plain(self, K, max_ratio):
        """Conditional + S_cond control variate vs plain payoff MC.
        Measured at the canonical set: ratio ~0.29 (K=80), ~0.47 (K=100).
        The conditional step alone gives only ~0.77 at rho = -0.9 (the
        B_perp share of the variance is 1 - rho^2 = 0.19); the exact-mean
        control variate on S_cond provides the rest — see _control_adjust."""
        m = make_canonical()
        _, se_c = m.price_european_conditional(
            S0, K, 1.0, 0.0, n_paths=100_000, n_steps=100, seed=5
        )
        res = m.price_mc(S0, K, 1.0, 0.0, n_paths=100_000, n_steps=100, seed=9)
        assert se_c / res.std_error < max_ratio

    def test_put_call_parity_exact(self):
        """C - P = S e^{-qT} - K e^{-rT} EXACTLY (puts are priced by
        parity per path; the control variate preserves it because the
        optimal beta is identical for call and put values)."""
        m = make_canonical(mc_paths=20_000, mc_steps=32)
        c = m.price(S0, 110.0, 0.75, 0.04, "call", 0.02)
        p = m.price(S0, 110.0, 0.75, 0.04, "put", 0.02)
        parity = S0 * np.exp(-0.02 * 0.75) - 110.0 * np.exp(-0.04 * 0.75)
        assert c - p == pytest.approx(parity, abs=1e-10)

    def test_strike_monotonicity_exact_on_shared_draws(self):
        """Without the control variate the per-path values are pointwise
        decreasing in K, so the sample means are monotone EXACTLY (same
        seed -> same draws). The in-sample beta reintroduces O(SE) noise
        across strikes, hence control_variate=False here."""
        m = make_canonical(mc_paths=20_000, mc_steps=32)
        strikes = [70.0, 85.0, 100.0, 115.0, 130.0]
        prices = [
            m.price_european_conditional(S0, K, 1.0, 0.03, control_variate=False)[0]
            for K in strikes
        ]
        assert np.all(np.diff(prices) < 0)

    def test_no_arbitrage_bounds(self):
        m = make_canonical(mc_paths=50_000, mc_steps=64)
        for K in [60.0, 100.0, 140.0]:
            p, se = m.price_european_conditional(S0, K, 1.0, 0.03, "call", 0.01)
            lb = max(S0 * np.exp(-0.01) - K * np.exp(-0.03), 0.0)
            ub = S0 * np.exp(-0.01)
            assert lb - 4 * se <= p <= ub + 4 * se

    def test_zero_strike(self):
        """K = 0: call = prepaid forward (deterministic), put worthless."""
        m = make_canonical(mc_paths=2_000, mc_steps=8)
        p, se = m.price_european_conditional(S0, 0.0, 2.0, 0.05, "call", 0.03)
        assert p == pytest.approx(S0 * np.exp(-0.03 * 2.0), abs=1e-12)
        assert se == 0.0
        assert m.price(S0, 0.0, 2.0, 0.05, "put", 0.03) == 0.0

    def test_price_deterministic_given_instance(self):
        m = make_canonical(mc_paths=10_000, mc_steps=16)
        assert m.price(S0, 100, 1.0, 0.05) == m.price(S0, 100, 1.0, 0.05)

    def test_price_broadcasting_shares_draws_per_maturity(self):
        """Array strikes at one maturity equal the scalar calls exactly
        (one simulation per distinct maturity, same seed)."""
        m = make_canonical(mc_paths=10_000, mc_steps=16)
        Ks = np.array([90.0, 100.0, 110.0])
        vec = m.price(S0, Ks, 1.0, 0.05)
        scal = np.array([m.price(S0, float(k), 1.0, 0.05) for k in Ks])
        np.testing.assert_allclose(vec, scal, atol=1e-14)
        assert vec.shape == (3,)

    def test_price_broadcasting_multiple_maturities(self):
        m = make_canonical(mc_paths=5_000, mc_steps=16)
        Ts = np.array([[0.5], [1.0]])
        Ks = np.array([90.0, 110.0])
        out = m.price(S0, Ks, Ts, 0.05)
        assert out.shape == (2, 2)
        assert out[0, 0] == m.price(S0, 90.0, 0.5, 0.05)


# ──────────────────────────────────────────────
# ABC Greeks
# ──────────────────────────────────────────────


class TestGreeks:
    @pytest.fixture(scope="class")
    def model(self):
        return make_canonical(mc_paths=50_000, mc_steps=64)

    ARGS = (S0, 100.0, 1.0, 0.05)
    Q = 0.02

    def test_delta_bounds_and_parity(self, model):
        d_call = model.delta(*self.ARGS, "call", self.Q)
        d_put = model.delta(*self.ARGS, "put", self.Q)
        disc_q = np.exp(-self.Q * 1.0)
        assert 0.0 < d_call < disc_q
        # Parity is exact on shared draws: put = call - e^{-qT}
        assert d_put == pytest.approx(d_call - disc_q, abs=1e-10)

    def test_gamma_positive_and_call_equals_put(self, model):
        """Parity makes gamma identical for call and put. The control
        variate's in-sample beta is computed on shifted (put) values, so
        the agreement is to rounding noise amplified by 1/h^2 (~1e-11),
        not to machine epsilon."""
        g_call = model.gamma(*self.ARGS, "call", self.Q)
        g_put = model.gamma(*self.ARGS, "put", self.Q)
        assert g_call > 0
        assert g_call == pytest.approx(g_put, abs=1e-9)

    def test_vega_positive(self, model):
        assert model.vega(*self.ARGS, "call", self.Q) > 0

    def test_theta_call_negative(self, model):
        assert model.theta(*self.ARGS, "call", self.Q) < 0

    def test_rho_signs(self, model):
        assert model.rho(*self.ARGS, "call", self.Q) > 0
        assert model.rho(*self.ARGS, "put", self.Q) < 0

    def test_delta_stable_in_bump_size(self, model):
        """The conditional price is smooth in S: an independent FD of
        price() with a 5x coarser bump must agree closely (same draws)."""
        d_api = model.delta(*self.ARGS, "call", self.Q)
        h = 5e-3 * S0
        d_fd = (
            model.price(S0 + h, 100.0, 1.0, 0.05, "call", self.Q)
            - model.price(S0 - h, 100.0, 1.0, 0.05, "call", self.Q)
        ) / (2 * h)
        assert d_api == pytest.approx(d_fd, abs=2e-4)

    def test_greeks_dict_matches_individual_methods(self, model):
        """Same seed -> deterministic -> the dict must reproduce the
        individual methods exactly."""
        g = model.greeks(*self.ARGS, "call", self.Q)
        assert g["price"] == model.price(*self.ARGS, "call", self.Q)
        assert g["delta"] == model.delta(*self.ARGS, "call", self.Q)
        assert g["gamma"] == model.gamma(*self.ARGS, "call", self.Q)
        assert g["vega"] == model.vega(*self.ARGS, "call", self.Q)
        assert g["theta"] == model.theta(*self.ARGS, "call", self.Q)
        assert g["rho"] == model.rho(*self.ARGS, "call", self.Q)

    def test_zero_strike_greeks(self, model):
        """K = 0 closed forms: the prepaid forward S e^{-qT}."""
        disc_q = np.exp(-self.Q * 1.0)
        assert model.delta(S0, 0.0, 1.0, 0.05, "call", self.Q) == pytest.approx(disc_q, abs=1e-12)
        assert model.gamma(S0, 0.0, 1.0, 0.05, "call", self.Q) == 0.0
        assert model.vega(S0, 0.0, 1.0, 0.05, "call", self.Q) == 0.0
        assert model.rho(S0, 0.0, 1.0, 0.05, "call", self.Q) == 0.0
        assert model.theta(S0, 0.0, 1.0, 0.05, "call", self.Q) == pytest.approx(
            self.Q * S0 * disc_q, rel=1e-6
        )
        assert model.delta(S0, 0.0, 1.0, 0.05, "put", self.Q) == 0.0
        assert model.theta(S0, 0.0, 1.0, 0.05, "put", self.Q) == 0.0

    def test_zero_strike_greeks_dict(self):
        """greeks() at K = 0 (cheap model): the deterministic branch of
        every kernel, including the dict's own price kernel."""
        m = make_canonical(mc_paths=1_000, mc_steps=8)
        g = m.greeks(S0, 0.0, 1.0, 0.05, "call", self.Q)
        assert g["price"] == pytest.approx(S0 * np.exp(-self.Q), abs=1e-12)
        assert g["gamma"] == 0.0 and g["vega"] == 0.0 and g["rho"] == 0.0


# ──────────────────────────────────────────────
# Model-parameter Greeks
# ──────────────────────────────────────────────


class TestModelGreeks:
    @pytest.fixture(scope="class")
    def model(self):
        return make_canonical(mc_paths=30_000, mc_steps=64)

    ARGS = (S0, 100.0, 1.0, 0.05)

    def test_consistency_vs_brute_force_bumping(self, model):
        """
        The CRN sharing tricks (xi0 by exact scaling, eta by recomputation
        from stored Volterra draws, rho in-formula) must reproduce the
        brute-force central differences of bumped-model price() calls with
        the same seed — mathematically identical, different code paths.
        """
        mg = model.model_greeks(*self.ARGS)

        def brute(name, value, h):
            up = model._bumped(**{name: value + h}).price(*self.ARGS)
            dn = model._bumped(**{name: value - h}).price(*self.ARGS)
            return (up - dn) / (2 * h)

        assert mg["xi0"] == pytest.approx(brute("xi0", model.xi0, 1e-3 * model.xi0), rel=1e-8)
        assert mg["eta"] == pytest.approx(brute("eta", model.eta, 1e-3 * model.eta), rel=1e-8)
        h_H = min(1e-3 * model.H, 0.5 * (0.5 - model.H))
        assert mg["H"] == pytest.approx(brute("H", model.H, h_H), rel=1e-8)
        h_rho = min(1e-3 * max(abs(model.rho_sv), 0.1), 0.5 * (1.0 - abs(model.rho_sv)))
        assert mg["rho"] == pytest.approx(brute("rho", model.rho_sv, h_rho), rel=1e-8)

    def test_vega_chain_rule(self, model):
        """vega = dV/d(sqrt(xi0)) = dV/dxi0 * 2 sqrt(xi0)."""
        vega = model.vega(*self.ARGS)
        dxi0 = model.model_greeks(*self.ARGS)["xi0"]
        assert vega == pytest.approx(dxi0 * 2 * np.sqrt(model.xi0), rel=1e-3)

    def test_xi0_sensitivity_positive(self, model):
        assert model.model_greeks(*self.ARGS)["xi0"] > 0

    def test_zero_strike_all_zero(self, model):
        mg = model.model_greeks(S0, 0.0, 1.0, 0.05)
        assert all(v == 0.0 for v in mg.values())

    def test_H_boundary_backward_difference(self):
        """H = 0.5: the central bump would cross the boundary — the
        one-sided backward branch must return finite sensitivities."""
        m = make_canonical(H=0.5, mc_paths=10_000, mc_steps=32)
        mg = m.model_greeks(S0, 100.0, 1.0, 0.05)
        assert all(np.isfinite(v) for v in mg.values())
        assert mg["xi0"] > 0
        assert m.model_greeks(S0, 0.0, 1.0, 0.05)["H"] == 0.0


# ──────────────────────────────────────────────
# The defining property: ATM skew power law psi(T) ~ T^(H - 1/2)
# ──────────────────────────────────────────────


def _atm_skew_heston(model: HestonModel, T: float, k_off: float) -> float:
    """Heston ATM skew by the same +-k_off log-moneyness stencil."""
    ivs = []
    for k in (-k_off, k_off):
        K = S0 * np.exp(k)
        p = model.price(S0, K, T, 0.0, "call")
        ivs.append(BlackScholesModel.implied_vol(p, S0, K, T, 0.0, "call"))
    return (ivs[1] - ivs[0]) / (2 * k_off)


class TestSkewPowerLaw:
    """
    psi(T) = d(sigma_IV)/dk at the money follows T^(H - 1/2) (Fukasawa
    2011) — the property that separates rough volatility from any
    classical stochastic-volatility model, and the empirical motivation
    of Phase 5.

    Comparison design (measured values; deviation from the original spec,
    documented): over the spec's window [0.05, 1.0] the SPX-calibrated
    Heston (kappa = 13 -> mean-reversion time 1/kappa ~ 0.08y) is already
    in its 1/T decay regime and its log-log slope (-0.58) is MORE negative
    than rBergomi's (-0.43) — the naive slope comparison over that window
    would pass Heston. The structural failure of Heston is at the SHORT
    end: as T -> 0 its skew saturates to a constant (measured slope -0.20
    on [0.01, 0.05]) while rBergomi keeps the power law (-0.41 on the same
    window). Equivalently, Heston's log-log skew curve is strongly convex
    (slope_short - slope_long = +0.51) while rBergomi's is a straight line
    (curvature ~0.03). Both effects are asserted.
    """

    K_OFF = 0.02
    MATS_MAIN = np.array([0.05, 0.10, 0.25, 0.50, 1.00])
    MATS_SHORT = np.array([0.01, 0.02, 0.05])

    @pytest.fixture(scope="class")
    def rb_skews(self):
        model = make_canonical(mc_paths=65_536, mc_steps=200)
        Ks = S0 * np.exp(np.array([-self.K_OFF, self.K_OFF]))
        ivs_main = model.iv_surface(S0, Ks, self.MATS_MAIN, 0.0)
        ivs_short = model.iv_surface(S0, Ks, self.MATS_SHORT, 0.0)
        psi_main = (ivs_main[:, 1] - ivs_main[:, 0]) / (2 * self.K_OFF)
        psi_short = (ivs_short[:, 1] - ivs_short[:, 0]) / (2 * self.K_OFF)
        return psi_main, psi_short

    @staticmethod
    def _slope(mats, psi):
        return float(np.polyfit(np.log(mats), np.log(np.abs(psi)), 1)[0])

    def test_skew_negative_everywhere(self, rb_skews):
        psi_main, psi_short = rb_skews
        assert np.all(psi_main < 0) and np.all(psi_short < 0)

    def test_power_law_slope(self, rb_skews):
        """log psi vs log T slope = H - 1/2 = -0.4 +- 0.10 (measured
        -0.433 — the small excess over -0.4 is the known higher-order
        term at eta = 1.9)."""
        psi_main, _ = rb_skews
        slope = self._slope(self.MATS_MAIN, psi_main)
        assert -0.5 < slope < -0.3

    def test_power_law_persists_at_short_maturities(self, rb_skews):
        """rBergomi maintains the power law where Heston saturates
        (measured -0.41 on [0.01, 0.05])."""
        _, psi_short = rb_skews
        assert self._slope(self.MATS_SHORT, psi_short) < -0.33

    def test_rbergomi_loglog_is_straight_line(self, rb_skews):
        psi_main, psi_short = rb_skews
        slope_short = self._slope(self.MATS_SHORT, psi_short)
        slope_long = self._slope(self.MATS_MAIN[2:], psi_main[2:])
        assert abs(slope_short - slope_long) < 0.15

    def test_heston_short_skew_saturates(self, rb_skews):
        """The Phase 4 SPX Heston cannot keep the short-skew explosion:
        its [0.01, 0.05] slope (-0.20, deterministic pricing) is well
        above rBergomi's (-0.41), and its log-log curvature (+0.51) marks
        the saturation-to-1/T crossover that the SPX data (and rBergomi)
        do not show."""
        _, psi_short_rb = rb_skews
        heston = make_heston_spx()
        psi_h_short = np.array([_atm_skew_heston(heston, T, self.K_OFF) for T in self.MATS_SHORT])
        psi_h_long = np.array([_atm_skew_heston(heston, T, self.K_OFF) for T in self.MATS_MAIN[2:]])
        assert np.all(psi_h_short < 0)
        slope_h_short = self._slope(self.MATS_SHORT, psi_h_short)
        slope_h_long = self._slope(self.MATS_MAIN[2:], psi_h_long)
        slope_rb_short = self._slope(self.MATS_SHORT, psi_short_rb)
        # Saturation: Heston's short-end slope is far less negative
        assert slope_h_short > -0.30
        assert slope_h_short - slope_rb_short > 0.10
        # Convexity: Heston bends from flat to ~1/T; rBergomi does not
        assert slope_h_short - slope_h_long > 0.30


# ──────────────────────────────────────────────
# Phase 3 exotics on rBergomi paths — zero-modification integration
# ──────────────────────────────────────────────


class TestExoticsUnderRoughBergomi:
    N_PATHS = 100_000
    N_STEPS = 100

    @pytest.fixture(scope="class")
    def setup(self):
        model = make_canonical()
        engine = MonteCarloEngine(n_paths=self.N_PATHS, n_steps=self.N_STEPS, seed=17)
        paths = model.simulate(S0, 1.0, 0.0, n_paths=self.N_PATHS, n_steps=self.N_STEPS, seed=17)
        return model, engine, paths

    def test_asian_differs_from_gbm(self, setup):
        """Same engine, same instrument, same sigma-equivalent GBM
        (sqrt(xi0) = 0.20): the rough smile must move the Asian put by
        many standard errors (measured z ~ -21)."""
        _, engine, paths = setup
        asian = AsianOption(K=100.0, option_type="put")
        res_rb = engine.price(asian.payoff, paths, 0.0, 1.0)
        paths_gbm = engine.simulate_gbm(S0, 1.0, 0.0, 0.20)
        res_gbm = engine.price(asian.payoff, paths_gbm, 0.0, 1.0)
        z = (res_rb.price - res_gbm.price) / np.hypot(res_rb.std_error, res_gbm.std_error)
        assert abs(z) > 5.0

    def test_barrier_dominance_and_in_out_parity(self, setup):
        """Pathwise: knock-out <= vanilla; down-out + down-in = vanilla
        exactly on the same paths."""
        _, engine, paths = setup
        do = BarrierOption(K=100.0, barrier=85.0, barrier_type="down-and-out")
        di = BarrierOption(K=100.0, barrier=85.0, barrier_type="down-and-in")
        res_do = engine.price(do.payoff, paths, 0.0, 1.0)
        res_di = engine.price(di.payoff, paths, 0.0, 1.0)
        vanilla = engine.price(lambda p: np.maximum(p[:, -1] - 100.0, 0.0), paths, 0.0, 1.0)
        assert res_do.price <= vanilla.price
        assert res_do.price + res_di.price == pytest.approx(vanilla.price, abs=1e-9)

    def test_vanilla_on_paths_matches_conditional(self, setup):
        """Plain payoff on the simulated paths vs the conditional
        estimator on an independent seed: |z| < 3."""
        model, engine, paths = setup
        res = engine.price(lambda p: np.maximum(p[:, -1] - 100.0, 0.0), paths, 0.0, 1.0)
        p_c, se_c = model.price_european_conditional(
            S0, 100.0, 1.0, 0.0, n_paths=self.N_PATHS, n_steps=self.N_STEPS, seed=5
        )
        z = (res.price - p_c) / np.hypot(res.std_error, se_c)
        assert abs(z) < 3.0


# ──────────────────────────────────────────────
# Simulation API contract
# ──────────────────────────────────────────────


class TestSimulate:
    def test_shapes_and_initial_values(self):
        m = make_canonical(seed=1)
        (paths, anti), (v, v_anti) = m.simulate(
            S0, 1.0, 0.05, n_paths=1_000, n_steps=32, antithetic=True, return_variance=True
        )
        for arr in (paths, anti, v, v_anti):
            assert arr.shape == (1_000, 33)
        assert np.all(paths[:, 0] == S0) and np.all(anti[:, 0] == S0)
        assert np.all(v[:, 0] == m.xi0) and np.all(v_anti[:, 0] == m.xi0)
        assert np.all(paths > 0) and np.all(anti > 0)
        assert np.all(v > 0) and np.all(v_anti > 0)

    def test_reproducibility(self):
        m = make_canonical()
        a = m.simulate(S0, 1.0, 0.05, n_paths=500, n_steps=16, seed=7)
        b = m.simulate(S0, 1.0, 0.05, n_paths=500, n_steps=16, seed=7)
        np.testing.assert_array_equal(a, b)
        # seed=None falls back to the model seed: also reproducible
        c = m.simulate(S0, 1.0, 0.05, n_paths=500, n_steps=16)
        d = m.simulate(S0, 1.0, 0.05, n_paths=500, n_steps=16)
        np.testing.assert_array_equal(c, d)
        e = m.simulate(S0, 1.0, 0.05, n_paths=500, n_steps=16, seed=8)
        assert not np.array_equal(a, e)

    def test_antithetic_variance_identity(self):
        """The Gaussian map is linear (D6): V(-Z) = -V(Z), so
        v * v_anti = xi0^2 exp(-eta^2 t^(2H)) EXACTLY, path by path —
        the sharpest possible check that antithetic negates every draw."""
        m = make_canonical(seed=3)
        (_, _), (v, v_anti) = m.simulate(
            S0, 1.0, 0.05, n_paths=200, n_steps=16, antithetic=True, return_variance=True
        )
        t = np.arange(1, 17) / 16.0
        expected = m.xi0**2 * np.exp(-(m.eta**2) * t ** (2 * m.H))
        np.testing.assert_allclose(
            v[:, 1:] * v_anti[:, 1:], np.broadcast_to(expected, (v.shape[0], 16)), rtol=1e-12
        )

    def test_antithetic_reduces_standard_error(self):
        m = make_canonical(seed=5)
        engine = MonteCarloEngine(n_paths=50_000, n_steps=50)
        paths, anti = m.simulate(S0, 1.0, 0.05, n_paths=50_000, n_steps=50, antithetic=True, seed=5)

        def payoff(p):
            return np.maximum(p[:, -1] - 100.0, 0.0)

        res_anti = engine.price(payoff, paths, 0.05, 1.0, paths_anti=anti)
        res_plain = engine.price(payoff, paths, 0.05, 1.0)
        assert res_anti.std_error < res_plain.std_error

    def test_single_step_mesh(self):
        """n_steps = 1: hybrid has no convolution term, Cholesky is 2x2."""
        m = make_canonical()
        for method in ("hybrid", "cholesky"):
            paths = m.simulate(S0, 1.0, 0.05, n_paths=100, n_steps=1, method=method, seed=1)
            assert paths.shape == (100, 2)
            assert np.all(paths > 0)


# ──────────────────────────────────────────────
# Implied volatility surface
# ──────────────────────────────────────────────


class TestIVSurface:
    def test_shape_and_skew_direction(self):
        m = make_canonical(mc_paths=30_000, mc_steps=64)
        strikes = np.array([90.0, 100.0, 110.0])
        mats = np.array([0.25, 1.0])
        ivs = m.iv_surface(S0, strikes, mats, 0.02, 0.01)
        assert ivs.shape == (2, 3)
        assert np.all(np.isfinite(ivs)) and np.all(ivs > 0)
        # rho < 0: downward skew at every maturity
        assert np.all(np.diff(ivs, axis=1) < 0)

    def test_rejects_zero_strike(self):
        m = make_canonical(mc_paths=1_000, mc_steps=8)
        with pytest.raises(ValueError):
            m.iv_surface(S0, [0.0, 100.0], [1.0], 0.05)

    def test_otm_side_consistency(self):
        """The surface inverts on the OTM side; parity is exact by
        construction, so inverting the call at an ITM-call strike must
        give the same vol (call/put symmetric check at one node)."""
        m = make_canonical(mc_paths=30_000, mc_steps=64)
        K, T = 90.0, 0.5  # K < forward -> surface uses the put
        iv_surf = m.iv_surface(S0, [K], [T], 0.03)[0, 0]
        p_call, _ = m.price_european_conditional(S0, K, T, 0.03, "call")
        iv_call = BlackScholesModel.implied_vol(p_call, S0, K, T, 0.03, "call")
        assert iv_surf == pytest.approx(iv_call, abs=5e-7)


# ──────────────────────────────────────────────
# Hurst roundtrip — closes the loop with Gatheral-Jaisson-Rosenbaum 2018
# ──────────────────────────────────────────────


class TestHurstRoundtrip:
    def test_H_recovered_from_log_variance_increments(self):
        """
        GJR 2018 estimate H from E[|log v_{t+D} - log v_t|^2] ~ D^(2H).
        Here log v increments are eta * (V_{t+D} - V_t) plus a
        deterministic drift, and the Riemann-Liouville process has the
        same local roughness as fBM away from t = 0. Regression over lags
        {1, 2, 4, 8, 16} on the second half of the grid (t >= 0.5, where
        the non-stationarity of RL increments is negligible):
        measured H_hat = 0.098 for H = 0.1.
        """
        m = make_canonical(seed=3)
        _, v = m.simulate(S0, 1.0, 0.0, n_paths=50_000, n_steps=512, return_variance=True)
        log_v = np.log(v[:, 256:])
        dt = 1.0 / 512
        lags = np.array([1, 2, 4, 8, 16])
        moments = [np.mean((log_v[:, lag:] - log_v[:, :-lag]) ** 2) for lag in lags]
        H_hat = 0.5 * np.polyfit(np.log(lags * dt), np.log(moments), 1)[0]
        assert abs(H_hat - m.H) < 0.05


# ──────────────────────────────────────────────
# Property-based tests
# ──────────────────────────────────────────────

MODEL_STRATEGY = dict(
    xi0=st.floats(0.01, 0.09),
    eta=st.floats(0.5, 2.5),
    H=st.floats(0.05, 0.45),
    rho=st.floats(-0.95, 0.5),
)


class TestHypothesisProperties:
    @given(
        T=st.floats(0.1, 1.5), r=st.floats(-0.02, 0.08), q=st.floats(0.0, 0.04), **MODEL_STRATEGY
    )
    @settings(max_examples=50, deadline=None)
    def test_martingale_and_positivity(self, xi0, eta, H, rho, T, r, q):
        m = RoughBergomiModel(xi0=xi0, eta=eta, H=H, rho=rho, seed=42)
        paths, v = m.simulate(S0, T, r, q, n_paths=20_000, n_steps=32, return_variance=True)
        assert np.all(paths > 0) and np.all(v > 0)
        assert np.all(np.isfinite(paths))
        disc = np.exp(-(r - q) * T) * paths[:, -1]
        se = disc.std(ddof=1) / np.sqrt(len(disc))
        assert abs(disc.mean() - S0) < 4 * se

    @given(
        moneyness=st.floats(0.7, 1.3),
        T=st.floats(0.1, 1.5),
        r=st.floats(-0.02, 0.08),
        q=st.floats(0.0, 0.04),
        **MODEL_STRATEGY,
    )
    @settings(max_examples=50, deadline=None)
    def test_parity_and_bounds(self, xi0, eta, H, rho, moneyness, T, r, q):
        m = RoughBergomiModel(xi0=xi0, eta=eta, H=H, rho=rho, mc_paths=20_000, mc_steps=32, seed=42)
        K = S0 * moneyness
        c, se = m.price_european_conditional(S0, K, T, r, "call", q)
        p, _ = m.price_european_conditional(S0, K, T, r, "put", q)
        parity = S0 * np.exp(-q * T) - K * np.exp(-r * T)
        assert c - p == pytest.approx(parity, abs=1e-9)
        lb = max(parity, 0.0)
        # Absolute floor: at rho = 0 / low vol the estimator is nearly
        # deterministic (se ~ 1e-16) and 4*se does not cover the rounding
        # noise between mean(vals) and the independently computed bound.
        tol = 4 * se + 1e-9
        assert lb - tol <= c <= S0 * np.exp(-q * T) + tol

    @given(T=st.floats(0.1, 1.5), **MODEL_STRATEGY)
    @settings(max_examples=25, deadline=None)
    def test_call_monotone_decreasing_in_strike(self, xi0, eta, H, rho, T):
        """Exact (not statistical): shared draws, no control variate."""
        m = RoughBergomiModel(xi0=xi0, eta=eta, H=H, rho=rho, mc_paths=20_000, mc_steps=32, seed=42)
        prices = [
            m.price_european_conditional(S0, K, T, 0.02, control_variate=False)[0]
            for K in (80.0, 100.0, 120.0)
        ]
        assert prices[0] > prices[1] > prices[2]


# ──────────────────────────────────────────────
# Performance guardrail
# ──────────────────────────────────────────────


class TestPerformance:
    def test_hybrid_throughput(self):
        """Spec target: 200k paths x 256 steps in a few seconds (FFT
        convolution batched over paths). Generous CI bound."""
        m = make_canonical()
        start = time.perf_counter()
        paths = m.simulate(S0, 1.0, 0.05, n_paths=200_000, n_steps=256, seed=1)
        elapsed = time.perf_counter() - start
        assert paths.shape == (200_000, 257)
        assert elapsed < 15.0, f"hybrid simulation took {elapsed:.1f}s"
