"""
Exotic Option Pricer — src/engines/variance_reduction.py

Variance reduction techniques for Monte Carlo pricing.

Implements:
- Antithetic variates: exploit symmetry of normal distribution to halve variance
- Control variates: use correlated variable with known expectation to reduce variance
- Combined antithetic + control variates
- Importance sampling: drift shift for deep OTM options

All functions are pure (no side effects, no state). They operate on NumPy arrays
and are designed to compose with MonteCarloEngine.

Mathematical Background
-----------------------
Standard MC converges as O(1/sqrt(N)). Variance reduction lowers the prefactor
without changing the rate, yielding equivalent accuracy at a fraction of the cost.

**Antithetic variates** (Hammersley & Handscomb, 1964):
    For monotone payoff f and standard normal Z:
        Cov[f(Z), f(-Z)] <= 0   (by FKG inequality / monotonicity argument)
    so Var[(f(Z) + f(-Z))/2] <= Var[f(Z)]/2.
    Typical reduction: 50-80% for vanilla calls/puts.

**Importance sampling** (Glasserman, 2003, Ch. 4.6):
    Shift the sampling distribution to increase the probability of
    rare events (e.g., deep OTM options exercising). For GBM with
    terminal payoff, the optimal drift shift theta centers the
    terminal distribution at the strike:
        theta* = (ln(K/S0) - (r-q)*T) / (sigma*sqrt(T))
    Each payoff is corrected by the likelihood ratio:
        L(Z) = exp(-theta*Z - theta^2/2)
    where Z is the shifted sample. This preserves unbiasedness while
    dramatically reducing variance for deep OTM/ITM options.

**Control variates** (Glasserman, 2003, Ch. 4):
    Given payoff Y and control C with known E[C]:
        Y_cv = Y - beta*(C - E[C])
        Var[Y_cv] = Var[Y](1 - rho_{Y,C}^2)
    where beta* = Cov[Y,C]/Var[C] minimizes variance.
    With BS analytical price as control for European MC, |rho| > 0.99 is typical,
    giving >98% variance reduction.

References
----------
.. [1] Glasserman (2003). Monte Carlo Methods in Financial Engineering. Ch. 4.
.. [2] Hammersley & Handscomb (1964). Monte Carlo Methods. Methuen.
.. [3] Hull (2018). Options, Futures & Other Derivatives, 10th ed. Ch. 21.
"""

import numpy as np


def generate_antithetic_normals(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate antithetic pairs from standard normal samples.

    For each sample Z_i, the antithetic counterpart is -Z_i. Returns both
    halves separately so that payoffs can be computed on each and then
    averaged **per pair** (essential for variance reduction — concatenating
    and averaging globally destroys the negative correlation).

    Parameters
    ----------
    Z : np.ndarray, shape (n_paths, ...) or (n_paths,)
        Standard normal samples. Any shape with first axis = n_paths.

    Returns
    -------
    Z_pos : np.ndarray
        Original samples (same as input Z).
    Z_neg : np.ndarray
        Antithetic samples (-Z), same shape as Z.

    Notes
    -----
    The pair (Z, -Z) has the same marginal distribution as (Z_1, Z_2) i.i.d.,
    but Cov[g(Z), g(-Z)] <= 0 for any monotone g (consequence of the FKG
    inequality for associated random variables). This negative covariance is
    what drives variance reduction.

    The estimator is:

        Y_hat_AV = (1/N) * sum_{i=1}^{N} [f(Z_i) + f(-Z_i)] / 2

    NOT (1/2N) * sum of all 2N payoffs — the per-pair averaging preserves
    the negative correlation structure.
    """
    return Z, -Z


def optimal_beta(payoffs: np.ndarray, control_values: np.ndarray) -> float:
    """
    Compute the optimal control variate coefficient.

    beta* = Cov[Y, C] / Var[C]

    This minimizes Var[Y - beta*(C - E[C])]. The coefficient is estimated
    from the same sample used for pricing (in-sample estimation), which is
    standard practice and does not introduce bias in the price estimator
    (Glasserman 2003, Theorem 4.1.2).

    Parameters
    ----------
    payoffs : np.ndarray, shape (n,)
        Discounted payoff values from MC simulation.
    control_values : np.ndarray, shape (n,)
        Control variate values (e.g., terminal stock prices, or payoffs
        computed under a model with known analytical price).

    Returns
    -------
    float
        Optimal coefficient beta*. Positive when payoff and control are
        positively correlated (typical for calls with S_T as control).

    Raises
    ------
    ValueError
        If arrays have different lengths or fewer than 2 elements.

    Notes
    -----
    When Var[C] is near zero (e.g., deterministic paths with sigma ~ 0),
    beta* is ill-defined. We return 0.0 in this case to fall back to the
    unadjusted estimator gracefully.
    """
    if payoffs.shape[0] != control_values.shape[0]:
        raise ValueError(
            f"payoffs and control_values must have same length, "
            f"got {payoffs.shape[0]} and {control_values.shape[0]}"
        )
    if payoffs.shape[0] < 2:
        raise ValueError("Need at least 2 samples to estimate beta")

    var_c = np.var(control_values, ddof=1)
    if var_c < 1e-30:
        return 0.0

    cov_yc = np.cov(payoffs, control_values, ddof=1)[0, 1]
    return float(cov_yc / var_c)


def control_variate_adjust(
    payoffs: np.ndarray,
    control_values: np.ndarray,
    control_expectation: float,
    beta: float | None = None,
) -> tuple[np.ndarray, float]:
    """
    Apply control variate adjustment to Monte Carlo payoffs.

    Y_adjusted_i = Y_i - beta* * (C_i - E[C])

    where beta* = Cov[Y, C] / Var[C] is estimated from the sample if not
    provided, and E[C] is the known analytical expectation of the control.

    Parameters
    ----------
    payoffs : np.ndarray, shape (n,)
        Discounted payoff values from MC simulation.
    control_values : np.ndarray, shape (n,)
        Observed control variate values for each path.
    control_expectation : float
        Known analytical expectation E[C]. For European options with BS
        as control, this is the BS analytical price.
    beta : float or None, default None
        Control variate coefficient. If None, optimal beta is computed
        from the sample via optimal_beta().

    Returns
    -------
    adjusted_payoffs : np.ndarray, shape (n,)
        Variance-reduced payoffs.
    beta_used : float
        The coefficient applied (either provided or computed).

    Notes
    -----
    The adjusted estimator is unbiased regardless of beta, because
    E[C_i - E[C]] = 0. The choice of beta only affects variance:

        Var[Y_cv] = Var[Y] - 2*beta*Cov[Y,C] + beta^2*Var[C]

    Minimized at beta* = Cov[Y,C] / Var[C], giving:

        Var[Y_cv] = Var[Y] * (1 - rho_{Y,C}^2)

    For European options priced via MC with BS analytical as control,
    |rho| > 0.99 is typical, yielding >98% variance reduction — often
    more effective than antithetic variates alone.
    """
    if beta is None:
        beta = optimal_beta(payoffs, control_values)

    adjusted = payoffs - beta * (control_values - control_expectation)
    return adjusted, float(beta)


def importance_sampling_shift(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> float:
    """
    Compute the optimal importance sampling drift shift for European options.

    The shift theta centers the terminal price distribution around the
    strike K under the sampling measure, maximizing the probability of
    the option finishing in-the-money. This is most effective for deep
    OTM options where standard MC produces mostly zero payoffs.

    Parameters
    ----------
    S0 : float
        Initial spot price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Risk-free rate.
    sigma : float
        Annualized volatility.
    q : float, default 0.0
        Continuous dividend yield.

    Returns
    -------
    float
        Optimal drift shift theta. Positive for OTM calls (K > F),
        negative for OTM puts (K < F), near zero for ATM.
        Clamped to [-5, 5] for numerical stability.

    Notes
    -----
    Under the shifted measure Q_theta, we sample Z ~ N(0,1) and compute:

        S_T = S_0 * exp((r - q - sigma^2/2)*T + sigma*sqrt(T)*(Z + theta))

    The likelihood ratio that corrects for the drift change is:

        L(Z) = exp(-theta*(Z + theta) + theta^2/2) = exp(-theta*Z - theta^2/2)

    where Z is the *original* (unshifted) standard normal. The corrected
    estimator is:

        V = e^{-rT} * (1/N) * sum payoff(S_T^(i)) * L(Z^(i))

    The optimal theta satisfies E_theta[S_T] = K:

        S_0 * exp((r-q)*T + sigma*theta*sqrt(T)) = K
        theta = (ln(K/S_0) - (r-q)*T) / (sigma*sqrt(T))

    References
    ----------
    .. [1] Glasserman (2003). Monte Carlo Methods in Financial Engineering, §4.6.
    """
    sqrt_T = np.sqrt(T)
    theta = (np.log(K / S0) - (r - q) * T) / (sigma * sqrt_T)
    return float(np.clip(theta, -5.0, 5.0))


def importance_sampling_likelihood(
    Z: np.ndarray,
    theta: float,
) -> np.ndarray:
    """
    Compute importance sampling likelihood ratios.

    Parameters
    ----------
    Z : np.ndarray
        Original standard normal samples (before shift).
    theta : float
        Drift shift applied to the samples.

    Returns
    -------
    np.ndarray
        Likelihood ratios exp(-theta*Z_shifted - theta^2/2) where
        Z_shifted = Z + theta.
    """
    Z_shifted = Z + theta
    return np.exp(-theta * Z_shifted + 0.5 * theta * theta)
