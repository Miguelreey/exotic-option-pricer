"""
Exotic Option Pricer — exotic_option_pricer/calibration/heston_calibrator.py

Heston calibration to an implied volatility surface.

Implements:
- Objective in IMPLIED VOL space (not price space): price errors over-weight
  deep ITM options whose vega is tiny; vol errors weight information content
  evenly across the surface, which is how desks quote and risk-manage.
- Unconstrained reparametrization: v0, kappa, theta, xi > 0 via exp;
  rho in (-1, 1) via tanh. The optimizer roams R^5 freely and every
  iterate maps to an admissible Heston model — no active-bound pathologies.
- Deterministic multi-start local least squares: the objective is
  non-convex with valleys (kappa and xi are weakly identified, and
  (kappa, xi) trade off against each other along curves of nearly constant
  total variance), so several starts are run and the best refit wins.
- Feller condition as an OPTIONAL soft penalty: equity surfaces routinely
  calibrate to 2*kappa*theta < xi^2; a hard constraint would reject valid
  fits. Off by default.
- Robustness: parameter vectors that produce unpriceable options (arbitrage
  violations, failed implied-vol inversion) contribute a large finite
  residual instead of crashing the optimizer mid-search.

Market data preparation (liquidity filtering) is a pure-NumPy function
here; the actual downloading via yfinance lives in ``market_data.py`` so
this module never needs network access or optional dependencies.

References
----------
.. [1] Gatheral (2006). The Volatility Surface. Wiley. Ch. 3.
.. [2] Cont & Tankov (2004). Financial Modelling with Jump Processes.
       Ch. 13 (calibration as an ill-posed inverse problem).
.. [3] Andersen & Piterbarg (2010). Interest Rate Modeling. Vol I, §9
       (objective choice and parameter identifiability).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from ..models.black_scholes import BlackScholesModel
from ..models.heston import HestonModel

# Residual assigned to options the trial model cannot price (vol points).
# Large enough to repel the optimizer, finite so the Jacobian stays usable.
_PENALTY_RESIDUAL = 1.0


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """
    Container for Heston calibration results.

    Attributes
    ----------
    params : dict of {str: float}
        Calibrated parameters: keys 'v0', 'kappa', 'theta', 'xi', 'rho'.
    model : HestonModel
        Ready-to-use model built from ``params``.
    rmse_iv : float
        Root-mean-square error in implied vol units (0.005 = 0.5 vol pts),
        unweighted, over the options the calibrated model can price.
    max_error_iv : float
        Worst absolute implied-vol error.
    feller_satisfied : bool
        Whether 2*kappa*theta >= xi^2 at the optimum (informative — a
        violation is common and not an error).
    cost : float
        Final value of the least-squares objective 0.5 * sum(residuals^2),
        including weights and any Feller penalty.
    n_starts : int
        Number of local optimizations run.
    success : bool
        Whether the winning local optimization reported convergence.
    """

    params: dict[str, float]
    model: HestonModel
    rmse_iv: float
    max_error_iv: float
    feller_satisfied: bool
    cost: float
    n_starts: int
    success: bool

    def __repr__(self) -> str:
        p = self.params
        return (
            f"CalibrationResult(v0={p['v0']:.4f}, kappa={p['kappa']:.4f}, "
            f"theta={p['theta']:.4f}, xi={p['xi']:.4f}, rho={p['rho']:.4f}, "
            f"rmse_iv={self.rmse_iv * 100:.3f} vol pts, "
            f"feller={'OK' if self.feller_satisfied else 'violated'}, "
            f"success={self.success})"
        )


def filter_option_quotes(
    strikes: np.ndarray,
    maturities: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    volumes: np.ndarray,
    S0: float,
    *,
    max_rel_spread: float = 0.25,
    min_volume: float = 1.0,
    moneyness_band: tuple[float, float] = (0.7, 1.3),
    maturity_band: tuple[float, float] = (0.02, 2.5),
) -> np.ndarray:
    """
    Liquidity mask for raw option quotes (pure NumPy — no data vendor).

    An equity option chain is mostly noise: stale quotes, zero-volume
    series, penny options with 100% relative spreads. Calibrating to them
    poisons the fit, so desks filter first and fit second.

    Parameters
    ----------
    strikes, maturities, bids, asks, volumes : np.ndarray
        Quote fields, equal length. ``maturities`` in years.
    S0 : float
        Spot price. Must be > 0.
    max_rel_spread : float, default 0.25
        Keep quotes with (ask - bid)/mid below this.
    min_volume : float, default 1.0
        Keep quotes with traded volume >= this.
    moneyness_band : tuple, default (0.7, 1.3)
        Keep strikes with K/S0 inside the band — beyond it, quotes are
        dominated by spread noise and pin little vega.
    maturity_band : tuple, default (0.02, 2.5)
        Keep maturities (years) inside the band: below ~1 week expiries
        are gamma-driven microstructure; far LEAPS quote on rates/divs.

    Returns
    -------
    np.ndarray of bool
        Mask of quotes to keep.
    """
    if S0 <= 0:
        raise ValueError(f"S0 must be > 0, got {S0}")
    strikes = np.asarray(strikes, dtype=np.float64)
    maturities = np.asarray(maturities, dtype=np.float64)
    bids = np.asarray(bids, dtype=np.float64)
    asks = np.asarray(asks, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)
    n = strikes.shape[0]
    if not all(x.shape == (n,) for x in (maturities, bids, asks, volumes)):
        raise ValueError("All quote fields must be 1-D arrays of equal length")

    mid = 0.5 * (bids + asks)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_spread = np.where(mid > 0, (asks - bids) / mid, np.inf)

    return (
        (bids > 0)
        & (asks > bids)
        & (volumes >= min_volume)
        & (rel_spread <= max_rel_spread)
        & (strikes / S0 >= moneyness_band[0])
        & (strikes / S0 <= moneyness_band[1])
        & (maturities >= maturity_band[0])
        & (maturities <= maturity_band[1])
    )


class HestonCalibrator:
    """
    Calibrate Heston parameters to market implied volatilities.

    Market data enters as flat arrays (one entry per option): strikes,
    maturities and implied vols — decoupled from any data vendor. Use
    ``market_data.fetch_spx_quotes()`` (optional yfinance extra) or any
    other source.

    Parameters
    ----------
    S0 : float
        Spot price. Must be > 0.
    r : float
        Risk-free rate (annualized, continuous compounding).
    q : float, default 0.0
        Continuous dividend yield.

    Examples
    --------
    >>> cal = HestonCalibrator(S0=100.0, r=0.03, q=0.01)
    >>> result = cal.calibrate(strikes, maturities, market_ivs)
    >>> result.model.price(100, 105, 0.5, 0.03, 'call', q=0.01)
    """

    def __init__(self, S0: float, r: float, q: float = 0.0) -> None:
        if not np.isfinite(S0) or S0 <= 0:
            raise ValueError(f"S0 must be finite and > 0, got {S0}")
        if not np.isfinite(r) or not np.isfinite(q):
            raise ValueError("r and q must be finite")
        self.S0 = float(S0)
        self.r = float(r)
        self.q = float(q)

    # ──────────────────────────────────────────────
    # Parameter transform: R^5 <-> admissible Heston set
    # ──────────────────────────────────────────────

    @staticmethod
    def _to_params(x: np.ndarray) -> dict[str, float]:
        """x in R^5 -> admissible (v0, kappa, theta, xi, rho)."""
        return {
            "v0": float(np.exp(x[0])),
            "kappa": float(np.exp(x[1])),
            "theta": float(np.exp(x[2])),
            "xi": float(np.exp(x[3])),
            "rho": float(np.tanh(x[4])),
        }

    @staticmethod
    def _to_x(params: dict[str, float]) -> np.ndarray:
        """(v0, kappa, theta, xi, rho) -> unconstrained x in R^5."""
        return np.array(
            [
                np.log(params["v0"]),
                np.log(params["kappa"]),
                np.log(params["theta"]),
                np.log(params["xi"]),
                np.arctanh(params["rho"]),
            ]
        )

    # ──────────────────────────────────────────────
    # Model implied vols
    # ──────────────────────────────────────────────

    def model_ivs(
        self, model: HestonModel, strikes: np.ndarray, maturities: np.ndarray
    ) -> np.ndarray:
        """
        Implied vols of ``model`` at the given (strike, maturity) points.

        Options are grouped by maturity so each group is priced with ONE
        Carr-Madan FFT (the whole point of the FFT route: an objective
        evaluation needs the entire chain), then inverted to implied vol
        with the Phase 1 Halley solver.

        Returns
        -------
        np.ndarray
            Implied vols; NaN where the model price is not invertible
            (arbitrage-violating or out of the solver's reach).
        """
        strikes = np.asarray(strikes, dtype=np.float64)
        maturities = np.asarray(maturities, dtype=np.float64)
        ivs = np.full(strikes.shape, np.nan)

        for T in np.unique(maturities):
            idx = np.flatnonzero(maturities == T)
            try:
                prices = model.price_surface(
                    self.S0,
                    strikes[idx],
                    float(T),
                    self.r,
                    q=self.q,
                )
            except ValueError:
                continue  # moment explosion at this T: leave NaN
            for j, price in zip(idx, prices):
                try:
                    iv = BlackScholesModel.implied_vol(
                        float(price),
                        self.S0,
                        float(strikes[j]),
                        float(T),
                        self.r,
                        "call",
                        q=self.q,
                    )
                except ValueError:
                    continue
                if np.isfinite(iv):
                    ivs[j] = iv
        return ivs

    # ──────────────────────────────────────────────
    # Calibration
    # ──────────────────────────────────────────────

    def _default_starts(
        self, market_ivs: np.ndarray, maturities: np.ndarray
    ) -> list[dict[str, float]]:
        """
        Heuristic start points. The variance anchors come from the data
        (short-end IV^2 for v0, global mean IV^2 for theta); the weakly
        identified pair (kappa, xi) and the skew direction rho are seeded
        over a small grid to escape their shared valleys.
        """
        t_min = np.min(maturities)
        short_iv = float(np.mean(market_ivs[maturities == t_min]))
        mean_iv = float(np.mean(market_ivs))
        v0_seed = float(np.clip(short_iv**2, 1e-4, 1.0))
        theta_seed = float(np.clip(mean_iv**2, 1e-4, 1.0))
        return [
            dict(v0=v0_seed, kappa=1.5, theta=theta_seed, xi=0.5, rho=-0.6),
            dict(v0=v0_seed, kappa=3.0, theta=theta_seed, xi=0.8, rho=-0.4),
            dict(v0=v0_seed, kappa=0.7, theta=theta_seed, xi=0.3, rho=-0.8),
            dict(v0=theta_seed, kappa=2.0, theta=v0_seed, xi=1.0, rho=-0.2),
        ]

    def calibrate(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_ivs: np.ndarray,
        weights: np.ndarray | None = None,
        *,
        starts: list[dict[str, float]] | None = None,
        n_starts: int = 3,
        feller_penalty: float = 0.0,
        max_nfev: int = 400,
    ) -> CalibrationResult:
        """
        Fit (v0, kappa, theta, xi, rho) to market implied vols.

        Minimizes sum_i [w_i (sigma_model_i - sigma_market_i)]^2 over the
        unconstrained reparametrization, with deterministic multi-start
        Levenberg-Marquardt-type local optimization
        (``scipy.optimize.least_squares``, TRF).

        Parameters
        ----------
        strikes, maturities, market_ivs : np.ndarray
            One entry per option (flat layout, ragged chains welcome).
            Maturities in years, implied vols as decimals.
        weights : np.ndarray or None, default None
            Per-option weights (e.g. vegas or 1/spread). None = equal.
            Keep them O(1) (e.g. normalized by their max): options the
            trial model cannot price contribute a FIXED residual of
            ``_PENALTY_RESIDUAL`` = 1.0 vol point, which must dominate
            the weighted vol errors to repel the optimizer from
            unpriceable parameter regions.
        starts : list of dict or None, default None
            Explicit start points (each with keys v0, kappa, theta, xi,
            rho). None = data-driven heuristic grid.
        n_starts : int, default 3
            How many starts to run (first ``n_starts`` of the list).
        feller_penalty : float, default 0.0
            Soft-penalty scale lambda: appends residual
            sqrt(lambda * max(0, xi^2 - 2*kappa*theta)). 0 disables.
            Keep it soft — equity fits legitimately violate Feller.
        max_nfev : int, default 400
            Max objective evaluations per start.

        Returns
        -------
        CalibrationResult

        Raises
        ------
        ValueError
            On inconsistent inputs or if every option is unpriceable at
            every start (hopeless data).
        """
        strikes = np.asarray(strikes, dtype=np.float64)
        maturities = np.asarray(maturities, dtype=np.float64)
        market_ivs = np.asarray(market_ivs, dtype=np.float64)
        n = strikes.shape[0]
        if maturities.shape != (n,) or market_ivs.shape != (n,):
            raise ValueError(
                "strikes, maturities and market_ivs must be 1-D arrays of equal length"
            )
        if n < 5:
            raise ValueError(f"Need at least 5 options to identify 5 parameters, got {n}")
        if np.any(strikes <= 0) or np.any(maturities <= 0):
            raise ValueError("strikes and maturities must be > 0")
        if np.any(market_ivs <= 0) or not np.all(np.isfinite(market_ivs)):
            raise ValueError("market_ivs must be finite and > 0")
        w: np.ndarray
        if weights is None:
            w = np.ones(n)
        else:
            w = np.asarray(weights, dtype=np.float64)
            if w.shape != (n,):
                raise ValueError("weights must match the number of options")
            if np.any(w < 0):
                raise ValueError("weights must be >= 0")
        if feller_penalty < 0:
            raise ValueError(f"feller_penalty must be >= 0, got {feller_penalty}")

        def residuals(x: np.ndarray) -> np.ndarray:
            params = self._to_params(x)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                model = HestonModel(**params)
            ivs = self.model_ivs(model, strikes, maturities)
            res = np.where(np.isnan(ivs), _PENALTY_RESIDUAL, w * (ivs - market_ivs))
            if feller_penalty > 0.0:
                violation = max(
                    params["xi"] ** 2 - 2.0 * params["kappa"] * params["theta"],
                    0.0,
                )
                res = np.append(res, np.sqrt(feller_penalty * violation))
            return res

        start_list = starts if starts is not None else self._default_starts(market_ivs, maturities)
        start_list = start_list[: max(1, n_starts)]
        if not start_list:
            raise ValueError("starts must contain at least one start point")

        best = None
        for start in start_list:
            fit = least_squares(
                residuals,
                self._to_x(start),
                method="trf",
                max_nfev=max_nfev,
            )
            if best is None or fit.cost < best.cost:
                best = fit
        assert best is not None  # narrowed: start_list validated non-empty

        params = self._to_params(best.x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model = HestonModel(**params)

        fitted_ivs = self.model_ivs(model, strikes, maturities)
        valid = ~np.isnan(fitted_ivs)
        if not np.any(valid):
            raise ValueError(
                "Calibration failed: the best-fit model cannot price any "
                "of the supplied options. Check the input surface."
            )
        errors = fitted_ivs[valid] - market_ivs[valid]
        return CalibrationResult(
            params=params,
            model=model,
            rmse_iv=float(np.sqrt(np.mean(errors**2))),
            max_error_iv=float(np.max(np.abs(errors))),
            feller_satisfied=(2.0 * params["kappa"] * params["theta"] >= params["xi"] ** 2),
            cost=float(best.cost),
            n_starts=len(start_list),
            success=bool(best.success),
        )
