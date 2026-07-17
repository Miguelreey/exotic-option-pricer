"""
Exotic Option Pricer — exotic_option_pricer/calibration/market_data.py

Live option-chain download for calibration (S&P 500 by default).

This module is the ONLY place that touches a data vendor. It requires the
optional dependency ``yfinance`` (``pip install yfinance``); everything in
``heston_calibrator.py`` works without it, so the core library and the test
suite never need network access.

Pipeline:
1. Select up to ``max_expiries`` listed expiries inside the maturity band,
   sampled roughly uniformly in sqrt(T) (vol time). SPX lists DAILY
   expiries: naively taking the first N yields a two-week wall of
   near-identical maturities with zero term information, after which
   (kappa, xi) are unidentifiable and the fit degenerates.
2. Infer the forward per expiry from put-call parity at the most liquid
   near-ATM strike pair: F = K + e^{rT}(C_mid - P_mid), hence an implied
   dividend yield q_T = r - ln(F/S0)/T. Inverting IVs with an assumed q
   instead leaves a visible vol jump where the chain switches from OTM
   puts to OTM calls (measured ~2 vol pts on SPX) — the market's funding
   and dividends, not ours, set the forward.
3. Keep out-of-the-money quotes only — puts below spot, calls above — the
   liquid side of the book; by put-call parity both map to the same
   implied vol, so nothing is lost.
4. Apply the liquidity mask of ``filter_option_quotes`` (positive bids,
   tight relative spreads, minimum volume, moneyness and maturity bands).
5. Invert mid prices to implied vol with the Phase 1 Halley solver —
   Yahoo's own IV column is stale and not trusted.

The output is the flat (strikes, maturities, ivs) layout consumed by
``HestonCalibrator.calibrate``, plus the per-option implied dividend
yield; pass its mean as the calibrator's q.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

import numpy as np

# Optional-dependency pattern: the module stays importable without
# yfinance (its pure helpers are used in tests and by other data sources);
# only fetch_option_quotes() requires the extra.
try:
    import yfinance as yf
except ImportError:  # pragma: no cover - exercised only without the extra
    yf = None  # type: ignore[assignment]

from ..models.black_scholes import BlackScholesModel
from .heston_calibrator import filter_option_quotes


@dataclass(frozen=True, slots=True)
class OptionQuotes:
    """
    Cleaned market quotes ready for calibration.

    Attributes
    ----------
    ticker : str
    S0 : float
        Spot at download time.
    strikes, maturities, ivs : np.ndarray
        One entry per surviving option; maturities in years, implied vols
        as decimals (from mid prices, OTM side, parity-implied forward).
    implied_q : np.ndarray
        Per-option implied dividend yield (constant within an expiry),
        backed out from put-call parity. Pass ``float(np.mean(implied_q))``
        as the calibrator's q.
    n_raw : int
        Quotes seen before filtering (for a liquidity sanity check).
    asof : datetime.date
        Download date (maturities are measured from here).
    """

    ticker: str
    S0: float
    strikes: np.ndarray
    maturities: np.ndarray
    ivs: np.ndarray
    implied_q: np.ndarray
    n_raw: int
    asof: _dt.date


def _select_expiries(
    listed: list[str],
    today: _dt.date,
    maturity_band: tuple[float, float],
    max_expiries: int,
) -> list[tuple[str, float]]:
    """
    Pick up to ``max_expiries`` expiries covering the band, spaced roughly
    uniformly in sqrt(T): short-dated vol moves faster, so vol time (not
    calendar time) is the natural sampling scale.
    """
    in_band = []
    for expiry_str in listed:
        T = (_dt.date.fromisoformat(expiry_str) - today).days / 365.0
        if maturity_band[0] <= T <= maturity_band[1]:
            in_band.append((expiry_str, T))
    if len(in_band) <= max_expiries:
        return in_band

    sqrt_ts = np.sqrt([t for _, t in in_band])
    targets = np.linspace(sqrt_ts[0], sqrt_ts[-1], max_expiries)
    chosen: list[int] = []
    for target in targets:
        idx = int(np.argmin(np.abs(sqrt_ts - target)))
        if idx not in chosen:
            chosen.append(idx)
    return [in_band[i] for i in chosen]


def _implied_dividend_yield(
    calls, puts, S0: float, T: float, r: float, fallback: float,
) -> float:
    """
    Back out q from put-call parity at the most liquid near-ATM pair:

        C - P = S0 e^{-qT} - K e^{-rT}
        F = K + e^{rT} (C_mid - P_mid),    q = r - ln(F / S0) / T

    Falls back to ``fallback`` if no two-sided near-ATM pair exists or the
    implied value is implausible (broken quotes).
    """
    call_mids: dict[float, float] = {}
    for K, bid, ask in zip(calls["strike"], calls["bid"], calls["ask"]):
        if bid > 0 and ask > bid:
            call_mids[float(K)] = 0.5 * (float(bid) + float(ask))
    # (strike, call_mid, put_mid) of the closest-to-spot two-sided pair
    best: tuple[float, float, float] | None = None
    for K, bid, ask in zip(puts["strike"], puts["bid"], puts["ask"]):
        K = float(K)
        if bid > 0 and ask > bid and K in call_mids:
            if best is None or abs(K - S0) < abs(best[0] - S0):
                best = (K, call_mids[K], 0.5 * (float(bid) + float(ask)))
    if best is None or abs(best[0] - S0) > 0.1 * S0:
        return fallback
    strike_atm, call_mid, put_mid = best
    forward = strike_atm + np.exp(r * T) * (call_mid - put_mid)
    if forward <= 0:
        return fallback
    q = r - np.log(forward / S0) / T
    return float(q) if abs(q) <= 0.15 else fallback


def fetch_option_quotes(
    ticker: str = "^SPX",
    *,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    max_expiries: int = 8,
    max_rel_spread: float = 0.25,
    min_volume: float = 1.0,
    moneyness_band: tuple[float, float] = (0.7, 1.3),
    maturity_band: tuple[float, float] = (0.02, 2.5),
) -> OptionQuotes:
    """
    Download and clean an option chain, returning calibration-ready arrays.

    Parameters
    ----------
    ticker : str, default '^SPX'
        Yahoo Finance ticker. '^SPX' is the S&P 500 index chain.
    risk_free_rate : float
        Continuous-compounding rate used to invert mids to implied vol.
        Keyword-only and mandatory: silently defaulting it would bias
        every vol in the surface.
    dividend_yield : float, default 0.0
        Continuous dividend yield of the underlying.
    max_expiries : int, default 8
        Number of listed expiries to download (nearest first).
    max_rel_spread, min_volume, moneyness_band, maturity_band
        Passed to ``filter_option_quotes`` — see its docstring.

    Returns
    -------
    OptionQuotes

    Raises
    ------
    ImportError
        If the optional dependency yfinance is not installed.
    ValueError
        If the download yields no usable quotes after filtering.
    """
    if yf is None:
        raise ImportError(
            "fetch_option_quotes requires the optional dependency yfinance. "
            "Install it with: pip install yfinance"
        )
    asset = yf.Ticker(ticker)
    history = asset.history(period="1d")
    if history.empty:
        raise ValueError(f"No price history returned for ticker '{ticker}'")
    S0 = float(history["Close"].iloc[-1])
    today = _dt.date.today()

    all_strikes: list[np.ndarray] = []
    all_maturities: list[np.ndarray] = []
    all_ivs: list[np.ndarray] = []
    all_q: list[np.ndarray] = []
    n_raw = 0

    expiries = _select_expiries(list(asset.options), today, maturity_band,
                                max_expiries)
    for expiry_str, T in expiries:
        chain = asset.option_chain(expiry_str)
        q_T = _implied_dividend_yield(
            chain.calls, chain.puts, S0, T, risk_free_rate,
            fallback=dividend_yield,
        )

        # OTM side only: puts below spot, calls at/above
        for frame, option_type in ((chain.puts, "put"), (chain.calls, "call")):
            strikes = frame["strike"].to_numpy(dtype=np.float64)
            bids = frame["bid"].to_numpy(dtype=np.float64)
            asks = frame["ask"].to_numpy(dtype=np.float64)
            volumes = np.nan_to_num(
                frame["volume"].to_numpy(dtype=np.float64), nan=0.0,
            )
            n_raw += strikes.shape[0]

            otm = strikes < S0 if option_type == "put" else strikes >= S0
            keep = otm & filter_option_quotes(
                strikes, np.full_like(strikes, T), bids, asks, volumes, S0,
                max_rel_spread=max_rel_spread, min_volume=min_volume,
                moneyness_band=moneyness_band, maturity_band=maturity_band,
            )

            mids = 0.5 * (bids[keep] + asks[keep])
            kept_strikes = strikes[keep]
            ivs = np.full(kept_strikes.shape, np.nan)
            for i, (K, mid) in enumerate(zip(kept_strikes, mids)):
                try:
                    ivs[i] = BlackScholesModel.implied_vol(
                        float(mid), S0, float(K), T, risk_free_rate,
                        option_type, q=q_T,
                    )
                except ValueError:
                    continue  # mid violates no-arbitrage bounds: drop quote

            good = np.isfinite(ivs)
            all_strikes.append(kept_strikes[good])
            all_maturities.append(np.full(int(good.sum()), T))
            all_ivs.append(ivs[good])
            all_q.append(np.full(int(good.sum()), q_T))

    if not all_strikes or sum(len(s) for s in all_strikes) == 0:
        raise ValueError(
            f"No usable quotes for '{ticker}' after liquidity filtering "
            f"({n_raw} raw quotes seen). Loosen the filters or check the ticker."
        )

    return OptionQuotes(
        ticker=ticker,
        S0=S0,
        strikes=np.concatenate(all_strikes),
        maturities=np.concatenate(all_maturities),
        ivs=np.concatenate(all_ivs),
        implied_q=np.concatenate(all_q),
        n_raw=n_raw,
        asof=today,
    )
