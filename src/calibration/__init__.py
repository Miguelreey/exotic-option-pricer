"""
Exotic Option Pricer — src.calibration

Model calibration to market data (volatility surface fitting).

``market_data`` (live download via the optional yfinance extra) is NOT
imported here on purpose: import it explicitly so the core package keeps
zero non-scientific dependencies.
"""

from .heston_calibrator import (
    CalibrationResult,
    HestonCalibrator,
    filter_option_quotes,
)

__all__ = ["CalibrationResult", "HestonCalibrator", "filter_option_quotes"]
