"""Compute features used by the HMM regime classifier and strategy layer."""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns fed into the HMM (ordered, z-score scaled).
# Strategy-only columns (atr_14, macd_*, bb_*) are excluded from HMM input.
HMM_FEATURE_COLS = [
    "log_ret", "vol_21", "vol_of_vol",
    "mom_5", "mom_21", "mom_63",
    "vol_zscore", "drawdown", "rsi_14", "atr_pct",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with OHLCV columns return a DataFrame of all features.
    HMM uses HMM_FEATURE_COLS; strategies also read macd_hist, bb_lower_pct, atr_14.
    All features are computed strictly from past data — zero lookahead.
    """
    out = pd.DataFrame(index=df.index)

    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # ── Returns & volatility ─────────────────────────────────────────────────
    out["log_ret"]    = np.log(close / close.shift(1))
    out["vol_21"]     = out["log_ret"].rolling(21).std()
    out["vol_of_vol"] = out["vol_21"].rolling(21).std()

    # ── Momentum ─────────────────────────────────────────────────────────────
    out["mom_5"]  = close.pct_change(5)
    out["mom_21"] = close.pct_change(21)
    out["mom_63"] = close.pct_change(63)

    # ── Volume z-score ────────────────────────────────────────────────────────
    vol_mean = volume.rolling(21).mean()
    vol_std  = volume.rolling(21).std().replace(0, np.nan)
    out["vol_zscore"] = (volume - vol_mean) / vol_std

    # ── Drawdown from 252-day high ────────────────────────────────────────────
    # min_periods=1 uses expanding window when history < 252 bars (back-test windows)
    roll_high       = close.rolling(252, min_periods=1).max()
    out["drawdown"] = (close - roll_high) / roll_high

    # ── RSI (14-day) ─────────────────────────────────────────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # ── ATR (14-day) ───────────────────────────────────────────────────────���─
    tr     = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_14         = tr.rolling(14).mean()
    out["atr_pct"] = atr_14 / close          # HMM feature (normalised)
    out["atr_14"]  = atr_14                   # raw ATR for stop-loss placement

    # ── MACD histogram (12-26-9) — strategy filter ────────────────────────────
    ema_12            = close.ewm(span=12, adjust=False).mean()
    ema_26            = close.ewm(span=26, adjust=False).mean()
    macd_line         = ema_12 - ema_26
    signal_line       = macd_line.ewm(span=9, adjust=False).mean()
    out["macd_hist"]     = macd_line - signal_line
    out["macd_hist_chg"] = out["macd_hist"].diff()   # positive = histogram growing

    # ── Bollinger Bands (20-period, 2σ) — strategy filter ────────────────────
    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std()
    bb_lower = bb_mid - 2.0 * bb_std
    bb_upper = bb_mid + 2.0 * bb_std
    band_width = (bb_upper - bb_lower).replace(0, np.nan)
    # 0.0 = price at lower band, 1.0 = price at upper band
    out["bb_lower_pct"] = (close - bb_lower) / band_width

    out = out.dropna()
    return out


def scale_features(features: pd.DataFrame) -> np.ndarray:
    """Z-score normalise every column independently. Returns ndarray of same shape."""
    arr  = features.values.astype(float)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    std[std == 0] = 1.0
    return (arr - mean) / std


def get_hmm_matrix(features: pd.DataFrame) -> np.ndarray:
    """Extract and z-score scale only the HMM-relevant columns."""
    cols = [c for c in HMM_FEATURE_COLS if c in features.columns]
    return scale_features(features[cols])
