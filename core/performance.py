"""
Performance metrics for backtest results.
Computes Sharpe, drawdown, win rate, regime breakdowns, confidence buckets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List


def compute_metrics(equity_curve: pd.Series, risk_free_rate: float = 0.05) -> Dict:
    """Compute core performance metrics from a daily equity curve."""
    returns = equity_curve.pct_change().dropna()
    if len(returns) == 0:
        return {}

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    ann_return   = (1 + total_return) ** (252 / len(returns)) - 1
    ann_vol      = returns.std() * np.sqrt(252)
    sharpe       = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

    # Drawdown
    roll_max   = equity_curve.cummax()
    drawdown   = (equity_curve - roll_max) / roll_max
    max_dd     = float(drawdown.min())
    calmar     = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    # Win rate (daily)
    wins     = (returns > 0).sum()
    win_rate = wins / len(returns) if len(returns) > 0 else 0.0

    return {
        "total_return":  round(total_return, 4),
        "ann_return":    round(ann_return, 4),
        "ann_vol":       round(ann_vol, 4),
        "sharpe":        round(sharpe, 4),
        "max_drawdown":  round(max_dd, 4),
        "calmar":        round(calmar, 4),
        "win_rate":      round(win_rate, 4),
        "n_days":        len(returns),
    }


def regime_breakdown(
    equity_curve: pd.Series,
    regimes: pd.Series,
) -> pd.DataFrame:
    """Return per-regime performance metrics."""
    aligned = equity_curve.align(regimes, join="inner")
    eq, reg = aligned[0], aligned[1]
    rets = eq.pct_change().dropna()

    rows = []
    for label in reg.unique():
        mask  = reg.reindex(rets.index) == label
        r_sub = rets[mask]
        if len(r_sub) == 0:
            continue
        ann_ret = r_sub.mean() * 252
        ann_vol = r_sub.std() * np.sqrt(252)
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0
        rows.append({
            "regime":    label,
            "n_days":    int(mask.sum()),
            "ann_return": round(ann_ret, 4),
            "ann_vol":   round(ann_vol, 4),
            "sharpe":    round(sharpe, 4),
        })
    return pd.DataFrame(rows)


def confidence_buckets(
    equity_curve: pd.Series,
    confidence: pd.Series,
    n_buckets: int = 5,
) -> pd.DataFrame:
    """
    Compare performance across confidence quintiles.
    High-confidence signals should outperform low-confidence ones.
    """
    aligned = equity_curve.align(confidence, join="inner")
    eq, conf = aligned[0], aligned[1]
    rets = eq.pct_change().dropna()

    conf_at_ret = conf.reindex(rets.index).dropna()
    rets = rets.reindex(conf_at_ret.index)

    labels_q = pd.qcut(conf_at_ret, q=n_buckets, labels=False, duplicates="drop")
    rows = []
    for bucket in range(n_buckets):
        mask  = labels_q == bucket
        r_sub = rets[mask]
        conf_sub = conf_at_ret[mask]
        if len(r_sub) == 0:
            continue
        ann_ret = r_sub.mean() * 252
        rows.append({
            "confidence_bucket": bucket,
            "conf_min": round(float(conf_sub.min()), 3),
            "conf_max": round(float(conf_sub.max()), 3),
            "n_days":   int(mask.sum()),
            "ann_return": round(ann_ret, 4),
        })
    return pd.DataFrame(rows)


def benchmark_comparison(
    strategy_equity: pd.Series,
    prices: pd.Series,
) -> Dict[str, Dict]:
    """
    Compare strategy against three benchmarks:
      1. Buy & hold
      2. 200-day SMA trend-following
      3. Random entry / random allocation (same risk rules)
    """
    results = {}

    # 1. Buy and hold
    bah = prices / prices.iloc[0] * strategy_equity.iloc[0]
    results["buy_and_hold"] = compute_metrics(bah)

    # 2. 200-day SMA
    sma = prices.rolling(200).mean()
    invested = prices > sma
    daily_ret = prices.pct_change().fillna(0)
    sma_ret   = daily_ret * invested.shift(1).fillna(False)
    sma_eq    = (1 + sma_ret).cumprod() * strategy_equity.iloc[0]
    results["sma_200"] = compute_metrics(sma_eq)

    # 3. Random baseline
    rng = np.random.default_rng(42)
    rand_pos  = rng.integers(0, 2, size=len(daily_ret)).astype(float)
    rand_ret  = daily_ret * rand_pos
    rand_eq   = (1 + rand_ret).cumprod() * strategy_equity.iloc[0]
    results["random_entry"] = compute_metrics(rand_eq)

    results["strategy"] = compute_metrics(strategy_equity)
    return results


def stress_test(
    equity_curve: pd.Series,
    shock_magnitudes: List[float],
) -> pd.DataFrame:
    """
    Inject synthetic crash shocks and measure recovery.
    """
    rows = []
    for shock in shock_magnitudes:
        shocked = equity_curve.copy()
        # inject at 1/3 and 2/3 through the period
        idx1 = len(shocked) // 3
        idx2 = (len(shocked) * 2) // 3
        for idx in [idx1, idx2]:
            shocked.iloc[idx:] *= (1 + shock)
        m = compute_metrics(shocked)
        m["shock_pct"] = shock
        rows.append(m)
    return pd.DataFrame(rows)
