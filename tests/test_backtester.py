"""Tests for walk-forward backtester and performance metrics."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from core.back_tester import WalkForwardBacktester
from core.performance import compute_metrics, regime_breakdown, benchmark_comparison, stress_test
from config import settings


def _make_ohlcv(n: int = 800, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    log_ret = rng.normal(0.0004, 0.011, n)
    close = 100 * np.exp(np.cumsum(log_ret))
    high  = close * (1 + rng.uniform(0, 0.008, n))
    low   = close * (1 - rng.uniform(0, 0.008, n))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    vol   = rng.integers(500_000, 3_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )


class TestPerformanceMetrics:
    def _equity(self, n=252):
        prices = np.cumprod(1 + np.random.default_rng(0).normal(0.0005, 0.01, n))
        return pd.Series(prices * 100_000, index=pd.date_range("2023-01-02", periods=n, freq="B"))

    def test_compute_metrics_keys(self):
        eq = self._equity()
        m = compute_metrics(eq)
        for key in ["total_return", "sharpe", "max_drawdown", "win_rate"]:
            assert key in m

    def test_max_drawdown_non_positive(self):
        eq = self._equity()
        m = compute_metrics(eq)
        assert m["max_drawdown"] <= 0

    def test_regime_breakdown_dataframe(self):
        eq = self._equity()
        regimes = pd.Series(
            ["bull"] * 100 + ["bear"] * 100 + ["neutral"] * 52,
            index=eq.index,
        )
        df = regime_breakdown(eq, regimes)
        assert isinstance(df, pd.DataFrame)
        assert "regime" in df.columns

    def test_benchmark_comparison_keys(self):
        eq = self._equity()
        prices = pd.Series(
            np.cumprod(1 + np.random.default_rng(1).normal(0.0003, 0.008, len(eq))),
            index=eq.index,
        )
        result = benchmark_comparison(eq, prices)
        for key in ["strategy", "buy_and_hold", "sma_200", "random_entry"]:
            assert key in result

    def test_stress_test_returns_dataframe(self):
        eq = self._equity()
        result = stress_test(eq, [-0.10, -0.15])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "shock_pct" in result.columns


class TestWalkForwardBacktester:
    def test_run_produces_report(self):
        df  = _make_ohlcv(800)
        bt  = WalkForwardBacktester(in_sample_days=252, out_sample_days=63)
        report = bt.run(df)
        assert len(report.folds) > 0

    def test_aggregate_metrics_present(self):
        df  = _make_ohlcv(800)
        bt  = WalkForwardBacktester(in_sample_days=252, out_sample_days=63)
        report = bt.run(df)
        assert "sharpe" in report.aggregate_metrics

    def test_no_lookahead_each_fold(self):
        """Verify out-of-sample starts strictly after in-sample ends."""
        df  = _make_ohlcv(800)
        bt  = WalkForwardBacktester(in_sample_days=252, out_sample_days=63)
        report = bt.run(df)
        for fold in report.folds:
            assert fold.test_start > fold.train_end

    def test_insufficient_data_raises(self):
        df = _make_ohlcv(100)
        bt = WalkForwardBacktester(in_sample_days=252, out_sample_days=126)
        with pytest.raises(ValueError):
            bt.run(df)

    def test_equity_curve_positive(self):
        df  = _make_ohlcv(800)
        bt  = WalkForwardBacktester(in_sample_days=252, out_sample_days=63)
        report = bt.run(df)
        for fold in report.folds:
            assert (fold.equity_curve > 0).all()
