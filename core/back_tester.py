"""
Walk-forward backtesting engine.
Splits historical data into rolling in-sample / out-of-sample windows,
trains the HMM on in-sample, then evaluates on out-of-sample.
No lookahead bias — the model never sees the out-of-sample window during training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from config import settings
from core.feature_engineering import compute_features
from core.hmm_engine import RegimeDetectionEngine
from core.performance import (
    benchmark_comparison, compute_metrics, confidence_buckets,
    regime_breakdown, stress_test,
)
from core.regime_strategies import StrategyOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_regimes: int
    metrics: dict
    regime_metrics: pd.DataFrame
    confidence_metrics: pd.DataFrame
    equity_curve: pd.Series


@dataclass
class BacktestReport:
    folds: List[FoldResult] = field(default_factory=list)
    aggregate_metrics: dict = field(default_factory=dict)
    benchmark_metrics: dict = field(default_factory=dict)
    stress_results: pd.DataFrame = field(default_factory=pd.DataFrame)


class WalkForwardBacktester:
    """Allocation-based walk-forward back test (not a traditional bar-by-bar sim)."""

    def __init__(
        self,
        in_sample_days: int = settings.BT_IN_SAMPLE_DAYS,
        out_sample_days: int = settings.BT_OUT_SAMPLE_DAYS,
        slippage_bps: float = settings.BT_SLIPPAGE_BPS,
    ):
        self.in_sample_days  = in_sample_days
        self.out_sample_days = out_sample_days
        self.slippage        = slippage_bps / 10_000
        self.orchestrator    = StrategyOrchestrator()

    def run(self, df: pd.DataFrame, initial_equity: float = 100_000.0) -> BacktestReport:
        """
        Run walk-forward test over df (OHLCV, daily).
        Returns a BacktestReport with per-fold and aggregate results.
        """
        report = BacktestReport()
        total = len(df)
        window = self.in_sample_days + self.out_sample_days

        if total < window:
            raise ValueError(
                f"Not enough data ({total} bars) for one fold "
                f"({self.in_sample_days} + {self.out_sample_days} = {window} bars)."
            )

        fold_id = 0
        all_equity: List[pd.Series] = []
        all_regimes: List[pd.Series] = []
        all_confidence: List[pd.Series] = []

        start = 0
        while start + window <= total:
            train_df = df.iloc[start : start + self.in_sample_days]
            test_df  = df.iloc[start + self.in_sample_days : start + window]

            logger.info(
                "Fold %d: train %s–%s | test %s–%s",
                fold_id,
                train_df.index[0].date(), train_df.index[-1].date(),
                test_df.index[0].date(), test_df.index[-1].date(),
            )

            # Train HMM on in-sample window
            engine = RegimeDetectionEngine()
            try:
                engine.train(train_df)
            except Exception as exc:
                logger.warning("Fold %d: HMM training failed — %s", fold_id, exc)
                start += self.out_sample_days
                fold_id += 1
                continue

            # Simulate out-of-sample
            eq, regimes, confs = self._simulate(engine, train_df, test_df, initial_equity)
            all_equity.append(eq)
            all_regimes.append(regimes)
            all_confidence.append(confs)

            fold_result = FoldResult(
                fold_id=fold_id,
                train_start=train_df.index[0],
                train_end=train_df.index[-1],
                test_start=test_df.index[0],
                test_end=test_df.index[-1],
                n_regimes=engine.n_regimes,
                metrics=compute_metrics(eq),
                regime_metrics=regime_breakdown(eq, regimes),
                confidence_metrics=confidence_buckets(eq, confs),
                equity_curve=eq,
            )
            report.folds.append(fold_result)

            start += self.out_sample_days
            fold_id += 1

        # Aggregate across folds
        if all_equity:
            combined_eq   = pd.concat(all_equity).sort_index()
            combined_reg  = pd.concat(all_regimes).sort_index()
            combined_conf = pd.concat(all_confidence).sort_index()

            report.aggregate_metrics   = compute_metrics(combined_eq)
            report.benchmark_metrics   = benchmark_comparison(combined_eq, df["close"].reindex(combined_eq.index))
            report.stress_results      = stress_test(combined_eq, settings.STRESS_SHOCKS)

        return report

    # ── Simulation ────────────────────────────────────────────────────────────

    def _simulate(
        self,
        engine: RegimeDetectionEngine,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        initial_equity: float,
    ):
        equity       = initial_equity
        equity_vals  = [initial_equity]
        regime_list  = []
        conf_list    = []
        prev_alloc   = 0.50  # start 50% deployed

        features     = compute_features(test_df)

        for i, (ts, row) in enumerate(test_df.iterrows()):
            # Features up to but NOT including this bar (use combined window)
            lookback = pd.concat([train_df, test_df.iloc[:i]])
            try:
                regime, conf = engine.predict(lookback)
            except Exception:
                regime, conf = "neutral", 0.5

            feat_slice = compute_features(lookback)
            signal     = self.orchestrator.get_signal(regime, conf, feat_slice)
            target_pct = signal.effective_exposure

            # Daily return of the underlying
            daily_ret  = row["close"] / test_df["close"].iloc[max(0, i - 1)] - 1

            # Rebalance cost
            rebal_cost = abs(target_pct - prev_alloc) * self.slippage
            portfolio_ret = target_pct * daily_ret - rebal_cost
            equity = equity * (1 + portfolio_ret)
            prev_alloc = target_pct

            equity_vals.append(equity)
            regime_list.append(regime)
            conf_list.append(conf)

        # Prepend equity at day-before-test-start (avoids duplicate index labels)
        prev_day = test_df.index[0] - pd.Timedelta(days=1)
        idx = pd.DatetimeIndex([prev_day] + list(test_df.index))
        eq  = pd.Series(equity_vals, index=idx, name="equity")
        reg = pd.Series(regime_list, index=test_df.index, name="regime")
        con = pd.Series(conf_list,   index=test_df.index, name="confidence")
        return eq, reg, con
