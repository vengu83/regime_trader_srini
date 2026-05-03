"""Tests for allocation/regime strategies."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from core.regime_strategies import StrategyOrchestrator, SignalData
from config import settings


def _dummy_features() -> pd.DataFrame:
    return pd.DataFrame({
        "log_ret":   [0.001],
        "vol_21":    [0.01],
        "mom_5":     [0.02],
        "mom_21":    [0.03],
        "mom_63":    [0.05],
        "rsi_14":    [55.0],
        "drawdown":  [-0.02],
        "vol_zscore":[0.5],
        "atr_pct":   [0.008],
        "vol_of_vol":[0.003],
    })


class TestStrategyOrchestrator:
    def setup_method(self):
        self.orch = StrategyOrchestrator()
        self.feat = _dummy_features()

    def test_returns_signal_data(self):
        sig = self.orch.get_signal("bull", 0.85, self.feat)
        assert isinstance(sig, SignalData)

    @pytest.mark.parametrize("regime", ["crash", "bear", "neutral", "bull", "euphoria"])
    def test_all_regimes_handled(self, regime):
        sig = self.orch.get_signal(regime, 0.80, self.feat)
        assert sig.regime == regime
        assert 0.0 <= sig.target_equity_pct <= 1.0
        assert sig.leverage > 0

    def test_low_confidence_returns_hold(self):
        sig = self.orch.get_signal("bull", 0.10, self.feat)
        assert sig.strategy_name == "hold_low_confidence"

    def test_crash_has_lowest_exposure(self):
        crash = self.orch.get_signal("crash", 0.90, self.feat)
        bull  = self.orch.get_signal("bull",  0.90, self.feat)
        assert crash.effective_exposure < bull.effective_exposure

    def test_effective_exposure_capped_at_2x(self):
        sig = self.orch.get_signal("bull", 0.95, self.feat)
        assert sig.effective_exposure <= 2.0

    def test_rebalance_threshold(self):
        assert self.orch.rebalance_needed(0.50, 0.60)   # 10% diff → True
        assert not self.orch.rebalance_needed(0.50, 0.52)  # 2% diff → False

    def test_unknown_regime_falls_back_gracefully(self):
        sig = self.orch.get_signal("xyz_unknown", 0.80, self.feat)
        assert isinstance(sig, SignalData)
