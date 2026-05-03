"""Tests for feature engineering and HMM engine."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from core.feature_engineering import compute_features, scale_features
from core.hmm_engine import RegimeDetectionEngine


def _make_ohlcv(n: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    log_ret = rng.normal(0.0005, 0.012, n)
    close = 100 * np.exp(np.cumsum(log_ret))
    high  = close * (1 + rng.uniform(0, 0.01, n))
    low   = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    vol   = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )


class TestFeatureEngineering:
    def test_compute_features_returns_dataframe(self):
        df = _make_ohlcv()
        feat = compute_features(df)
        assert isinstance(feat, pd.DataFrame)
        assert len(feat) > 0

    def test_no_nan_in_features(self):
        df = _make_ohlcv()
        feat = compute_features(df)
        assert not feat.isnull().any().any()

    def test_expected_columns_present(self):
        df = _make_ohlcv()
        feat = compute_features(df)
        for col in ["log_ret", "vol_21", "mom_5", "mom_21", "rsi_14", "drawdown"]:
            assert col in feat.columns, f"Missing column: {col}"

    def test_scale_features_zero_mean(self):
        df = _make_ohlcv()
        feat = compute_features(df)
        X = scale_features(feat)
        assert X.shape == feat.shape
        means = np.abs(X.mean(axis=0))
        assert np.all(means < 1e-10), "Features not zero-centred after scaling"

    def test_feature_index_monotonic(self):
        df = _make_ohlcv()
        feat = compute_features(df)
        assert feat.index.is_monotonic_increasing


class TestHMMEngine:
    @pytest.fixture
    def trained_engine(self):
        df = _make_ohlcv(600)
        eng = RegimeDetectionEngine()
        eng.train(df)
        return eng, df

    def test_trains_without_error(self, trained_engine):
        eng, _ = trained_engine
        assert eng.is_trained

    def test_n_regimes_in_valid_range(self, trained_engine):
        eng, _ = trained_engine
        from config import settings
        assert settings.HMM_MIN_REGIMES <= eng.n_regimes <= settings.HMM_MAX_REGIMES

    def test_regime_labels_assigned(self, trained_engine):
        eng, _ = trained_engine
        assert len(eng.regime_labels) == eng.n_regimes
        for lbl in eng.regime_labels:
            assert isinstance(lbl, str) and len(lbl) > 0

    def test_predict_returns_valid_regime(self, trained_engine):
        eng, df = trained_engine
        regime, conf = eng.predict(df)
        assert regime in eng.regime_labels
        assert 0.0 <= conf <= 1.0

    def test_no_lookahead_bias(self, trained_engine):
        """Predict on a strictly smaller slice than training data."""
        eng, df = trained_engine
        half = df.iloc[: len(df) // 2]
        regime, conf = eng.predict(half)
        assert regime in eng.regime_labels

    def test_classify_states_length(self, trained_engine):
        eng, df = trained_engine
        series = eng.classify_states(df)
        from core.feature_engineering import compute_features
        feat = compute_features(df)
        assert len(series) == len(feat)

    def test_stability_filter_dampens_confidence(self):
        eng = RegimeDetectionEngine()
        # Simulate a flickering regime history
        for i in range(25):
            eng._apply_stability_filter("bull" if i % 2 == 0 else "bear", 0.9)
        _, conf = "bull", eng._apply_stability_filter("bull", 0.9)
        assert conf < 0.9, "Stability filter should reduce confidence during flickers"
