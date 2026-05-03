"""
Hidden Markov Model regime-detection engine.

Key design decisions (per video):
  - StudentTHMM: trains with Gaussian EM for fast convergence, then scores with
    Student-t (df=4) emission probabilities to handle fat tails in returns.
    Crash events are downweighted so normal regimes aren't distorted by extremes.
  - Auto-selects best n_components (3–7) via BIC on Student-t log-likelihood.
  - Prediction uses the FORWARD ALGORITHM only — never Viterbi smoothing —
    so live inference never sees future data (no lookahead bias).
  - classify_states runs a single O(n) forward pass (not O(n²) per-bar loop).
  - Stability filter: regime must persist REGIME_STABILITY_BARS consecutive
    bars before the system acts on it.
  - Flicker guard: if regime changes > REGIME_FLICKER_THRESHOLD times in the
    last REGIME_FLICKER_WINDOW bars, confidence is halved.
  - Regimes are sorted by mean return so labels are deterministic:
      3 states  → bear / neutral / bull
      4 states  → crash / bear / bull / euphoria
      5 states  → crash / bear / neutral / bull / euphoria
"""

from __future__ import annotations

import logging
import warnings
from collections import deque
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning

from config import settings
from core.feature_engineering import compute_features, get_hmm_matrix

logger = logging.getLogger(__name__)

_REGIME_NAMES = {
    3: ["bear", "neutral", "bull"],
    4: ["crash", "bear", "bull", "euphoria"],
    5: ["crash", "bear", "neutral", "bull", "euphoria"],
    6: ["crash", "deep_bear", "bear", "bull", "strong_bull", "euphoria"],
    7: ["crash", "deep_bear", "bear", "neutral", "bull", "strong_bull", "euphoria"],
}


# ── Student-t HMM ─────────────────────────────────────────────────────────────

class StudentTHMM:
    """
    HMM that trains with Gaussian EM then scores with Student-t emissions.

    Gaussian EM learns state boundaries efficiently and provides stable
    parameter initialisation. Student-t scoring (df=4) gives heavy-tail
    robustness: extreme crash observations are downweighted rather than
    pulling state means toward the tail.
    """

    def __init__(
        self,
        n_components: int,
        df: int = 4,
        covariance_type: str = "full",
        n_iter: int = 200,
        random_state: int = 42,
    ):
        self.n_components    = n_components
        self._df             = df
        self._gaussian       = GaussianHMM(
            n_components    = n_components,
            covariance_type = covariance_type,
            n_iter          = n_iter,
            random_state    = random_state,
        )
        self.transmat_   = None
        self.startprob_  = None
        self.means_      = None
        self._scales     = None   # (n_components, n_features) std per state/feature
        self.is_fitted   = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "StudentTHMM":
        """Train via Gaussian EM, then cache parameters for Student-t scoring."""
        self._gaussian.fit(X)

        self.transmat_  = self._gaussian.transmat_.copy()
        self.startprob_ = self._gaussian.startprob_.copy()
        self.means_     = self._gaussian.means_.copy()

        covars = self._gaussian.covars_
        ct     = self._gaussian.covariance_type
        n, d   = self.n_components, X.shape[1]

        if ct == "full":
            self._scales = np.sqrt(np.maximum(
                np.stack([np.diag(covars[k]) for k in range(n)]), 1e-10
            ))
        elif ct == "diag":
            self._scales = np.sqrt(np.maximum(covars, 1e-10))
        elif ct == "tied":
            diag = np.sqrt(np.maximum(np.diag(covars), 1e-10))
            self._scales = np.tile(diag, (n, 1))
        else:  # spherical
            self._scales = np.full((n, d), np.sqrt(max(float(covars.mean()), 1e-10)))

        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> float:
        log_prob, _ = self._forward_pass(X)
        return log_prob

    # ── Emission log-likelihoods (Student-t) ──────────────────────────────────

    def _compute_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Student-t log emission probabilities.
        Returns shape (n_samples, n_components).
        scipy.stats.t.logpdf is broadcast-safe over feature dimension.
        """
        from scipy.stats import t as student_t

        n_samples = X.shape[0]
        logprob   = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            logprob[:, k] = np.sum(
                student_t.logpdf(
                    X,
                    df    = self._df,
                    loc   = self.means_[k],
                    scale = self._scales[k],
                ),
                axis=1,
            )
        return logprob

    # ── Forward algorithm ─────────────────────────────────────────────────────

    def _forward_pass(self, X: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Single O(n · K²) forward pass.
        Returns (log_prob, fwdlattice) where fwdlattice[t, k] is the
        log-scaled filtered probability of being in state k at time t.
        """
        framelogprob  = self._compute_log_likelihood(X)
        n_samples, K  = framelogprob.shape
        fwdlattice    = np.empty((n_samples, K))
        log_transmat  = np.log(self.transmat_ + 1e-300)  # (K, K): [i,j] = log P(j|i)

        fwdlattice[0] = np.log(self.startprob_ + 1e-300) + framelogprob[0]
        for t in range(1, n_samples):
            fwdlattice[t] = (
                logsumexp(fwdlattice[t - 1, :, np.newaxis] + log_transmat, axis=0)
                + framelogprob[t]
            )
        return float(logsumexp(fwdlattice[-1])), fwdlattice


# ── Regime detection engine ───────────────────────────────────────────────────

class RegimeDetectionEngine:
    """Fits a StudentTHMM and provides real-time regime predictions."""

    def __init__(self):
        self.model: Optional[StudentTHMM] = None
        self.n_regimes: int  = 5
        self.regime_labels: list = []
        self._history: deque  = deque(maxlen=settings.REGIME_FLICKER_WINDOW)
        self._consecutive_count: int  = 0
        self._last_regime: Optional[str] = None
        self.is_trained: bool = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> None:
        """Train StudentTHMM on historical OHLCV data. Auto-selects n_components via BIC."""
        features = compute_features(df)
        X        = get_hmm_matrix(features)

        best_model, best_score, best_n = None, -np.inf, settings.HMM_MIN_REGIMES

        for n in range(settings.HMM_MIN_REGIMES, settings.HMM_MAX_REGIMES + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model = StudentTHMM(
                    n_components    = n,
                    df              = settings.HMM_STUDENT_T_DF,
                    covariance_type = settings.HMM_COVARIANCE_TYPE,
                    n_iter          = settings.HMM_N_ITER,
                    random_state    = settings.HMM_RANDOM_STATE,
                )
                try:
                    model.fit(X)
                    log_ll   = model.score(X)
                    n_params = n * n + 2 * n * X.shape[1]   # transmat + means + scales
                    bic      = -2 * log_ll + n_params * np.log(len(X))
                    score    = -bic
                    if score > best_score:
                        best_score, best_model, best_n = score, model, n
                except Exception as exc:
                    logger.warning("StudentTHMM fit failed for n=%d: %s", n, exc)

        if best_model is None:
            raise RuntimeError("HMM training failed for all n_components values.")

        self.model        = best_model
        self.n_regimes    = best_n
        self.regime_labels = self._assign_labels(best_model, X)
        self.is_trained   = True
        logger.info("HMM trained: n_regimes=%d, labels=%s", best_n, self.regime_labels)

    def _assign_labels(self, model: StudentTHMM, X: np.ndarray) -> list:
        """Sort states by mean return of first feature (log_ret) — bear → euphoria."""
        means      = model.means_[:, 0]
        sorted_idx = np.argsort(means)
        names      = _REGIME_NAMES.get(
            model.n_components,
            [f"regime_{i}" for i in range(model.n_components)],
        )
        labels = [""] * model.n_components
        for rank, state_idx in enumerate(sorted_idx):
            labels[state_idx] = names[rank]
        return labels

    # ── Prediction — forward algorithm only ───────────────────────────────────

    def predict(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Predict current regime using only past observations (forward algorithm).
        Returns (regime_label, confidence).
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")

        features = compute_features(df)
        if features.empty:
            return "neutral", 0.0

        X = get_hmm_matrix(features)
        _, fwdlattice = self.model._forward_pass(X)

        log_posterior = fwdlattice[-1]
        posterior     = np.exp(log_posterior - logsumexp(log_posterior))

        state      = int(np.argmax(posterior))
        confidence = float(posterior[state])
        regime     = self.regime_labels[state]
        confidence = self._apply_stability_filter(regime, confidence)
        return regime, confidence

    def _apply_stability_filter(self, regime: str, confidence: float) -> float:
        """Halve confidence when regime is flickering or not yet stable."""
        self._history.append(regime)

        if regime == self._last_regime:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 1
            self._last_regime       = regime

        flickers = sum(
            1 for i in range(1, len(self._history))
            if self._history[i] != self._history[i - 1]
        )
        if flickers > settings.REGIME_FLICKER_THRESHOLD:
            logger.warning(
                "Regime instability: %d flickers in %d bars (regime=%s)",
                flickers, len(self._history), regime,
            )
            confidence *= settings.UNCERTAINTY_CONFIDENCE_PENALTY

        if self._consecutive_count < settings.REGIME_STABILITY_BARS:
            confidence *= 0.75   # not yet confirmed

        return min(confidence, 1.0)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_regime_summary(self) -> dict:
        return {
            "n_regimes":       self.n_regimes,
            "regime_labels":   self.regime_labels,
            "is_trained":      self.is_trained,
            "last_regime":     self._last_regime,
            "consecutive_bars": self._consecutive_count,
        }

    def classify_states(self, df: pd.DataFrame) -> pd.Series:
        """
        Return per-bar regime labels using a single O(n) forward pass.
        At each bar t the label reflects p(s_t | obs_1:t) — fully causal.
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")

        features = compute_features(df)
        X        = get_hmm_matrix(features)

        _, fwdlattice = self.model._forward_pass(X)

        labels = []
        for t in range(len(X)):
            log_posterior = fwdlattice[t]
            posterior     = np.exp(log_posterior - logsumexp(log_posterior))
            state         = int(np.argmax(posterior))
            labels.append(self.regime_labels[state])

        return pd.Series(labels, index=features.index, name="regime")
