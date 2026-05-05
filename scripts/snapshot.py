"""
One-shot regime + signal snapshot for GitHub Actions.
Trains HMMs on all tickers using yfinance, computes current regime and signal,
writes memory/regime_insights.md, and sends a Telegram notification.

Usage:
  python scripts/snapshot.py "Market Open"
  python scripts/snapshot.py "Midday Scan"
  python scripts/snapshot.py "End of Day"
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from core.feature_engineering import compute_features
from core.hmm_engine import RegimeDetectionEngine
from core.market_data import MarketDataProvider
from core.regime_strategies import StrategyOrchestrator
import notify


def main(label: str = "Snapshot") -> None:
    data_prov    = MarketDataProvider()
    orchestrator = StrategyOrchestrator()
    today        = datetime.utcnow().strftime("%Y-%m-%d")

    notify.send(f"[RegimeTrader] {label} starting — {today}")

    lines        = []
    md_lines     = [f"# Regime Insights\n\n_Updated {today} by {label}_\n\n## Current Regimes\n"]

    for ticker in settings.TICKERS:
        try:
            df      = data_prov.get_history(ticker, days=settings.HMM_TRAINING_DAYS)
            engine  = RegimeDetectionEngine()
            engine.train(df)

            recent   = data_prov.get_history(ticker, days=90)  # mom_63 needs ≥63 bars
            regime, conf = engine.predict(recent)
            features = compute_features(recent)
            sig      = orchestrator.get_signal(regime, conf, features)

            line = (
                f"{ticker}: {regime} ({conf:.0%}) "
                f"→ {sig.effective_exposure:.0%} [{sig.strategy_name}]"
            )
            lines.append(line)
            md_lines.append(f"- **{ticker}**: {regime} | conf={conf:.0%} | "
                            f"exposure={sig.effective_exposure:.0%} | {sig.strategy_name}")
            print(line, flush=True)
        except Exception as exc:
            err = f"{ticker}: ERROR — {exc}"
            lines.append(err)
            md_lines.append(f"- **{ticker}**: ERROR — {exc}")
            print(err, flush=True)

    # Write regime_insights.md
    os.makedirs("memory", exist_ok=True)
    with open("memory/regime_insights.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    summary = f"[RegimeTrader] {label} ✅\n{today}\n" + "\n".join(lines)
    notify.send(summary)
    print(summary, flush=True)


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "Snapshot"
    main(label)
