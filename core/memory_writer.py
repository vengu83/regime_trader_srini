"""
Writes structured markdown files to the /memory folder after each trading session.
These files persist context between Claude Code routine executions, which are stateless.

Files written:
  memory/portfolio_state.md   — equity, cash, positions, drawdown, circuit breaker
  memory/regime_insights.md   — per-ticker regime history and current HMM state
  memory/trade_history.md     — last N trades with regime context and P&L
  memory/session_log.md       — latest session summary and next-session plan
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

from core.state_persistence import StateDB

logger = logging.getLogger(__name__)

MEMORY_DIR = "memory"


def _write(filename: str, content: str) -> None:
    os.makedirs(MEMORY_DIR, exist_ok=True)
    path = os.path.join(MEMORY_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.debug("Memory updated: %s", path)


class MemoryWriter:
    """Translates live trading state into Claude-readable markdown memory files."""

    def __init__(self, db: StateDB):
        self._db = db

    def write_portfolio_state(
        self,
        equity: float,
        cash: float,
        positions: dict,
        daily_pnl_pct: float = 0.0,
        weekly_pnl_pct: float = 0.0,
        peak_drawdown_pct: float = 0.0,
        circuit_breaker: bool = False,
    ) -> None:
        pos_lines = ""
        if positions:
            for ticker, pos in positions.items():
                pos_lines += (
                    f"| {ticker} | {pos.qty} | ${pos.notional:,.2f} "
                    f"| ${pos.avg_entry:.2f} "
                    f"| ${pos.unrealized_pnl:+.2f} ({pos.unrealized_pnl_pct:+.1%}) |\n"
                )
        else:
            pos_lines = "| — | — | — | — | — |\n"

        cb_flag = "🔴 ACTIVE — all trading halted" if circuit_breaker else "🟢 OK"
        deployed = equity - cash

        content = f"""# Portfolio State
_Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_

## Account
| Metric | Value |
|---|---|
| Equity | ${equity:,.2f} |
| Cash | ${cash:,.2f} |
| Deployed | ${deployed:,.2f} ({deployed / equity:.1%} of equity) |
| Daily P&L | {daily_pnl_pct:+.2%} |
| Weekly P&L | {weekly_pnl_pct:+.2%} |
| Peak Drawdown | {peak_drawdown_pct:.2%} |

## Circuit Breaker
{cb_flag}

## Open Positions
| Ticker | Qty | Notional | Avg Entry | Unrealized P&L |
|---|---|---|---|---|
{pos_lines}"""
        _write("portfolio_state.md", content)

    def write_regime_insights(self, engines: dict) -> None:
        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
        sections = [f"# Regime Insights\n_Last updated: {now}_\n"]

        for ticker, engine in engines.items():
            if not engine.is_trained:
                sections.append(f"## {ticker}\nNot trained.\n")
                continue

            summary = engine.get_regime_summary()
            recent = self._db.get_regime_history(ticker, limit=10)

            rows = ""
            for row in recent:
                confirmed = "✓" if row["is_confirmed"] else "·"
                rows += (
                    f"| {row['ts'][:16]} | {row['regime']} "
                    f"| {row['confidence']:.2f} | {confirmed} |\n"
                )

            labels_str = ", ".join(summary["regime_labels"])
            sections.append(f"""## {ticker}
- **Current regime**: `{summary['last_regime'] or 'unknown'}`
- **Consecutive bars in regime**: {summary['consecutive_bars']}
- **HMM states**: {summary['n_regimes']} ({labels_str})

### Recent detections (last 10 bars)
| Time (UTC) | Regime | Confidence | Confirmed |
|---|---|---|---|
{rows or '| — | — | — | — |\n'}
""")

        _write("regime_insights.md", "\n".join(sections))

    def write_trade_history(self, n: int = 20) -> None:
        trades = self._db.get_recent_trades(n)
        rows = ""
        for t in trades:
            pnl = t["realized_pnl"] or 0.0
            pnl_str = f"${pnl:+.2f}" if pnl != 0 else "—"
            rows += (
                f"| {t['ts'][:16]} | {t['ticker']} | {t['side'].upper()} "
                f"| {t['qty']} | ${t['fill_price']:.2f} | {t['regime']} "
                f"| {t['confidence']:.2f} | {pnl_str} |\n"
            )

        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
        content = f"""# Trade History
_Last updated: {now} — showing last {n} trades_

| Time (UTC) | Ticker | Side | Qty | Fill | Regime | Conf | Realized P&L |
|---|---|---|---|---|---|---|---|
{rows or '| — | — | — | — | — | — | — | — |\n'}"""
        _write("trade_history.md", content)

    def write_session_log(
        self,
        summary: str,
        next_actions: Optional[List[str]] = None,
    ) -> None:
        actions_md = ""
        if next_actions:
            actions_md = "\n## Next Session Actions\n" + "\n".join(f"- {a}" for a in next_actions)

        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
        content = f"""# Session Log
_Session ended: {now}_

## Summary
{summary}
{actions_md}"""
        _write("session_log.md", content)

    def write_all(
        self,
        equity: float,
        cash: float,
        positions: dict,
        daily_pnl_pct: float = 0.0,
        weekly_pnl_pct: float = 0.0,
        peak_drawdown_pct: float = 0.0,
        circuit_breaker: bool = False,
        engines: Optional[dict] = None,
        session_summary: str = "Routine bar-close update.",
        next_actions: Optional[List[str]] = None,
    ) -> None:
        """Write all memory files. Call once per bar-close cycle."""
        self.write_portfolio_state(
            equity, cash, positions,
            daily_pnl_pct, weekly_pnl_pct, peak_drawdown_pct, circuit_breaker,
        )
        if engines:
            self.write_regime_insights(engines)
        self.write_trade_history()
        self.write_session_log(session_summary, next_actions)
        logger.info("Memory files updated.")
