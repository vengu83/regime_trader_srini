"""
RegimeTrader — main orchestration loop.

Usage:
  python main.py                          # live / paper trading
  python main.py --dry-run               # simulate signals without placing orders
  python main.py --train-only            # train HMMs, persist state, exit
  python main.py --backtest              # walk-forward backtest on all tickers, exit
  python main.py --backtest --stress-test

Startup sequence (live mode):
  1. Parse CLI args.
  2. Initialise SQLite DB.
  3. Connect to Alpaca, verify account.
  4. Check lock file.
  5. Train HMMs (or load if --train-only exits after this).
  6. Initialise risk manager and strategy orchestrator.
  7. Enter main bar-close loop.
  8. Graceful shutdown on KeyboardInterrupt.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime

from config import settings
from brokers.alpaca_broker import AlpacaBroker
from brokers.order_executor import OrderExecutor
from brokers.position_tracker import PositionTracker, TradeRecord
from core.alerts import AlertManager
from core.feature_engineering import compute_features
from core.hmm_engine import RegimeDetectionEngine
from core.market_data import MarketDataProvider
from core.memory_writer import MemoryWriter
from core.regime_strategies import StrategyOrchestrator
from core.risk_manager import RiskManager
from core.state_persistence import StateDB, init_db

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/regime_trader.log"),
    ],
)
logger = logging.getLogger("main")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RegimeTrader — HMM-based automated trading bot")
    p.add_argument("--dry-run",     action="store_true", help="Generate signals but do not place orders")
    p.add_argument("--live",        action="store_true", help="Enable real order placement (default is paper/dry-run)")
    p.add_argument("--train-only",  action="store_true", help="Train HMMs and exit")
    p.add_argument("--backtest",    action="store_true", help="Run walk-forward backtest and exit")
    p.add_argument("--stress-test", action="store_true", help="Include stress tests in backtest report")
    p.add_argument("--research",    action="store_true", help="Run pre-market Perplexity research and write memory/market_research.md, then exit")
    return p.parse_args()


# ── Backtest entry point ───────────────────────────────────────────────────────

def run_backtest(stress: bool = False) -> None:
    from core.back_tester import WalkForwardBacktester
    data_prov = MarketDataProvider()
    for ticker in settings.TICKERS:
        logger.info("Backtesting %s …", ticker)
        try:
            df     = data_prov.get_history(ticker, days=settings.HMM_TRAINING_DAYS + settings.BT_OUT_SAMPLE_DAYS * 4)
            bt     = WalkForwardBacktester()
            report = bt.run(df)
            m      = report.aggregate_metrics
            logger.info(
                "[%s] Sharpe=%.2f | Ann.Ret=%.1f%% | MaxDD=%.1f%% | Folds=%d",
                ticker,
                m.get("sharpe", 0),
                m.get("ann_return", 0) * 100,
                m.get("max_drawdown", 0) * 100,
                len(report.folds),
            )
            if stress and not report.stress_results.empty:
                logger.info("[%s] Stress results:\n%s", ticker, report.stress_results.to_string())
        except Exception as exc:
            logger.error("Backtest failed for %s: %s", ticker, exc)


# ── Main trading loop ─────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # ── 1. SQLite init ────────────────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    init_db()
    db = StateDB()

    # ── 2a. Research shortcut ─────────────────────────────────────────────────
    if args.research:
        from core.research_provider import PerplexityResearchProvider
        logger.info("Running pre-market Perplexity research …")
        rp = PerplexityResearchProvider()
        rp.run()
        db.close()
        return

    # ── 2b. Backtest shortcut ─────────────────────────────────────────────────
    if args.backtest:
        logger.info("Running walk-forward backtest …")
        run_backtest(stress=args.stress_test)
        db.close()
        return

    # ── Paper-mode default: real orders require explicit --live flag ──────────
    # Without --live the bot NEVER places orders, regardless of .env settings.
    if not args.live:
        args.dry_run = True
        logger.info(
            "Paper mode active (--live not passed). "
            "Signals will be logged but NO orders will be placed."
        )
    else:
        logger.warning(
            "⚠  LIVE TRADING ENABLED (--live flag). Real orders will be placed. "
            "Ensure .env credentials are correct and circuit breaker is clear."
        )

    logger.info("=" * 60)
    logger.info(
        "RegimeTrader starting up … live=%s dry_run=%s",
        args.live, args.dry_run,
    )
    logger.info("=" * 60)

    # ── 3. Connect to Alpaca ──────────────────────────────────────────────────
    broker = AlpacaBroker()
    if not broker.connect():
        logger.critical("Cannot connect to Alpaca. Check your .env credentials.")
        sys.exit(1)

    account = broker.get_account()
    equity  = account["equity"]
    logger.info("Account equity: $%.2f | buying power: $%.2f", equity, account["buying_power"])

    executor  = OrderExecutor(broker)
    tracker   = PositionTracker(broker)
    alerts    = AlertManager()
    data_prov = MarketDataProvider()

    # ── 4. Lock file check ────────────────────────────────────────────────────
    if os.path.exists(settings.LOCK_FILE_PATH):
        logger.critical(
            "Lock file found at %s — bot is halted. "
            "Review strategy and delete the file to restart.",
            settings.LOCK_FILE_PATH,
        )
        sys.exit(1)

    # ── 5. Train HMMs ─────────────────────────────────────────────────────────
    engines: dict[str, RegimeDetectionEngine] = {}
    for ticker in settings.TICKERS:
        logger.info("Training HMM for %s …", ticker)
        try:
            df     = data_prov.get_history(ticker, days=settings.HMM_TRAINING_DAYS)
            engine = RegimeDetectionEngine()
            engine.train(df)
            engines[ticker] = engine
            logger.info(
                "HMM ready for %s (%d regimes: %s)",
                ticker, engine.n_regimes, engine.regime_labels,
            )
        except Exception as exc:
            logger.error("HMM training failed for %s: %s", ticker, exc)

    if not engines:
        logger.critical("No HMMs trained. Exiting.")
        sys.exit(1)

    if args.train_only:
        logger.info("--train-only flag set — exiting after training.")
        db.close()
        return

    # ── 6. Initialise risk manager ────────────────────────────────────────────
    risk_mgr      = RiskManager(initial_equity=equity)
    orchestrator  = StrategyOrchestrator()
    memory_writer = MemoryWriter(db)

    # ── 6a. Pre-market research (best-effort — skipped if key not set) ────────
    from core.research_provider import PerplexityResearchProvider
    regime_map = {t: e._last_regime or "unknown" for t, e in engines.items()}
    PerplexityResearchProvider().run(regime_map=regime_map)

    logger.info("RegimeTrader initialised. Entering main loop …")
    alerts.send("RegimeTrader Started", f"Equity: ${equity:,.2f} | dry_run={args.dry_run}")

    # ── 7. Main bar-close loop ────────────────────────────────────────────────
    try:
        while True:
            account = broker.get_account()
            if account is None:
                logger.warning("Could not fetch account. Sleeping.")
                time.sleep(60)
                continue

            equity    = account["equity"]
            positions = tracker.get_positions()

            risk_status = risk_mgr.check_all(equity, positions)
            for msg in risk_status.messages:
                logger.warning("RISK: %s", msg)
                alerts.send("Risk Alert", msg)

            if risk_status.circuit_breaker_active:
                logger.critical("Circuit breaker active — closing all positions.")
                if not args.dry_run:
                    executor.close_all_positions()
                alerts.send("Circuit Breaker Triggered", "\n".join(risk_status.messages))
                db.upsert_snapshot(equity, account.get("cash", 0), "HALTED", True)
                if risk_status.lock_file_exists:
                    logger.critical("Lock file written. Bot halted until manual review.")
                    break
                time.sleep(300)
                continue

            if not broker.is_market_open():
                logger.info("Market closed — waiting …")
                time.sleep(60)
                continue

            # Process each ticker
            for ticker in settings.TICKERS:
                engine = engines.get(ticker)
                if engine is None:
                    continue

                try:
                    recent_df = data_prov.build_live_window(
                        ticker,
                        lookback_days=settings.MARKET_DATA_LOOKBACK_DAYS,
                        alpaca_client=broker,
                    )
                    regime, confidence = engine.predict(recent_df)
                    features           = compute_features(recent_df)

                    if features.empty or len(features) < 20:
                        logger.warning(
                            "[%s] Insufficient features after NaN drop (%d rows) — skipping bar.",
                            ticker, len(features),
                        )
                        continue

                    # Extract ATR and recent returns for risk manager
                    atr = float(features["atr_14"].iloc[-1]) if "atr_14" in features else None
                    recent_returns = (
                        features["log_ret"].tail(settings.KELLY_LOOKBACK_DAYS)
                        if "log_ret" in features else None
                    )

                    signal = orchestrator.get_signal(regime, confidence, features)

                    is_confirmed = confidence >= settings.MIN_CONFIDENCE
                    db.record_regime(ticker, regime, confidence, is_confirmed)

                    logger.info(
                        "[%s] regime=%s conf=%.2f exposure=%.0f%% strategy=%s",
                        ticker, regime, confidence,
                        signal.effective_exposure * 100, signal.strategy_name,
                    )

                    current_pos  = positions.get(ticker)
                    current_qty  = current_pos.qty if current_pos else 0
                    price        = broker.get_current_price(ticker)
                    if price is None:
                        continue

                    current_pct = (current_qty * price) / equity if equity > 0 else 0.0
                    if not orchestrator.rebalance_needed(current_pct, signal.effective_exposure):
                        continue

                    size_result = risk_mgr.size_position(
                        equity             = equity,
                        price              = price,
                        signal_equity_pct  = signal.effective_exposure,
                        risk_status        = risk_status,
                        existing_positions = {k: {"notional": v.notional} for k, v in positions.items()},
                        ticker             = ticker,
                        recent_returns     = recent_returns,
                        atr                = atr,
                        buying_power       = account.get("buying_power"),
                    )

                    if not size_result.approved:
                        logger.info("[%s] Order rejected: %s", ticker, size_result.reason)
                        continue

                    target_qty = size_result.shares
                    delta_qty  = target_qty - current_qty

                    if not args.dry_run and delta_qty != 0:
                        if delta_qty > 0:
                            order = executor.buy(ticker, delta_qty)
                        else:
                            order = executor.sell(ticker, abs(delta_qty))

                        if order and delta_qty > 0 and atr is not None:
                            executor.place_atr_stop(ticker, price, atr, regime, qty=target_qty)

                        risk_mgr.record_trade(ticker)

                    elif args.dry_run and delta_qty != 0:
                        action = "BUY" if delta_qty > 0 else "SELL"
                        logger.info("[DRY-RUN] Would %s %d %s", action, abs(delta_qty), ticker)

                    if delta_qty != 0:
                        tracker.record_trade(TradeRecord(
                            ticker         = ticker,
                            side           = "buy" if delta_qty > 0 else "sell",
                            qty            = abs(delta_qty),
                            entry_price    = price,
                            exit_price     = None,
                            realized_pnl   = 0.0,
                            regime         = regime,
                            confidence     = confidence,
                            allocation_pct = signal.effective_exposure,
                        ))
                        db.record_trade(
                            ticker         = ticker,
                            side           = "buy" if delta_qty > 0 else "sell",
                            qty            = abs(delta_qty),
                            fill_price     = price,
                            regime         = regime,
                            confidence     = confidence,
                            kelly_fraction = size_result.kelly_fraction,
                            order_type     = "dry_run" if args.dry_run else "limit",
                        )

                except Exception as exc:
                    logger.error("[%s] Error in trading loop: %s", ticker, exc)

            db.record_equity(equity)
            cash = account.get("cash", 0)
            db.upsert_snapshot(equity, cash, regime if engines else "unknown", False)

            memory_writer.write_all(
                equity             = equity,
                cash               = cash,
                positions          = positions,
                daily_pnl_pct      = risk_status.daily_pnl_pct,
                weekly_pnl_pct     = risk_status.weekly_pnl_pct,
                peak_drawdown_pct  = risk_status.peak_drawdown_pct,
                circuit_breaker    = risk_status.circuit_breaker_active,
                engines            = engines,
                session_summary    = (
                    f"Bar closed. Equity=${equity:,.2f} | "
                    f"daily={risk_status.daily_pnl_pct:+.2%} | "
                    f"weekly={risk_status.weekly_pnl_pct:+.2%} | "
                    f"positions={len(positions)}"
                ),
            )

            logger.info("Bar processed. Sleeping %d seconds …", 300)
            time.sleep(300)

    except KeyboardInterrupt:
        logger.info("Shutdown requested by user.")
        alerts.send("RegimeTrader Stopped", "Manual shutdown via KeyboardInterrupt.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
