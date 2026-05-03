# RegimeTrader — Claude Code Agent Guide

## What This System Does
RegimeTrader is an HMM-based automated trading bot for US equities (SPY, QQQ, TLT, GLD).
It detects market regimes (crash / bear / neutral / bull / euphoria) using a Hidden Markov
Model with Student-t emissions, then applies regime-appropriate position sizing via half-Kelly.
All risk limits are enforced by `core/risk_manager.py` and cannot be bypassed.

## Memory Files — Read These First
Before every routine, read these files **in this order**:

| File | Purpose |
|---|---|
| `memory/strategy_rules.md` | Hard limits — absolute constraints on every decision |
| `memory/portfolio_state.md` | Current equity, positions, circuit breaker status |
| `memory/regime_insights.md` | Latest HMM regime per ticker and confidence |
| `memory/session_log.md` | What happened last session, pending actions |
| `memory/trade_history.md` | Last 20 trades with regime context and P&L |
| `memory/lessons_learned.md` | Accumulated strategy insights |
| `memory/market_research.md` | Today's macro context and news (pre-market only) |

## WHAT CLAUDE MUST NEVER DO
> These rules are non-negotiable. They live here AND in every routine prompt.

- **NEVER place real orders without `--live` flag.** Default mode is always paper/dry-run.
- **NEVER trade options, futures, or leveraged ETFs** (TQQQ, SQQQ, UVXY, etc.) — equities only.
- **NEVER exceed 6 trades per day** (MAX_DAILY_TRADES enforced in code).
- **NEVER act when circuit breaker is active.** Read `memory/portfolio_state.md` FIRST. If circuit breaker shows 🔴, stop immediately.
- **NEVER override or delete `logs/CIRCUIT_BREAKER.lock`** — that is a human-only action after manual review.
- **NEVER modify `memory/strategy_rules.md`** — it is human-maintained. Suggestions go in `session_log.md`.
- **NEVER send alerts/notifications unless a trade actually executed** or a circuit breaker fired.
- **NEVER trade during market-closed hours** (outside 09:30–16:00 ET, Mon–Fri).
- **NEVER remove stop-loss orders** — every long position must have an ATR stop in place.
- **NEVER use leverage above 1.25×** total portfolio, regardless of regime signal strength.
- **Read `memory/strategy_rules.md` FIRST at the start of every session.** No exceptions.

## CLI Commands

```bash
# Analysis only — NO orders placed (DEFAULT — use this most of the time)
python main.py --dry-run

# Same as above — paper mode is the default even without --dry-run
python main.py

# REAL orders — requires explicit --live flag (do NOT run without reviewing dry-run first)
python main.py --live

# Pre-market Perplexity research → writes memory/market_research.md
python main.py --research

# Train HMMs on all tickers and exit
python main.py --train-only

# Walk-forward backtest across all tickers
python main.py --backtest

# Backtest with stress tests
python main.py --backtest --stress-test
```

## Daily Routines

### 1. Pre-Market (08:30 ET — before open)
**Goal**: Research and validate the trading plan for the day.

1. Read all memory files.
2. Search for macro news, Fed commentary, scheduled data releases (CPI, NFP, FOMC).
3. Check for material news on SPY, QQQ, TLT, GLD.
4. Update `memory/market_research.md` with findings.
5. Cross-check: does today's news align with the current HMM regime?
6. If there is a conflict (e.g., HMM says bull but macro is deteriorating), note it in
   `memory/session_log.md` and plan to reduce confidence threshold for the day.
7. Run `python main.py --dry-run` to preview what signals the system would generate.

### 2. Market Open (09:30–10:00 ET)
**Goal**: Execute the opening rebalance if signals warrant it.

1. Read memory files.
2. Check `memory/portfolio_state.md` — is the circuit breaker active?
   - If active: do NOT proceed. Note in session log.
3. Run `python main.py --dry-run` to see pending orders.
4. If signals look valid and align with market_research notes, run `python main.py`.
5. Let the bot handle execution — do not place manual orders outside the system.
6. After the first bar: update `memory/session_log.md` with what was executed.

### 3. Midday Scan (12:00 ET)
**Goal**: Monitor positions, check for stop-loss triggers or regime shifts.

1. Read `memory/portfolio_state.md` and `memory/regime_insights.md`.
2. Check if any regime changed since open — if so, note in session log.
3. If the circuit breaker has activated, alert via Alpaca dashboard and stop.
4. No new entries at midday unless a significant regime shift occurred.
5. Update `memory/session_log.md` with midday status.

### 4. End of Day (15:45 ET — before close)
**Goal**: Review the day's performance and update memory.

1. Read all memory files.
2. Summarise: which regimes were active, which trades fired, daily P&L.
3. Append any new observations to `memory/lessons_learned.md` if warranted.
4. Update `memory/session_log.md` with end-of-day summary and tomorrow's plan.
5. The bot auto-updates `portfolio_state.md`, `regime_insights.md`, and `trade_history.md`
   after each bar — no manual file updates needed for those.

### 5. Weekly Review (Friday 16:30 ET)
**Goal**: Assess performance, tune rules, plan next week.

1. Read all memory files.
2. Run `python main.py --backtest` for the latest walk-forward metrics.
3. Assess: Sharpe, max drawdown, regime accuracy, win rate by regime.
4. Review `memory/trade_history.md` — any patterns in losers?
5. Update `memory/lessons_learned.md` with weekly insights.
6. Confirm `memory/strategy_rules.md` reflects current intentions.
7. Write next-week plan to `memory/session_log.md`.

## Architecture Overview

```
main.py
  ├── MarketDataProvider          — fetches OHLCV from Alpaca
  ├── RegimeDetectionEngine       — StudentT-HMM, forward algorithm only
  │     └── compute_features()   — 10 HMM features + MACD/BB/ATR strategy features
  ├── StrategyOrchestrator        — routes regime → allocation → technical filter
  ├── RiskManager                 — half-Kelly sizing, circuit breakers, correlation gate
  ├── OrderExecutor               — limit orders with ATR stops
  ├── PositionTracker             — mirrors live Alpaca positions
  ├── StateDB (SQLite WAL)        — equity_curve, regime_history, trade_log, snapshot
  └── MemoryWriter                — writes memory/*.md after each bar-close
```

## Key Parameters (config/settings.py)

| Parameter | Value | Meaning |
|---|---|---|
| TICKERS | SPY, QQQ, TLT, GLD | Trading universe |
| HMM_TRAINING_DAYS | 504 | ~2 years of training data |
| MIN_CONFIDENCE | 0.60 | Minimum HMM confidence to act |
| KELLY_FRACTION | 0.50 | Half-Kelly multiplier |
| KELLY_MAX_POSITION_PCT | 0.15 | Hard cap per position |
| MAX_LEVERAGE | 1.25 | Portfolio-level leverage cap |
| CB_PEAK_DRAWDOWN_STOP | -0.10 | Lock file threshold |

## Safety Checklist Before Any Live Run
- [ ] `memory/strategy_rules.md` reflects your current risk tolerance
- [ ] `logs/CIRCUIT_BREAKER.lock` does NOT exist (or was reviewed and deleted)
- [ ] `.env` has paper-trading credentials (not live) for initial runs
- [ ] `--dry-run` has been run and signals look correct
- [ ] No pending macro events (FOMC, CPI) in the next 2 hours

## Modifying the System
- To change risk limits: edit `config/settings.py` — parameters are documented inline.
- To add a new ticker: add to `settings.TICKERS` and extend the correlation pairs table
  in `core/risk_manager.py:_correlation_action`.
- To add a new regime strategy: subclass `BaseStrategy` in `core/regime_strategies.py`
  and register it in `_STRATEGY_MAP`.
- Do not modify `core/hmm_engine.py` without running the full test suite:
  `python -m pytest tests/ -v`
