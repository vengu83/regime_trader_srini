# Strategy Rules
> **STATIC — never auto-overwritten by the bot. Edit manually or append lessons at the bottom.**

## Hard Rules (absolute — never override)

### Position Sizing
| Rule | Value |
|---|---|
| Max position size | 15% of portfolio (KELLY_MAX_POSITION_PCT) |
| Max portfolio risk per trade | 1% (MAX_POSITION_RISK_PCT) |
| Max total leverage | 1.25× (MAX_LEVERAGE) |
| Max open positions | 6 (MAX_CONCURRENT_POSITIONS) |
| Max trades per day | 6 across all tickers (MAX_DAILY_TRADES) |

### Circuit Breakers
| Trigger | Action |
|---|---|
| Daily loss ≥ 2% | Cut all new position sizes by 50% |
| Daily loss ≥ 3% | Close all positions, halt trading for the day |
| Weekly loss ≥ 5% | Resize all positions by 50% |
| Weekly loss ≥ 7% | Close all positions, halt for the week |
| Peak drawdown ≥ 10% | Write `logs/CIRCUIT_BREAKER.lock`, full stop — manual review required |

### Regime Allocations
| Regime | Equity % | Leverage | Posture |
|---|---|---|---|
| crash | 10% | 0.50× | Defensive — capital preservation above all |
| bear | 40% | 0.80× | Defensive — reduced exposure |
| neutral | 70% | 1.00× | Balanced — mean-reversion entries only |
| bull | 95% | 1.25× | Aggressive — momentum confirmation required |
| euphoria | 60% | 1.00× | Trim — market extended, reduce risk |

### Technical Confirmation Gates
- **Bull / strong_bull**: RSI(14) must be in [50, 75] AND MACD histogram must be positive and growing
- **Neutral**: Price must be in the lower 20% of the Bollinger Band (mean-reversion entry only)
- **Crash / Bear / Euphoria**: No technical gate — regime signal is sufficient

### Correlated Position Rules
| Pair | Action |
|---|---|
| SPY + QQQ, SPY + IVV, QQQ + TQQQ, TLT + IEF, GLD + IAU | Blocked — same risk factor |
| SPY + TLT | Allowed but scaled down 30% |

### Confidence Threshold
- Act on signal only when HMM confidence ≥ 0.60
- Below threshold → hold current allocation, no new orders
- Flickering regime (> 4 changes in 20 bars) → confidence auto-halved

### Stop-Loss Placement
- Stop distance = ATR(14) × multiplier (placed immediately after every buy)

| Regime | ATR Multiplier |
|---|---|
| crash | 3.0× |
| bear / neutral | 2.0× |
| bull / euphoria | 1.5× |

### Order Execution
- Default: limit order at ask + 10 bps
- Timeout: 30 seconds → fall back to market order
- Never place orders during market-closed hours

## What Claude Must NEVER Do
- Override or bypass the circuit breaker
- Delete or modify `logs/CIRCUIT_BREAKER.lock` programmatically
- Place orders exceeding MAX_DAILY_TRADES
- Exceed KELLY_MAX_POSITION_PCT for any position
- Trade outside 09:30–16:00 ET (US market hours)
- Skip `--dry-run` validation before placing real orders in a new session
- Modify `memory/strategy_rules.md` automatically (this file is human-maintained)
