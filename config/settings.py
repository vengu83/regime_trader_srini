"""Central configuration for the regime_trader system."""
from typing import List, Dict

# ── Broker ────────────────────────────────────────────────────────────────────
BROKER = "alpaca"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# ── Universe ──────────────────────────────────────────────────────────────────
TICKERS: List[str] = ["SPY", "QQQ", "TLT", "GLD"]
BENCHMARK_TICKER: str = "SPY"

# ── HMM parameters ────────────────────────────────────────────────────────────
HMM_MIN_REGIMES: int = 3
HMM_MAX_REGIMES: int = 7
HMM_TRAINING_DAYS: int = 504          # ~2 years of daily bars
HMM_COVARIANCE_TYPE: str = "full"
HMM_N_ITER: int = 200
HMM_RANDOM_STATE: int = 42
HMM_STUDENT_T_DF: int = 4             # Student-t degrees of freedom (fat-tail robustness)

# Stability filter – regime must persist this many bars before acting
REGIME_STABILITY_BARS: int = 3
REGIME_FLICKER_THRESHOLD: int = 4
REGIME_FLICKER_WINDOW: int = 20

# ── Allocation / strategy ──────────────────────────────────────────────────────
REGIME_ALLOCATIONS: Dict[str, Dict] = {
    "crash":    {"equity_pct": 0.10, "leverage": 0.50, "strategy": "defensive",  "atr_stop_mult": 3.0},
    "bear":     {"equity_pct": 0.40, "leverage": 0.80, "strategy": "defensive",  "atr_stop_mult": 2.0},
    "neutral":  {"equity_pct": 0.70, "leverage": 1.00, "strategy": "balanced",   "atr_stop_mult": 2.0},
    "bull":     {"equity_pct": 0.95, "leverage": 1.25, "strategy": "aggressive", "atr_stop_mult": 1.5},
    "euphoria": {"equity_pct": 0.60, "leverage": 1.00, "strategy": "trim",       "atr_stop_mult": 1.5},
}

# Technical signal filter thresholds
TECH_RSI_BULL_LO: float = 50.0        # bull confirmation: RSI must be above this
TECH_RSI_BULL_HI: float = 75.0        # bull confirmation: RSI must be below this
TECH_BB_ENTRY_PCT: float = 0.20       # neutral entry: price in lower 20% of BB band

# Minimum HMM confidence before acting on a signal
MIN_CONFIDENCE: float = 0.60
UNCERTAINTY_CONFIDENCE_PENALTY: float = 0.50

# ── Kelly position sizing ──────────────────────────────────────────────────────
KELLY_LOOKBACK_DAYS: int = 63         # rolling window for win-rate / payoff estimation
KELLY_FRACTION: float = 0.50          # half-Kelly to reduce estimation error sensitivity
KELLY_MAX_POSITION_PCT: float = 0.15  # hard cap: 15% of portfolio per position
MAX_CORRELATION_BLOCK: float = 0.85   # reject new position if correlation >= this
MAX_CORRELATION_SCALE: float = 0.70   # scale position if correlation >= this

# ── Risk management ────────────────────────────────────────────────────────────
MAX_POSITION_RISK_PCT: float = 0.01   # 1% of portfolio at risk per trade
MIN_ORDER_NOTIONAL: float = 500.0     # reject orders below this dollar value
MAX_LEVERAGE: float = 1.25
MAX_DAILY_TRADES: int = 6             # daily trade cap across all tickers
MAX_CONCURRENT_POSITIONS: int = 6     # max open positions at once
ATR_GAP_MULTIPLIER: float = 3.0       # overnight gap protection multiplier on stop distance

# Circuit breakers
CB_DAILY_LOSS_HALVE: float = -0.02    # −2% daily  → cut all sizes 50%
CB_DAILY_LOSS_CLOSE: float = -0.03    # −3% daily  → close all, halt day
CB_WEEKLY_LOSS_RESIZE: float = -0.05  # −5% weekly → resize all 50%
CB_WEEKLY_LOSS_CLOSE: float = -0.07   # −7% weekly → close all, halt week
CB_PEAK_DRAWDOWN_STOP: float = -0.10  # −10% peak  → write lock file, stop bot
LOCK_FILE_PATH: str = "logs/CIRCUIT_BREAKER.lock"

MAX_CORRELATION: float = 0.70

# ── Order execution ────────────────────────────────────────────────────────────
ORDER_LIMIT_OFFSET_BPS: float = 10.0  # 0.1% offset above ask for limit buys
ORDER_TIMEOUT_SECS: int = 30          # seconds before falling back to market order

# ── Backtester ─────────────────────────────────────────────────────────────────
BT_IN_SAMPLE_DAYS: int = 252
BT_OUT_SAMPLE_DAYS: int = 126
BT_SLIPPAGE_BPS: float = 5.0
BT_COMMISSION_BPS: float = 0.0
STRESS_SHOCKS: List[float] = [-0.10, -0.15, -0.20]

# ── Main loop ──────────────────────────────────────────────────────────────────
BAR_TIMEFRAME: str = "5Min"
MARKET_DATA_LOOKBACK_DAYS: int = 30

# ── Persistence ────────────────────────────────────────────────────────────────
SQLITE_DB_PATH: str = "logs/trader_state.db"
TRADE_LOG_PATH: str = "logs/trade_history.csv"

# ── Perplexity research ────────────────────────────────────────────────────────
PERPLEXITY_MODEL: str = "sonar-pro"          # sonar-pro has real-time web search
PERPLEXITY_MAX_TOKENS: int = 2048
PERPLEXITY_TEMPERATURE: float = 0.1          # low temp for factual research

# ── Monitoring / alerting ──────────────────────────────────────────────────────
LOG_LEVEL: str = "INFO"
DASHBOARD_REFRESH_SECS: int = 60
ALERT_EMAIL: str = ""
ALERT_WEBHOOK: str = ""
