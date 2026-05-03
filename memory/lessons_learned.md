# Lessons Learned
> Accumulated insights across trading sessions.
> **Append new lessons — do not delete old ones.**
> Format: `## YYYY-MM-DD — <topic>`

## Initial — HMM Design
- Student-t emissions (df=4) handle crash fat-tails better than Gaussian; extreme returns
  no longer distort the bear/bull state boundary.
- Stability filter (3 consecutive bars before acting) reduces whipsaw from single-bar
  regime flips, especially around macro announcements.
- BIC model selection (3–7 states) lets the data choose complexity; forcing 5 states
  sometimes merges bull and euphoria into one ambiguous state on calm markets.

## Initial — Risk Management
- Half-Kelly (fraction=0.50) meaningfully reduces drawdown vs full Kelly without
  sacrificing much upside; estimation error on a 63-day window is large.
- ATR-based stops adapt to current volatility; fixed-percentage stops are too wide in
  calm regimes and too tight in crash regimes.
- Correlation gate between SPY/QQQ prevents doubling the same beta exposure even
  when both show bull signals simultaneously.

## Initial — Strategy Layer
- Bull technical filter (RSI 50–75 AND MACD histogram positive + growing) prevents
  chasing moves that are already extended; it blocks ~20% of bull signals in backtests
  but improves the average trade quality.
- Bollinger Band mean-reversion entry for neutral regime works well in range-bound
  periods but lags in strong trending markets — watch bb_lower_pct carefully.
- Euphoria trim (60% equity, 1.0× leverage) historically captures most of the final
  leg while reducing exposure before the eventual drawdown.

## Open Questions
- Does Bollinger Band entry for neutral regime degrade in persistent trending markets?
- Is 63 days the right Kelly lookback, or should it adapt to the current volatility regime?
- Should euphoria trigger a short hedge (e.g., small TLT position) rather than just trimming?
