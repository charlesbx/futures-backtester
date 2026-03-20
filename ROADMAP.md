# SRS Backtester — Roadmap

## Completed

### Infrastructure
- [x] Data pipeline: Databento DBN → Parquet cache (MES/MNQ 1-min, 2019-2026)
- [x] Backtester engine: bar-by-bar simulation with slippage + commissions
- [x] Metrics: win rate, PF, Sharpe, max drawdown, by-session/instrument/weekday
- [x] Optimizer: grid search with precomputation + walk-forward validation
- [x] Trade-level analysis: 12-chart diagnostic script (`scripts/analyze_trades.py`)

### Strategy 1: SRS (Session Range Strategy)
- [x] Baseline: PF=0.965, -$6,530 (not profitable)
- [x] Grid search (432 combos): best PF=1.101, +$15,642 (thin edge)
- [x] Walk-forward: OOS PF=1.029 (essentially breakeven)
- [x] Trade-level analysis: edge exists only in large ranges (30+ ticks), best config exits 80% at session end not TP
- [x] **Conclusion: not tradeable.** Breakout continuation is weak on MES/MNQ.

### Strategy 2: Intraday Momentum (Gao et al. JFE 2018)
- [x] Baseline: PF=0.869, -$12,739 (signal does not exist on futures)
- [x] Grid search (1,024 combos): zero combos profitable with 100+ trades
- [x] **Conclusion: dead signal on MES/MNQ.** The paper studied SPY ETF; the closing auction mechanism does not transfer to futures.

### Strategy 3: Overnight Gap Fade
- [x] Baseline: PF=0.952, WR=67.8% (mechanism works but edge too thin)
- [x] Grid search (180 combos): best PF=1.083, +$14,749, 37% of combos profitable
- [x] Walk-forward (9 windows): OOS PF=0.988 (breakeven, does not survive OOS)
- [x] **Key finding: 9:15 AM gap measurement (pre-market) >> 9:30 AM (regular open).** Overnight session close captures a stronger signal than the cash open.
- [x] **Conclusion: closest to viable but still not tradeable.** High win rate (67%), gaps do fill, but per-trade edge (~$2) is eaten by transaction costs.

## Learnings

1. **Simple single-factor intraday strategies on liquid index futures are picked clean.** SRS breakouts, intraday momentum, and gap fades are all well-known — the edge is priced in.
2. **Exit management matters more than entry.** SRS improved dramatically by disabling range reentry. Gap Fade's high win rate was offset by uncapped losers.
3. **The backtester infrastructure is production-ready.** Three strategies designed, implemented, optimized, and walk-forward validated. The framework handles everything from data loading to multi-parameter optimization.

## Next Steps (Potential)

### Multi-Factor Combinations
- Gap fade filtered by prior-day trend direction
- Gap fade filtered by VIX regime (elevated vol = stronger mean reversion)
- SRS breakout only when aligned with overnight gap direction
- Requires: VIX/VIX futures data, multi-factor signal framework

### Structural Strategy Changes
- Trailing stop instead of fixed TP/SL (SRS analysis showed most trades exit at session end)
- Adaptive position sizing based on gap size or volatility
- Partial exits (scale out at 50% fill, let rest ride)

### Different Market Microstructure
- Test on less liquid micro futures (MYM, M2K) where edges may persist longer
- Test during specific regimes only (high VIX, FOMC weeks, quarter-end)
- Overnight-only holds (buy close, sell open) — documented "overnight drift" anomaly

### Data Enhancements
- Add VIX/VIX futures data for regime filtering
- Add economic calendar data for event-based filtering
- Consider tick data for more precise entry/exit simulation
