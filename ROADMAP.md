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

### Strategy 4: Asian Range (Breakout + Fade)
- [x] Baseline (breakout, midnight, opposite stop, RR=2.0): **PF=1.80, +$157,123, Sharpe=3.86, 3,486 trades**
- [x] Grid search (4,050 combos): **57.5% profitable**, mean PF=1.365, max PF=2.914, best PnL=$182k
- [x] **Breakout mode dominant**: 80% profitable (mean PF=1.65) vs fade 24% (mean PF=0.94)
- [x] **Best combo**: breakout, midnight, 9:30→2PM, opposite stop, RR=3.0 → PF=2.06, Sharpe=4.48
- [x] Walk-forward (9 windows): **OOS PF=2.985**, 100% positive (90/90), IS→OOS degradation only 5.6%
- [x] **Conclusion: TRADEABLE.** First strategy to survive OOS validation. Breakout of overnight Asian range (6PM-midnight ET) has a robust, persistent edge on MES/MNQ. Edge weakening in recent windows (2025 PF=1.20) but still profitable. Fade mode does not work (PF=0.94).

## Learnings

1. **Simple single-factor intraday strategies on liquid index futures are picked clean.** SRS breakouts, intraday momentum, and gap fades are all well-known — the edge is priced in.
2. **Exit management matters more than entry.** SRS improved dramatically by disabling range reentry. Gap Fade's high win rate was offset by uncapped losers.
3. **The backtester infrastructure is production-ready.** Four strategies designed, implemented, optimized, and walk-forward validated.
4. **Overnight range breakout is structurally different from intraday patterns.** The Asian range (6PM-midnight) captures institutional positioning during low-liquidity hours. The breakout during US hours exploits this information asymmetry — unlike SRS (same-session range) which competes against full market participation.
5. **Higher R:R ratios (3:1) outperform lower ones on range breakouts.** The best combos use RR=3.0 with ~50% win rate, earning 3x on winners vs 1x on losers. This contrasts with Gap Fade where high win rate + low R:R failed.
6. **Edge decay is real.** Asian range OOS PF declined from 6.1 (2022) to 1.2 (2025) — the strategy is becoming more crowded. Monitor ongoing performance closely.

### Engine Refactor: Strategy-Agnostic Architecture
- [x] `BaseStrategy` ABC with plugin interface (`generate_signals`, `simulate_trade`, `build_param_grid`, etc.)
- [x] Auto-discovery registry: strategies in `src/strategies/` are auto-registered
- [x] Generic backtest engine (`src/backtester.py`): runs any BaseStrategy on OHLCV data
- [x] Generic optimizer (`src/optimizer.py`): grid search + walk-forward for any strategy
- [x] All 4 strategies migrated to BaseStrategy interface (SRS, Asian Range, IntMom, GapFade)
- [x] Per-strategy optimized `run_grid_search` with precomputation preserved
- [x] Unified CLI: `python run.py <strategy> [--optimize|--walk-forward|--report]`
- [x] Zero test regression (33/33 tests pass)
- [x] Old monolithic strategy files removed; all imports updated

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
