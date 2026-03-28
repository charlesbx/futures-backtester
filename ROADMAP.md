# Futures Backtester — ROADMAP

## High Priority

- [x] **Coarse-to-fine grid search** — Add `coarse_to_fine_grid_search()` to `optimizer.py`. Phase 1: sample every Nth combination (stride parameter). Early exit if first 25% of coarse has no PF > threshold. Phase 2: build refined grid around top 5 coarse results. Same output format as `generic_grid_search()`. Add `--coarse-to-fine` flag to `run.py` CLI. Expected ~85% reduction in total backtests for large grids (6,561 → ~980).
- [x] **Grid search progress file** — Both `generic_grid_search()` and `coarse_to_fine_grid_search()` should write a `_progress.json` file during execution with: phase, total combos, completed count, pct_complete, best_pf_so_far, elapsed_seconds, estimated_remaining_seconds. Add `progress_file` parameter to both functions. Add `--progress-file` flag to `run.py`.
- [x] **Walk-forward progress file** — `generic_walk_forward()` should write progress to a `_progress.json` file: current window, total windows, pct_complete, elapsed_seconds.
- [x] **Fix `binom_test` deprecation in optimizer.py** — Replace `binom_test` with `scipy.stats.binomtest(...).pvalue` for SciPy 1.12+ compatibility.
- [ ] **Add logging instead of print() statements** — Replace all strategy-level print() calls with a proper logging module (Python's `logging`). Add `--log-level` CLI arg for controlling verbosity.
- [ ] **Implement comprehensive error handling in data loading** — Add proper error hierarchy, guard clauses on all pipeline functions, DST boundary handling in data_loader.py.
- [ ] **Add test coverage for all core modules** — Write tests for backtester.py, metrics.py, optimizer.py, data_loader.py with synthetic fixtures. Target 80%+ coverage.
- [ ] **Validate timezone handling across DST transitions** — Add DST-specific tests covering spring-forward, fall-back, session detection, and rollover near DST boundaries.
- [ ] **Add integration tests for end-to-end backtest workflows** — E2E tests covering full pipeline for each strategy with synthetic data.
- [ ] **Extract hardcoded parameters into configuration** — Create a config module (dataclass + TOML file) so users can tune parameters without editing source. Add `--config` CLI flag.

## Medium Priority

- [ ] **Rename `moyenne` → `mean` in metrics.py** — French naming inconsistency in all aggregation groups.
- [ ] **Add statistical significance testing framework** — Bootstrap Sharpe CI, paired bootstrap tests, permutation tests, BH FDR correction. Fix Sharpe CI to use daily PnL (not trade-level).
- [ ] **Standardize report generation and formatting** — Self-contained HTML reports with equity curve, drawdown, monthly heatmap, and exit reason charts (inline SVG).
- [ ] **Implement continuous front-month series validation** — Tests for rollover: no duplicate timestamps, monotonic index, volume-based roll selection, price continuity.
- [ ] **Create contributor guide for adding new strategies** — CONTRIBUTING.md with plugin architecture overview, skeleton strategy, testing checklist.

## Low Priority

- [ ] **Fix stale error message in run.py** — Update to reference current data paths and modules.
- [ ] **Add caching layer for processed Parquet files** — In-memory cache with get/put/clear/stats, module-level singleton, `--no-cache` CLI flag.
- [ ] **Implement results database (SQLite)** — Structured results storage with runs, metrics, trades tables. Dedup detection, WAL mode, backward-compatible with flat files.
- [ ] **Add command-line progress indicators** — Centralized progress bar module wrapping tqdm. Auto-detect TTY.
- [ ] **Implement Monte Carlo resampling for robustness testing** — Bootstrap and shuffle methods, CI bounds, probability of ruin, `--monte-carlo` CLI flag.
