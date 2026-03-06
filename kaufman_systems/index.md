# Kaufman Systems — Master Implementation Plan (~70 Systems)

Research-grade trading systems library based on Perry Kaufman's
*Trading Systems and Methods*. Each system exposes a standard interface:

```python
class System(TradingSystem):
    def signal(self, data) -> int        # 1 long, -1 short, 0 flat
    def position_sizing(self, data, risk) -> float
    def risk_filter(self, data) -> bool
    def indicators(self, data) -> dict   # optional diagnostics
```

Base class: `kaufman_systems/base.py`

---

## Phase 1 — Core Systems (10)

Baseline signal primitives: trend, momentum, breakout, volatility.

| # | Directory | Module | System | Status |
|---|-----------|--------|--------|--------|
| 1 | trend/ | er_trend_system.py | Efficiency Ratio Trend | Done |
| 2 | trend/ | linear_regression_trend.py | Linear Regression Trend | Done |
| 3 | moving_average/ | dual_ma_system.py | Dual Moving Average Crossover | Done |
| 4 | moving_average/ | triple_ma_system.py | Triple Moving Average | Done |
| 5 | moving_average/ | kama_system.py | KAMA Crossover | Done |
| 6 | momentum/ | dual_roc_system.py | Dual Rate of Change | Done |
| 7 | momentum/ | rsi_reversal_system.py | RSI Reversal | Done |
| 8 | breakout/ | donchian_breakout_system.py | Donchian Channel Breakout | Done |
| 9 | breakout/ | atr_breakout_system.py | ATR Breakout | Done |
| 10 | volatility/ | bollinger_breakout_system.py | Bollinger Band Breakout | Done |

---

## Phase 2A — Structural Systems (10)

Channels, swing structure, and price patterns.

| # | Directory | Module | System | Status |
|---|-----------|--------|--------|--------|
| 11 | channel/ | price_channel_breakout.py | Price Channel Breakout | Done |
| 12 | channel/ | moving_channel_system.py | Moving Channel | Done |
| 13 | channel/ | regression_channel_system.py | Regression Channel | Done |
| 14 | channel/ | high_low_channel_system.py | High-Low Channel | Done |
| 15 | swing/ | swing_reversal_system.py | Swing Reversal | Done |
| 16 | swing/ | outside_day_system.py | Outside Day | Done |
| 17 | swing/ | thrust_system.py | Thrust Method | Done |
| 18 | pattern/ | congestion_breakout_system.py | Congestion Breakout | Done |
| 19 | pattern/ | range_expansion_system.py | Range Expansion | Done |
| 20 | pattern/ | inside_day_breakout.py | Inside Day Breakout | Done |

---

## Phase 2B — Volatility & Contraction Systems (10)

Volatility compression → expansion detection.

| # | Directory | Module | System | Status |
|---|-----------|--------|--------|--------|
| 21 | volatility_contraction/ | bollinger_squeeze_system.py | Bollinger Squeeze | Done |
| 22 | volatility_contraction/ | keltner_squeeze_system.py | Keltner Squeeze | Done |
| 23 | volatility_contraction/ | atr_contraction_system.py | ATR Contraction | Done |
| 24 | volatility_contraction/ | volatility_ratio_system.py | Volatility Ratio | Done |
| 25 | volatility_contraction/ | standard_deviation_breakout.py | Std Dev Breakout | Done |
| 26 | range_expansion/ | narrow_range_breakout.py | Narrow Range Breakout | Done |
| 27 | range_expansion/ | opening_range_breakout.py | Opening Range Breakout | Done |
| 28 | range_expansion/ | volatility_expansion_breakout.py | Volatility Expansion | Done |
| 29 | range_expansion/ | vix_expansion_system.py | VIX Expansion | Done |
| 30 | range_expansion/ | range_percentile_system.py | Range Percentile | Done |

---

## Phase 3 — Pattern Recognition Systems (10)

Calendar effects, event patterns, and time-of-day systems.

| # | Directory | Module | System | Status |
|---|-----------|--------|--------|--------|
| 31 | patterns/ | opening_gap_system.py | Opening Gap | Done |
| 32 | patterns/ | weekend_effect_system.py | Weekend Effect | Done |
| 33 | patterns/ | weekday_pattern_system.py | Weekday Pattern | Done |
| 34 | patterns/ | reversal_pattern_system.py | Reversal Pattern | Done |
| 35 | patterns/ | time_of_day_pattern.py | Time of Day | Done |
| 36 | patterns/ | intraday_range_pattern.py | Intraday Range | Done |
| 37 | patterns/ | seasonal_pattern_system.py | Seasonal Pattern | Done |
| 38 | patterns/ | monthly_turn_system.py | Monthly Turn | Done |
| 39 | patterns/ | holiday_effect_system.py | Holiday Effect | Done |
| 40 | patterns/ | earnings_drift_system.py | Earnings Drift | Done |

---

## Phase 4 — Spread & Relative Value Systems (10)

Substitute products, location spreads, carrying-charge spreads, arbitrage.

| # | Directory | Module | System | Status |
|---|-----------|--------|--------|--------|
| 41 | spread/ | intermarket_spread_system.py | Intermarket Spread | Done |
| 42 | spread/ | ratio_spread_system.py | Ratio Spread | Done |
| 43 | spread/ | pairs_trading_system.py | Pairs Trading (cointegration) | Done |
| 44 | spread/ | mean_reversion_spread.py | Mean Reversion Spread | Done |
| 45 | spread/ | carry_spread_system.py | Carry Spread | Done |
| 46 | spread/ | term_structure_spread.py | Term Structure Spread | Done |
| 47 | arbitrage/ | cash_and_carry_system.py | Cash and Carry | Done |
| 48 | arbitrage/ | calendar_spread_system.py | Calendar Spread | Done |
| 49 | arbitrage/ | commodity_crack_spread.py | Commodity Crack Spread | Done |
| 50 | arbitrage/ | interexchange_arbitrage.py | Inter-Exchange Arbitrage | Done |

---

## Phase 5 — Behavioral & Sentiment Systems (10)

News, COT positioning, contrary opinion, Fibonacci, Elliott waves, Gann timing.

| # | Directory | Module | System | Status |
|---|-----------|--------|--------|--------|
| 51 | behavioral/ | cot_positioning_system.py | COT Positioning | Done |
| 52 | behavioral/ | news_event_system.py | News Event | Done |
| 53 | behavioral/ | sentiment_contrarian_system.py | Sentiment Contrarian | Done |
| 54 | behavioral/ | opinion_indicator_system.py | Opinion Indicator | Done |
| 55 | behavioral/ | fibonacci_projection_system.py | Fibonacci Projection | Done |
| 56 | behavioral/ | elliott_wave_filter.py | Elliott Wave Filter | Done |
| 57 | cycle/ | gann_time_cycle_system.py | Gann Time Cycle | Done |
| 58 | cycle/ | price_time_cycle_system.py | Price-Time Cycle | Done |
| 59 | cycle/ | market_sentiment_index_system.py | Market Sentiment Index | Done |
| 60 | cycle/ | crowd_behavior_system.py | Crowd Behavior | Done |

---

## Phase 6 — Adaptive & Multi-Timeframe Systems (10)

Adaptive techniques (Ch.17), multiple time frames (Ch.19),
price distributions (Ch.18), advanced techniques (Ch.20).

| # | Directory | Module | System | Status |
|---|-----------|--------|--------|--------|
| 61 | adaptive/ | adaptive_trend_system.py | Adaptive Trend | Done |
| 62 | adaptive/ | adaptive_momentum_system.py | Adaptive Momentum | Done |
| 63 | adaptive/ | adaptive_channel_system.py | Adaptive Channel | Done |
| 64 | adaptive/ | kaufman_adaptive_system.py | Kaufman Adaptive (KAMA) | Done |
| 65 | multi_timeframe/ | dual_timeframe_trend.py | Dual Timeframe Trend | Done |
| 66 | multi_timeframe/ | triple_screen_system.py | Triple Screen | Done |
| 67 | multi_timeframe/ | multi_resolution_momentum.py | Multi-Resolution Momentum | Done |
| 68 | advanced/ | volatility_weighted_trend.py | Volatility Weighted Trend | Done |
| 69 | advanced/ | noise_filtered_trend.py | Noise Filtered Trend | Done |
| 70 | advanced/ | price_volume_distribution_system.py | Price Volume Distribution | Done |

---

## Repository Structure

```
kaufman_systems/
    base.py                      # TradingSystem ABC

    # Phase 1
    trend/
    moving_average/
    momentum/
    breakout/
    volatility/

    # Phase 2A
    channel/
    swing/
    pattern/

    # Phase 2B
    volatility_contraction/
    range_expansion/

    # Phase 3
    patterns/

    # Phase 4
    spread/
    arbitrage/

    # Phase 5
    behavioral/
    cycle/

    # Phase 6
    adaptive/
    multi_timeframe/
    advanced/
```

---

## Design Principles

1. **Separate signal families** — trend, breakout, mean-reversion, behavioral,
   adaptive — to avoid correlated alpha.

2. **Standard interface** — every system inherits `TradingSystem` with
   `signal()`, `position_sizing()`, `risk_filter()`, `indicators()`.

3. **ATR-based sizing** — Phase 6 systems use ATR-normalized position sizing;
   earlier phases use simpler capital-fraction sizing.

4. **Research-grade** — designed for backtesting and parameter research,
   not live execution. Connect to Lean/UP execution layer separately.

---

## Kaufman Book Mapping

| Book Chapter | Repo Phase |
|---|---|
| Ch.1-5: Trend, regression, momentum | Phase 1 |
| Ch.6-8: Channels, breakout, charting | Phase 2A |
| Ch.9-10: Volatility, distribution | Phase 2B |
| Ch.11-13: Patterns, calendar effects | Phase 3 |
| Ch.14-16: Spreads, arbitrage | Phase 4 |
| Ch.15: Behavioral techniques | Phase 5 |
| Ch.17: Adaptive techniques | Phase 6 (adaptive/) |
| Ch.18: Price distributions | Phase 6 (advanced/) |
| Ch.19: Multiple time frames | Phase 6 (multi_timeframe/) |
| Ch.20: Advanced techniques | Phase 6 (advanced/) |

---

## Potential Phase 7 — AI / Optimization / Advanced Models

Future expansion aligned with Kaufman's final chapters:

```
optimization/
    genetic_algorithm_system.py
    neural_network_predictor.py
    fuzzy_logic_system.py
    expert_system.py
    fractal_chaos_system.py
```

Overlaps with existing fractal/nonlinear research in the UP architecture.
