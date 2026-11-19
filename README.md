# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tukiyatlampu/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tukiyatlampu/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for each of GTA VI's 55 PS5 trophies, the percent of
Exophase-tracked players who will have earned it at the case target,
as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95). Python 3.13,
standard library only, no network, deterministic.

## Architecture

Earners and players are fitted separately, because a refresh that
adds tracked players must dilute a rare trophy's rate even when
nobody new earned it. Earner counts are reconstructed from each
two-decimal displayed rate intersected with integer counts at the
simultaneous player count. The denominator follows a decaying-influx
curve in three decay variants (beta 0.5 / 1 / 1.5, weights
0.25 / 0.5 / 0.25). Each trophy gets two anchored curves (log-count
and direct logit) whose slopes shrink toward family-peer medians
(story, side, exploration, grind, completion). A refresh-timing
mixture — a no-change atom plus eight last-refresh nodes, three
denominator shocks, two normal scales (at most 144 components per
trophy) — is inverted by 28-step bisection to give the requested
quantiles. Forward log-time steps are damped
(`D_eff = 0.3*(1-exp(-D/0.3))`) so a log-linear earner slope fitted
on the steep launch window cannot compound unchecked over a
4.5-day horizon; near D = 0 the damping is ~identity. Startup (a
page frozen since launch) uses a Weibull
first-refresh wait and denominator jumps of 2x / 4x / 8x. Output is
validated against the contract; on failure it falls back to
last-observed-rate rows, then to an all-zero fixed vector.

## Run it

```sh
MAKEFUN_INPUT_DIR=demo MAKEFUN_OUTPUT_DIR=demo/out python3 main.py
```

Grader form: `python3 -P -s /bot/main.py` with `MAKEFUN_INPUT_DIR`
/ `MAKEFUN_OUTPUT_DIR` (defaults `/input`, `/output`). One `<id>.out`
per `<id>.in`: `{"trophies":[{"id":N,"q":[q05,q25,q50,q75,q95]}]}`,
one row per trophy in input order, five finite numbers in [0, 100],
non-decreasing, at most 65,536 bytes.

## Tests

```sh
python3 tests/edge_test.py   # 58 checks: edges, contract, determinism, retry paths
python3 tests/backtest.py    # practice pseudo-answers, 10:17Z replay, scenario G, synthetic + dense
python3 tests/validate.py <input_dir> <output_dir> [answers.json]
```

Results in `TEST_RESULTS.md`. Practice (kit p00/p01) vs the
dilution pseudo-answer: 0.5844 vs starter 1.6871; vs
dilution+growth: 0.5295 vs 1.5955. 10:17Z replay (9 trophies):
0.2052 vs 1.0028. Scenario G (79.94% -> 13.19% truth): 0.9082 vs
3.1764. Synthetic pooled (6 scenarios): 0.3746 vs starter 0.5731;
dense benchmark: 0.4131 vs 0.5968 in 0.24 s. A second independent
synthetic generator agrees: pooled 0.6861 vs starter 1.0478,
dense 0.6645 vs 1.0125. See
`TEST_RESULTS.md` for the full table and the fixture note.

## Files

- `main.py` — the forecaster (entrypoint, stdlib only).
- `tests/edge_test.py` — edge-case and contract tests.
- `tests/backtest.py` — offline backtest with its own seeded
  synthetic generator (arrival convolution, batched refreshes,
  backfills, missing reads).
- `tests/validate.py` — output-contract validator and pinball
  scorer against an answers file.
- `demo/demo.in` — 5-trophy smoke case (frozen launch-night page).
- `TEST_RESULTS.md` — measured results.
- `LICENSE` — MIT.
