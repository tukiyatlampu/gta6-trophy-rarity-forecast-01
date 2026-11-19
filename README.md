[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tukiyatlampu/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tukiyatlampu/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# Denominator-Aware Trophy Rarity Forecast

Forecasts each of GTA VI's 55 PS5 trophies' Exophase earn rate (%) at the case
target, as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95). Python 3.13,
standard library only, no network, deterministic.

## Architecture

- **Earners over players, not rates.** Each two-decimal displayed rate plus its
  simultaneous players-tracked count reconstructs an integer earner count
  (rounding-interval midpoint; on practice p00 the reconstructed earners sum
  to exactly 943 = `trophies_earned`). A sixfold jump in tracked players with
  frozen earners then correctly predicts a sixfold rate drop.
- **Denominator with decaying influx.** Players-tracked is fit with three
  curves (β = 0.5 / 1 / 1.5, ridge + Huber), kept as a 0.25 / 0.50 / 0.25
  mixture, so the forecast carries denominator uncertainty instead of a
  single growth guess.
- **Anchored per-trophy curves.** Each trophy fits two curves in
  x = ln(1 + age in days) — log earners and direct logit rate — anchored at
  its last valid read and shrunk toward a family-pooled slope (story / side /
  exploration / grind / completion, by description keywords).
- **Refresh timing.** The answer is the rate as of the last refresh before
  the target, so the model forecasts the next refresh: a Weibull waiting time
  when the page has been frozen since launch, otherwise an exponential
  cadence; eight equal-mass time nodes, denominator shocks, and a two-scale
  normal mixture in scoring-logit space. Quantiles are the left inverse of
  the mixture CDF (28 fixed bisection steps).
- **Fallbacks.** A never-observed trophy takes the median projected logit of
  its family peers (scale 2); with no peers, a family prior median rate.

## Quick test

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=demo MAKEFUN_OUTPUT_DIR=out python3 main.py
python3 tests/validate.py out/demo.out
python3 tests/backtest.py    # practice + replay + synthetic stress
python3 tests/edge_test.py   # 21 edge cases + ledger/mids/stale/thin/contract/batch checks
```

## Validation (backtest estimates, not kit scores)

| Check | Ours | Persistence baseline |
|---|---|---|
| Practice p00/p01 vs dilution pseudo-answer | 0.59 | 1.69 |
| Practice vs dilution+growth pseudo-answer | 0.54 | 1.60 |
| 10:17Z replay (9 known snapshot rates) | 0.20 | 1.00 |
| Synthetic stress, pooled (6 scenarios) | 0.42 | 0.50 |
| High-rate dilution (80% -> 13.19% truth) | 0.91 | 3.18 |
| Dense benchmark, 289 reads, 6510-min horizon | 0.26 | 0.25 |

Logit-scale pinball loss, lower is better. House baselines score ~1.51 on
practice. Runtime ~0.3 s/case (limit 8 s); two runs byte-identical; worst
case 289 reads on 1 CPU: 0.36 s. See `TEST_RESULTS.md` for the full numbers,
the pseudo-answer construction, the seven judge passes (60, 72, 84, 89, 86,
94, 95/100 — latest pass: 95 SHIP, pass-6 blocker fixed and verified in
pass 7), and the known limitations — including the
dense benchmark where persistence still wins narrowly (0.26 vs 0.25);
the H-round cap rework had regressed it to 0.29, and restoring the old
broad cap recovered it to 0.26 (applied in this revision).

## Files

- `main.py` — the entry (run as `python3 -P -s /bot/main.py` by the grader)
- `tests/validate.py` — output-contract validator
- `tests/backtest.py` — pinball scoring vs persistence + synthetic stress
- `tests/edge_test.py` — edge cases, determinism, permutation-invariance
- `TEST_RESULTS.md` — full numbers, judge passes, known limitations
- `demo/demo.in` — two-read smoke-test case
- `.makefun/owner.json` — repo-owner claim (wallet `G9dDsodYieJBpsE5wBZS2drwBBRrRCF6ait6z7fhdod`)
