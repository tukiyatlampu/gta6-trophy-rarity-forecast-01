# TEST_RESULTS.md — gta6-trophy-rarity-forecast-01

Measured 2026-11-19 (WIB) from this package, Python 3.13, 1 CPU.

## R3 fix (2026-11-19 ~23:50 WIB, judge session)

Diagnosis correction, verified against the code: the design-doc
saturation `g(v)` (§2) **does exist** in `main.py` — `saturate()`
(lines 150-154, exactly `v` through 0.8 then
`1-0.2e^(-(v-0.8)/0.2)`), applied at the count-branch projection
and in branch residuals. The claim that the cap was "missing from
the rewrite" is **wrong**. What is true: `g(v)` only bites above an
80% rate, so it cannot damp the 4.5-day overshoot at mid-range
rates. Measured on the pre-fix code: median q50-vs-truth logit
bias at 6510 min was +1.1 to +3.1 (e.g. anchor 3.70% -> truth
8.58% -> q50 89.36%; 51-52 of 55 trophies above truth), i.e. the
undamped log-linear earner slope `exp(b*D)` compounds unchecked
once earners saturate.

Fix applied in this session (`main.py`, `project_z`): damp the
forward log-time step, `D_eff = 0.3*(1-exp(-D/0.3))` for `D > 0`
(≈ identity near D = 0, effective step capped at 0.3 as D -> inf).
Tuned on both synthetic generators over A ∈ {0.15, 0.2, 0.25,
0.3, 0.5, 0.75, 1.0}: A = 0.3 wins every 6510-min scenario on
both generators and leaves practice / replay / scenario G slightly
better, not worse. 90-min scenarios are byte-identical in loss
(damping ≈ identity at small D). Before -> after:

| check | before | after | starter |
|---|---|---|---|
| practice vs dilution | 0.5866 | **0.5844** | 1.6871 |
| practice vs dilution+growth | 0.5316 | **0.5295** | 1.5955 |
| 10:17Z replay (9 trophies) | 0.2084 | **0.2052** | 1.0028 |
| scenario G | 0.9124 | **0.9082** | 3.1764 |
| synthetic pooled (backtest gen) | 1.0027 | **0.3746** | 0.5731 |
| dense benchmark (backtest gen) | 0.9584 | **0.4131** | 0.5968 |
| synthetic pooled (2nd generator) | 1.3095 | **0.6861** | 1.0478 |
| dense benchmark (2nd generator) | 1.4903 | **0.6645** | 1.0125 |

Every number below was re-run against the fixed `main.py` on
disk and reproduced exactly. Commands (run from the package root):

```sh
python3 tests/edge_test.py
python3 tests/backtest.py
python3 tests/regression_test.py
python3 tests/synthetic_test.py
MAKEFUN_INPUT_DIR=<kit input> MAKEFUN_OUTPUT_DIR=<out> python3 main.py
python3 tests/validate.py <input_dir> <output_dir> [answers.json]
MAKEFUN_INPUT_DIR=demo MAKEFUN_OUTPUT_DIR=demo/out python3 main.py
```

## Package identity (verified this session)

- `main.py`: md5 `6d68f355d18108dafab2eba4cebeceea`,
  sha256 `2e3c4eb24b3f2b5ae7c1cfa519f5a71180010b2fd0f2149193417b8039a2ca78`,
  43,117 bytes, 1207 lines, syntax OK.
- `.makefun/owner.json`: 58 bytes, `{"wallet":
  "G9dDsodYieJBpsE5wBZS2drwBBRrRCF6ait6z7fhdod"}` + newline.
- `LICENSE`: line 1 `SPDX-License-Identifier: MIT`.
- `README.md`: 3 make.fun badges (bounty, rank, reward).
- Package files (11): `main.py`, `tests/validate.py`,
  `tests/edge_test.py`, `tests/backtest.py`,
  `tests/regression_test.py`, `tests/synthetic_test.py`,
  `demo/demo.in`, `TEST_RESULTS.md`, `README.md`, `LICENSE`,
  `.makefun/owner.json`.

## Verification (re-run this session, post-fix)

- `tests/edge_test.py`: 58 checks PASS, 0 FAIL, exit code 0.
  Practice runtime 0.52 s for the 2 cases (limit 8 s/case).
- `tests/validate.py` on kit practice outputs: p00 VALID, p01 VALID,
  both under the byte cap, exit code 0.
- Determinism post-fix: `main.py` run twice over the kit practice
  cases; outputs byte-identical (p00 md5
  `c9f8a52de53568a9cf60e2ddc3748bf2`, p01 md5
  `3b27a6873e9b2b5e72cf1e4521689028` in both runs).
- Demo smoke: `demo/demo.in` runs clean (exit 0, empty stderr) and
  validates; output byte-identical to the JSON quoted below.

## Edge + contract suite (`tests/edge_test.py`)

58 checks PASS, 0 FAIL, exit code 0. Covered: empty history, no
trophies (`{"trophies":[]}`), single read, null rates, 0%/100% rates,
player jump 638 -> 3868, duplicate reads (byte-identical forecast),
90-min vs 6510-min horizons (wider spread on the long horizon),
zero horizon (quantiles equal the current rate), two-run
byte-identical output, contract validation (accepts good rows;
rejects wrong order, decreasing quantiles, out-of-range values,
extra keys, wrong row count, NaN), rounding (half-up, ints stay
ints), earner reconstruction (platinum interval is exactly [4, 4]),
oversized-output retry to the all-zero fixed vector, all-serialize-
fails returns nothing, practice I/O (p00/p01 valid, <= 65,536
bytes, empty stderr, malformed `.in` skipped), missing input dir
(exit 0, no output). Practice runtime 0.52 s for the 2 cases
(limit 8 s/case).

## Regression tests (`tests/regression_test.py`)

20 checks PASS, 0 FAIL, exit code 0.

- (c) Oversized recovery: 55 trophies with 1162-digit integer ids.
  Real primary forecast valid but serializes past the cap;
  emergency rows valid but also oversized; fixed `[0,0,0,0,0]`
  vector serializes to exactly **65,245 bytes** (fits; 291 bytes
  spare). `process_case` returns exactly the fixed text, logs
  `TERMINAL: primary and emergency produced no compliant output;
  trying fixed rows` on stderr, the output validates and carries all
  55 ids with fixed zero quantiles, and no temporary file is left
  behind (checked by running inside a fresh temp working dir).
- (d) Empty-case validation gate: with `validate_output`
  monkeypatched to reject everything, `process_case` on the empty
  case returns None, and `main()` run in-process leaves a
  pre-existing output file untouched (`SENTINEL` preserved). After
  restoring validation, the same case yields exactly
  `{"trophies":[]}` + newline, both from `process_case` and through
  `main()` writing the file.

## Backtest (`tests/backtest.py`, 2.92 s) — PASS, exit code 0

Logit-scale pinball loss (lower is better). `starter` is the kit
persistence baseline. `target` is the previous implementation's
number, shown for reference; `MATCH` means equal at 4 decimals.
Post-fix the model beats the starter on every check; the small
gaps to the old targets (≤ 0.007) are the same pseudo-answer
construction differences noted before, not model errors.

| check | ours | starter | target ours / starter |
|---|---|---|---|
| practice vs dilution (p00 0.6063, p01 0.5626) | 0.5844 | 1.6871 | 0.5917 / 1.6871 — starter MATCH, ours MISS by 0.0073 |
| practice vs dilution+growth (p00 0.5508, p01 0.5081) | 0.5295 | 1.5955 | 0.5367 / 1.5955 — starter MATCH, ours MISS by 0.0072 |
| 10:17Z replay (9 trophies) | 0.2052 | 1.0028 | 0.2034 / 1.0028 — starter MATCH, ours MISS by 0.0018 |
| scenario G (79.94% -> truth 13.19%), q=[10.0637, 20.5995, 46.5672, 79.94, 79.94] | 0.9082 | 3.1764 | 0.9105 / 3.1764 — starter MATCH, ours MISS by 0.0023 |
| synthetic pooled (A-F) | 0.3746 | 0.5731 | 0.4221 / 0.5043 — MISS (see note) |
| dense benchmark (289 reads, 0.24 s) | 0.4131 | 0.5968 | 0.2569 / 0.2482 — MISS (see note) |

Per-scenario synthetic (ours / starter): A launch-surge 90min
0.0303 / 0.0340; B launch-surge 6510min 0.5035 / 0.7216; C steady
6510min 0.3790 / 0.5343; D missing-reads 6510min 0.4476 / 0.6214;
E slow-grind 6510min 0.3604 / 0.6953; F 2nd-backfill 6510min
0.5267 / 0.8320. The model now wins all six scenarios, including
all five 6510-min ones (pre-fix it lost all five).

Practice per-level loss vs dilution (ours): 0.030 / 0.613 /
1.215 / 0.888 / 0.178 at tau 0.05 / 0.25 / 0.5 / 0.75 / 0.95
(starter: 2.991 / 2.541 / 1.776 / 0.929 / 0.198).

Synthetic per-level (pooled A-F, ours): 0.199 / 0.460 / 0.404 /
0.515 / 0.295 (starter: 0.239 / 0.777 / 0.971 / 0.624 / 0.254).

Diagnostics: sum of reconstructed earners = 943.0 vs
`trophies_earned` = 943 (exact match).

Backtest exit code is 0 (PASS). Note on the old targets: the
0.4221 / 0.2569 figures were measured on the previous
implementation's synthetic generator, which no longer exists; the
current generator's starter baselines differ (pooled 0.5731 vs
0.5043; dense 0.5968 vs 0.2482), so those two rows are not
directly comparable to the old targets. The comparable criterion
— beat the starter on the same fixture — now passes on both
generators.

## Synthetic stress, independent generator (`tests/synthetic_test.py`, 3.29 s)

A second, independently written generator per design-doc section 5:
held-out seeds 201-207, decaying arrival surge, cohort unlock
delays, batched publication, denominator backfill, rounded zeros
(7-13 zero rates at cutoff in the 6510-min scenarios; 39 in the
90-min scenario), a saturated opener (share 0.97-0.995), missing
reads (30% in S4), a decline-to-recovery crossover probe, and both
90-minute and 4.5-day horizons. Pooled loss is also scored against
an anchored log-linear fitted curve with starter spread.

| scenario | ours | starter | fitted curve |
|---|---|---|---|
| S1 launch-surge 90min (25 reads) | 0.3385 | 0.3460 | 0.2790 |
| S2 launch-surge 6510min (97 reads) | 0.6741 | 0.9492 | 2.9152 |
| S3 steady 6510min (97 reads) | 0.5648 | 1.0315 | 2.2626 |
| S4 missing-reads 6510min (97 reads) | 0.7975 | 1.2314 | 2.5835 |
| S5 slow-grind 6510min (97 reads) | 0.7215 | 1.1814 | 2.1837 |
| S6 backfill 6510min (97 reads) | 1.0202 | 1.5472 | 2.5031 |
| **synthetic pooled** | **0.6861** | **1.0478** | **2.1212** |
| **dense benchmark (289 reads, 0.25 s)** | **0.6645** | **1.0125** | **2.4395** |

Targets (previous implementation, old generator): pooled 0.4221,
dense 0.2569 — both still MISSED, but the gap closed from
0.89 / 1.23 to 0.26 / 0.41, and the old-generator targets are not
directly comparable (see backtest note). Crossover probe (true
rates at cutoff-36h, -12h, cutoff, target) confirms the
construction: S2 0.00 / 9.90 / 19.12 / 55.53, S3 0.00 / 3.22 /
7.02 / 33.66 — the probe rate first falls behind the surging
denominator, then recovers sharply. (In S1 the probe is still
inside its 18 h unlock delay at the 6 h cutoff, so all four probe
rates are 0.00.)

`synthetic_test.py` exit code is 1: all forecasts were valid,
deterministic, and under the byte cap (22 checks PASS), the
pooled/dense comparisons against the starter and the fitted curve
now PASS, but the 2 checks against the old-generator targets
FAIL (ours does not reach 0.4221 / 0.2569 on this different
fixture). Nothing here was tuned to hit the old targets; the
numbers are reported as measured.

Held-out check (fresh seeds, never used to select A = 0.3):
backtest-generator seeds 21-26 pooled damped 0.4087 vs undamped
1.1987 vs starter 0.5985; second-generator seeds 211-216 pooled
0.7434 vs 1.3748 vs 1.0517; dense seed 217: 0.7034 vs 1.8084 vs
1.0744. Median q50-vs-truth logit bias at 6510 min, damped:
+0.04 / +0.32 / +0.31 / -0.37 / -0.12 (gen 1) and +0.06 / -0.16 /
-0.16 / -0.44 / -1.16 (gen 2), vs undamped +1.15 to +2.93; 90%
interval coverage damped 0.93-0.98 (gen 1) and 0.76-0.84
(gen 2, plus S6 under-shoot bias -1.16 on the backfill regime),
vs undamped 0.24-0.87. The S1 90-min held-out case scores
coverage 0.00 for damped and undamped alike (truths are rounded
zeros while quantiles sit above zero) and starter wins its
pinball 0.2551 vs 0.2649 — a fixture artifact, not a damping
effect. The damping generalizes to unseen seeds on both
generators.

## Demo smoke test

`demo/demo.in` (5 trophies, frozen launch-night page) runs clean
through `main.py` (exit 0, empty stderr) and validates:

```json
{"trophies":[{"id":1,"q":[0.0772,0.1585,0.3842,0.63,0.63]},{"id":2,"q":[99.9874,99.9899,99.99,99.99,99.99]},{"id":3,"q":[0.1574,0.3232,0.7845,1.25,1.25]},{"id":4,"q":[0.1378,0.2829,0.687,1.1,1.1]},{"id":5,"q":[0.1378,0.2829,0.687,1.1,1.1]}]}
```

`tests/validate.py` on the kit practice outputs: p00 VALID,
p01 VALID, both under the byte cap.
