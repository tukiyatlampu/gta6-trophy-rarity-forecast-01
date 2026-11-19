# TEST_RESULTS — gta6-trophy-rarity-forecast-01

Run: 2026-11-19 ~21:02 WIB, Python 3.12.3 locally (grader uses 3.13),
`python3 -P -s main.py` grader-style, `PYTHONHASHSEED=0`.
Commands: `python3 tests/backtest.py`, `python3 tests/edge_test.py`,
`python3 tests/validate.py <out>`. All outputs below are the real outputs
of those runs. Scores are logit-scale pinball loss (lower is better);
the exact scoring clamp (0.01..99.99) and pinball formula from the bounty
are used by `tests/backtest.py`. This run covers the K-round revision
(judge pass-6 blocker fix: fixed-vector retry after any terminal
failure + validated empty literal); the forecast is untouched, so the
backtest numbers are identical to the J-round run, and the H-round
numbers it replaces are kept in the judge-pass history below for
comparison.

## Practice cases (p00/p01)

Both practice outputs VALID (55 rows, 5 quantiles each, ~2.6 KB).
Runtime 0.61 s for both cases (limit 8 s/case).

| Check | Ours | Persistence (starter) |
|---|---|---|
| Practice vs dilution pseudo-answer | 0.5917 (p00 0.6081, p01 0.5753) | 1.6871 |
| Practice vs dilution+growth pseudo-answer | 0.5367 (p00 0.5526, p01 0.5209) | 1.5955 |

House baselines score ~1.51 on practice. The dilution pseudo-answer
freezes earners at their launch-night rounding-interval midpoints and
moves players 638 -> 3868 at the 04:30Z refresh (opener stays 100%);
the growth variant lets earners grow at the family prior slope to the
target. These are approximations, not the actual 04:30Z answers
(the kit ships no answers); they test surviving the refresh jump.

Per-level loss, practice vs dilution pseudo-answer:

| tau | Ours | Persistence |
|---|---|---|
| 0.05 | 0.026 | 2.991 |
| 0.25 | 0.652 | 2.541 |
| 0.50 (median) | 1.214 | 1.776 |
| 0.75 | 0.888 | 0.929 |
| 0.95 | 0.178 | 0.198 |

The median is the weakest level: with a frozen launch state the
no-change atom still holds ~44% mass at a 90-min horizon, so the median
sits near the top of the refreshed distribution. The lower quantiles
carry the win (they price the refresh that persistence misses).

Diagnostic: sum of reconstructed earners = 943.0, exactly equal to
`trophies_earned` = 943 in p00 — the rounding-interval reconstruction
is exactly consistent with the house aggregate.

## 10:17Z replay (real snapshot, 9 trophies with known rates)

p01 retargeted to 2026-11-19T10:17Z (horizon 407 min), scored only on
trophies whose live snapshot rate is known exactly (platinum 0.10%,
opener 100%, Chapter 1 0.49%, Chapters 2-4 + final ~0.23%,
"Baited My Tackle" 1.16%, "Stick Fiddler" 0.39%):

| Ours | Persistence |
|---|---|
| 0.2034 | 1.0028 |

## Synthetic stress (exact ground truth, 6 scenarios)

| Scenario | Ours | Persistence |
|---|---|---|
| A launch-surge, 90-min horizon | 0.0384 | 0.0454 |
| B launch-surge, 6510-min horizon | 0.5304 | 0.6434 |
| C steady state, 6510 min | 0.5839 | 0.6922 |
| D missing reads (30%), 6510 min | 0.4838 | 0.5754 |
| E slow-grind unlocks, 6510 min | 0.4553 | 0.5616 |
| F second backfill (x1.8 at 36h), 6510 min | 0.4406 | 0.5077 |
| **Pooled** | **0.4221** | **0.5043 (16.3% lower)** |

Per-level, synthetic pooled: ours tau 0.05/0.25/0.5/0.75/0.95 =
0.096 / 0.430 / 0.793 / 0.646 / 0.146; persistence =
0.099 / 0.469 / 0.885 / 0.803 / 0.267. We win every scenario and
every level including tau=0.05 (0.096 vs 0.099) — an earlier draft
of this file wrongly called tau=0.05 a tie; the judge caught it and
the numbers here correct it.

Scenario G — high-rate dilution (added after the first judge pass):
trophy at 79.94% (E=510 of 638) with frozen earners and a x6.06
denominator jump; truth 13.19%. Ours q=[10.21, 20.89, 46.23, 79.94,
79.94], loss 0.9105; persistence q=[76.44, 78.55, 79.94, 81.26, 83.04],
loss 3.1764. (Before the blend fix the loss was 2.0350.) The median
misses the truth by ~1.77 scoring logits even though the lower tail
covers the outcome — the no-change atom still holds the top quantiles
at the anchor on this shape.

## Worst-case runtime (dense benchmark)

48h of 10-minute reads (289 reads, 12 refreshes), 6510-min horizon,
pinned to one CPU with `taskset -c 0`: **0.36 s** (0.38 s in the
clean-directory re-run), output VALID. Limit is 8 s/case. Loss on
this benchmark: ours 0.2569 vs persistence 0.2482 — the H-round
regression (0.2852) is RECOVERED by restoring the G-round broad cap
(`anchored_slope(recent, 0.0, 0.35)`, condition `b_raw < b_count`,
old eligibility without the direct_only skip; `raw_recent_slope`
deleted). The restored cap recovers dense exactly as the judge
measured (0.2569), with practice unchanged (0.5917) and synthetic
pooled essentially unchanged (0.4221 vs 0.4215).
Recent-guard engagement on this benchmark (reported by
`tests/backtest.py` on every run): count branch eligible 55/55
trophies, recent-constrained 43 (was 37 under the H3 rework); direct
branch eligible 55, recent-constrained 17. Note the raw ledger scales
with polls x trophies and retains runs before the fit window, so
0.36 s on this 289-read case is no worst-case bound for a much longer
history.

## Edge cases (tests/edge_test.py) — all pass

empty-history, single-read, null-rates, all-null-last, zero-and-hundred,
player-jump-down, player-jump-up, duplicate-reads, no-players,
horizon-zero, horizon-6510, short-rates, unordered-history, bad-levels
(caller levels [0.1, 0.9] ignored, 5 quantiles still emitted),
missing-at (fallback rows), null-then-change (A=1.0, B=None, C=2.0
retains the change), nan-rates (non-finite values rejected),
launch-anchor (single read exactly at launch — recent-window start
clamps to launch_ref, no log(0)), partial-poll-simultaneity (Poll A
P=100 rate=10, Poll B P=null rate=10 — the simultaneous pair survives
in the count ledger, identical polls collapse to one run),
stale-count-anchor (exact old count anchor E=10 of P=100, later
rate-only observations reach 50% — forecast median must follow the
newer direct evidence: observed q50 62.03, threshold 40),
plateau-mids (polls at h0/h30/h40/h48, rate 10% flat — the anchor
run must retain h30/h40 as mids),
permutation-invariance (shuffled trophy order gives identical per-id
quantiles). Ledger check (direct, on `build_ledgers`), ledger mids
check (anchor run carries mids (t30, t40)), stale-anchor
check, thin check (00:00/01:00/01:01 observations thin to 00:00 and
01:01), output-contract check (`validate_output` / `serialize_output`
unit tests: short rows, decreasing q, NaN rejected; a
`forecast_case` that raises, and one that returns an invalid
forecast, both yield the 55-row fallback and no stray `.tmp`),
terminal-contract check (an oversized 2200-trophy case preserves
any existing target instead of writing an invalid `.out`, and creates
no `.out` when none exists; forcing emergency-row construction to
raise still yields 55 rows with the fixed [50, 50, 50, 50, 50]
quantiles, which validate),
and batch isolation (a top-level list JSON file gets an
empty-trophies `.out` and cannot kill the batch) also pass. Two runs
byte-identical.

Demo case: `demo/demo.in` (two-read smoke test) runs and validates.

## Judge passes (Champion Upgrade, 2-pass critique)

- Pass 1 (SolAPI gpt-6.1-sol, harshest judge): **60/100 — DO NOT SHIP**.
  13 findings; all deterministic defects fixed in this session:
  (1) levels forced to the fixed scored levels; (2) timestamp/NaN
  robustness + per-file failure isolation with conservative fallback;
  (3) count/direct blend no longer vanishes at 80%+ rates (count
  branch dominates below 90%, direct only near saturation);
  (4) null reads no longer erase later valid observations (last-known
  finite comparison + per-field observation times, simultaneity
  required for earner reconstruction); (5) anchors taken from the
  full state list before thinning; (6) residual scales computed in
  two phases (all slopes first, then cached scales) — permutation
  test added; (7) Huber reweighting from base weights (no
  compounding); (8) Weibull CDF via expm1 + 12h outage age cap;
  (9) earner interval uses exact half-up bounds (upper-exclusive);
  (10) gamma documented as a denominator-shock elasticity;
  (12) round4 via Decimal half-up; (13) 289-read 1-CPU benchmark.
  After the fixes, practice improved 0.6805 -> 0.5973 and scenario G
  2.0350 -> 0.9191; synthetic pooled unchanged (0.3869).
- A 12h recent-slope window was tried and rejected (dense 0.3099 vs
  0.3025 at 24h — noisier with 4h refreshes); the 24h window stays.
- Denominator-prior sensitivity: weakening the ridge prior 0.15 -> 0.5
  leaves practice unchanged (0.5973) and worsens dense (0.3505 vs
  0.3025); the design's 0.15 prior stays.
- Pass 2 (SolAPI gpt-6.1-sol, harshest judge): **72/100 — DO NOT SHIP**.
  Of the 13 original findings, 9 fixed as specified, 4 partially
  (2, 4, 11, 13). Seven remaining weaknesses, all fixed in this
  session (F1–F8):
  - F1 launch-time anchor crash: the recent-window start now clamps to
    launch_ref (`max(anchor_t - 24h, launch_ref)`); regression case
    `launch-anchor` (single read exactly at launch) added to edge tests.
  - F2 collapsed states lose valid observations / manufacture repeated
    evidence: per-trophy run ledgers (`build_ledgers`, one point per
    maximal run of equal values) replace collapsed states for fitting;
    anchors are the ledgers' last runs; pooling eligibility is
    branch-specific (count points for the count branch, direct points
    for the direct branch).
  - F3 count projections used the rate anchor's clock: `d_count` /
    `d_direct` (and `h_ic` / `h_i`) are now separate in
    `trophy_components`; `project_z` receives the count horizon.
  - F4 `fit_branch` returned mismatched (slope, info) and the 0.02
    threshold was weak vs the ridge prior: slope and info stay paired,
    and the recent slope requires >=2 distinct non-anchor observations
    with info_rec >= 0.15.
  - F5 high-rate path distortions: `saturate()` removed, replaced by an
    anchor-preserving clip of E/P to [0, 1]; `direct_only` now means
    "no usable player count" or the opening trophy (explicit exception),
    not "anchor >= 100%"; the count branch keeps its vote through the
    100% boundary; the blend weight is identification-based
    (0.5 x anchor interval log-width) instead of rate-gated.
  - F6 denominator fit bypassed the fitting segment and used the last
    poll time as recency reference: player observations now come from
    the segment only, density-bounded to <=1/hour (`thin_p_obs`), and
    recency decays from the latest actual players observation.
  - F7 `main()` isolation incomplete: the whole per-file pipeline
    (`process_file`) sits inside one outer boundary; fallback,
    serialization, the size guard, and the write are each guarded;
    `classify` / `fallback_row` / `fallback_forecast` never raise on
    non-dict trophies or cases (missing ids become "").
  - F8 edge cases added: `launch-anchor`, `partial-poll-simultaneity`
    (ledger check + end-to-end), batch isolation for top-level list
    JSON. (A `trophy_id` regression in this round — ids coerced to
    strings, failing the validator's int-id check — was caught by the
    edge suite and fixed the same session.)
  After the pass-2 fixes, re-run 2026-11-19 ~19:16 WIB: practice
  unchanged (0.5973 / 0.5423, replay 0.2088, scenario G 0.9191),
  synthetic pooled 0.3869 -> 0.3866, dense benchmark improved
  0.3025 -> 0.2789 (still behind persistence 0.2482 — see Known
  limitations). During this round one defect was caught by the demo
  run: `raw_residual_scales` lost its `out = []` initializer in the
  ledger rewrite, so every case fell back to the conservative
  fallback forecast (backtest then showed fallback numbers: synthetic
  pooled 0.9833, G 1.4881). Fixed the same session; the numbers above
  are from the real model, verified by the diagnostic (reconstructed
  earners 943.0 = trophies_earned 943) and by the demo running without
  a fallback warning.
- Pass 3 (SolAPI gpt-6.1-sol, harshest judge; focused re-review of the
  changed functions — a full-code call timed out upstream twice,
  `stream_timeout` / HTTP 524, so the prompt carried only the changed
  extracts): **84/100 — DO NOT SHIP** (axes: contract 27/30,
  forecast 23/30, edge/fallbacks 12/15, determinism 9/10, code 13/15).
  Of the 7 pass-2 weaknesses: FIXED = 1, 2, 4, 5; PARTIALLY = 3, 6, 7.
  Six remaining weaknesses, all addressed in this session (G1–G6):
  - G1 stale count anchor overrides newer rates: `w_blend` is now
    freshness- and consistency-aware — besides the identification
    weight, doubt grows with the count anchor's age
    (`1 - 2^(-age_h/12)`) and with the discrepancy between the count
    branch's own projection at the rate anchor and the observed rate
    (`disc / (disc + 1)` in logit units). Regression case
    `stale-count-anchor` added (old E=10/100, later rates 50%:
    q50 62.03, threshold 40). Isolated-variant tests showed G1 costs
    nothing on dense (0.2838 with it disabled vs 0.2827).
  - G2 ledger compression erases plateau duration / drops in-window
    evidence: runs now keep first AND last poll times; fits emit up to
    two points per run (first + last), and a run is kept when EITHER
    poll falls inside the window. The judge's example (rate at h0/h24,
    change at h25, window from h12) is now retained via the h24 point.
    Halving the two points' weights was tried and rejected twice over:
    it weakens the data against the positive family prior (dense
    0.3069 vs 0.2827) and halves the recent window's information, so
    the saturation cap clears its bar less often (dense 0.2964 vs
    0.2540 with the broad cap).
  - G3 recent-slope guard is mitigation, not demonstrated protection:
    added a POST-POOLING saturation cap in `fit_trophies` — when the
    last-24h count observations identify their own slope, `b_count`
    is capped at the unshrunk recent slope (floored at 0), so the
    family prior cannot pull a saturating trophy back up. A
    near-zero-only variant was tried and rejected (dense 0.2827 vs
    0.2540 for the broad form; synthetic ~unchanged either way).
    Engagement is now instrumented: `tests/backtest.py` reports
    recent-guard eligibility/selection on the dense benchmark every
    run (count 55 eligible / 43 constrained; direct 38 / 17).
    Diagnosis note: on dense, count-anchor age gaps are 0 and anchor
    count-run spans max 3.3h, while 17 trophies have rate-run spans
    >=24h — saturation evidence lives in the rate runs, which is why
    the cap keys on the recent count slope, not on run spans.
  - G4 fallback visibility: `forecast_case` now logs every whole-case
    fallback exit with its reason (non-dict case, invalid cutoff, no
    usable history) plus a per-case count of unobserved trophies;
    `safe_fallback` logs the exception it swallows. Output schema
    unchanged (logging is stderr-only).
  - G5 serialization/size recovery: NEW `serialize_output()` guards
    serialization AND the 65536-byte check in one helper, applied to
    every candidate output (primary and fallback); `process_file`
    falls back and re-validates, and logs an explicit terminal failure
    if no compliant output can be produced (last resort: the 17-byte
    `{"trophies":[]}`).
  - G6 denominator thinning: `thin_p_obs` now selects BACKWARD from
    the newest observation, enforcing the 1-hour spacing between
    every kept pair (00:00/01:00/01:01 -> 00:00 + 01:01); `p_at`
    values are filtered against the segment start, keeping the single
    latest pre-segment value (the count carried into the segment's
    first state). Isolated-variant test: old thinning gives the same
    dense loss (0.3069 at the time), so thinning is not the dense
    driver.
  - Small items from pass 3: `fit_branch` now counts DISTINCT
    observation times for its "two observations" requirement (two
    changing polls sharing a timestamp no longer qualify); the
    OPENING comment no longer claims the direct branch "stays at
    100%" (it fits a slope from 100% with uncertainty).
  - Trade-off measured and decided explicitly: the G-round moves the
    synthetic pooled loss 0.3866 -> 0.4176 (still wins every scenario
    and every level) while the dense benchmark improves
    0.2789 -> 0.2540 (near parity with persistence 0.2482). The final
    case (48h cutoff, 6510-min horizon) matches the dense benchmark's
    shape, not the 12h-cutoff synthetic scenarios; on the real-data
    proxies (practice 0.5942, replay 0.2048) the G-round is the best
    version yet. Re-run 2026-11-19 ~20:12 WIB: all numbers in the
    G-round section of this file are from that run; the headline
    numbers above are from the later H-round re-run (~20:35 WIB),
    which supersedes them.
- Pass 4 (SolAPI gpt-6.1-sol, harshest judge; focused re-review of
  the changed functions): **89/100 — Do not ship** (axes: contract
  28/30, forecast 25/30, edge/fallbacks 13/15, determinism 9/10,
  code 14/15). All 6 findings marked PARTIALLY, with ranked fixes
  1–5; all addressed in this session (H1–H6):
  - H1 discrepancy penalty only when the count-anchor age gap > 0
    (same-time observed players is the consistency check, not the
    fitted denominator); compared rates clamped to [0.5, 99.5] so a
    fitted P of 103 vs observed 100 at a 100% anchor no longer reads
    as an ~8-logit discrepancy.
  - H2 ledger mids (up to 2 intermediate poll times per run, anchor
    run only) + only the anchor TIME bypasses the segment/48h window
    filters (first poll and mids must pass them). Regression case
    `plateau-mids` + `ledger_mids_check()` added to edge tests.
  - H3 cap uses the UNPENALIZED recent slope (`raw_recent_slope`,
    no ridge/Huber) and binds only when lower than the pooled slope
    by more than its own standard error (`b_raw + se < b_count`,
    se from recent residual scatter); direct-only trophies skipped;
    `info_count` replaced by the recent info when the cap binds.
  - H4 logging of empty denominator curves and
    atom-only-with-positive-horizon counts; the ledger tie rule
    (input-index order; last same-timestamp poll wins the run's
    value) documented in the `build_ledgers` docstring.
  - H5 `validate_output` (exact contract check, never raises) +
    emergency fallback from validated case IDs (family prior
    medians, scale 2) + atomic writes (`.out.tmp` + `os.replace`)
    + terminal empty-literal path with a loud TERMINAL log. Unit
    tests for `validate_output` / `serialize_output` and in-process
    `process_file` recovery tests added to edge tests.
  - H6 when in-segment counts exist, the denominator fits those
    alone (a carried pre-segment count seeds the level only); only
    when the segment has no counts does the latest carried count
    seed the fit. `thin_p_obs` docstring now states thinning also
    strengthens effective shrinkage toward the denominator prior.
  - H-round re-run 2026-11-19 ~20:35 WIB: practice 0.5917 (was
    0.5942), synthetic pooled 0.4215 (was 0.4176), replay 0.2034
    (was 0.2048), scenario G 0.9105 (was 0.9169), dense 0.2852
    (was 0.2540 — REGRESSION, see the dense section).
- Pass 5 (SolAPI gpt-6.1-sol, harshest judge; one final focused
  call, reasoning high, 16k tokens, returned in ~88 s):
  **86/100 — DO-NOT-SHIP** (axes: contract 26/30, forecast 23/30,
  edge/fallbacks 14/15, determinism 10/10, code 13/15). Two
  must-fix issues, both NOT applied: the wrap-up stop order froze
  judge->fix iteration after this call returned, so they ship
  documented here for the parent to decide.
  - (1) H3 is an avoidable predictive regression: dense 0.2540 ->
    0.2852 (+12.3%, now ~14.9% worse than persistence); restore the
    exact tested old broad-cap form (`anchored_slope` ridge 0.35,
    condition `b_raw < b_count`, old eligibility) while keeping
    H1/H2/H4/H5/H6. Measured: restoring only the old cap recovers
    dense to 0.2569 with synthetic 0.4221 and practice 0.5917
    unchanged; removing the se bar alone gives 0.2846, removing
    mids alone 0.2860, anchored estimator + new conditions 0.2689.
    The judge notes the residual-based SE treats retained poll
    points as independent evidence while mids from one run are not
    independent refreshes, and `len(recent) >= 3` proves three
    retained points, not three independent updates.
  - (2) The terminal `{"trophies":[]}` violates the contract for a
    valid nonempty case: once valid case IDs are available, retain
    them independently of forecasting and make the terminal forecast
    contain every required ID in order (fixed bounded ordered
    five-number vector); validate the payload actually committed
    before `os.replace`; on serialization/disk failure preserve any
    existing target instead of replacing it with an invalid result.
    Note: the emergency fallback rows themselves ARE contract-valid
    (the edge suite asserts it); the terminal literal is reachable
    only if serializing those fixed rows fails (55 rows, ~2.6 KB,
    finite fixed quantiles — not reachable in practice).
  - H1, H2, H4, H6 explicitly judged sound; no other must-fix found.
  Full reply: `~/workspace/makefun-bounty/work/judge5-reply.md`
  (prompt: `judge5-prompt.md`). Champion Upgrade target >=95/100
  is NOT met (best completed score 89/100, pass 4; pass 5: 86/100).
- Pass 6 (SolAPI gpt-6.1-sol, harshest judge; one focused call,
  reasoning high, returned in ~89 s): **94/100** (axes: contract
  29/30, forecast 28/30, edge/fallbacks 13/15, determinism 10/10,
  code 14/15). Both pass-5 must-fixes VERIFIED fixed by the judge:
  (1) the restored broad cap matches the tested form (prior 0.0,
  ridge 0.15, Huber 0.35, `b_raw < b_count`, `info_raw >= 0.15`, no
  direct_only skip) and dense recovers 0.2852 -> 0.2569 (~91% of the
  regression from 0.2540), practice unchanged, synthetic +0.0006;
  (2) the nonempty commit paths validate against the retained trophy
  list before serialization, and the new terminal test covers both
  described outcomes. ONE remaining blocker, NOT fixed in this
  revision (parent stop order: one fix round + one judge call only):
  the fixed [50, 50, 50, 50, 50] vector is tried only when
  fallback-row construction RAISES; if the emergency rows build
  successfully but fail validation or the size guard, the code
  returns without trying the fixed vector — even though a compliant
  fixed-vector payload can fit where float fallback rows cannot
  (judge's example: 55 distinct 1,162-digit integer IDs take 65,520
  bytes with the fixed vector vs >=65,795 bytes as floats). Also the
  empty-literal branch for a genuinely empty case bypasses
  validate_output/serialize_output, so validation before os.replace
  is not universal. The judge's verdict: DO NOT SHIP on this one
  blocker; no other new blocker demonstrated. Champion Upgrade
  target >=95/100 is NOT met (94/100, best score so far).
  Full reply: `~/workspace/makefun-bounty/work/judge6-reply.md`
  (prompt: `judge6-prompt.md`).
- Pass 7 (SolAPI gpt-6.1-sol, harshest judge; one focused call,
  reasoning high, returned in ~67 s): **95/100 — SHIP** (axes:
  contract 30/30, forecast 26/30, edge/fallbacks 15/15, determinism
  10/10, code 14/15). The pass-6 must-fix VERIFIED fixed: the
  fixed-vector retry covers both failure modes (emergency rows
  failing validation skips serialization so text stays None and the
  fixed vector is tried; serialization returning None also triggers
  the retry; construction raising is handled by the exception
  handler), and the empty-literal branch now calls validate_output
  before serialize_output and returns if no serialized text survives
  — the unvalidated literal write is gone. Tests (c) and (d)
  verified as effective end-to-end regressions: (c) proves the real
  primary forecast is valid but oversized, the emergency rows are
  oversized, the fixed payload fits at 65,520 bytes, and the written
  output carries all 55 retained IDs with fixed quantiles; (d)
  proves forced validation failure prevents an empty-case
  replacement and preserves the target, and the restored path writes
  exactly `{"trophies":[]}\n`. No new demonstrated blocker.
  Advisories only: (i) terminal comments overclaim minimality —
  [0,0,0,0,0] is also legal and saves 275 bytes across 55 rows, so
  the final comment should say "no attempted candidate produced
  compliant output" (NOT applied — comment wording only, stop order
  after one fix round; code behavior unchanged); (ii) forecast
  evidence remains the principal uncertainty (dense trails
  persistence by 0.0087; practice uses pseudo-answers); (iii) the
  new tests establish recovery correctness, not additional forecast
  skill. Champion Upgrade target >=95/100 MET (95/100).
  Full reply: `~/workspace/makefun-bounty/work/judge7-reply.md`
  (prompt: `judge7-prompt.md`).
- Judge pass 3 infrastructure note: the upstream kept killing long
  streams (`stream_timeout`, `Unable to forward upstream stream`) and
  a non-streaming call hit Cloudflare HTTP 524 (120 s proxy timeout);
  the rebuilt client `~/workspace/makefun-bounty/work/sol-responses.py`
  now retries a failed stream through the effort/token ladder and
  refuses to save incomplete replies. Pass-3 prompts/replies live in
  `~/workspace/makefun-bounty/work/` (judge3-prompt.md,
  judge3b-prompt.md, judge3b-reply.md) — the earlier /tmp copies were
  lost in a runtime restart that wiped /tmp, along with the pass-1/2
  reply files (their substance is preserved in this file).

## Known limitations (honest)

- No actual 04:30Z answers ship in the kit; practice scores are
  against dilution pseudo-answers, and the 10:17Z replay covers only
  9 trophies. The house AI baseline is unknown — nothing here proves
  a win against it, only against the two reproducible house baselines.
- Dense benchmark: on a 48h-cutoff case whose earners saturate early,
  log-linear-in-x extrapolation overshoots and persistence still wins
  narrowly (0.2569 vs 0.2482 — recovered from the H-round regression
  0.2852 by restoring the broad cap; the G-round 0.2540 is not fully
  recovered because H2 ledger mids stay in place).
  The recent-slope guard constrains 43/55 count slopes and 17/55
  direct slopes on this benchmark. The final case (60h cutoff) sits
  between the 12h scenarios (clear wins) and this dense case. The
  simulator's saturation speed (story tau 0.5-1.5 days) is a chosen
  parameter, not a measured one; the real 10:17Z replay favors us 5x.
- Terminal fallback gap (judge pass 6) FIXED in the K-round revision
  (judge pass 7 verified, 95/100 SHIP): the fixed [50, 50, 50, 50, 50]
  vector is now tried after ANY terminal candidate fails validation
  or serialization, and the empty-literal branch goes through
  validate + serialize. End-to-end regression test (55 trophies,
  1162-digit ids) proves recovery: float rows 66,425 bytes rejected,
  fixed-vector payload 65,520 bytes written and validated. Advisory
  left open (comment wording only): [0,0,0,0,0] would also be legal
  and 275 bytes smaller, so the "smallest legal payload" comment
  overclaims; behavior is unaffected.
- The startup Weibull scale (4h) and the jump set {2,4,8} are priors,
  not calibrated: only one first-refresh interval (launch -> 04:30Z,
  4.67h) is observed. The median can stay at the anchor on short
  horizons when the no-change atom exceeds 50%.
- Spread constants (0.5/sqrt-day random walk, 10% x2.5 tail,
  fallback scale 2) are priors validated only on the replays above.

## Zero-fail QA (final, clean directory, 2026-11-19 ~21:05 WIB)

Package copied to /tmp/qa7-final/pkg (no `__pycache__`), practice
cases p00.in/p01.in + demo/demo.in run from the clean directory with
`MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR` env vars, TWICE:
`python3 -P -s main.py` rc=0 both runs, stderr 0 bytes both runs,
outputs byte-identical across the two runs, all three outputs VALID
(`tests/validate.py`). Backtest + edge suite re-run from the clean
copy: dense 0.2569 (1 CPU, 0.40 s), ALL EDGE CASES PASS (including
the new terminal-contract gap + empty-validation checks). Imports
are stdlib only (json, math, os, sys, datetime, decimal, pathlib);
grep for api_key/secret/token/password/http/urllib/socket/requests
across main.py and .makefun/owner.json is clean (no matches). No
network, no config, no vendored answers. `.makefun/owner.json`
present in the package (58 bytes, sha256
2fa59095823d3ea016372f445fa71eb4f3e0a18f451ac8875941e9469b31fcdb),
byte-identical to the format verified passing on the
gta6-cover-art-01 entry. `__pycache__` removed from the package (0
remain).
