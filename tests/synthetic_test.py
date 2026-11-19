"""Synthetic stress suite for main.py, built from the design doc's
validation methodology (sol-design-rarity-reply.md section 5).

    python3 tests/synthetic_test.py

Held-out seeds (201-207, disjoint from tests/backtest.py) generate a
decaying arrival surge, cohort-specific unlock delays, batched
publication, occasional denominator backfills, rounded zeros, a
saturated opener, and missing reads. One probe trophy is built for a
rate-decline-to-recovery crossover, and both a 90-minute and a
4.5-day horizon are exercised. Every forecast must be contract-valid
and deterministic; pooled pinball loss is reported against the
persistence starter and against a simple anchored log-linear fitted
curve, per regime and horizon, so launch cases cannot hide weak
steady-state forecasts.

Exit code is 0 only when every forecast is valid and deterministic.
The pooled comparisons against the baselines are reported as PASS /
FAIL lines; they do not change the exit code, because this file is a
measurement harness and its numbers are reported as measured.
Uses only the standard library.
"""
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
import main  # noqa: E402

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SPREAD = [-1.645, -0.674, 0.0, 0.674, 1.645]
LAUNCH = datetime(2026, 11, 18, 23, 50, tzinfo=timezone.utc)
P_LAUNCH = 638.0
MAX_BYTES = 65536
# Previous implementation's numbers, for reference only.
TARGET_POOLED = 0.4221
TARGET_DENSE = 0.2569

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("PASS %s" % name)
    else:
        print("FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def stamp(moment):
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def slogit(rate):
    r = min(99.99, max(0.01, float(rate)))
    return math.log(r / (100.0 - r))


def pinball(answer, rows):
    """Mean logit-scale pinball loss over trophies x levels."""
    total = 0.0
    count = 0
    for y, row in zip(answer, rows):
        for li, tau in enumerate(LEVELS):
            d = slogit(y) - slogit(row["q"][li])
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
            count += 1
    return total / count


def spread_rows(case, centres):
    """Point forecasts widened with the kit starter's spread."""
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0)
    rows = []
    for trophy, centre in zip(case["trophies"], centres):
        lg = slogit(centre)
        rows.append({"id": trophy["id"],
                     "q": [main.round_out(100.0 / (1.0 + math.exp(-(lg + z * spread))))
                           for z in Z_SPREAD]})
    return rows


def starter_rows(case):
    """Kit persistence baseline: last observed rate, starter spread."""
    last = case["history"][-1]
    centres = [r if r is not None else 0.0 for r in last["rates"]]
    return spread_rows(case, centres)


def fitted_curve_rows(case):
    """Anchored log-linear baseline: per trophy, regress logit-rate
    differences on log-time differences from the last valid anchor
    and project to the target; starter spread around the point."""
    history = case["history"]
    cutoff = main.parse_time(case["at"])
    target = main.parse_time(case["target"])
    t_tgt = main.days_since_launch(target)
    x_tgt = main.x_of_t(t_tgt)
    centres = []
    for i in range(len(case["trophies"])):
        anchor_rate = None
        anchor_t = None
        points = []
        for read in history:
            rates = read.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            moment = main.parse_time(read.get("at"))
            if moment is None or moment > cutoff:
                continue
            t_days = main.days_since_launch(moment)
            points.append((t_days, rates[i]))
        if points:
            anchor_t, anchor_rate = points[-1]
        if anchor_rate is None or len(points) < 2:
            centres.append(anchor_rate if anchor_rate is not None else 0.0)
            continue
        x_a = main.x_of_t(anchor_t)
        y_a = slogit(anchor_rate)
        num = 0.0
        den = 0.0
        for t_days, rate in points[:-1]:
            d = main.x_of_t(t_days) - x_a
            num += d * (slogit(rate) - y_a)
            den += d * d
        slope = max(-6.0, min(6.0, num / den)) if den > 1e-12 else 0.0
        z = y_a + slope * (x_tgt - x_a)
        centres.append(100.0 / (1.0 + math.exp(-z)))
    return spread_rows(case, centres)


def build_roster():
    """55 trophies of my own wording, spanning the classifier's
    families: 1 saturated opener, 1 platinum, 12 story chapters,
    10 side missions, 12 exploration/collectibles, 12 grinds,
    3 completion markers, 4 unclassified."""
    roster = []
    roster.append(("Just Like Paradise", "Just like paradise.",
                   "gold", 90))
    roster.append(("Leonida Legend", "Earn all trophies.",
                   "platinum", 90))
    for n in range(1, 13):
        roster.append(("Chapter %d" % n,
                       "Complete Chapter %d." % n, "bronze", 15))
    for n in range(1, 11):
        roster.append(("Side Mission %d" % n,
                       "Complete a side mission in Vice City.", "bronze", 15))
    for n in range(1, 13):
        roster.append(("Hidden Package %d" % n,
                       "Collect all hidden packages in zone %d." % n,
                       "silver", 30))
    for n in range(1, 13):
        roster.append(("Street Racer %d" % n,
                       "Win 100 street races.", "silver", 30))
    for n in range(1, 4):
        roster.append(("Completionist %d" % n,
                       "Complete everything in the game.", "gold", 90))
    for name in ("Sunset Watcher", "Pier Stroller", "Neon Nights",
                 "Ocean Drive"):
        roster.append((name, "Watch the sunset from the pier.",
                       "bronze", 15))
    trophies = []
    for k, (name, desc, tier, points) in enumerate(roster):
        trophies.append({"id": k + 1, "name": name, "description": desc,
                         "tier": tier, "points": points, "secret": False})
    return trophies


# Per-family truth parameters: eventual unlock share, unlock delay
# in hours, and exponential unlock time constant in days.
FAMILY_PARAMS = {
    "opening": ((0.97, 0.995), (0.0, 0.0), (0.05, 0.20)),
    "story": ((0.25, 0.85), (0.0, 8.0), (0.5, 1.5)),
    "side": ((0.05, 0.35), (0.0, 24.0), (1.0, 3.0)),
    "exploration": ((0.02, 0.25), (0.0, 24.0), (1.0, 4.0)),
    "grind": ((0.001, 0.08), (6.0, 72.0), (3.0, 10.0)),
    "completion": ((0.0005, 0.005), (48.0, 120.0), (5.0, 15.0)),
    "unknown": ((0.01, 0.20), (0.0, 24.0), (1.0, 3.0)),
}


def generate(seed, horizon_min, cutoff_hours, read_minutes,
             surge_g=2.0, surge_tau_d=2.0, drop_rate=0.0,
             backfill_at_h=None, backfill_factor=1.0, tau_scale=1.0,
             label="syn"):
    """One synthetic case with exact-truth answers plus probe rates
    for the designated crossover trophy (first story chapter).

    Truth model, per trophy i: players arrive along a decaying
    influx P(t); each arrival cohort unlocks trophy i with eventual
    probability share_i once its delay has passed, then
    exponentially with mean tau_i. Earners follow
    E_i(t) = share_i * B_i(t - delay_i), where B_i is the
    exponentially smoothed arrival mass, updated by the exact
    per-step recurrence for a uniform-within-step arrival flow.
    """
    rng = random.Random(seed)
    trophies = build_roster()
    families = [main.classify_trophy(t) for t in trophies]
    share = []
    delay_h = []
    tau_d = []
    for fam in families:
        (slo, shi), (dlo, dhi), (tlo, thi) = FAMILY_PARAMS[fam]
        share.append(rng.uniform(slo, shi))
        delay_h.append(rng.uniform(dlo, dhi))
        tau_d.append(rng.uniform(tlo, thi) * tau_scale)
    # Crossover probe: first story chapter, long delay + large share,
    # so its true rate first falls (denominator surge outruns
    # earners) and later recovers (earners catch up).
    cross = families.index("story")
    share[cross] = 0.70
    delay_h[cross] = 18.0
    tau_d[cross] = 2.0
    cutoff = LAUNCH + timedelta(hours=cutoff_hours)
    target = cutoff + timedelta(minutes=horizon_min)

    def players_true(moment):
        t_days = (moment - LAUNCH).total_seconds() / 86400.0
        p = P_LAUNCH * (1.0 + surge_g * (1.0 - math.exp(-t_days / surge_tau_d)))
        if backfill_at_h is not None:
            if (moment - LAUNCH).total_seconds() / 3600.0 >= backfill_at_h:
                p *= backfill_factor
        return p

    # Fine truth grid, 5-minute steps from launch to the target.
    step = timedelta(minutes=5)
    grid = []
    moment = LAUNCH
    while moment <= target:
        grid.append(moment)
        moment += step
    if grid[-1] < target:
        grid.append(target)
    p_grid = [players_true(m) for m in grid]
    n_grid = len(grid)
    # Earners per trophy on the grid, via the smoothing recurrence.
    earners = []
    for i in range(len(trophies)):
        tau_s = tau_d[i] * 86400.0
        dt_s = 300.0
        decay = math.exp(-dt_s / tau_s)
        flow = (tau_s / dt_s) * (1.0 - decay)
        smooth = 0.0  # C(t): arrival mass still pending unlock
        b_col = []
        for gi in range(n_grid):
            if gi:
                d_p = p_grid[gi] - p_grid[gi - 1]
                smooth = smooth * decay + d_p * flow
            b_col.append((p_grid[gi] - P_LAUNCH) - smooth)
        shift = int(round(delay_h[i] * 3600.0 / 300.0))
        col = []
        for gi in range(n_grid):
            src = gi - shift
            col.append(share[i] * b_col[src] if src >= 0 else 0.0)
        earners.append(col)

    def idx_at(moment):
        idx = int((moment - LAUNCH).total_seconds() // 300)
        return max(0, min(idx, n_grid - 1))

    def true_rate(i, moment):
        gi = idx_at(moment)
        return 100.0 * earners[i][gi] / p_grid[gi]

    # Batched publication: the displayed page only changes at seeded
    # refresh times; reads between refreshes repeat the last page.
    refreshes = [LAUNCH]
    moment = LAUNCH
    while moment < cutoff:
        gap_h = min(10.0, max(1.0, rng.expovariate(1.0 / 4.0)))
        moment = moment + timedelta(hours=gap_h)
        if moment <= cutoff:
            refreshes.append(moment)
    refreshes.append(cutoff)
    history = []
    shown_p = None
    shown_rates = None
    shown_earned = 0
    ri = 0
    read = LAUNCH
    first = True
    while read <= cutoff:
        while ri < len(refreshes) and refreshes[ri] <= read:
            shown_p = players_true(refreshes[ri])
            gi = idx_at(refreshes[ri])
            shown_rates = [round(100.0 * earners[i][gi] / shown_p, 2)
                           for i in range(len(trophies))]
            shown_earned = sum(int(round(earners[i][gi]))
                               for i in range(len(trophies)))
            ri += 1
        if not first and drop_rate and rng.random() < drop_rate:
            read += timedelta(minutes=read_minutes)
            continue
        first = False
        history.append({
            "slot": stamp(read), "at": stamp(read),
            "players_tracked": int(round(shown_p)),
            "trophies_earned": shown_earned,
            "rates": list(shown_rates),
        })
        read += timedelta(minutes=read_minutes)
    answer = [true_rate(i, target) for i in range(len(trophies))]
    case = {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": "syn-%s-%d" % (label, seed),
        "at": stamp(cutoff), "horizon_minutes": horizon_min,
        "target": stamp(target), "levels": LEVELS,
        "game": {"title": "Grand Theft Auto VI"},
        "trophies": trophies, "history": history,
        "notes": "synthetic %s seed=%d" % (label, seed),
    }
    probes = {
        "crossover": [true_rate(cross, cutoff - timedelta(hours=h))
                      for h in (36, 12, 0)]
        + [true_rate(cross, target)],
        "opener_at_target": true_rate(0, target),
        "rounded_zeros": sum(1 for r in shown_rates if r == 0.0),
    }
    return case, answer, probes


SCENARIOS = [
    ("S1 launch-surge 90min",
     dict(seed=201, horizon_min=90, cutoff_hours=6, read_minutes=15,
          surge_g=6.0, surge_tau_d=0.5, label="surge90")),
    ("S2 launch-surge 6510min",
     dict(seed=202, horizon_min=6510, cutoff_hours=48, read_minutes=30,
          surge_g=6.0, surge_tau_d=0.5, label="surge6510")),
    ("S3 steady 6510min",
     dict(seed=203, horizon_min=6510, cutoff_hours=48, read_minutes=30,
          surge_g=1.5, surge_tau_d=2.0, label="steady")),
    ("S4 missing-reads 6510min",
     dict(seed=204, horizon_min=6510, cutoff_hours=48, read_minutes=30,
          surge_g=1.5, surge_tau_d=2.0, drop_rate=0.30, label="missing")),
    ("S5 slow-grind 6510min",
     dict(seed=205, horizon_min=6510, cutoff_hours=48, read_minutes=30,
          surge_g=1.5, surge_tau_d=2.0, tau_scale=2.5, label="grind")),
    ("S6 backfill 6510min",
     dict(seed=206, horizon_min=6510, cutoff_hours=48, read_minutes=30,
          surge_g=1.5, surge_tau_d=2.0, backfill_at_h=30.0,
          backfill_factor=2.2, label="backfill")),
]


def run_case(label, kwargs):
    case, answer, probes = generate(**kwargs)
    rows = main.forecast_case(case)
    ids = [t["id"] for t in case["trophies"]]
    check(label + "-valid", main.validate_output(rows, ids, LEVELS))
    again = main.forecast_case(case)
    check(label + "-deterministic",
          json.dumps(rows) == json.dumps(again))
    text = main.serialize_output(rows)
    check(label + "-under-cap",
          text is not None and len(text.encode("utf-8")) <= MAX_BYTES)
    ours = pinball(answer, rows)
    base = pinball(answer, starter_rows(case))
    curve = pinball(answer, fitted_curve_rows(case))
    print("%s: ours %.4f starter %.4f fitted-curve %.4f (reads %d, "
          "rounded-zero rates at cutoff %d, opener at target %.2f%%)"
          % (label, ours, base, curve, len(case["history"]),
             probes["rounded_zeros"], probes["opener_at_target"]))
    cr = probes["crossover"]
    print("  crossover probe true rates (cutoff-36h, -12h, cutoff, "
          "target): %s" % " ".join("%.2f" % v for v in cr))
    return ours, base, curve


def run_all():
    start = time.time()
    pooled = {"ours": [], "starter": [], "curve": []}
    for label, kwargs in SCENARIOS:
        try:
            ours, base, curve = run_case(label, kwargs)
        except Exception as exc:  # noqa: BLE001 - report, don't abort
            check(label + "-no-raise", False, repr(exc))
            continue
        pooled["ours"].append(ours)
        pooled["starter"].append(base)
        pooled["curve"].append(curve)
    n = len(pooled["ours"])
    if n:
        p_ours = sum(pooled["ours"]) / n
        p_base = sum(pooled["starter"]) / n
        p_curve = sum(pooled["curve"]) / n
        print("synthetic pooled: ours %.4f starter %.4f fitted-curve "
              "%.4f (target ours %.4f)" % (p_ours, p_base, p_curve,
                                          TARGET_POOLED))
        check("pooled-beats-starter", p_ours < p_base,
              "%.4f vs %.4f" % (p_ours, p_base))
        check("pooled-beats-fitted-curve", p_ours < p_curve,
              "%.4f vs %.4f" % (p_ours, p_curve))
        check("pooled-beats-target", p_ours < TARGET_POOLED,
              "%.4f vs %.4f" % (p_ours, TARGET_POOLED))
    print("== dense benchmark ==")
    try:
        start_d = time.time()
        case, answer, _probes = generate(
            seed=207, horizon_min=6510, cutoff_hours=48, read_minutes=10,
            surge_g=1.5, surge_tau_d=2.0, label="dense")
        print("dense reads: %d" % len(case["history"]))
        rows = main.forecast_case(case)
        elapsed = time.time() - start_d
        ids = [t["id"] for t in case["trophies"]]
        check("dense-valid", main.validate_output(rows, ids, LEVELS))
        check("dense-deterministic",
              json.dumps(rows) == json.dumps(main.forecast_case(case)))
        d_ours = pinball(answer, rows)
        d_base = pinball(answer, starter_rows(case))
        d_curve = pinball(answer, fitted_curve_rows(case))
        print("dense: %.2fs ours %.4f starter %.4f fitted-curve %.4f "
              "(target ours %.4f)" % (elapsed, d_ours, d_base, d_curve,
                                      TARGET_DENSE))
        check("dense-beats-starter", d_ours < d_base,
              "%.4f vs %.4f" % (d_ours, d_base))
        check("dense-beats-target", d_ours < TARGET_DENSE,
              "%.4f vs %.4f" % (d_ours, TARGET_DENSE))
    except Exception as exc:  # noqa: BLE001 - report, don't abort
        check("dense-no-raise", False, repr(exc))
    print("synthetic runtime %.2fs" % (time.time() - start))
    print("synthetic_test: %d checks failed" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(run_all())
