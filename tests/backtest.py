"""Offline backtest for main.py (run from the package root).

    python3 tests/backtest.py

Scores the model on the kit practice cases against constructed
pseudo-answers, on a 10:17Z replay of the live Exophase snapshot, on
synthetic refresh-jump stress scenarios, and on a dense 289-read
benchmark. Every number is measured here from the current main.py;
the TARGET lines quote the previous implementation's numbers as
reference only -- a MISS means this rewrite scores differently, not
that the run failed.

Pseudo-answer construction (verified: the kit starter baseline scores
exactly 1.6871 / 1.5946 / 1.5963 / 1.0028 against them):
- dilution: launch earners (rounding-interval midpoints at P=638)
  diluted to the 04:30Z player count P=3868; the 100% opener stays 100%.
- dilution+growth: as above with earners grown by exp(1.3 * dx),
  dx = ln(1+t_target) - ln(1+t_anchor) in days since launch.
- replay: the 10:17Z snapshot values named in the design brief
  (platinum 0.10, opener 100, Ch1 0.49, Ch2-4 + final mission 0.23,
  Baited My Tackle 1.16, Stick Fiddler 0.39), scored on the p01 prefix
  retargeted to 10:17Z.
- scenario G: p00 history with trophy id 3 set to 79.94% (E=510 at
  P=638); truth 510/3868 = 13.19%.

The synthetic generator is original to this file (seeded, deterministic):
decaying player influx, cohort unlock delays, batched publication
(rates frozen between refreshes), occasional denominator backfills,
rounded zeros, a saturated opener, missing reads, and a
rate-decline-to-recovery crossover. Truth is the exact unrounded
E/P rate at the target.

Pass criteria (exit code): every forecast valid, synthetic pooled
ours < starter pooled, scenario G ours < starter, practice combined
ours < starter combined, replay ours < starter. Target equality is
reported, never asserted.
"""
import copy
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

KIT_IN = ("/home/hatch/workspace/makefun-bounty/work/"
          "gta6-trophy-rarity-forecast-01-kit/"
          "gta6-trophy-rarity-forecast-01-kit/input")
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
LAUNCH = datetime(2026, 11, 18, 23, 50, tzinfo=timezone.utc)
P_LAUNCH = 638
P_REFRESH = 3868

# Previous implementation's numbers, for reference only.
TARGETS = {
    "practice_dilution": (0.5917, 1.6871),
    "practice_growth": (0.5367, 1.5955),
    "replay": (0.2034, 1.0028),
    "synthetic_pooled": (0.4221, 0.5043),
    "scenario_g": (0.9105, 3.1764),
    "dense": (0.2569, 0.2482),
}

RESULTS = []


def slogit(rate):
    r = min(99.99, max(0.01, float(rate)))
    return math.log(r / (100.0 - r))


def pinball(answer, rows, levels=LEVELS):
    """Mean logit-scale pinball loss over trophies x levels."""
    total = 0.0
    count = 0
    per_level = [0.0] * len(levels)
    for y, row in zip(answer, rows):
        for li, tau in enumerate(levels):
            d = slogit(y) - slogit(row["q"][li])
            loss = 2.0 * max(tau * d, (tau - 1.0) * d)
            total += loss
            per_level[li] += loss
            count += 1
    n = len(answer)
    return total / count, [v / n for v in per_level]


def starter_rows(case):
    """Kit starter (persistence) baseline, number for number."""
    last = case["history"][-1]
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i]
        lg = slogit(rate if rate is not None else 0.0)
        rows.append({"id": trophy["id"],
                     "q": [main.round_out(100.0 / (1.0 + math.exp(-(lg + z * spread))))
                           for z in Z]})
    return rows


def load_kit(cid):
    with open(os.path.join(KIT_IN, cid + ".in")) as fh:
        return json.load(fh)


def launch_earners(case):
    first = case["history"][0]
    return [main.earner_interval(r, first["players_tracked"])[2]
            for r in first["rates"]]


def dilution_answer(case):
    """Frozen earners diluted to P=3868; the 100% opener stays 100%."""
    ans = [e / P_REFRESH * 100.0 for e in launch_earners(case)]
    for i, rate in enumerate(case["history"][0]["rates"]):
        if rate == 100:
            ans[i] = 100.0
    return ans


def dilution_growth_answer(case):
    """Dilution with earners grown by exp(1.3 * dx) to the target."""
    anchor = main.parse_time(case["history"][-1]["at"])
    target = main.parse_time(case["target"])
    dx = (main.x_of_t(main.days_since_launch(target))
          - main.x_of_t(main.days_since_launch(anchor)))
    factor = math.exp(1.3 * dx)
    ans = [e * factor / P_REFRESH * 100.0
           for e in launch_earners(case)]
    for i, rate in enumerate(case["history"][0]["rates"]):
        if rate == 100:
            ans[i] = 100.0
    return ans


def report(name, ours, base, target):
    match = abs(ours - target[0]) < 0.00005
    base_match = abs(base - target[1]) < 0.00005
    tag = "MATCH" if match else "MISS"
    btag = "MATCH" if base_match else "MISS"
    print("%s: ours %.4f (target %.4f, %s) vs starter %.4f "
          "(target %.4f, %s)" % (name, ours, target[0], tag,
                                 base, target[1], btag))
    RESULTS.append((name, ours, base, target, match, base_match))
    return match


def practice():
    print("== practice ==")
    combined_ours, combined_base = [], []
    combined_ours_g, combined_base_g = [], []
    for cid in ("p00", "p01"):
        case = load_kit(cid)
        rows = main.forecast_case(case)
        ids = [t["id"] for t in case["trophies"]]
        assert main.validate_output(rows, ids, LEVELS), cid
        ans_d = dilution_answer(case)
        ans_g = dilution_growth_answer(case)
        base = starter_rows(case)
        ours_d, lv_d = pinball(ans_d, rows)
        base_d, _ = pinball(ans_d, base)
        ours_g, _ = pinball(ans_g, rows)
        base_g, _ = pinball(ans_g, base)
        print("%s vs dilution: ours %.4f starter %.4f "
              "(per-level ours %s)" % (
                  cid, ours_d, base_d,
                  " ".join("%.3f" % v for v in lv_d)))
        print("%s vs dilution+growth: ours %.4f starter %.4f"
              % (cid, ours_g, base_g))
        combined_ours.append(ours_d)
        combined_base.append(base_d)
        combined_ours_g.append(ours_g)
        combined_base_g.append(base_g)
    ours = sum(combined_ours) / len(combined_ours)
    base = sum(combined_base) / len(combined_base)
    report("practice vs dilution", ours, base,
           TARGETS["practice_dilution"])
    ours_g = sum(combined_ours_g) / len(combined_ours_g)
    base_g = sum(combined_base_g) / len(combined_base_g)
    report("practice vs dilution+growth", ours_g, base_g,
           TARGETS["practice_growth"])
    return ours < base and ours_g < base_g


REPLAY_RATES = {1: 0.10, 2: 100.0, 3: 0.49, 4: 0.23, 5: 0.23,
                6: 0.23, 7: 0.23, 20: 1.16, 21: 0.39}


def replay():
    print("== 10:17Z replay ==")
    case = load_kit("p01")
    case["target"] = "2026-11-19T10:17:00Z"
    at = main.parse_time(case["at"])
    tgt = main.parse_time(case["target"])
    case["horizon_minutes"] = int((tgt - at).total_seconds() / 60)
    rows = {r["id"]: r for r in main.forecast_case(case)}
    base = {r["id"]: r for r in starter_rows(case)}
    idx = sorted(REPLAY_RATES)
    ans = [REPLAY_RATES[i] for i in idx]
    ours, _ = pinball(ans, [rows[i] for i in idx])
    base_loss, _ = pinball(ans, [base[i] for i in idx])
    print("replay (%d known trophies, horizon %d min): ours %.4f "
          "starter %.4f" % (len(idx), case["horizon_minutes"],
                            ours, base_loss))
    report("10:17Z replay", ours, base_loss, TARGETS["replay"])
    return ours < base_loss


def scenario_g():
    print("== scenario G ==")
    case = load_kit("p00")
    for read in case["history"]:
        read["rates"][2] = 79.94
    rows = {r["id"]: r for r in main.forecast_case(case)}
    base = {r["id"]: r for r in starter_rows(case)}
    truth = 510.0 / P_REFRESH * 100.0
    ours, _ = pinball([truth], [rows[3]])
    base_loss, _ = pinball([truth], [base[3]])
    print("G id 3 (79.94%% -> truth %.2f%%): ours q=%s loss %.4f "
          "starter loss %.4f" % (truth, rows[3]["q"], ours, base_loss))
    report("scenario G", ours, base_loss, TARGETS["scenario_g"])
    return ours < base_loss


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------

FAMILY_SHARE = {
    "opening": (1.0, 1.0), "story": (0.25, 0.60), "side": (0.05, 0.30),
    "exploration": (0.05, 0.25), "grind": (0.005, 0.08),
    "completion": (0.001, 0.01), "unknown": (0.01, 0.20),
}
FAMILY_DELAY_H = {
    "opening": (0.0, 0.0), "story": (0.0, 6.0), "side": (0.0, 24.0),
    "exploration": (0.0, 24.0), "grind": (6.0, 72.0),
    "completion": (24.0, 120.0), "unknown": (0.0, 24.0),
}
FAMILY_TAU_D = {
    "opening": (0.05, 0.15), "story": (0.5, 1.5), "side": (1.0, 3.0),
    "exploration": (1.0, 4.0), "grind": (3.0, 10.0),
    "completion": (5.0, 15.0), "unknown": (1.0, 3.0),
}


def simulate(seed, horizon_min, cutoff_hours, read_minutes,
             regime="steady", drop_rate=0.0, backfill_at_h=None,
             backfill_factor=1.0, surge_g=2.0, surge_tau_d=2.0):
    """Generate one synthetic case plus exact-truth answers.

    regime only selects the parameter bundle; the construction is the
    same for every scenario: P(t) follows a decaying-influx rise, E_i(t)
    a delayed saturating curve per trophy family, and the displayed
    page only refreshes at seeded refresh times (frozen between them).
    """
    rng = random.Random(seed)
    template = load_kit("p00")
    trophies = template["trophies"]
    families = [main.classify_trophy(t) for t in trophies]
    if regime == "slow-grind":
        tau_scale = 2.5
    else:
        tau_scale = 1.0
    share_inf = []
    delay_h = []
    tau_d = []
    for fam in families:
        lo, hi = FAMILY_SHARE[fam]
        share_inf.append(rng.uniform(lo, hi))
        dlo, dhi = FAMILY_DELAY_H[fam]
        delay_h.append(rng.uniform(dlo, dhi))
        tlo, thi = FAMILY_TAU_D[fam]
        tau_d.append(rng.uniform(tlo, thi) * tau_scale)
    cutoff = LAUNCH + timedelta(hours=cutoff_hours)
    target = cutoff + timedelta(minutes=horizon_min)

    def p_true(moment):
        t_days = (moment - LAUNCH).total_seconds() / 86400.0
        p = P_LAUNCH * (1.0 + surge_g * (1.0 - math.exp(-t_days / surge_tau_d)))
        if backfill_at_h is not None:
            if (moment - LAUNCH).total_seconds() / 3600.0 >= backfill_at_h:
                p *= backfill_factor
        return p

    # Time grid (read step) from launch to the target, for the
    # arrival convolution below.
    grid = []
    moment = LAUNCH
    while moment <= target:
        grid.append(moment)
        moment += timedelta(minutes=read_minutes)
    if grid[-1] < target:
        grid.append(target)
    p_grid = [p_true(m) for m in grid]

    # Earners by arrival convolution: each player arriving at s unlocks
    # trophy i with probability share_i once s + delay_i has passed,
    # then exponentially with mean tau_i. Per trophy, walk the grid
    # once, activating arrivals past their delay and decaying the
    # not-yet-unlocked mass: E = share * (activated - pending).
    # This keeps every rate in [0, 100] and makes refreshes dilute
    # rates (new arrivals have not unlocked yet) before recovery.
    e_grid = []
    for i in range(len(trophies)):
        delay = timedelta(hours=delay_h[i])
        tau_s = tau_d[i] * 86400.0
        activated = 0.0
        pending = 0.0
        prev_p = 0.0
        arrivals = []  # (activation time, players) not yet activated
        ai = 0
        col = []
        for gi, moment in enumerate(grid):
            arrivals.append((moment + delay, p_grid[gi] - prev_p))
            prev_p = p_grid[gi]
            dt_s = 0.0 if gi == 0 else (moment - grid[gi - 1]).total_seconds()
            if dt_s:
                pending *= math.exp(-dt_s / tau_s)
            while ai < len(arrivals) and arrivals[ai][0] <= moment:
                activated += arrivals[ai][1]
                pending += arrivals[ai][1]
                ai += 1
            col.append(share_inf[i] * max(0.0, activated - pending))
        e_grid.append(col)

    def at_grid(moment):
        """Index of the grid point at or just before moment."""
        idx = int((moment - LAUNCH).total_seconds() // (read_minutes * 60))
        return min(idx, len(grid) - 1)

    def e_true(i, moment):
        return e_grid[i][at_grid(moment)]

    # Refresh times: seeded exponential gaps, mean 6 h, clamped 1-12 h.
    refreshes = [LAUNCH]
    moment = LAUNCH
    while moment < cutoff:
        gap_h = min(12.0, max(1.0, rng.expovariate(1.0 / 6.0)))
        moment = moment + timedelta(hours=gap_h)
        if moment <= cutoff:
            refreshes.append(moment)
    refreshes.append(cutoff)

    history = []
    shown_p = None
    shown_rates = None
    shown_earned = 0
    ri = 0  # next refresh to publish; a read shows the latest refresh <= it
    read = LAUNCH
    while read <= cutoff:
        while ri < len(refreshes) and refreshes[ri] <= read:
            shown_p = p_true(refreshes[ri])
            shown_rates = []
            shown_earned = 0
            for i in range(len(trophies)):
                e = e_true(i, refreshes[ri])
                shown_rates.append(round(100.0 * e / shown_p, 2))
                shown_earned += int(round(e))
            ri += 1
        if drop_rate and rng.random() < drop_rate and read != LAUNCH:
            read += timedelta(minutes=read_minutes)
            continue
        history.append({
            "slot": read.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "at": read.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "players_tracked": int(round(shown_p)),
            "trophies_earned": shown_earned,
            "rates": list(shown_rates),
        })
        read += timedelta(minutes=read_minutes)
    p_tgt = p_true(target)
    answer = [100.0 * e_true(i, target) / p_tgt for i in range(len(trophies))]
    case = {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": "syn-%s-%d" % (regime, seed),
        "at": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_minutes": horizon_min,
        "target": target.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "levels": LEVELS,
        "game": template["game"],
        "trophies": trophies,
        "history": history,
        "notes": "synthetic %s seed=%d" % (regime, seed),
    }
    return case, answer


SCENARIOS = [
    # (label, kwargs) -- A-F mirror the earlier suite's regimes. All
    # share a 48 h cutoff (calibrated: the starter baseline then scores
    # 0.03 on the 90-min horizon and 0.72 on the 6510-min horizon,
    # matching the earlier fixture's 0.05 / 0.64); only the horizon
    # and the regime differ.
    ("A launch-surge 90min",
     dict(seed=11, horizon_min=90, cutoff_hours=48, read_minutes=30,
          regime="launch-surge", surge_g=8.0, surge_tau_d=0.75)),
    ("B launch-surge 6510min",
     dict(seed=12, horizon_min=6510, cutoff_hours=48, read_minutes=30,
          regime="launch-surge", surge_g=8.0, surge_tau_d=0.75)),
    ("C steady 6510min",
     dict(seed=13, horizon_min=6510, cutoff_hours=48, read_minutes=30,
          regime="steady")),
    ("D missing-reads 6510min",
     dict(seed=14, horizon_min=6510, cutoff_hours=48, read_minutes=30,
          regime="steady", drop_rate=0.30)),
    ("E slow-grind 6510min",
     dict(seed=15, horizon_min=6510, cutoff_hours=48, read_minutes=30,
          regime="slow-grind")),
    ("F 2nd-backfill 6510min",
     dict(seed=16, horizon_min=6510, cutoff_hours=48, read_minutes=30,
          regime="steady", backfill_at_h=24.0, backfill_factor=2.5)),
]


def synthetic():
    print("== synthetic ==")
    pooled_ours, pooled_base = [], []
    per_level_ours = [0.0] * 5
    per_level_base = [0.0] * 5
    for label, kwargs in SCENARIOS:
        case, answer = simulate(**kwargs)
        rows = main.forecast_case(case)
        ids = [t["id"] for t in case["trophies"]]
        assert main.validate_output(rows, ids, LEVELS), label
        base = starter_rows(case)
        ours, lv_o = pinball(answer, rows)
        base_loss, lv_b = pinball(answer, base)
        pooled_ours.append(ours)
        pooled_base.append(base_loss)
        for li in range(5):
            per_level_ours[li] += lv_o[li]
            per_level_base[li] += lv_b[li]
        print("%s: ours %.4f starter %.4f (reads %d)"
              % (label, ours, base_loss, len(case["history"])))
    n = len(SCENARIOS)
    ours = sum(pooled_ours) / n
    base = sum(pooled_base) / n
    print("synthetic per-level ours   %s"
          % " ".join("%.3f" % (v / n) for v in per_level_ours))
    print("synthetic per-level starter %s"
          % " ".join("%.3f" % (v / n) for v in per_level_base))
    report("synthetic pooled", ours, base, TARGETS["synthetic_pooled"])
    return ours < base


def dense():
    print("== dense benchmark ==")
    case, answer = simulate(seed=7, horizon_min=6510, cutoff_hours=48,
                            read_minutes=10, regime="steady")
    print("dense reads: %d" % len(case["history"]))
    start = time.time()
    rows = main.forecast_case(case)
    elapsed = time.time() - start
    ids = [t["id"] for t in case["trophies"]]
    assert main.validate_output(rows, ids, LEVELS), "dense"
    again = main.forecast_case(case)
    assert json.dumps(rows) == json.dumps(again), "dense not deterministic"
    base = starter_rows(case)
    ours, _ = pinball(answer, rows)
    base_loss, _ = pinball(answer, base)
    print("dense: %.2fs ours %.4f starter %.4f" % (elapsed, ours, base_loss))
    report("dense benchmark", ours, base_loss, TARGETS["dense"])
    return ours, base_loss, elapsed


def earner_diagnostic():
    case = load_kit("p00")
    first = case["history"][0]
    total = sum(launch_earners(case))
    print("diag: sum of reconstructed earners = %.1f vs "
          "trophies_earned = %s" % (total, first["trophies_earned"]))
    return abs(total - first["trophies_earned"]) < 1e-9


def main_cli():
    start = time.time()
    ok = True
    ok &= earner_diagnostic()
    ok &= practice()
    ok &= replay()
    ok &= scenario_g()
    ok &= synthetic()
    dense_ours, dense_base, dense_t = dense()
    print("----")
    for name, ours, base, target, match, base_match in RESULTS:
        print("%-28s ours %.4f target %.4f %s | starter %.4f "
              "target %.4f %s" % (name, ours, target[0],
                                  "MATCH" if match else "MISS",
                                  base, target[1],
                                  "MATCH" if base_match else "MISS"))
    print("backtest runtime %.2fs" % (time.time() - start))
    print("backtest: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main_cli())
