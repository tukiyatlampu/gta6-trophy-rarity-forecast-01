"""Backtest harness for gta6-trophy-rarity-forecast-01 (checklist C3/C5).

Scores our main.py against the starter (persistence) baseline with logit-scale
pinball loss, on three honest proxies (no ground truth ships in the kit):

  1. Practice p00/p01 vs a dilution pseudo-answer: earners frozen at their
     launch-night rounding-interval midpoints, players 638 -> 3868 at the
     04:30Z refresh; the opening trophy stays 100%. A second, growth-adjusted
     pseudo-answer lets earners grow at the family prior slope to the target.
  2. A 10:17Z replay: p01 retargeted to the live snapshot (~10:17Z), scored
     only on the trophies whose snapshot rate is known exactly.
  3. Synthetic stress scenarios with exact ground truth (decaying-influx
     surge, backfill jumps, cohort unlock delays, batched publication,
     missing reads, 90-min and 6510-min horizons).

All numbers are backtest estimates, never kit scores.

Usage: python3 tests/backtest.py   (run from the package root)
"""
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main as model  # noqa: E402  (reuse classify + slope defaults)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = os.path.expanduser("~/workspace/makefun-bounty/work/gta6-trophy-rarity-"
                         "forecast-01-kit/gta6-trophy-rarity-forecast-01-kit")
STARTER = os.path.join(KIT, "starter", "main.py")
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

LAUNCH = datetime(2026, 11, 18, 23, 50, tzinfo=timezone.utc)


def clogit(pct):
    r = min(99.99, max(0.01, pct))
    return math.log(r / (100.0 - r))


def pinball(y, qs):
    ly = clogit(y)
    tot = 0.0
    for tau, q in zip(LEVELS, qs):
        d = ly - clogit(q)
        tot += 2.0 * max(tau * d, (tau - 1.0) * d)
    return tot / len(LEVELS)


def run_entry(entry_py, cases, timeout=60):
    """Run an entry grader-style on {id: case} dict; returns {id: output}."""
    with tempfile.TemporaryDirectory() as td:
        inp, out = os.path.join(td, "in"), os.path.join(td, "out")
        os.makedirs(inp)
        os.makedirs(out)
        for cid, case in cases.items():
            with open(os.path.join(inp, cid + ".in"), "w") as f:
                json.dump(case, f)
        env = dict(os.environ, MAKEFUN_INPUT_DIR=inp, MAKEFUN_OUTPUT_DIR=out,
                   PYTHONHASHSEED="0", TZ="UTC")
        t0 = datetime.now()
        r = subprocess.run([sys.executable, "-P", "-s", entry_py], env=env,
                           capture_output=True, text=True, timeout=timeout)
        wall = (datetime.now() - t0).total_seconds()
        if r.returncode != 0:
            raise RuntimeError(f"{entry_py} rc={r.returncode}: {r.stderr[-800:]}")
        res = {}
        for cid in cases:
            with open(os.path.join(out, cid + ".out")) as f:
                res[cid] = json.load(f)
        return res, wall


def score_per_level(outputs, answers):
    """Mean pinball per quantile level over trophies x cases."""
    tot = {tau: 0.0 for tau in LEVELS}
    n = 0
    for cid, out in outputs.items():
        ans = answers[cid]
        for row in out["trophies"]:
            if row["id"] not in ans:
                continue
            ly = clogit(ans[row["id"]])
            for tau, q in zip(LEVELS, row["q"]):
                d = ly - clogit(q)
                tot[tau] += 2.0 * max(tau * d, (tau - 1.0) * d)
            n += 1
    return {tau: tot[tau] / n for tau in LEVELS} if n else {}


def score_all(outputs, answers):
    """Mean pinball over trophies x levels; answers: {id: {trophy_id: rate}}."""
    tot, n = 0.0, 0
    per = {}
    for cid, out in outputs.items():
        ans = answers[cid]
        s, m = 0.0, 0
        for row in out["trophies"]:
            if row["id"] not in ans:
                continue
            s += pinball(ans[row["id"]], row["q"])
            m += 1
        per[cid] = s / m if m else float("nan")
        tot += s
        n += m
    return tot / n if n else float("nan"), per


def load_practice():
    cases = {}
    for n in ("p00", "p01"):
        with open(os.path.join(KIT, "input", n + ".in")) as f:
            cases[n] = json.load(f)
    return cases


def dilution_answers(cases, growth=False):
    """Pseudo-answers for p00/p01 from the 04:30Z refresh (3868 players)."""
    answers = {}
    for cid, case in cases.items():
        first = case["history"][0]
        p0 = first["players_tracked"]
        target = datetime.fromisoformat(case["target"].replace("Z", "+00:00"))
        anchor = datetime.fromisoformat(
            case["history"][-1]["at"].replace("Z", "+00:00"))
        d = math.log(1.0 + (target - LAUNCH).total_seconds() / 86400.0) - \
            math.log(1.0 + (anchor - LAUNCH).total_seconds() / 86400.0)
        ans = {}
        for i, tr in enumerate(case["trophies"]):
            r = first["rates"][i]
            if r is None:
                continue
            if r >= 100.0:
                ans[tr["id"]] = 100.0
                continue
            emin, emax, _w = model.earner_interval(r, p0)
            e_mid = 0.5 * (emin + emax) if emin is not None else p0 * r / 100.0
            if growth:
                fam = model.classify(tr)
                e_mid = e_mid * math.exp(
                    model.COUNT_SLOPE_DEFAULT[fam] * d)
            ans[tr["id"]] = 100.0 * e_mid / 3868.0
        answers[cid] = ans
    return answers


SNAPSHOT_1017 = {1: 0.10, 2: 100.0, 3: 0.49, 4: 0.23, 5: 0.23, 6: 0.23,
                 7: 0.23, 20: 1.16, 21: 0.39}  # ids 4-7 approx (~0.23)


def replay_1017(case_p01):
    case = json.loads(json.dumps(case_p01))
    case["id"] = "r1017"
    case["target"] = "2026-11-19T10:17:00Z"
    case["horizon_minutes"] = 407
    return case


# --- synthetic stress simulator --------------------------------------------

def simulate(seed, horizon_min, jump=6.06, first_refresh_h=4.5,
             refresh_h=4.0, missing=0.0, second_jump=None, slow_grind=False,
             cutoff_hours=12, read_minutes=30):
    """One synthetic case + exact published answer at the target."""
    rng = random.Random(seed)
    trophies = json.load(open(os.path.join(KIT, "input", "p00.in")))["trophies"]
    cutoff = LAUNCH + timedelta(hours=cutoff_hours)
    target = cutoff + timedelta(minutes=horizon_min)
    # Denominator: smooth decaying-influx growth, x6 backfill at 1st refresh.
    p0 = 638.0
    v_den = 0.8
    beta = 1.0

    def p_true(t_days):
        a = 0.5
        integ = a * math.log((0.5 + t_days) / a) if beta == 1.0 else 0.0
        return p0 * (1.0 + v_den * integ)

    def p_pub(t):
        t_days = (t - LAUNCH).total_seconds() / 86400.0
        j = 1.0
        if t >= LAUNCH + timedelta(hours=first_refresh_h):
            j = jump
        if second_jump and t >= LAUNCH + timedelta(hours=second_jump[0]):
            j *= second_jump[1]
        return max(1.0, p_true(t_days) * j)

    p_target = p_pub(target)
    # Earners: unlock curves per trophy; opening trophy tracks players.
    earners = {}
    for tr in trophies:
        fam = model.classify(tr)
        if fam == model.FAMILY_OPENING:
            earners[tr["id"]] = ("open", None, None, None)
            continue
        r_inf = {"story": rng.uniform(0.08, 0.45),
                 "side": rng.uniform(0.03, 0.12),
                 "exploration": rng.uniform(0.01, 0.08),
                 "grind": rng.uniform(0.002, 0.03),
                 "completion": rng.uniform(0.0005, 0.005),
                 "unknown": rng.uniform(0.005, 0.05)}[fam]
        delay = {"story": rng.uniform(0.05, 0.6),
                 "side": rng.uniform(0.2, 1.0),
                 "exploration": rng.uniform(0.3, 1.5),
                 "grind": rng.uniform(1.0, 3.0) if slow_grind else rng.uniform(0.5, 2.0),
                 "completion": rng.uniform(2.0, 4.0),
                 "unknown": rng.uniform(0.3, 1.5)}[fam]
        tau = {"story": rng.uniform(0.5, 1.5),
               "side": rng.uniform(1.0, 2.0),
               "exploration": rng.uniform(1.5, 3.0),
               "grind": rng.uniform(2.0, 4.0),
               "completion": 4.0,
               "unknown": rng.uniform(1.0, 2.5)}[fam]
        c_inf = r_inf / 100.0 * p_target
        earners[tr["id"]] = ("curve", c_inf, delay, tau)

    def e_true(tr_id, t):
        kind, c_inf, delay, tau = earners[tr_id]
        if kind == "open":
            return p_pub(t)
        t_days = (t - LAUNCH).total_seconds() / 86400.0
        if t_days <= delay:
            return c_inf * 0.02
        return c_inf * (0.02 + 0.98 * (1.0 - math.exp(-(t_days - delay) / tau)))

    # Refreshes publish the latest true state; reads sample it every 30 min.
    refresh_times = []
    t = LAUNCH + timedelta(hours=first_refresh_h)
    while t <= target:
        refresh_times.append(t)
        t += timedelta(hours=refresh_h)

    def published(t):
        pub = None
        for rt in refresh_times:
            if rt <= t:
                pub = rt
            else:
                break
        if pub is None:
            pub = LAUNCH
        p = p_pub(pub)
        rates = []
        tot_e = 0
        for tr in trophies:
            e = e_true(tr["id"], pub)
            tot_e += e
            rates.append(round(100.0 * e / p, 2))
        return p, rates, int(round(tot_e))

    history = []
    slot = LAUNCH
    while slot < cutoff:
        if rng.random() >= missing:
            p, rates, tot_e = published(slot)
            rates = [None if (r is not None and rng.random() < missing * 0.5)
                     else r for r in rates]
            history.append({"slot": slot.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "at": (slot + timedelta(seconds=1)).strftime(
                                "%Y-%m-%dT%H:%M:%S.000Z"),
                            "players_tracked": int(round(p)),
                            "trophies_earned": tot_e,
                            "rates": rates})
        slot += timedelta(minutes=read_minutes)
    p_ans, rates_ans, _ = published(target)
    answer = {tr["id"]: rates_ans[i] for i, tr in enumerate(trophies)}
    case = {"schema": "makefun/gta6-trophies-case/v1",
            "id": f"syn{seed}",
            "at": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "horizon_minutes": horizon_min,
            "target": target.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "levels": LEVELS,
            "game": {"name": "Grand Theft Auto VI", "platform": "PS5"},
            "trophies": trophies, "history": history, "notes": "synthetic"}
    return case, answer


def main():
    cases = load_practice()
    # Diagnostic: does trophies_earned equal the sum of reconstructed earners?
    first = cases["p00"]["history"][0]
    tot_e = 0
    for i, tr in enumerate(cases["p00"]["trophies"]):
        emin, emax, _ = model.earner_interval(first["rates"][i], 638)
        tot_e += 0.5 * (emin + emax)
    print(f"diag: sum of reconstructed earners = {tot_e:.1f} "
          f"vs trophies_earned = {first['trophies_earned']}")
    ours, wall = run_entry(os.path.join(HERE, "main.py"), cases)
    base, _ = run_entry(STARTER, cases)
    print(f"runtime: ours {wall:.2f}s for 2 practice cases "
          f"(limit 8s/case)")
    for label, answers in (("dilution", dilution_answers(cases, growth=False)),
                           ("dilution+growth",
                            dilution_answers(cases, growth=True))):
        s_ours, per_ours = score_all(ours, answers)
        s_base, per_base = score_all(base, answers)
        print(f"practice vs {label} pseudo-answer: ours {s_ours:.4f} "
              f"(p00 {per_ours['p00']:.4f}, p01 {per_ours['p01']:.4f}) vs "
              f"persistence {s_base:.4f} "
              f"(p00 {per_base['p00']:.4f}, p01 {per_base['p01']:.4f})")
    # 10:17Z replay on the p01 prefix.
    replay = replay_1017(cases["p01"])
    ours_r, _ = run_entry(os.path.join(HERE, "main.py"), {"r1017": replay})
    base_r, _ = run_entry(STARTER, {"r1017": replay})
    s_ours, _ = score_all(ours_r, {"r1017": SNAPSHOT_1017})
    s_base, _ = score_all(base_r, {"r1017": SNAPSHOT_1017})
    print(f"10:17Z replay (9 known trophies): ours {s_ours:.4f} vs "
          f"persistence {s_base:.4f}")
    # Synthetic stress scenarios.
    scenarios = [
        ("A launch-surge 90min", dict(seed=1, horizon_min=90)),
        ("B launch-surge 6510min", dict(seed=2, horizon_min=6510)),
        ("C steady 6510min", dict(seed=3, horizon_min=6510, jump=1.2)),
        ("D missing-reads 6510min",
         dict(seed=4, horizon_min=6510, missing=0.3)),
        ("E slow-grind 6510min",
         dict(seed=5, horizon_min=6510, slow_grind=True)),
        ("F 2nd-backfill 6510min",
         dict(seed=6, horizon_min=6510, second_jump=(36.0, 1.8))),
    ]
    syn_cases, syn_ans = {}, {}
    for name, kw in scenarios:
        case, ans = simulate(**kw)
        syn_cases[case["id"]] = case
        syn_ans[case["id"]] = ans
    ours_s, wall_s = run_entry(os.path.join(HERE, "main.py"), syn_cases)
    base_s, _ = run_entry(STARTER, syn_cases)
    s_ours, per_ours = score_all(ours_s, syn_ans)
    s_base, per_base = score_all(base_s, syn_ans)
    print(f"synthetic pooled: ours {s_ours:.4f} vs persistence {s_base:.4f} "
          f"({(1 - s_ours / s_base) * 100:.1f}% lower) "
          f"[{wall_s:.2f}s for {len(syn_cases)} cases]")
    for (name, _kw), cid in zip(scenarios, syn_cases):
        print(f"  {name}: ours {per_ours[cid]:.4f} vs "
              f"persistence {per_base[cid]:.4f}")
    # Per-level breakdown (median behaviour reported separately: the
    # aggregate must not hide a median locked to the no-change atom).
    pl_ours = score_per_level(ours, dilution_answers(cases, growth=False))
    pl_base = score_per_level(base, dilution_answers(cases, growth=False))
    print("practice per-level loss (dilution): ours " +
          ", ".join(f"tau={t}: {pl_ours[t]:.3f}" for t in LEVELS) +
          " | persistence " +
          ", ".join(f"tau={t}: {pl_base[t]:.3f}" for t in LEVELS))
    pl_syn_o = score_per_level(ours_s, syn_ans)
    pl_syn_b = score_per_level(base_s, syn_ans)
    print("synthetic per-level loss: ours " +
          ", ".join(f"tau={t}: {pl_syn_o[t]:.3f}" for t in LEVELS) +
          " | persistence " +
          ", ".join(f"tau={t}: {pl_syn_b[t]:.3f}" for t in LEVELS))
    # Scenario G: high-rate trophy (79.94%, E=510 of 638) with frozen
    # earners and a x6.06 denominator jump. A model that drops the count
    # branch at high rates forecasts ~80%; the truth is 510/3868 = 13.19%.
    case_g = json.loads(json.dumps(cases["p00"]))
    case_g["id"] = "highrate"
    for rec in case_g["history"]:
        rec["rates"][2] = 79.94
    ans_g = dilution_answers({"highrate": case_g}, growth=False)["highrate"]
    ours_g, _ = run_entry(os.path.join(HERE, "main.py"),
                          {"highrate": case_g})
    base_g, _ = run_entry(STARTER, {"highrate": case_g})
    q_ours = [r for r in ours_g["highrate"]["trophies"] if r["id"] == 3][0]["q"]
    q_base = [r for r in base_g["highrate"]["trophies"] if r["id"] == 3][0]["q"]
    print(f"G high-rate dilution (id 3, truth 13.19%): ours q={q_ours} "
          f"loss {pinball(13.185, q_ours):.4f} vs persistence q={q_base} "
          f"loss {pinball(13.185, q_base):.4f}")
    # Worst-case runtime: 48h of 10-minute reads (~289 reads, dense
    # changes), 6510-min horizon, pinned to one CPU.
    dense_case, dense_ans = simulate(seed=7, horizon_min=6510,
                                     cutoff_hours=48, read_minutes=10)
    t0 = datetime.now()
    cmd = ["taskset", "-c", "0", sys.executable, "-P", "-s",
           os.path.join(HERE, "main.py")]
    with tempfile.TemporaryDirectory() as td:
        inp, out = os.path.join(td, "in"), os.path.join(td, "out")
        os.makedirs(inp)
        os.makedirs(out)
        with open(os.path.join(inp, "dense.in"), "w") as f:
            json.dump(dense_case, f)
        env = dict(os.environ, MAKEFUN_INPUT_DIR=inp, MAKEFUN_OUTPUT_DIR=out,
                   PYTHONHASHSEED="0", TZ="UTC")
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                               timeout=60)
            pinned = True
        except FileNotFoundError:
            r = subprocess.run(cmd[3:], env=env, capture_output=True,
                               text=True, timeout=60)
            pinned = False
        wall = (datetime.now() - t0).total_seconds()
        if r.returncode != 0:
            print(f"dense benchmark: CRASH rc={r.returncode} {r.stderr[-300:]}")
        else:
            with open(os.path.join(out, "dense.out")) as f:
                dense_out = json.load(f)
            loss = score_all({"dense": dense_out}, {"dense": dense_ans})[0]
            base_d, _ = run_entry(STARTER, {"dense": dense_case})
            loss_b = score_all(base_d, {"dense": dense_ans})[0]
            print(f"dense benchmark (289 reads, 6510-min horizon, "
                  f"1 CPU{' pinned' if pinned else ' (taskset missing)'}): "
                  f"{wall:.2f}s, loss {loss:.4f} vs persistence {loss_b:.4f}")
            # Recent-guard engagement on the dense case (pass-3 weakness
            # 3): rebuild the fits in-process and count, per branch, how
            # often the recent data constrained the slope below the
            # full-window slope (recent guard winning, or the post-pooling
            # saturation cap firing).
            _st, _lref, _gm = model.build_states(
                dense_case, model.parse_ts(dense_case["at"]))
            _sg = model.fitting_segment(_st)
            _ld = model.build_ledgers(dense_case,
                                      model.parse_ts(dense_case["at"]))
            _fams = [model.classify(t) for t in dense_case["trophies"]]
            _fts = model.fit_trophies(
                dense_case, model.thin_states(_sg, _st[-1]["t"]),
                _lref, _fams, _ld)
            for _branch, _pts, _prior, _drec, _slope, _hub in (
                    ("count", "_c_points", "b_count_prior", "_d_rec_c",
                     "b_count", 0.35),
                    ("direct", "_d_points", "b_direct_prior", "_d_rec_d",
                     "b_direct", 0.60)):
                _elig = _won = 0
                for _f in _fts:
                    _points = getattr(_f, _pts)
                    if not _f.observed or not _points:
                        continue
                    _sub = [p for p in _points
                            if p[0] >= getattr(_f, _drec)]
                    _non_anchor = [p for p in _sub if p[0] < -1e-9]
                    if len({round(p[0], 9) for p in _non_anchor}) < 2:
                        continue
                    _elig += 1
                    _b_full, _i = model.anchored_slope(
                        _points, getattr(_f, _prior), _hub)
                    if getattr(_f, _slope) < _b_full - 1e-12:
                        _won += 1
                print(f"dense recent-guard {_branch}: eligible {_elig}, "
                      f"recent-constrained {_won} trophies")


if __name__ == "__main__":
    main()
