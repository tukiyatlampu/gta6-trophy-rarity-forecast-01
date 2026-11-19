"""Edge-case and contract tests for main.py (run from the package root).

    python3 tests/edge_test.py

Exits 0 only when every check passes. Uses only the standard library.
The kit practice cases are read from the unpacked kit under work/ when
present; every other fixture is built inline so the suite is
self-contained. Temporary directories live under /tmp.
"""
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import time

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
import main  # noqa: E402

KIT_IN = ("/home/hatch/workspace/makefun-bounty/work/"
          "gta6-trophy-rarity-forecast-01-kit/"
          "gta6-trophy-rarity-forecast-01-kit/input")
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("PASS %s" % name)
    else:
        print("FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def base_case(n=3, horizon=90):
    trophies = [
        {"id": i + 1, "name": "Trophy %d" % (i + 1),
         "description": "Complete Chapter %d." % (i + 1) if i < 2
         else "Collect all hidden packages.",
         "tier": "bronze", "points": 15, "secret": False}
        for i in range(n)
    ]
    return {
        "schema": "makefun/gta6-trophies-case/v1", "id": "edge",
        "at": "2026-11-19T03:00:00Z", "horizon_minutes": horizon,
        "target": "2026-11-19T04:30:00Z", "levels": LEVELS,
        "game": {"title": "Grand Theft Auto VI"},
        "trophies": trophies,
        "history": [
            {"slot": "2026-11-18T23:50:00Z",
             "at": "2026-11-18T23:50:49Z", "players_tracked": 638,
             "trophies_earned": 100,
             "rates": [0.63, 1.25, 0.78][:n]},
            {"slot": "2026-11-19T02:30:00Z",
             "at": "2026-11-19T02:30:00Z", "players_tracked": 638,
             "trophies_earned": 100,
             "rates": [0.63, 1.25, 0.78][:n]},
        ],
        "notes": "",
    }


def rows_valid(case, rows):
    ids = [t["id"] for t in case["trophies"]]
    return main.validate_output(rows, ids, case["levels"])


def test_empty_history():
    case = base_case()
    case["history"] = []
    rows = main.forecast_case(case)
    check("empty-history-valid", rows_valid(case, rows))
    check("empty-history-finite",
          all(0.0 <= q <= 100.0 for r in rows for q in r["q"]))


def test_no_trophies():
    case = base_case()
    case["trophies"] = []
    text = main.process_case(case)
    check("no-trophies-empty-payload", text == '{"trophies":[]}\n',
          repr(text))


def test_single_read():
    case = base_case()
    case["history"] = case["history"][:1]
    rows = main.forecast_case(case)
    check("single-read-valid", rows_valid(case, rows))


def test_null_rates():
    case = base_case()
    case["history"][0]["rates"] = [None, 1.25, None]
    case["history"][1]["rates"] = [0.63, None, 0.78]
    rows = main.forecast_case(case)
    check("null-rates-valid", rows_valid(case, rows))
    case2 = base_case()
    for read in case2["history"]:
        read["rates"] = [None, None, None]
    rows2 = main.forecast_case(case2)
    check("all-null-rates-valid", rows_valid(case2, rows2))


def test_zero_and_hundred_rates():
    case = base_case()
    case["history"][0]["rates"] = [0, 100, 0.78]
    case["history"][1]["rates"] = [0, 100, 0.78]
    rows = main.forecast_case(case)
    check("zero-hundred-valid", rows_valid(case, rows))
    check("hundred-stays-high", rows[1]["q"][2] > 90.0,
          str(rows[1]["q"]))
    check("zero-stays-low", rows[0]["q"][2] < 10.0, str(rows[0]["q"]))


def test_player_jump():
    case = base_case()
    jumped = copy.deepcopy(case["history"][1])
    jumped["players_tracked"] = 3868
    jumped["rates"] = [0.10, 0.21, 0.13]
    jumped["at"] = "2026-11-19T02:45:00Z"
    case["history"].append(jumped)
    rows = main.forecast_case(case)
    check("player-jump-valid", rows_valid(case, rows))
    check("player-jump-dilutes",
          rows[0]["q"][2] < case["history"][0]["rates"][0],
          str(rows[0]["q"]))


def test_duplicate_reads():
    # Repeating the same reads must not change the forecast: the last
    # read time (the anchor) stays the same and duplicate polls add
    # only zero-length gaps. (A single-read prefix is a different
    # anchor, so it is not compared here.)
    case = base_case()
    dup = copy.deepcopy(case)
    dup["history"] = dup["history"] + copy.deepcopy(dup["history"])
    a = main.forecast_case(case)
    b = main.forecast_case(dup)
    check("duplicate-reads-same",
          json.dumps(a) == json.dumps(b), "dup changed the forecast")


def test_horizons():
    short = base_case(horizon=90)
    longc = base_case(horizon=6510)
    longc["target"] = "2026-11-26T00:00:00Z"
    rs = main.forecast_case(short)
    rl = main.forecast_case(longc)
    check("horizon-90-valid", rows_valid(short, rs))
    check("horizon-6510-valid", rows_valid(longc, rl))
    spread_s = rs[0]["q"][4] - rs[0]["q"][0]
    spread_l = rl[0]["q"][4] - rl[0]["q"][0]
    check("long-horizon-wider-spread", spread_l >= spread_s,
          "%.4f vs %.4f" % (spread_l, spread_s))


def test_zero_horizon():
    case = base_case(horizon=0)
    case["target"] = case["at"]
    rows = main.forecast_case(case)
    check("zero-horizon-valid", rows_valid(case, rows))
    check("zero-horizon-is-current-rate",
          all(abs(r["q"][2] - rate) < 1e-9 for r, rate in
              zip(rows, case["history"][-1]["rates"])),
          str([r["q"] for r in rows]))


def test_determinism():
    case = base_case(horizon=6510)
    case["target"] = "2026-11-26T00:00:00Z"
    a = main.forecast_case(case)
    b = main.forecast_case(case)
    check("forecast-deterministic", json.dumps(a) == json.dumps(b))
    with tempfile.TemporaryDirectory() as tmp:
        in_dir = os.path.join(tmp, "in")
        out1 = os.path.join(tmp, "out1")
        out2 = os.path.join(tmp, "out2")
        os.makedirs(in_dir)
        with open(os.path.join(in_dir, "edge.in"), "w") as fh:
            json.dump(case, fh)
        env = dict(os.environ, MAKEFUN_INPUT_DIR=in_dir,
                   PYTHONHASHSEED="0", TZ="UTC")
        for out_dir in (out1, out2):
            env["MAKEFUN_OUTPUT_DIR"] = out_dir
            proc = subprocess.run(
                [sys.executable, "-P", "-s",
                 os.path.join(PKG, "main.py")],
                env=env, capture_output=True, text=True, timeout=120)
            check("main-rc0-%s" % os.path.basename(out_dir),
                  proc.returncode == 0, proc.stderr[-300:])
        with open(os.path.join(out1, "edge.out"), "rb") as fh:
            first = fh.read()
        with open(os.path.join(out2, "edge.out"), "rb") as fh:
            second = fh.read()
        check("two-runs-byte-identical", first == second)


def test_contract_validation():
    ids = [1, 2]
    good = [{"id": 1, "q": [0.1, 0.2, 0.3, 0.4, 0.5]},
            {"id": 2, "q": [1, 1, 1, 1, 1]}]
    check("validate-accepts-good",
          main.validate_output(good, ids, LEVELS))
    bad_cases = {
        "wrong-order": [{"id": 2, "q": [1, 1, 1, 1, 1]},
                        {"id": 1, "q": [1, 1, 1, 1, 1]}],
        "decreasing": [{"id": 1, "q": [0.5, 0.2, 0.3, 0.4, 0.5]},
                       {"id": 2, "q": [1, 1, 1, 1, 1]}],
        "out-of-range": [{"id": 1, "q": [0.1, 0.2, 0.3, 0.4, 100.5]},
                         {"id": 2, "q": [1, 1, 1, 1, 1]}],
        "extra-key": [{"id": 1, "q": [1, 1, 1, 1, 1], "x": 0},
                      {"id": 2, "q": [1, 1, 1, 1, 1]}],
        "wrong-count": [{"id": 1, "q": [1, 1, 1, 1, 1]}],
        "nan": [{"id": 1, "q": [1, 1, float("nan"), 1, 1]},
                {"id": 2, "q": [1, 1, 1, 1, 1]}],
    }
    for name, rows in bad_cases.items():
        check("validate-rejects-" + name,
              not main.validate_output(rows, ids, LEVELS))


def test_rounding_and_helpers():
    check("round-half-up", main.round_out(1.23455) == 1.2346,
          str(main.round_out(1.23455)))
    check("round-int-stays-int",
          main.round_out(100.0) == 100
          and isinstance(main.round_out(100.0), int))
    check("score-logit-50", abs(main.score_logit(50.0)) < 1e-12)
    check("logit-roundtrip",
          abs(main.inv_score_logit(main.score_logit(3.7)) - 3.7) < 1e-9)
    check("saturate-identity", main.saturate(0.5) == 0.5)
    check("saturate-bounded", 0.8 < main.saturate(5.0) < 1.0)
    emin, emax, ehat, empty, _w = main.earner_interval(0.63, 638)
    check("earner-interval-platinum",
          (emin, emax, ehat) == (4, 4, 4.0), str((emin, emax, ehat)))
    check("earner-interval-not-empty", empty is False)


def test_earner_sum_943():
    if not os.path.exists(os.path.join(KIT_IN, "p00.in")):
        print("SKIP earner-sum-943 (kit not present)")
        return
    with open(os.path.join(KIT_IN, "p00.in")) as fh:
        case = json.load(fh)
    first = case["history"][0]
    total = sum(main.earner_interval(r, first["players_tracked"])[2]
                for r in first["rates"])
    check("earner-sum-943", abs(total - first["trophies_earned"]) < 1e-9,
          "sum=%.1f earned=%s" % (total, first["trophies_earned"]))


def test_oversized_retry():
    # Judge-7 regression (c): a valid primary forecast and valid
    # emergency rows that both serialize past 65,536 bytes must fall
    # through to the all-zero fixed vector, which fits.
    n = 2000
    trophies = [{"id": i + 1, "name": "Trophy %d" % (i + 1),
                 "description": "Complete Chapter 1.",
                 "tier": "bronze", "points": 15, "secret": False}
                for i in range(n)]
    case = base_case(n=n)
    case["trophies"] = trophies
    rates = [1.1] * n
    for read in case["history"]:
        read["rates"] = list(rates)
    ids = [t["id"] for t in trophies]
    primary = [{"id": tid,
                "q": [12.3456, 23.4567, 34.5678, 45.6789, 56.789]}
               for tid in ids]
    check("oversized-primary-valid",
          main.validate_output(primary, ids, LEVELS))
    check("oversized-primary-too-big",
          main.serialize_output(primary) is None)
    emergency = main.emergency_rows(case, ids, LEVELS)
    check("oversized-emergency-too-big",
          main.serialize_output(emergency) is None)
    fixed = main.fixed_rows(ids, LEVELS)
    fixed_text = main.serialize_output(fixed)
    check("fixed-vector-fits",
          fixed_text is not None and len(fixed_text.encode()) <= 65536,
          "len=%s" % (None if fixed_text is None
                      else len(fixed_text.encode())))
    orig_forecast = main.forecast_case
    try:
        main.forecast_case = lambda _case: primary
        text = main.process_case(case)
    finally:
        main.forecast_case = orig_forecast
    check("oversized-retry-returns-fixed",
          text == fixed_text, "got %r" % (None if text is None
                                          else text[:80]))
    if text is not None:
        parsed = json.loads(text)
        check("oversized-retry-all-ids",
              [r["id"] for r in parsed["trophies"]] == ids)
        check("oversized-retry-all-zero",
              all(r["q"] == [0, 0, 0, 0, 0]
                  for r in parsed["trophies"]))


def test_all_serialize_fails():
    # Judge-7 regression (d): when no candidate serializes, nothing is
    # written; the empty case still yields {"trophies":[]}.
    orig_serialize = main.serialize_output
    try:
        main.serialize_output = lambda _rows: None
        check("all-serialize-fail-returns-none",
              main.process_case(base_case()) is None)
    finally:
        main.serialize_output = orig_serialize
    check("empty-case-literal",
          main.process_case(base_case(n=0)) == '{"trophies":[]}\n')


def test_main_io_and_malformed():
    if not os.path.exists(os.path.join(KIT_IN, "p00.in")):
        print("SKIP main-io (kit not present)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        in_dir = os.path.join(tmp, "in")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(in_dir)
        for name in ("p00.in", "p01.in"):
            with open(os.path.join(KIT_IN, name)) as fh:
                json.dump(json.load(fh),
                          open(os.path.join(in_dir, name), "w"))
        with open(os.path.join(in_dir, "bad.in"), "w") as fh:
            fh.write("{not json")
        with open(os.path.join(in_dir, "note.txt"), "w") as fh:
            fh.write("ignored")
        env = dict(os.environ, MAKEFUN_INPUT_DIR=in_dir,
                   MAKEFUN_OUTPUT_DIR=out_dir, PYTHONHASHSEED="0",
                   TZ="UTC")
        start = time.time()
        proc = subprocess.run(
            [sys.executable, "-P", "-s", os.path.join(PKG, "main.py")],
            env=env, capture_output=True, text=True, timeout=120)
        elapsed = time.time() - start
        check("main-io-rc0", proc.returncode == 0, proc.stderr[-300:])
        check("main-io-no-stderr", proc.stderr == "",
              proc.stderr[-200:])
        for name in ("p00.out", "p01.out"):
            path = os.path.join(out_dir, name)
            check("main-io-writes-" + name, os.path.exists(path))
            with open(path, "rb") as fh:
                raw = fh.read()
            check("main-io-size-" + name, len(raw) <= 65536,
                  "len=%d" % len(raw))
            parsed = json.loads(raw)
            with open(os.path.join(in_dir, name[:-4] + ".in")) as fh:
                case = json.load(fh)
            check("main-io-valid-" + name,
                  rows_valid(case, parsed["trophies"]))
        check("main-io-skips-bad", not os.path.exists(
            os.path.join(out_dir, "bad.out")))
        check("main-io-fast", elapsed < 8.0, "%.2fs" % elapsed)
        print("INFO practice runtime %.2fs for 2 cases" % elapsed)


def test_missing_input_dir():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = os.path.join(tmp, "out")
        env = dict(os.environ,
                   MAKEFUN_INPUT_DIR=os.path.join(tmp, "nope"),
                   MAKEFUN_OUTPUT_DIR=out_dir, PYTHONHASHSEED="0",
                   TZ="UTC")
        proc = subprocess.run(
            [sys.executable, "-P", "-s", os.path.join(PKG, "main.py")],
            env=env, capture_output=True, text=True, timeout=60)
        check("missing-input-dir-rc0", proc.returncode == 0,
              proc.stderr[-200:])
        check("missing-input-dir-no-output",
              os.listdir(out_dir) == [])


def run_all():
    tests = [test_empty_history, test_no_trophies, test_single_read,
             test_null_rates, test_zero_and_hundred_rates,
             test_player_jump, test_duplicate_reads, test_horizons,
             test_zero_horizon, test_determinism,
             test_contract_validation, test_rounding_and_helpers,
             test_earner_sum_943, test_oversized_retry,
             test_all_serialize_fails, test_main_io_and_malformed,
             test_missing_input_dir]
    for fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - report, don't abort
            check(fn.__name__ + "-no-raise", False, repr(exc))
    print("edge_test: %d checks failed" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(run_all())
