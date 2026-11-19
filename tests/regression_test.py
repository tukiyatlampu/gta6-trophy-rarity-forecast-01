"""Regression tests for the terminal fallback paths in main.py.

    python3 tests/regression_test.py

Exits 0 only when every check passes. Uses only the standard library.

Test (c) -- oversized recovery (judge-7 regression): a case of 55
trophies whose ids are 1162-digit integers. The real primary forecast
is contract-valid but serializes past the 65,536-byte cap, and so do
the emergency rows, so process_case must fall through to the all-zero
fixed vector, which fits (65,245 bytes). The run must log the
TERMINAL retry line, the written payload must validate and carry all
55 ids with fixed [0, 0, 0, 0, 0] quantiles, and no temporary file
may be left behind.

Test (d) -- empty-case validation gate (judge-7 regression): while
validate_output is monkeypatched to reject everything, process_case
on the empty case must return None, so a caller writing only on a
non-None result leaves an existing output file untouched. Once
validation is restored, the same case must yield exactly
{"trophies":[]} plus a newline.
"""
import contextlib
import io
import json
import os
import sys
import tempfile

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
import main  # noqa: E402

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
MAX_BYTES = 65536
# Measured fixed payload for the 55 x 1162-digit-id case below:
# 55 rows x (6 + 1162 + 17) chars + 54 commas + 13 prefix + 2 suffix
# + 1 newline = 65,245 bytes.
FIXED_BYTES = 65245
# Family-flavoured descriptions, cycled so the 55 trophies span the
# classifier's story / side / exploration / grind / completion pools.
DESCRIPTIONS = [
    "Complete Chapter 1.",
    "Complete a side mission.",
    "Collect all hidden packages.",
    "Defeat 100 enemies.",
    "Earn every trophy.",
]

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("PASS %s" % name)
    else:
        print("FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def oversized_case():
    """55 trophies with 1162-digit integer ids and two frozen reads."""
    trophies = []
    for k in range(55):
        trophies.append({
            "id": 10 ** 1161 + k,
            "name": "Trophy %d" % (k + 1),
            "description": DESCRIPTIONS[k % len(DESCRIPTIONS)],
            "tier": "bronze", "points": 15, "secret": False,
        })
    rates = [round(0.10 + 0.05 * (k % 17), 2) for k in range(55)]
    reads = []
    for slot, at in (("2026-11-18T23:50:00Z", "2026-11-18T23:50:49Z"),
                     ("2026-11-19T02:30:00Z", "2026-11-19T02:30:00Z")):
        reads.append({
            "slot": slot, "at": at, "players_tracked": 638,
            "trophies_earned": 100, "rates": list(rates),
        })
    return {
        "schema": "makefun/gta6-trophies-case/v1", "id": "regression-c",
        "at": "2026-11-19T03:00:00Z", "horizon_minutes": 90,
        "target": "2026-11-19T04:30:00Z", "levels": LEVELS,
        "game": {"title": "Grand Theft Auto VI"},
        "trophies": trophies, "history": reads, "notes": "",
    }


def empty_case():
    return {
        "schema": "makefun/gta6-trophies-case/v1", "id": "regression-d",
        "at": "2026-11-19T03:00:00Z", "horizon_minutes": 90,
        "target": "2026-11-19T04:30:00Z", "levels": LEVELS,
        "game": {"title": "Grand Theft Auto VI"},
        "trophies": [], "history": [], "notes": "",
    }


def test_oversized_recovery():
    case = oversized_case()
    ids = [t["id"] for t in case["trophies"]]
    check("oversized-55-ids", len(ids) == 55
          and all(len(str(tid)) == 1162 for tid in ids))
    primary = main.forecast_case(case)
    check("oversized-primary-valid",
          main.validate_output(primary, ids, LEVELS))
    check("oversized-primary-too-big",
          main.serialize_output(primary) is None)
    emergency = main.emergency_rows(case, ids, LEVELS)
    check("oversized-emergency-valid",
          main.validate_output(emergency, ids, LEVELS))
    check("oversized-emergency-too-big",
          main.serialize_output(emergency) is None)
    fixed = main.fixed_rows(ids, LEVELS)
    check("oversized-fixed-valid",
          main.validate_output(fixed, ids, LEVELS))
    fixed_text = main.serialize_output(fixed)
    check("oversized-fixed-fits",
          fixed_text is not None
          and len(fixed_text.encode("utf-8")) == FIXED_BYTES,
          "len=%s" % (None if fixed_text is None
                      else len(fixed_text.encode("utf-8"))))
    with tempfile.TemporaryDirectory() as tmp:
        old_cwd = os.getcwd()
        err = io.StringIO()
        try:
            os.chdir(tmp)
            with contextlib.redirect_stderr(err):
                text = main.process_case(case)
        finally:
            os.chdir(old_cwd)
        check("oversized-no-temp-files", os.listdir(tmp) == [],
              str(os.listdir(tmp)))
    check("oversized-retry-returns-fixed", text == fixed_text,
          "got %r" % (None if text is None else text[:80]))
    log = err.getvalue()
    check("oversized-terminal-logged",
          "TERMINAL:" in log and "trying fixed rows" in log,
          repr(log))
    if text is not None:
        parsed = json.loads(text)
        check("oversized-all-ids",
              [r["id"] for r in parsed["trophies"]] == ids)
        check("oversized-all-fixed-zero",
              all(r["q"] == [0, 0, 0, 0, 0]
                  for r in parsed["trophies"]))
        check("oversized-output-under-cap",
              len(text.encode("utf-8")) <= MAX_BYTES)


def test_empty_case_validation_gate():
    case = empty_case()
    real_validate = main.validate_output
    try:
        main.validate_output = lambda *args: False
        with contextlib.redirect_stderr(io.StringIO()):
            check("empty-rejected-returns-none",
                  main.process_case(case) is None)
        # End to end: with validation failing, main() must not touch
        # an existing output file (the old target is preserved).
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)
            os.makedirs(out_dir)
            with open(os.path.join(in_dir, "empty.in"), "w") as fh:
                json.dump(case, fh)
            out_path = os.path.join(out_dir, "empty.out")
            with open(out_path, "w") as fh:
                fh.write("SENTINEL\n")
            old_env = dict(os.environ)
            os.environ["MAKEFUN_INPUT_DIR"] = in_dir
            os.environ["MAKEFUN_OUTPUT_DIR"] = out_dir
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    rc = main.main()
            finally:
                os.environ.clear()
                os.environ.update(old_env)
            check("empty-rejected-main-rc0", rc == 0, "rc=%r" % rc)
            with open(out_path) as fh:
                kept = fh.read()
            check("empty-rejected-target-preserved",
                  kept == "SENTINEL\n", repr(kept))
    finally:
        main.validate_output = real_validate
    check("empty-validate-restored",
          main.validate_output([], [], LEVELS) is True)
    with contextlib.redirect_stderr(io.StringIO()):
        text = main.process_case(case)
    check("empty-case-exact-payload",
          text == '{"trophies":[]}\n', repr(text))
    # Same, through main(): the file is now written with that payload.
    with tempfile.TemporaryDirectory() as tmp:
        in_dir = os.path.join(tmp, "in")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(in_dir)
        with open(os.path.join(in_dir, "empty.in"), "w") as fh:
            json.dump(case, fh)
        old_env = dict(os.environ)
        os.environ["MAKEFUN_INPUT_DIR"] = in_dir
        os.environ["MAKEFUN_OUTPUT_DIR"] = out_dir
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                rc = main.main()
        finally:
            os.environ.clear()
            os.environ.update(old_env)
        check("empty-main-rc0", rc == 0, "rc=%r" % rc)
        with open(os.path.join(out_dir, "empty.out")) as fh:
            written = fh.read()
        check("empty-main-writes-exact-payload",
              written == '{"trophies":[]}\n', repr(written))


def run_all():
    for fn in (test_oversized_recovery, test_empty_case_validation_gate):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - report, don't abort
            check(fn.__name__ + "-no-raise", False, repr(exc))
    print("regression_test: %d checks failed" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(run_all())
