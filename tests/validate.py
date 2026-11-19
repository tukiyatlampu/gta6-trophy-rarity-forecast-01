"""Validate make.fun gta6-trophies-case/v1 outputs and score them.

    python3 tests/validate.py <input_dir> <output_dir> [answers.json]

<input_dir> holds <id>.in cases, <output_dir> holds <id>.out forecasts.
answers.json (optional) maps each case id to the answer rates in trophy
order, e.g. {"p00": [0.1, 100.0, ...]}. With answers, each case is
scored with logit-scale pinball loss against both the forecast and the
kit starter (persistence) baseline, per level and pooled.

Exits 0 only when every .out file satisfies the output contract and is
at most 65,536 bytes. Scoring never changes the exit code; it is
reported, not asserted.
"""
import json
import math
import os
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
import main  # noqa: E402

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
MAX_BYTES = 65536


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


def validate_one(case, out_path):
    """Return (ok, problems, rows)."""
    problems = []
    if not os.path.exists(out_path):
        return False, ["missing output file"], None
    with open(out_path, "rb") as fh:
        raw = fh.read()
    if len(raw) > MAX_BYTES:
        problems.append("output %d bytes > %d" % (len(raw), MAX_BYTES))
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, ["unparsable output: %r" % (exc,)], None
    if set(parsed.keys()) != {"trophies"}:
        problems.append("top-level keys %r" % (sorted(parsed.keys()),))
        return False, problems, None
    rows = parsed["trophies"]
    ids = [t["id"] for t in case["trophies"]]
    if not main.validate_output(rows, ids, list(case.get("levels") or LEVELS)):
        problems.append("contract validation failed")
    return (not problems), problems, rows


def main_cli(argv):
    if len(argv) < 3:
        print(__doc__.strip().splitlines()[0])
        print("usage: validate.py <input_dir> <output_dir> [answers.json]")
        return 2
    in_dir, out_dir = argv[1], argv[2]
    answers = {}
    if len(argv) > 3:
        with open(argv[3]) as fh:
            answers = json.load(fh)
    ok_all = True
    names = sorted(n[:-3] for n in os.listdir(in_dir) if n.endswith(".in"))
    if not names:
        print("no .in cases in %s" % in_dir)
        return 2
    for cid in names:
        with open(os.path.join(in_dir, cid + ".in")) as fh:
            case = json.load(fh)
        ok, problems, rows = validate_one(
            case, os.path.join(out_dir, cid + ".out"))
        status = "VALID" if ok else "INVALID"
        print("%s %s: %s" % (cid, status, "; ".join(problems) or "contract ok"))
        if not ok:
            ok_all = False
            continue
        if cid in answers:
            ans = answers[cid]
            ours, ours_lv = pinball(ans, rows)
            base, base_lv = pinball(ans, starter_rows(case))
            print("  pinball ours %.4f starter %.4f" % (ours, base))
            print("  per-level ours   %s"
                  % " ".join("%.3f" % v for v in ours_lv))
            print("  per-level starter %s"
                  % " ".join("%.3f" % v for v in base_lv))
    print("validate: %s" % ("OK" if ok_all else "FAILED"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main_cli(sys.argv))
