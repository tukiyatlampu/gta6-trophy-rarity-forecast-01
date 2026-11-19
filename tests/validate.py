"""Output validator for gta6-trophy-rarity-forecast-01 (checklist C2/C4).

Checks the exact output contract: {"trophies": [{"id": N, "q": [5 numbers]}]},
one row per case trophy, each id once, no other keys, every q finite,
0..100, non-decreasing, file at most 65,536 bytes.

Usage: python3 tests/validate.py <out.json> [...]
"""
import json
import math
import sys
from pathlib import Path

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def validate(path, n_trophies=55):
    errors = []
    data = Path(path).read_bytes()
    if len(data) > 65536:
        errors.append(f"size {len(data)} > 65536")
    try:
        obj = json.loads(data)
    except Exception as e:
        return [f"not JSON: {e}"]
    if set(obj.keys()) != {"trophies"}:
        errors.append(f"top-level keys {sorted(obj.keys())} != ['trophies']")
        return errors
    rows = obj["trophies"]
    if not isinstance(rows, list) or len(rows) != n_trophies:
        errors.append(f"{len(rows) if isinstance(rows, list) else 'not-list'} "
                      f"rows != {n_trophies}")
        return errors
    seen = []
    for row in rows:
        if set(row.keys()) != {"id", "q"}:
            errors.append(f"row keys {sorted(row.keys())}")
            continue
        seen.append(row["id"])
        q = row["q"]
        if not isinstance(q, list) or len(q) != 5:
            errors.append(f"id {row['id']}: q len != 5")
            continue
        prev = None
        for v in q:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                errors.append(f"id {row['id']}: non-number {v!r}")
                continue
            if not math.isfinite(v):
                errors.append(f"id {row['id']}: non-finite {v!r}")
            if v < 0 or v > 100:
                errors.append(f"id {row['id']}: out of range {v!r}")
            if prev is not None and v < prev:
                errors.append(f"id {row['id']}: decreasing {prev} -> {v}")
            prev = v
    if seen != list(range(1, n_trophies + 1)):
        errors.append(f"ids {seen[:8]}... != 1..{n_trophies} in order")
    return errors


if __name__ == "__main__":
    ok = True
    for p in sys.argv[1:]:
        errs = validate(p)
        print(f"{p}: {'VALID' if not errs else 'INVALID: ' + '; '.join(errs)}")
        ok = ok and not errs
    raise SystemExit(0 if ok else 1)
