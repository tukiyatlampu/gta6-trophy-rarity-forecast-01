#!/usr/bin/env python3
"""Denominator-aware GTA VI trophy rarity forecast.

Reads makefun/gta6-trophies-case/v1 cases from the input directory and
writes one forecast row per trophy to the output directory.

Model (summary): earners and players are fitted separately. Earners are
reconstructed from two-decimal displayed rates intersected with integer
counts; the denominator follows a decaying-influx curve in three decay
variants. Each trophy gets two anchored curves (log-count and direct
logit) whose slopes are shrunk toward family-peer medians. A refresh
timing mixture (no-change atom plus eight last-refresh nodes, three
denominator shocks, two normal scales) is inverted by bisection to give
the requested quantiles.

Standard library only. No network, no clock reads, no randomness: two
runs on the same input produce identical bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

LAUNCH = datetime(2026, 11, 18, 23, 50, tzinfo=timezone.utc)
DAY = 86400.0
HOUR = 3600.0
DEFAULT_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
BETA_VALUES = (0.5, 1.0, 1.5)
BETA_WEIGHTS = (0.25, 0.50, 0.25)
JUMP_VALUES = (2, 4, 8)
JUMP_WEIGHTS = (0.25, 0.50, 0.25)
SHOCK_POINTS = (-1.7320508075688772, 0.0, 1.7320508075688772)
SHOCK_WEIGHTS = (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)
NODES = 8
BISECT_STEPS = 28
MAX_BYTES = 65536
SQRT2 = 1.4142135623730951
LOGIT_LO = -9.210340371976184  # scoring logit of 0.01
LOGIT_HI = 9.210340371976184  # scoring logit of 99.99
CENT = Decimal("0.0001")


def clip(value, low, high):
    """Clamp value into [low, high]."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def safe_exp(value):
    """exp with overflow guard."""
    return math.exp(clip(value, -700.0, 700.0))


def parse_time(value):
    """Parse an ISO 8601 timestamp to an aware UTC datetime, or None."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def days_since_launch(moment):
    """Age in days since the 2026-11-18T23:50Z launch."""
    return (moment - LAUNCH).total_seconds() / DAY


def x_of_t(t_days):
    """Log-time transform x(t) = ln(1 + t), t in days since launch."""
    return math.log1p(max(0.0, t_days))


def score_logit(rate):
    """Scoring logit l(r) with r clamped to [0.01, 99.99]."""
    r = clip(float(rate), 0.01, 99.99)
    return math.log(r / (100.0 - r))


def inv_score_logit(z):
    """Inverse scoring logit, returning a percentage in (0, 100)."""
    z = clip(float(z), -30.0, 30.0)
    return 100.0 / (1.0 + math.exp(-z))


def median(values):
    """Median of a non-empty sequence, or None when empty."""
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def mad_scale(values):
    """1.4826 * MAD of values, or None when empty."""
    med = median(values)
    if med is None:
        return None
    devs = [abs(v - med) for v in values]
    mad = median(devs)
    if mad is None:
        return None
    return 1.4826 * mad


def norm_cdf(z):
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + math.erf(z / SQRT2))


def norm_ppf(p):
    """Standard normal quantile by bisection (fallback path only)."""
    p = clip(float(p), 1e-9, 1.0 - 1e-9)
    low, high = -10.0, 10.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        if norm_cdf(mid) >= p:
            high = mid
        else:
            low = mid
    return high


def round_out(value):
    """Round to 4 decimals, half up; integral values stay ints."""
    dec = Decimal(repr(float(value))).quantize(CENT, rounding=ROUND_HALF_UP)
    if dec == dec.to_integral_value():
        return int(dec)
    return float(dec)


def saturate(v):
    """Bounded growth g(v): identity through 0.8, then saturating to 1."""
    if v <= 0.8:
        return v
    return 1.0 - 0.2 * math.exp(-(v - 0.8) / 0.2)


# ---------------------------------------------------------------------------
# Page states
# ---------------------------------------------------------------------------

def _clean_players(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0 or not math.isfinite(value):
        return None
    return int(value)


def _clean_int(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0 or not math.isfinite(value):
        return None
    return int(value)


def rate_at(state, index):
    """Rate of trophy index in a state, or None when absent."""
    rates = state["rates"]
    if index < 0 or index >= len(rates):
        return None
    value = rates[index]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def build_states(history, cutoff):
    """Collapse reads into page states.

    Reads are sorted by timestamp (input order breaks ties) and reads
    after the cutoff are discarded. A change in a valid player count or
    in a comparable non-null rate starts a new state; a null or a newly
    available value alone supplies no change evidence. Repeated reads
    merge into the current state but keep their timestamps for exposure.
    """
    reads = []
    for order, raw in enumerate(history or []):
        if not isinstance(raw, dict):
            continue
        at = parse_time(raw.get("at"))
        if at is None or at > cutoff:
            continue
        reads.append((at, order, raw))
    reads.sort(key=lambda item: (item[0], item[1]))
    states = []
    for at, _order, raw in reads:
        players = _clean_players(raw.get("players_tracked"))
        rates = raw.get("rates")
        if not isinstance(rates, list):
            rates = []
        earned = _clean_int(raw.get("trophies_earned"))
        if not states:
            states.append({
                "at": at, "first_seen": at, "players": players,
                "rates": list(rates), "earned": earned, "polls": [at],
            })
            continue
        cur = states[-1]
        changed = False
        if players is not None and cur["players"] is not None:
            if players != cur["players"]:
                changed = True
        if not changed:
            limit = min(len(rates), len(cur["rates"]))
            for k in range(limit):
                new_r = rates[k]
                old_r = cur["rates"][k]
                if new_r is None or old_r is None:
                    continue
                if isinstance(new_r, bool) or isinstance(old_r, bool):
                    continue
                if not isinstance(new_r, (int, float)):
                    continue
                if not isinstance(old_r, (int, float)):
                    continue
                if new_r != old_r:
                    changed = True
                    break
        if changed:
            states.append({
                "at": at, "first_seen": at, "players": players,
                "rates": list(rates), "earned": earned, "polls": [at],
            })
        else:
            cur["polls"].append(at)
            if cur["players"] is None and players is not None:
                cur["players"] = players
            if cur["earned"] is None and earned is not None:
                cur["earned"] = earned
            for k, r in enumerate(rates):
                if k < len(cur["rates"]):
                    if cur["rates"][k] is None and r is not None:
                        cur["rates"][k] = r
                else:
                    cur["rates"].append(r)
    return states


def state_weight(state, anchor_time):
    """Fitting weight 2^(-age in days) measured from the series anchor."""
    age = (anchor_time - state["at"]).total_seconds() / DAY
    return 2.0 ** (-max(0.0, age))


def fitting_states(states, anchor_time):
    """Thin states for fitting: last per UTC hour, 48h window, segments.

    A downward player correction greater than 20% starts a new fitting
    segment, and when the first doubling of P happens within launch day
    trend fitting starts at that state. Earlier change times are kept
    by the caller for cadence estimation.
    """
    if not states:
        return []
    by_hour = {}
    for st in states:
        key = (st["at"].year, st["at"].month, st["at"].day, st["at"].hour)
        by_hour[key] = st
    kept = sorted(by_hour.values(), key=lambda s: s["at"])
    floor_time = anchor_time.timestamp() - 48.0 * HOUR
    kept = [st for st in kept if st["at"].timestamp() >= floor_time]
    if not kept:
        kept = [states[-1]]
    start = 0
    first_p = None
    for idx, st in enumerate(kept):
        if st["players"] is None:
            continue
        if first_p is None:
            first_p = st["players"]
            continue
        if first_p > 0 and st["players"] >= 2 * first_p:
            if (st["at"] - LAUNCH).total_seconds() <= DAY:
                start = max(start, idx)
            break
    prev_p = None
    for idx, st in enumerate(kept):
        if st["players"] is None:
            continue
        if prev_p is not None and prev_p > 0:
            if st["players"] < 0.8 * prev_p:
                start = max(start, idx)
        prev_p = st["players"]
    return kept[start:]


# ---------------------------------------------------------------------------
# Earner reconstruction and denominator fit
# ---------------------------------------------------------------------------

def earner_interval(rate, players):
    """Feasible integer earner counts for a two-decimal rate.

    Returns (emin, emax, ehat, empty, width) where width is the raw
    interval width in counts and empty marks a rounding-tie miss.
    """
    low = players * (rate - 0.005) / 100.0
    high = players * (rate + 0.005) / 100.0
    emin = math.ceil(low - 1e-9)
    emax = math.floor(high + 1e-9)
    emin = max(0, emin)
    emax = min(players, emax)
    width = high - low
    if emin > emax:
        return 0, -1, players * rate / 100.0, True, width
    return emin, emax, 0.5 * (emin + emax), False, width


def influx_integral(beta, t_p, s):
    """Integral of ((0.5+u)/(0.5+t_p))^-beta from t_p to s (days)."""
    base = max(0.5 + t_p, 1e-9)
    top = max(0.5 + s, 1e-9)
    if beta == 1.0:
        return base * math.log(top / base)
    return (base ** beta) * (top ** (1.0 - beta) - base ** (1.0 - beta)) / (1.0 - beta)


def _ridge(xs, ys, ws, lam, prior):
    """Weighted ridge solve with prior mean: (sum wxy + lam*prior) /
    (sum wx^2 + lam)."""
    num = lam * prior
    den = lam
    for x, y, w in zip(xs, ys, ws):
        num += w * x * y
        den += w * x * x
    if den <= 0.0:
        return prior
    return num / den


def fit_denominator(fit_states, anchor_time):
    """Fit the decaying-influx denominator for beta in {0.5, 1, 1.5}.

    Returns None when no usable player observation exists.
    """
    obs = []
    for st in fit_states:
        if st["players"] is None:
            continue
        obs.append((st, days_since_launch(st["at"]), st["players"]))
    if not obs:
        return None
    t_p, p_star = obs[-1][1], obs[-1][2]
    weights = [state_weight(st, anchor_time) for st, _t, _p in obs]
    curves = []
    for beta in BETA_VALUES:
        lam = 0.0225  # (0.15 day)^2 penalty weight
        xs = [influx_integral(beta, t_p, t) for _st, t, _p in obs]
        ys = [p / float(p_star) - 1.0 for _st, _t, p in obs]
        v = _ridge(xs, ys, weights, lam, 0.5)
        for _ in range(2):
            res = [y - v * x for x, y in zip(xs, ys)]
            w2 = [w * (1.0 if abs(r) <= 0.15 else 0.15 / abs(r))
                  for w, r in zip(weights, res)]
            v = _ridge(xs, ys, w2, lam, 0.5)
        v = max(0.0, v)
        logres = []
        for (_st, _t, p), x in zip(obs, xs):
            pred = p_star * (1.0 + v * x)
            if pred > 0.0 and p > 0:
                logres.append(math.log(p) - math.log(pred))
        if len(logres) >= 4:
            sp = mad_scale(logres)
            sp = 0.15 if sp is None else max(0.05, sp)
        else:
            sp = 0.15
        curves.append({
            "beta": beta, "v": v, "t_p": t_p,
            "p_star": p_star, "s_p": sp,
        })
    return {"curves": curves, "t_p": t_p, "p_star": p_star}


def denom_curve(curve, s_days):
    """Projected players Pbar_beta(s) at log-time age s (days)."""
    return curve["p_star"] * (
        1.0 + curve["v"] * influx_integral(curve["beta"], curve["t_p"], s_days)
    )


def sigma_p_of(curve, h_p):
    """Denominator log-scale at horizon h_p days past the last player obs."""
    h_p = max(0.0, h_p)
    var = 0.0625 * h_p + (curve["s_p"] ** 2) * h_p / (h_p + 0.25)
    return math.sqrt(max(0.0, var))


# ---------------------------------------------------------------------------
# Trophy families and anchored curve fits
# ---------------------------------------------------------------------------

COUNT_DEFAULTS = {
    "story": 1.5, "side": 1.25, "exploration": 1.25,
    "grind": 1.4, "completion": 1.4, "unknown": 1.0, "opening": 1.5,
}
FALLBACK_MEDIAN = {
    "story": 1.0, "side": 0.3, "exploration": 0.3,
    "grind": 0.05, "completion": 0.05, "unknown": 0.2, "opening": 1.0,
}


def classify_trophy(trophy):
    """Family from name/description keywords, with explicit overrides.

    Ordered rules: opening trophy first, then platinum/completion,
    chapters, side missions, exploration/collectibles, mandatory
    missions, high-count grinds, completion markers, else unknown.
    """
    name = str(trophy.get("name") or "")
    desc = str(trophy.get("description") or "")
    text = (name + " " + desc).lower()
    if "just like paradise" in text:
        return "opening"
    if ("platinum" in text or "earn all trophies" in text
            or "earn every trophy" in text or "unlock all trophies" in text):
        return "completion"
    if "chapter" in text:
        return "story"
    if ("side mission" in text or "side quest" in text
            or "strangers and freaks" in text or "stranger" in text):
        return "side"
    if ("collectible" in text or "collect all" in text
            or "collect " in text or "hidden package" in text
            or "find all" in text or "discover" in text
            or "explore" in text):
        return "exploration"
    if "complete the" in text or "finish the" in text or "mission" in text:
        return "story"
    if ("kill" in text or "defeat" in text or "earn $" in text
            or "reach level" in text or "win " in text
            or "drive " in text or " 100 " in text or "100x" in text):
        return "grind"
    if "100%" in text or "complete everything" in text:
        return "completion"
    return "unknown"


def family_key(trophy, family):
    """Pooling key: the family, or the tier for unclassified trophies."""
    if family != "unknown":
        return ("family", family)
    return ("tier", str(trophy.get("tiern") or trophy.get("tier") or ""))


def trophy_points(fit_states, index, anchor_time):
    """Per-state observations for one trophy.

    Each point carries log-time age, rate, players, the feasible earner
    interval (when a simultaneous player count exists), and the fitting
    weight. Points are ordered oldest first.
    """
    points = []
    last_state = None
    for st in fit_states:
        rate = rate_at(st, index)
        if rate is None:
            continue
        t = days_since_launch(st["at"])
        weight = state_weight(st, anchor_time)
        point = {
            "t": t, "rate": rate, "players": st["players"],
            "weight": weight, "emin": None, "emax": None,
            "ehat": None, "empty": False, "width": None,
        }
        if st["players"] is not None and st["players"] > 0:
            emin, emax, ehat, empty, width = earner_interval(rate, st["players"])
            point["emin"] = emin
            point["emax"] = emax
            point["ehat"] = ehat
            point["empty"] = empty
            point["width"] = width
        points.append(point)
        last_state = st
    # The anchor is the last VALID observation: a merged state's value
    # was last confirmed at its last poll, so time the anchor there.
    # Earlier states keep their first-seen time (a visible change is
    # timestamped at its first observing poll).
    if points and last_state is not None and last_state.get("polls"):
        points[-1]["t"] = days_since_launch(last_state["polls"][-1])
    return points


def count_weight(point):
    """Fitting weight for a count point: state weight, interval-width
    downweight, quarter weight on empty rounding intervals."""
    w = point["weight"]
    if point["empty"]:
        w *= 0.25
    if (not point["empty"] and point["emin"] is not None
            and point["emax"] is not None and point["emax"] >= point["emin"]):
        log_width = math.log((point["emax"] + 0.5) / (point["emin"] + 0.5))
        w /= 1.0 + (log_width / 0.5) ** 2
    return w


def fit_slope(pairs, y_anchor, b_f, clip_lo, clip_hi, huber):
    """Anchored slope fit: one ridge solve plus two Huber passes.

    pairs is a list of (D, y, w). Returns (slope, sum w D^2).
    """
    def solve(ws):
        num = 0.15 * b_f
        den = 0.15
        for (d, y, _w), w in zip(pairs, ws):
            num += w * d * (y - y_anchor)
            den += w * d * d
        if den <= 0.0:
            return b_f
        return num / den

    ws = [p[2] for p in pairs]
    b = solve(ws)
    for _ in range(2):
        new_ws = []
        for (d, y, w0), _w in zip(pairs, ws):
            res = (y - y_anchor) - b * d
            mult = 1.0 if abs(res) <= huber else huber / abs(res)
            new_ws.append(w0 * mult)
        ws = new_ws
        b = solve(ws)
    b = clip(b, clip_lo, clip_hi)
    info = 0.0
    for (d, _y, _w0), w in zip(pairs, ws):
        info += w * d * d
    return b, info


def fit_trophy_branches(points, anchor, b_f_count, b_f_direct):
    """Fit both anchored curves for one trophy.

    anchor is the last point (latest valid rate). Returns a dict with
    the count and direct slopes, their fit information, and whether the
    count branch is usable (anchor has a simultaneous player count).
    """
    t_i = anchor["t"]
    x_i = x_of_t(t_i)
    count_pairs = []
    direct_pairs = []
    for pt in points:
        d = x_of_t(pt["t"]) - x_i
        direct_pairs.append((d, score_logit(pt["rate"]), pt["weight"]))
        if pt["ehat"] is not None:
            count_pairs.append((d, math.log(pt["ehat"] + 0.5),
                                count_weight(pt)))
    b_direct, info_direct = fit_slope(
        direct_pairs, score_logit(anchor["rate"]), b_f_direct,
        -6.0, 6.0, 0.60)
    result = {
        "t_i": t_i, "r_i": anchor["rate"],
        "b_direct": b_direct, "info_direct": info_direct,
        "count": False, "b_count": None, "info_count": 0.0,
        "ehat_i": None,
    }
    if anchor["ehat"] is not None and count_pairs:
        y_anchor = math.log(anchor["ehat"] + 0.5)
        b_count, info_count = fit_slope(
            count_pairs, y_anchor, b_f_count, 0.0, 6.0, 0.35)
        result["count"] = True
        result["b_count"] = b_count
        result["info_count"] = info_count
        result["ehat_i"] = anchor["ehat"]
    return result


def branch_residuals(points, fit, branch):
    """Scoring-logit residuals of one branch against observed rates.

    Count residuals use fitted earners divided by observed players so
    denominator forecast error is not charged twice.
    """
    res = []
    t_i = fit["t_i"]
    x_i = x_of_t(t_i)
    for pt in points:
        d = x_of_t(pt["t"]) - x_i
        if branch == "count":
            if not fit["count"] or pt["players"] is None or pt["players"] <= 0:
                continue
            if pt["ehat"] is None:
                continue
            e_fit = max(0.0, (fit["ehat_i"] + 0.5)
                        * safe_exp(fit["b_count"] * d) - 0.5)
            z_fit = score_logit(100.0 * saturate(e_fit / float(pt["players"])))
        else:
            z_fit = score_logit(fit["r_i"]) + fit["b_direct"] * d
        res.append(score_logit(pt["rate"]) - z_fit)
    return res


def shrunk_scale(local_res, family_sq_scales):
    """Local squared MAD scale shrunk toward the family median squared
    scale with four pseudo-observations; default scale 0.25."""
    n = len(local_res)
    loc_sq = None
    if n >= 2:
        sc = mad_scale(local_res)
        if sc is not None:
            loc_sq = sc * sc
    fam = median(family_sq_scales) if family_sq_scales else None
    if loc_sq is not None and fam is not None:
        return (n * loc_sq + 4.0 * fam) / (n + 4.0)
    if loc_sq is not None:
        return loc_sq
    if fam is not None:
        return fam
    return 0.25 * 0.25


def family_slope_scale(peer_slopes):
    """Family slope MAD scale for one branch, floored at 0.25,
    default 0.75 when no peers exist."""
    if not peer_slopes:
        return 0.75
    sc = mad_scale(peer_slopes)
    if sc is None:
        return 0.75
    return max(0.25, sc)


# ---------------------------------------------------------------------------
# Refresh timing, projection, and quantile inversion
# ---------------------------------------------------------------------------

def estimate_timing(states, cutoff):
    """Estimate the refresh cadence from observed change times.

    Returns None when there is no usable history. mu is the regular
    mean interval in days, floored at the median positive poll gap.
    startup is true for a single frozen state watched since within one
    hour of launch.
    """
    if not states:
        return None
    change_times = [st["first_seen"] for st in states[1:]]
    first_seen = states[0]["first_seen"]
    backfill = (first_seen - LAUNCH).total_seconds() <= HOUR
    if change_times:
        ref = change_times[0]
        n_after = len(change_times) - 1
    else:
        ref = first_seen
        n_after = 0
    t_days = max(0.0, (cutoff - ref).total_seconds() / DAY)
    mu = (t_days + 6.0 / 24.0) / (n_after + 2)
    polls = []
    for st in states:
        polls.extend(st["polls"])
    polls.sort()
    gaps = []
    prev = None
    for moment in polls:
        if prev is not None:
            gap = (moment - prev).total_seconds() / DAY
            if gap > 0.0:
                gaps.append(gap)
        prev = moment
    med_gap = median(gaps)
    if med_gap is not None:
        mu = max(mu, med_gap)
    startup = len(states) == 1 and backfill
    return {"mu": mu, "startup": startup}


def future_cdf(u_days, timing, cutoff):
    """CDF of the first future refresh, u days after the cutoff."""
    if timing["startup"]:
        a_h = (cutoff - LAUNCH).total_seconds() / HOUR
        u_h = u_days * 24.0
        # S(a+u)/S(a) = exp(-(2*a*u + u^2)/16) for the Weibull scale 4h.
        ratio = safe_exp(-(2.0 * a_h * u_h + u_h * u_h) / 16.0)
        return 1.0 - ratio
    mu = timing["mu"]
    if mu <= 0.0:
        return 1.0
    return 1.0 - math.exp(-u_days / mu)


def refresh_nodes(h_days, timing, cutoff):
    """Eight equal-weight last-refresh nodes.

    Returns (node ages in days after the cutoff, refreshed mass).
    Nodes solve G(u_j) = F_W(H)(j-0.5)/8 with
    G(u) = F_W(u) exp(-(H-u)/mu_eff).
    """
    if h_days <= 0.0:
        return [], 0.0
    fwh = future_cdf(h_days, timing, cutoff)
    if fwh <= 0.0:
        return [], 0.0
    mu_eff = (4.0 / 24.0) if timing["startup"] else timing["mu"]

    def gfun(u):
        if mu_eff > 0.0:
            return future_cdf(u, timing, cutoff) * math.exp(-(h_days - u) / mu_eff)
        return future_cdf(u, timing, cutoff)

    nodes = []
    for j in range(1, NODES + 1):
        target = fwh * (j - 0.5) / NODES
        low, high = 0.0, h_days
        for _ in range(60):
            mid = 0.5 * (low + high)
            if gfun(mid) >= target:
                high = mid
            else:
                low = mid
        nodes.append(high)
    return nodes, fwh


def project_z(fit, s_days, p_s, eps_p, gamma, jump=None, no_players=False):
    """Projected scoring logit at refresh time s (days since launch)."""
    d = x_of_t(s_days) - x_of_t(fit["t_i"])
    if d > 0.0:
        # Damp forward log-time extrapolation: earners saturate, so a
        # log-linear slope fitted on the steep launch window must not
        # compound unchecked over a 4.5-day horizon. D_eff = A(1-e^-D/A)
        # with A = 0.3: near D = 0 it is ~identity (90-min forecasts
        # untouched), while D -> inf caps the effective step at 0.3, so
        # extrapolated earner growth stays bounded. The rate cap g(v)
        # in saturate() only bites above an 80% rate and cannot fix
        # this: at 6510 min the undamped median q50 sat 80-94% against
        # 9-25% truth on both synthetic generators, while the damped
        # model wins every 6510-min scenario and leaves practice,
        # replay and scenario G slightly better, not worse.
        d = 0.3 * (1.0 - math.exp(-d / 0.3))
    w = fit["w"]
    if fit["count"] and not no_players:
        e_s = max(0.0, (fit["ehat_i"] + 0.5)
                  * safe_exp(fit["b_count"] * d + gamma * eps_p) - 0.5)
        z_count = score_logit(100.0 * saturate(e_s / max(1.0, p_s)))
    else:
        z_count = None
    z_direct = score_logit(fit["r_i"]) + fit["b_direct"] * d
    if no_players and jump is not None:
        # Dilution prior for a frozen launch page without players.
        z_direct = z_direct - (1.0 - w) * math.log(jump)
    if z_count is None:
        return z_direct
    return (1.0 - w) * z_count + w * z_direct


def sigma_i_of(fit, s_days):
    """Forecast logit scale at refresh time s (days since launch)."""
    h_i = max(0.0, s_days - fit["t_i"])
    d = x_of_t(s_days) - x_of_t(fit["t_i"])
    var = (0.25 * h_i
           + fit["s_i2"] * h_i / (h_i + 0.25)
           + fit["sb2"] * d * d / (1.0 + fit["info"] / 0.15))
    return math.sqrt(max(0.0, var))


def invert_quantiles(components, atom_mass, atom_value, levels):
    """Left-inverse quantiles of the atom plus normal mixture.

    components is a list of (weight, location, sigma). Bisection runs
    28 steps over [logit(0.01), logit(99.99)] per level.
    """
    def cdf(z):
        total = 0.0
        if atom_mass > 0.0 and z >= atom_value:
            total += atom_mass
        for w, loc, sig in components:
            total += w * norm_cdf((z - loc) / sig)
        return total

    out = []
    for tau in levels:
        low, high = LOGIT_LO, LOGIT_HI
        for _ in range(BISECT_STEPS):
            mid = 0.5 * (low + high)
            if cdf(mid) >= tau:
                high = mid
            else:
                low = mid
        q = inv_score_logit(high)
        out.append(round_out(clip(q, 0.0, 100.0)))
    return out


# ---------------------------------------------------------------------------
# Case forecast
# ---------------------------------------------------------------------------

def forecast_case(case):
    """Forecast every trophy in one case. Returns rows in input order."""
    trophies = case.get("trophies") or []
    if not trophies:
        return []
    levels = list(case.get("levels") or DEFAULT_LEVELS)
    history = case.get("history") or []
    cutoff = parse_time(case.get("at"))
    if cutoff is None:
        times = [parse_time(h.get("at")) for h in history
                 if isinstance(h, dict)]
        times = [t for t in times if t is not None]
        cutoff = max(times) if times else LAUNCH
    target = parse_time(case.get("target"))
    horizon_min = case.get("horizon_minutes")
    if target is not None:
        h_days = max(0.0, (target - cutoff).total_seconds() / DAY)
    elif isinstance(horizon_min, (int, float)) and not isinstance(horizon_min, bool):
        h_days = max(0.0, float(horizon_min) / 1440.0)
    else:
        h_days = 0.0

    states = build_states(history, cutoff)
    anchor_time = states[-1]["at"] if states else cutoff
    fit_list = fitting_states(states, anchor_time)
    denom = fit_denominator(fit_list, anchor_time)
    n = len(trophies)
    families = [classify_trophy(t) for t in trophies]
    keys = [family_key(t, f) for t, f in zip(trophies, families)]
    points_list = [trophy_points(fit_list, k, anchor_time) for k in range(n)]

    # Case-wide direct-logit flag: many sub-count intervals with a high
    # empty share mean the rounded rates cannot pin earners down.
    narrow = 0
    narrow_empty = 0
    for pts in points_list:
        for pt in pts:
            if pt["width"] is not None and pt["width"] < 1.0:
                narrow += 1
                if pt["empty"]:
                    narrow_empty += 1
    direct_case = narrow >= 20 and narrow > 0 and (narrow_empty / narrow) > 0.10

    # Preliminary (unpooled) fits and peer-pool eligibility.
    fits = [None] * n
    prelim_count = [None] * n
    prelim_direct = [None] * n
    eligible = [False] * n
    for k in range(n):
        pts = points_list[k]
        if not pts:
            continue
        b0 = COUNT_DEFAULTS[families[k]]
        fit = fit_trophy_branches(pts, pts[-1], b0, 0.0)
        prelim_count[k] = fit["b_count"]
        prelim_direct[k] = fit["b_direct"]
        if fit["count"]:
            n_count = sum(1 for pt in pts if pt["ehat"] is not None)
            eligible[k] = n_count >= 3 and fit["info_count"] >= 0.04
        else:
            eligible[k] = len(pts) >= 3 and fit["info_direct"] >= 0.04
        fits[k] = fit

    def peer_slopes(k, branch_values):
        out = []
        for j in range(n):
            if j == k or not eligible[j]:
                continue
            if keys[j] != keys[k]:
                continue
            value = branch_values[j]
            if value is not None:
                out.append(value)
        return out

    # Family-pooled priors, then final fits.
    for k in range(n):
        fit = fits[k]
        if fit is None:
            continue
        b0 = COUNT_DEFAULTS[families[k]]
        peers_c = peer_slopes(k, prelim_count)
        peers_d = peer_slopes(k, prelim_direct)
        if peers_c:
            b_f_c = (len(peers_c) * median(peers_c) + 4.0 * b0) / (len(peers_c) + 4.0)
        else:
            b_f_c = b0
        if peers_d:
            b_f_d = (len(peers_d) * median(peers_d)) / (len(peers_d) + 4.0)
        else:
            b_f_d = 0.0
        fits[k] = fit_trophy_branches(points_list[k], points_list[k][-1],
                                      b_f_c, b_f_d)

    # Residual scales, shrunk toward family medians; slope scales.
    local_sq = {"count": [None] * n, "direct": [None] * n}
    for k in range(n):
        fit = fits[k]
        if fit is None:
            continue
        for branch in ("count", "direct"):
            if branch == "count" and not fit["count"]:
                continue
            res = branch_residuals(points_list[k], fit, branch)
            if len(res) >= 2:
                sc = mad_scale(res)
                if sc is not None:
                    local_sq[branch][k] = sc * sc
    for k in range(n):
        fit = fits[k]
        if fit is None:
            continue
        use_count = fit["count"] and not direct_case
        w = 1.0
        if use_count:
            # Direct branch takes over only in the top quintile
            # (80-100%), where count saturation actually bites; below
            # that the (well-identified) count branch leads, so a
            # denominator jump dilutes the forecast instead of being
            # ignored by the jump-blind direct curve.
            w = clip((fit["r_i"] - 80.0) / 20.0, 0.0, 1.0)
        s2 = 0.0
        sb = 0.0
        info = 0.0
        for branch, weight_b in (("count", 1.0 - w), ("direct", w)):
            if branch == "count" and not use_count:
                continue
            fam_sq = [local_sq[branch][j] for j in range(n)
                      if j != k and keys[j] == keys[k]
                      and local_sq[branch][j] is not None]
            res = branch_residuals(points_list[k], fit, branch)
            s2 += weight_b * shrunk_scale(res, fam_sq)
            peers = peer_slopes(
                k, prelim_count if branch == "count" else prelim_direct)
            sb += weight_b * (family_slope_scale(peers) ** 2)
            info += weight_b * (fit["info_count"] if branch == "count"
                                else fit["info_direct"])
        fit["w"] = w
        fit["s_i2"] = s2
        fit["sb2"] = sb
        fit["info"] = info
        if not use_count:
            fit["count"] = False

    timing = estimate_timing(states, cutoff) if states else None
    nodes = []
    fwh = 0.0
    if timing is not None and h_days > 0.0:
        nodes, fwh = refresh_nodes(h_days, timing, cutoff)
    t_cut = days_since_launch(cutoff)

    def atom_value(k, fit):
        latest_rate = rate_at(states[-1], k) if states else None
        if latest_rate is not None:
            return score_logit(latest_rate)
        # No valid rate in the latest page state: project the anchor
        # forward to that state's timestamp for the no-change branch.
        s_days = days_since_launch(states[-1]["at"])
        gamma = 0.0 if (timing and timing["startup"]) else 0.25
        if denom is not None:
            mid = denom["curves"][1]
            p_s = max(1.0, denom_curve(mid, s_days))
            return project_z(fit, s_days, p_s, 0.0, gamma)
        return project_z(fit, s_days, 1.0, 0.0, gamma, no_players=True)

    rows = [None] * n
    median_logits = [None] * n
    mid_idx = min(range(len(levels)), key=lambda i: abs(levels[i] - 0.5))
    for k in range(n):
        fit = fits[k]
        if fit is None:
            continue
        if h_days <= 0.0 or timing is None or fwh <= 0.0:
            # Current page state carries full weight.
            aval = atom_value(k, fit)
            q = [round_out(clip(inv_score_logit(aval), 0.0, 100.0))
                 for _ in levels]
            rows[k] = {"id": trophies[k].get("id"), "q": q}
            median_logits[k] = aval
            continue
        components = []
        gamma = 0.0 if timing["startup"] else 0.25
        for u in nodes:
            s_days = t_cut + u
            sig_i = sigma_i_of(fit, s_days)
            if timing["startup"]:
                if denom is not None:
                    curve = denom["curves"][1]
                    pbar = denom_curve(curve, s_days)
                    h_p = max(0.0, s_days - denom["t_p"])
                    sp = sigma_p_of(curve, h_p)
                    for jump, w_jump in zip(JUMP_VALUES, JUMP_WEIGHTS):
                        for eps_sign, w_eps in zip(SHOCK_POINTS, SHOCK_WEIGHTS):
                            eps = eps_sign * sp
                            p_s = max(1.0, pbar * safe_exp(eps)
                                      + (jump - 1) * denom["p_star"])
                            z = project_z(fit, s_days, p_s, eps, gamma)
                            base = fwh * (1.0 / NODES) * w_jump * w_eps
                            components.append((base * 0.9, z, sig_i))
                            components.append((base * 0.1, z, 2.5 * sig_i))
                else:
                    for jump, w_jump in zip(JUMP_VALUES, JUMP_WEIGHTS):
                        z = project_z(fit, s_days, 1.0, 0.0, gamma,
                                      jump=jump, no_players=True)
                        base = fwh * (1.0 / NODES) * w_jump
                        components.append((base * 0.9, z, sig_i))
                        components.append((base * 0.1, z, 2.5 * sig_i))
            else:
                if denom is not None:
                    h_p = max(0.0, s_days - denom["t_p"])
                    for curve, w_beta in zip(denom["curves"], BETA_WEIGHTS):
                        pbar = denom_curve(curve, s_days)
                        sp = sigma_p_of(curve, h_p)
                        for eps_sign, w_eps in zip(SHOCK_POINTS, SHOCK_WEIGHTS):
                            eps = eps_sign * sp
                            p_s = max(1.0, pbar * safe_exp(eps))
                            z = project_z(fit, s_days, p_s, eps, gamma)
                            base = fwh * (1.0 / NODES) * w_beta * w_eps
                            components.append((base * 0.9, z, sig_i))
                            components.append((base * 0.1, z, 2.5 * sig_i))
                else:
                    z = project_z(fit, s_days, 1.0, 0.0, gamma,
                                  no_players=True)
                    base = fwh * (1.0 / NODES)
                    components.append((base * 0.9, z, sig_i))
                    components.append((base * 0.1, z, 2.5 * sig_i))
        aval = atom_value(k, fit)
        q = invert_quantiles(components, 1.0 - fwh, aval, levels)
        rows[k] = {"id": trophies[k].get("id"), "q": q}
        median_logits[k] = score_logit(q[mid_idx])

    # Never-observed trophies: median projected logit of observed
    # family peers at scale 2, else the family default median rate.
    for k in range(n):
        if rows[k] is not None:
            continue
        peers = [median_logits[j] for j in range(n)
                 if j != k and keys[j] == keys[k]
                 and median_logits[j] is not None]
        if peers:
            loc = median(peers)
        else:
            loc = score_logit(FALLBACK_MEDIAN[families[k]])
        q = []
        for tau in levels:
            z = loc + 2.0 * norm_ppf(tau)
            q.append(round_out(clip(inv_score_logit(z), 0.0, 100.0)))
        rows[k] = {"id": trophies[k].get("id"), "q": q}
        median_logits[k] = loc
    return rows


# ---------------------------------------------------------------------------
# Output validation, serialization, and entrypoint
# ---------------------------------------------------------------------------

def validate_output(rows, ids, levels):
    """Contract check: one row per trophy, ids once in input order,
    five finite non-decreasing quantiles in [0, 100] per row."""
    if rows is None or len(rows) != len(ids):
        return False
    for row, want_id in zip(rows, ids):
        if not isinstance(row, dict):
            return False
        if set(row.keys()) != {"id", "q"}:
            return False
        if row["id"] != want_id:
            return False
        q = row["q"]
        if not isinstance(q, list) or len(q) != len(levels):
            return False
        prev = None
        for value in q:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            if not math.isfinite(value) or value < 0.0 or value > 100.0:
                return False
            if prev is not None and value < prev:
                return False
            prev = value
    return True


def serialize_output(rows):
    """Compact JSON plus newline, or None when it exceeds the size cap."""
    try:
        text = json.dumps({"trophies": rows}, separators=(",", ":"),
                          ensure_ascii=True) + "\n"
    except Exception:
        return None
    if len(text.encode("utf-8")) > MAX_BYTES:
        return None
    return text


def emergency_rows(case, ids, levels):
    """Fallback rows: each trophy's last observed rate repeated, else 50."""
    history = case.get("history") or []
    cutoff = parse_time(case.get("at"))
    if cutoff is None:
        times = [parse_time(h.get("at")) for h in history
                 if isinstance(h, dict)]
        times = [t for t in times if t is not None]
        cutoff = max(times) if times else LAUNCH
    states = build_states(history, cutoff)
    rows = []
    for k, tid in enumerate(ids):
        rate = None
        for st in reversed(states):
            found = rate_at(st, k)
            if found is not None:
                rate = found
                break
        if rate is None:
            rate = 50.0
        rows.append({"id": tid, "q": [round_out(rate) for _ in levels]})
    return rows


def fixed_rows(ids, levels):
    """Smallest legal payload: all-zero quantiles for every trophy."""
    return [{"id": tid, "q": [0 for _ in levels]} for tid in ids]


def process_case(case):
    """Forecast one case and return compliant output text, or None when
    no attempted candidate produced compliant output."""
    trophies = case.get("trophies") or []
    ids = [t.get("id") for t in trophies]
    levels = list(case.get("levels") or DEFAULT_LEVELS)
    text = None
    rows = None
    try:
        rows = forecast_case(case)
    except Exception:
        rows = None
    if rows is not None and validate_output(rows, ids, levels):
        text = serialize_output(rows)
    if text is None:
        try:
            emergency = emergency_rows(case, ids, levels)
            if validate_output(emergency, ids, levels):
                text = serialize_output(emergency)
        except Exception:
            print("TERMINAL: emergency rows failed; trying fixed rows",
                  file=sys.stderr)
            try:
                fixed = fixed_rows(ids, levels)
                if validate_output(fixed, ids, levels):
                    text = serialize_output(fixed)
            except Exception:
                text = None
    if text is None:
        print("TERMINAL: primary and emergency produced no compliant "
              "output; trying fixed rows", file=sys.stderr)
        try:
            fixed = fixed_rows(ids, levels)
            if validate_output(fixed, ids, levels):
                text = serialize_output(fixed)
        except Exception:
            text = None
    return text


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    try:
        names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    except OSError:
        names = []
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError:
        return 1
    for name in names:
        in_path = os.path.join(in_dir, name)
        try:
            with open(in_path, "r", encoding="utf-8") as handle:
                case = json.load(handle)
        except Exception:
            continue
        if not isinstance(case, dict):
            continue
        text = process_case(case)
        if text is None:
            continue
        out_path = os.path.join(out_dir, name[:-3] + ".out")
        try:
            with open(out_path, "w", encoding="utf-8") as handle:
                handle.write(text)
        except OSError:
            continue
    return 0


if __name__ == "__main__":
    sys.exit(main())
