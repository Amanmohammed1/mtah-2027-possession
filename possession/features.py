"""Per-frame features from the tracking: ball kinematics, proximity, and team shape.

All arrays are indexed by video frame. Positions are metres from the pitch centre.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from . import data

HALF_L = data.PITCH[0] / 2.0
HALF_W = data.PITCH[1] / 2.0
FPS = data.FPS


# -- small helpers ---------------------------------------------------------------------------------

def runs(mask):
    """[(start, stop)] half-open runs where mask is True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def nan_smooth(x, width):
    """Centred moving average that ignores NaN. width in frames."""
    valid = ~np.isnan(x)
    num = uniform_filter1d(np.where(valid, x, 0.0).astype(np.float64), width, mode="nearest")
    den = uniform_filter1d(valid.astype(np.float64), width, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0.2, num / den, np.nan)


def attack_sign(n=data.N_FRAMES, home_right_first=True):
    """+1 where the home team attacks towards +x, -1 where towards -x, per frame."""
    half = data.half_of(n)
    sign = np.ones(n, dtype=np.float32)
    first = 1.0 if home_right_first else -1.0
    # Ungraded frames: before the match and halftime take the first half, after the match the second.
    sign[:] = first
    sign[data.HALVES[1][0]:] = -first
    return sign


# -- ball ------------------------------------------------------------------------------------------

def clean_ball(bx, by, max_speed=40.0, max_gap=12):
    """Remove teleports, smooth, and fill short gaps.

    Returns (x, y, observed, filled): x, y are smoothed with short gaps linearly interpolated;
    observed is the raw visibility after outlier removal; filled marks interpolated frames.
    """
    x = bx.astype(np.float64).copy()
    y = by.astype(np.float64).copy()
    seen = ~np.isnan(x)

    # A point that jumps away from both neighbours faster than a struck ball can travel is noise.
    for _ in range(2):
        idx = np.flatnonzero(seen)
        if len(idx) < 3:
            break
        dt = np.diff(idx) / FPS
        step = np.hypot(np.diff(x[idx]), np.diff(y[idx])) / np.maximum(dt, 1.0 / FPS)
        bad_in = np.concatenate(([False], step > max_speed))
        bad_out = np.concatenate((step > max_speed, [False]))
        spikes = idx[bad_in & bad_out]
        x[spikes] = np.nan
        y[spikes] = np.nan
        seen = ~np.isnan(x)

    # Isolated detections (runs shorter than 3 frames) are mostly false positives.
    for start, stop in runs(seen):
        if stop - start < 3:
            x[start:stop] = np.nan
            y[start:stop] = np.nan
    seen = ~np.isnan(x)

    # Smooth inside visible runs only (5-frame centred mean).
    xs, ys = x.copy(), y.copy()
    for start, stop in runs(seen):
        if stop - start >= 5:
            xs[start:stop] = uniform_filter1d(x[start:stop], 5, mode="nearest")
            ys[start:stop] = uniform_filter1d(y[start:stop], 5, mode="nearest")

    # Fill short gaps linearly.
    filled = np.zeros(len(x), dtype=bool)
    for start, stop in runs(~seen):
        if start == 0 or stop == len(x) or stop - start > max_gap:
            continue
        a, b = start - 1, stop
        w = (np.arange(start, stop) - a) / float(b - a)
        xs[start:stop] = xs[a] + w * (xs[b] - xs[a])
        ys[start:stop] = ys[a] + w * (ys[b] - ys[a])
        filled[start:stop] = True
    return xs, ys, seen, filled


def gap_features(visible):
    """Length of the current visibility gap, frames since last / until next sighting."""
    n = len(visible)
    idx = np.arange(n)
    last = np.where(visible, idx, -1)
    last = np.maximum.accumulate(last)
    nxt = np.where(visible, idx, n + 10 ** 6)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    since = np.where(last >= 0, idx - last, 10 ** 6)
    until = nxt - idx
    gap_len = np.where(visible, 0, np.minimum(since + until, 10 ** 6))
    return since.astype(np.int32), until.astype(np.int32), gap_len.astype(np.int32)


# -- the main builder ------------------------------------------------------------------------------

def build(t):
    """DataFrame of per-frame features (n rows)."""
    n = len(t["bx"])
    px, py = t["px"].astype(np.float64), t["py"].astype(np.float64)
    team = t["team"]
    home, away = team == data.HOME, team == data.AWAY
    sign = attack_sign(n, bool(t["home_attacks_right_1st"]))

    f = {}
    bx, by, seen, filled = clean_ball(t["bx"], t["by"])
    vis = ~np.isnan(bx)
    f["ball_seen"] = seen
    f["ball_ok"] = vis
    f["ball_filled"] = filled
    f["bx"], f["by"] = bx, by

    vx = np.gradient(bx) * FPS
    vy = np.gradient(by) * FPS
    speed = np.hypot(vx, vy)
    f["ball_v"] = speed
    f["ball_vx"], f["ball_vy"] = vx, vy
    f["ball_acc"] = np.abs(np.gradient(speed)) * FPS
    # Direction change over +/- 3 frames, in radians.
    ang = np.arctan2(vy, vx)
    dang = np.abs(np.angle(np.exp(1j * (np.roll(ang, -3) - np.roll(ang, 3)))))
    f["ball_turn"] = dang
    f["ball_v_s1"] = nan_smooth(speed, 25)

    out_x = np.abs(bx) - HALF_L
    out_y = np.abs(by) - HALF_W
    f["ball_out"] = np.maximum(out_x, out_y)  # >0 means outside the lines, metres
    f["ball_line_dist"] = -np.maximum(out_x, out_y)

    since, until, gap_len = gap_features(vis)
    f["gap_since"], f["gap_until"], f["gap_len"] = since, until, gap_len

    # Distance from each slot to the ball.
    d = np.hypot(px - bx[:, None], py - by[:, None])
    dH = np.where(home, d, np.inf)
    dA = np.where(away, d, np.inf)
    dH = np.where(np.isnan(dH), np.inf, dH)
    dA = np.where(np.isnan(dA), np.inf, dA)
    f["dH"] = dH.min(1)
    f["dA"] = dA.min(1)
    f["nH3"] = (dH < 3).sum(1)
    f["nA3"] = (dA < 3).sum(1)
    f["nH8"] = (dH < 8).sum(1)
    f["nA8"] = (dA < 8).sum(1)

    # Player kinematics from positions over +/- 5 frames.
    pvx = (np.roll(px, -5, 0) - np.roll(px, 5, 0)) * (FPS / 10.0)
    pvy = (np.roll(py, -5, 0) - np.roll(py, 5, 0)) * (FPS / 10.0)
    psp = np.hypot(pvx, pvy)
    psp[psp > 12] = np.nan  # identity switches produce impossible speeds
    with np.errstate(invalid="ignore"):
        f["spd_all"] = np.nanmean(psp, 1)
        f["spd_H"] = np.nanmean(np.where(home, psp, np.nan), 1)
        f["spd_A"] = np.nanmean(np.where(away, psp, np.nan), 1)
        f["spd_p90"] = np.nanpercentile(psp, 90, axis=1)
        # Speed towards the team's attacking goal.
        f["adv_H"] = np.nanmean(np.where(home, pvx, np.nan), 1) * sign
        f["adv_A"] = -np.nanmean(np.where(away, pvx, np.nan), 1) * sign
    f["n_players"] = (~np.isnan(px)).sum(1)

    # Team shape, in each team's own attacking frame (+x = towards the goal it attacks).
    with np.errstate(invalid="ignore"):
        for name, mask, s in (("H", home, sign), ("A", away, -sign)):
            xs = np.where(mask, px, np.nan) * s[:, None]
            ys = np.where(mask, py, np.nan)
            f["cx_" + name] = np.nanmean(xs, 1)
            f["cy_" + name] = np.nanmean(ys, 1)
            f["sx_" + name] = np.nanstd(xs, 1)
            f["sy_" + name] = np.nanstd(ys, 1)
            f["xmax_" + name] = np.nanpercentile(xs, 90, axis=1)
            f["xmin_" + name] = np.nanpercentile(xs, 10, axis=1)
            f["n_" + name] = (~np.isnan(xs)).sum(1)
        # Where the ball is, in the home attacking frame.
        f["bx_att"] = bx * sign
    df = pd.DataFrame(f)
    df["half"] = data.half_of(n)
    df["graded"] = data.graded_mask(n)
    df["sign"] = sign
    return df


# -- windowed context for the learned models -------------------------------------------------------

SHAPE_COLS = ["cx_H", "cx_A", "sx_H", "sx_A", "sy_H", "sy_A", "xmax_H", "xmax_A", "xmin_H",
              "xmin_A", "spd_all", "spd_H", "spd_A", "spd_p90", "adv_H", "adv_A", "n_players"]


def shape_features(df, windows=(25, 125)):
    """Ball-free features describing both teams, with rolling context. For blackout frames."""
    out = {}
    for col in SHAPE_COLS:
        x = df[col].values.astype(np.float64)
        out[col] = x
        for w in windows:
            out["{}_m{}".format(col, w)] = nan_smooth(x, w)
    o = pd.DataFrame(out)
    o["cx_diff"] = o["cx_H"] - o["cx_A"]
    o["sx_diff"] = o["sx_H"] - o["sx_A"]
    o["sy_diff"] = o["sy_H"] - o["sy_A"]
    o["adv_diff"] = o["adv_H"] - o["adv_A"]
    for w in windows:
        o["cx_diff_m{}".format(w)] = o["cx_H_m{}".format(w)] - o["cx_A_m{}".format(w)]
        o["adv_diff_m{}".format(w)] = o["adv_H_m{}".format(w)] - o["adv_A_m{}".format(w)]
        o["sy_diff_m{}".format(w)] = o["sy_H_m{}".format(w)] - o["sy_A_m{}".format(w)]
    # Derivative of global speed over 1 s: stoppages show as a sustained deceleration.
    o["dspd_25"] = np.gradient(nan_smooth(df["spd_all"].values, 25)) * FPS
    return o
