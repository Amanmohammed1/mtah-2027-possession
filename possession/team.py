"""Team in possession during live play: last team to hold the ball, with hysteresis."""

import numpy as np

from . import data
from .features import runs

H, A, D, X = 0, 1, 2, 3
HOME, AWAY = H, A


def control(df, radius):
    """Per frame: 0/1 for the team whose nearest player is within `radius` of the observed
    ball (and nearer than the other team's), -1 otherwise."""
    seen = df.ball_seen.values
    dH, dA = df.dH.values, df.dA.values
    out = np.full(len(df), -1, dtype=np.int8)
    out[seen & (dH < radius) & (dH <= dA)] = H
    out[seen & (dA < radius) & (dA < dH)] = A
    return out


def hysteresis(ctrl, alive, k, initial=None):
    """Possession = the last team to control the ball for >= k consecutive frames.

    `initial` maps the first frame of a live stretch to the team taking the restart; that team
    holds the ball until the other one takes control. Without it, the label before the first
    qualifying control is back-filled with that team. Returns H/A on live frames, D elsewhere.
    Frames of a live stretch with no control and no known restart team are left X.
    """
    n = len(ctrl)
    lab = np.full(n, D, dtype=np.int8)
    initial = initial or {}
    for start, stop in runs(alive):
        cur, run_team, run_len = initial.get(start, -1), -1, 0
        first_set = None
        seg = np.full(stop - start, X, dtype=np.int8)
        for i in range(start, stop):
            c = ctrl[i]
            if c >= 0:
                if c == run_team:
                    run_len += 1
                else:
                    run_team, run_len = c, 1
                if c != cur and run_len >= k:
                    switch_at = i - run_len + 1  # the switch happens where the control began
                    if cur == -1:
                        first_set = (switch_at, c)
                    seg[switch_at - start:i - start] = c
                    cur = c
            if cur >= 0:
                seg[i - start] = cur
        if first_set is not None:
            seg[:first_set[0] - start] = first_set[1]
        lab[start:stop] = seg
    return lab


def restart_team(df, start, before=50, after=10, radius=3.0):
    """Team taking the restart that begins a live stretch at `start`: the team of the player
    nearest the visible ball just before and at the restart (opponents must stand back at
    free kicks, corners and throw-ins). -1 if the ball is not seen near anyone."""
    lo, hi = max(0, start - before), min(len(df), start + after)
    seen = df.ball_seen.values[lo:hi]
    dH, dA = df.dH.values[lo:hi], df.dA.values[lo:hi]
    votes = np.zeros(2)
    for s, h, a in zip(seen, dH, dA):
        if not s or min(h, a) > radius:
            continue
        votes[0 if h < a else 1] += 1
    if votes.sum() == 0:
        return -1
    return int(np.argmax(votes))


HALF_L, HALF_W = 105.3 / 2, 68.0 / 2
BOX_X, BOX_Y = HALF_L - 16.5, 20.16


def infer_restart(df, t, start, before=50, after=3, crowd=9):
    """Team taking the restart at `start`, and which rule decided it.

    1. ball (seen on the pitch just before the restart) at a corner -> the team attacking that
       goal; in a penalty area -> goal kick / defensive free kick by the team defending it.
    2. otherwise the team of the player within 3 m of the ball, nearest first (opponents must
       stand back at free kicks, corners and throw-ins).
    3. ball unseen: a crowded penalty area -> the team attacking that goal.
    `df.sign` is +1 where the home team attacks towards +x.
    """
    lo, hi = max(0, start - before), min(len(df), start + after)
    sign = df.sign.values[start]
    bxs, bys = df.bx.values[lo:hi], df.by.values[lo:hi]
    seen = df.ball_seen.values[lo:hi] & (np.abs(bxs) < HALF_L + 0.5) & (np.abs(bys) < HALF_W + 1.5)
    if seen.sum() >= 5:
        bx, by = np.median(bxs[seen]), np.median(bys[seen])
        side = np.sign(bx)
        attacker = HOME if side == sign else AWAY
        if abs(bx) > HALF_L - 3 and abs(by) > HALF_W - 5:
            return attacker, "corner"
        if abs(bx) > BOX_X and abs(by) < BOX_Y:
            return 1 - attacker, "box"
    team = restart_team(df, start, 50, 10, 3.0)
    if team >= 0:
        return team, "nearest"
    px, py = t["px"][start], t["py"][start]
    ok = ~np.isnan(px)
    for side in (1, -1):
        n_box = np.sum(ok & (px * side > BOX_X) & (np.abs(py) < BOX_Y))
        if n_box >= crowd:
            return (HOME if side == sign else AWAY), "crowd"
    return -1, "none"

