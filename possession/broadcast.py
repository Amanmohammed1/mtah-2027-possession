"""Align the broadcast recording to the tactical timeline by matching player layouts.

Both tracking files report positions in the same normalised pitch frame, so a broadcast frame
showing live play has a tactical twin whose players sit on top of it. Replays, close-ups and
crowd shots have no twin. The mapping found here is used for two things: borrowing broadcast ball
observations where the tactical camera lost the ball, and reading the director's cuts away from
live play as evidence of a stoppage.
"""

import os

import numpy as np
import pandas as pd
from numba import njit, prange

from . import data

FPS = data.FPS


def _compact(px, py, team, max_per_team=16):
    """(n, 2 teams, max_per_team, 2) positions with NaN padding, for fast numba loops."""
    n = len(px)
    out = np.full((n, 2, max_per_team, 2), np.nan, dtype=np.float32)
    for k in (0, 1):
        cols = np.flatnonzero(team == k)
        x, y = px[:, cols], py[:, cols]
        valid = ~np.isnan(x)
        order = np.argsort(~valid, axis=1, kind="stable")  # valid slots first
        xs = np.take_along_axis(x, order, 1)[:, :max_per_team]
        ys = np.take_along_axis(y, order, 1)[:, :max_per_team]
        m = min(max_per_team, xs.shape[1])
        out[:, k, :m, 0] = xs[:, :m]
        out[:, k, :m, 1] = ys[:, :m]
    return out


@njit(cache=True)
def _cost(bpos, tpos, bball, tball):
    """Mean distance from each broadcast player to the nearest same-team tactical player,
    capped at 5 m, plus the ball when both see it."""
    total = 0.0
    count = 0
    for k in range(2):
        for i in range(bpos.shape[1]):
            bx = bpos[k, i, 0]
            if np.isnan(bx):
                continue
            by = bpos[k, i, 1]
            best = 25.0
            for j in range(tpos.shape[1]):
                tx = tpos[k, j, 0]
                if np.isnan(tx):
                    continue
                d = (tx - bx) ** 2 + (tpos[k, j, 1] - by) ** 2
                if d < best:
                    best = d
            total += np.sqrt(best)
            count += 1
    if not np.isnan(bball[0]) and not np.isnan(tball[0]):
        d = np.sqrt((bball[0] - tball[0]) ** 2 + (bball[1] - tball[1]) ** 2)
        total += 2.0 * min(d, 5.0)
        count += 2
    if count == 0:
        return 1e9
    return total / count


@njit(parallel=True, cache=True)
def _search(banchors, bpos, bball, tpos, tball, cands):
    """For each broadcast anchor, the cost against every candidate tactical frame."""
    out = np.empty((len(banchors), len(cands)), dtype=np.float32)
    for a in prange(len(banchors)):
        b = banchors[a]
        for c in range(len(cands)):
            t = cands[c]
            out[a, c] = _cost(bpos[b], tpos[t], bball[b], tball[t])
    return out


@njit(parallel=True, cache=True)
def _local(bframes, guess, bpos, bball, tpos, tball, radius):
    """Best tactical frame within +/- radius of a guess, per broadcast frame."""
    best_t = np.full(len(bframes), -1, dtype=np.int64)
    best_c = np.full(len(bframes), 1e9, dtype=np.float32)
    n = tpos.shape[0]
    for i in prange(len(bframes)):
        b = bframes[i]
        for t in range(max(0, guess[i] - radius), min(n, guess[i] + radius + 1)):
            c = _cost(bpos[b], tpos[t], bball[b], tball[t])
            if c < best_c[i]:
                best_c[i] = c
                best_t[i] = t
    return best_t, best_c


def align(step_anchor=50, step_cand=5, good_cost=1.2):
    """Map every broadcast frame to a tactical frame (or -1). Cached."""
    cache = os.path.join(data.CACHE_DIR, "broadcast_alignment.parquet")
    if os.path.isfile(cache):
        return pd.read_parquet(cache)

    tac = data.load_tracking("tactical")
    bro = data.load_tracking("broadcast")
    tpos = _compact(tac["px"], tac["py"], tac["team"])
    bpos = _compact(bro["px"], bro["py"], bro["team"])
    tball = np.stack([tac["bx"], tac["by"]], 1).astype(np.float32)
    bball = np.stack([bro["bx"], bro["by"]], 1).astype(np.float32)
    nb = len(bpos)
    nplayers = (~np.isnan(bro["px"])).sum(1)

    # 1. Global search on sparse anchors.
    anchors = np.array([b for b in range(0, nb, step_anchor) if nplayers[b] >= 10], dtype=np.int64)
    cands = np.arange(0, len(tpos), step_cand, dtype=np.int64)
    costs = _search(anchors, bpos, bball, tpos, tball, cands)
    best = cands[np.argmin(costs, 1)]
    best_cost = costs.min(1)

    # 2. Refine each anchor to the exact frame.
    t_ref, c_ref = _local(anchors, best, bpos, bball, tpos, tball, step_cand)
    ok = c_ref < good_cost
    offset = np.where(ok, t_ref - anchors, np.iinfo(np.int64).min)

    # 3. Propagate to every broadcast frame: within a run of consistent anchors the offset is
    #    constant (live play); search locally around the interpolated guess, frame by frame.
    guess = np.full(nb, -1, dtype=np.int64)
    good = np.flatnonzero(ok)
    for i in range(nb):
        j = np.searchsorted(anchors[good], i)
        cand = []
        for k in (j - 1, j):
            if 0 <= k < len(good):
                cand.append(offset[good[k]])
        if cand:
            guess[i] = i + cand[0]
    frames = np.arange(nb, dtype=np.int64)
    has = (guess >= 0) & (nplayers >= 6)
    t_all = np.full(nb, -1, dtype=np.int64)
    c_all = np.full(nb, 1e9, dtype=np.float32)
    tt, cc = _local(frames[has], guess[has], bpos, bball, tpos, tball, 12)
    t_all[has], c_all[has] = tt, cc

    out = pd.DataFrame({"tac_frame": t_all, "cost": c_all, "n_players": nplayers})
    out["anchor_cost"] = np.nan
    out.loc[anchors, "anchor_cost"] = best_cost
    os.makedirs(data.CACHE_DIR, exist_ok=True)
    out.to_parquet(cache)
    return out


def offset_per_frame(al, max_cost=0.8, tol=3):
    """Tactical-minus-broadcast offset for every broadcast frame: one constant per half.

    The broadcast clock runs in step with the tactical one, so within a half the offset is a
    single number. Replays match the earlier moment they show and must not move it, which is
    why this is the median of the matches near the two dominant offsets rather than a rolling
    estimate.
    """
    t = al.tac_frame.values
    ok = (t >= 0) & (al.cost.values < max_cost)
    b = np.arange(len(t))
    o = t - b
    vals, cnt = np.unique(o[ok], return_counts=True)
    m1 = vals[np.argmax(cnt)]
    far = np.abs(vals - m1) > 1000
    m2 = vals[far][np.argmax(cnt[far])]
    halves = []
    for m in (m1, m2):
        sel = ok & (np.abs(o - m) <= tol)
        halves.append((np.median(b[sel]), int(np.round(np.median(o[sel]))), b[sel].min(), b[sel].max()))
    halves.sort()
    split = (halves[0][3] + halves[1][2]) // 2
    out = np.empty(len(t), dtype=np.int64)
    out[:split] = halves[0][1]
    out[split:] = halves[1][1]
    return out


def on_tactical(n_tac=data.N_FRAMES):
    """Per tactical frame: broadcast ball (metres), whether the broadcast shows the live pitch,
    and the reprojection error. NaN / False where the broadcast has no frame."""
    cache = os.path.join(data.CACHE_DIR, "broadcast_on_tactical.parquet")
    if os.path.isfile(cache):
        return pd.read_parquet(cache)
    al = align()
    off = offset_per_frame(al)
    bro = data.load_tracking("broadcast")
    hom = pd.read_csv(data.asset("homography", "broadcast"), usecols=["reprojection_error", "valid"])
    tac_idx = np.arange(len(off)) + off
    keep = (tac_idx >= 0) & (tac_idx < n_tac)
    out = pd.DataFrame({
        "bro_has": np.zeros(n_tac, bool),
        "bro_live": np.zeros(n_tac, bool),
        "bro_bx": np.full(n_tac, np.nan, np.float32),
        "bro_by": np.full(n_tac, np.nan, np.float32),
        "bro_nplayers": np.zeros(n_tac, np.int16),
    })
    ti = tac_idx[keep]
    out.loc[ti, "bro_has"] = True
    out.loc[ti, "bro_live"] = hom.valid.values[keep].astype(bool)
    out.loc[ti, "bro_bx"] = bro["bx"][keep]
    out.loc[ti, "bro_by"] = bro["by"][keep]
    out.loc[ti, "bro_nplayers"] = (~np.isnan(bro["px"][keep])).sum(1)
    out.to_parquet(cache)
    return out
