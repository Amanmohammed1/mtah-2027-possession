"""Dead-ball detection: rule-based stoppage intervals, windowed features, and a learned model."""

import numpy as np
import pandas as pd

from . import data
from .features import nan_smooth, runs

FPS = data.FPS


def restart_after(df, start, limit=60 * FPS, min_speed=6.0, in_margin=1.5):
    """First plausible restart after `start`: the ball, visible and (nearly) in bounds, is struck.

    A restart needs the ball to have been quiet first -- out of sight, or slow -- so the search
    begins once the ball has spent at least one second invisible, out of bounds or slow.
    """
    n = len(df)
    stop = min(n, start + limit)
    seen = df.ball_ok.values
    speed = df.ball_v.values
    out = df.ball_out.values
    quiet = 0
    for f in range(start, stop):
        if not seen[f] or out[f] > 0 or speed[f] < 2.0:
            quiet += 1
            continue
        if quiet >= FPS and out[f] < in_margin and speed[f] > min_speed:
            # back up to where the ball started accelerating
            g = f
            while g > start and seen[g - 1] and speed[g - 1] > 1.5 and speed[g - 1] < speed[g]:
                g -= 1
            return g
    return None


def out_of_bounds_stoppages(df, min_frames=5, margin=0.3):
    """[(start, restart)] for every episode where the visible ball leaves the pitch."""
    out = (df.ball_out.values > margin) & df.graded.values
    episodes = []
    last_end = -1
    for a, b in runs(out):
        if b - a < min_frames or a < last_end:
            continue
        # The crossing: walk back to where the ball was last inside the lines.
        c = a
        while c > 0 and df.ball_ok.values[c - 1] and df.ball_out.values[c - 1] > 0:
            c -= 1
        r = restart_after(df, b)
        if r is None:
            continue
        episodes.append((c, r))
        last_end = r
    return episodes


# -- windowed features -----------------------------------------------------------------------------

def _past(x, w):
    """Mean over the previous w frames (NaN-aware)."""
    s = nan_smooth(x, w)
    return np.concatenate((np.full(w // 2, np.nan), s[:-(w // 2)]))


def _future(x, w):
    s = nan_smooth(x, w)
    return np.concatenate((s[w // 2:], np.full(w // 2, np.nan)))


def window_features(df):
    """Context features for the alive/dead decision. Uses past and future, offline."""
    o = {}
    spd = df.spd_all.values.astype(np.float64)
    vis = df.ball_ok.values.astype(np.float64)
    bv = np.where(df.ball_ok.values, df.ball_v.values, np.nan)
    fast = np.where(df.ball_ok.values, (df.ball_v.values > 5).astype(np.float64), np.nan)
    still = np.where(df.ball_ok.values, (df.ball_v.values < 1.0).astype(np.float64), np.nan)
    out = np.where(df.ball_ok.values, (df.ball_out.values > 0).astype(np.float64), np.nan)

    for w in (25, 75, 150, 300):
        o["spd_m%d" % w] = nan_smooth(spd, w)
        o["vis_m%d" % w] = nan_smooth(vis, w)
        o["fast_m%d" % w] = nan_smooth(fast, w)
        o["still_m%d" % w] = nan_smooth(still, w)
        o["out_m%d" % w] = nan_smooth(out, w)
    for w in (75, 250):
        o["spd_past%d" % w] = _past(spd, w)
        o["spd_fut%d" % w] = _future(spd, w)
        o["fast_past%d" % w] = _past(fast, w)
        o["fast_fut%d" % w] = _future(fast, w)
        o["vis_past%d" % w] = _past(vis, w)
        o["vis_fut%d" % w] = _future(vis, w)
    o["spd_p90_m75"] = nan_smooth(df.spd_p90.values.astype(np.float64), 75)
    o["bv_med"] = nan_smooth(bv, 25)
    o["ball_line_dist"] = df.ball_line_dist.values
    o["gap_len"] = np.minimum(df.gap_len.values, 3000)
    o["gap_since"] = np.minimum(df.gap_since.values, 3000)
    o["gap_until"] = np.minimum(df.gap_until.values, 3000)
    o["dmin"] = np.minimum(df.dH.values, df.dA.values)
    o["dmin"][~np.isfinite(o["dmin"])] = np.nan
    # frames since the visible ball was last outside the lines
    last_out = np.where(df.ball_ok.values & (df.ball_out.values > 0.3), np.arange(len(df)), -10 ** 6)
    o["since_out"] = np.minimum(np.arange(len(df)) - np.maximum.accumulate(last_out), 3000)
    # crowding: how spread the whole set of players is (walls and boxes at set pieces)
    o["n_players"] = df.n_players.values
    o["sx_sum"] = df.sx_H.values + df.sx_A.values
    o["sy_sum"] = df.sy_H.values + df.sy_A.values
    return pd.DataFrame(o)


def smart_tag_features(tags, n):
    """Distance (frames) to the nearest Smart Tagging set-piece start, before and after."""
    starts = np.sort(tags.loc[tags.code == "SET PIECES", "start"].values)
    idx = np.arange(n)
    pos = np.searchsorted(starts, idx)
    nxt = np.where(pos < len(starts), starts[np.minimum(pos, len(starts) - 1)] - idx, 10 ** 5)
    prv = np.where(pos > 0, idx - starts[np.maximum(pos - 1, 0)], 10 ** 5)
    return pd.DataFrame({"sp_next": np.minimum(nxt, 3000), "sp_prev": np.minimum(prv, 3000)})


def broadcast_features(bt):
    """Is the TV director showing live play? Cuts to close-ups and replays mark stoppages."""
    live = np.where(bt.bro_has.values, bt.bro_live.values.astype(np.float64), np.nan)
    o = {"bro_has": bt.bro_has.values.astype(np.float64)}
    for w in (25, 75, 150, 300):
        o["live_m%d" % w] = nan_smooth(live, w)
    for w in (75, 250):
        o["live_past%d" % w] = _past(live, w)
        o["live_fut%d" % w] = _future(live, w)
    return pd.DataFrame(o)


def refine_boundaries(df, dead, onset_shift=0, search_back=250, search_fwd=50, kick_window=50):
    """Sharpen the edges of predicted stoppages.

    Onset: if the visible ball crossed a line shortly before (or just after) the predicted
    onset, the stoppage starts at the crossing. Otherwise the evidence the model relies on --
    players slowing, the broadcast cutting away -- trails the whistle, so the onset is pulled
    back by `onset_shift` frames.
    Restart: if the visible ball is struck from (near) rest close to the predicted restart, the
    stoppage ends on that frame.
    """
    n = len(dead)
    out = dead.copy()
    seen = df.ball_ok.values
    ball_out = df.ball_out.values
    speed = df.ball_v.values
    for a, b in runs(dead):
        # onset
        lo, hi = max(0, a - search_back), min(n - 1, a + search_fwd)
        # The earliest crossing in the window where the ball then stays out for >= 5 frames.
        crossing = None
        for f in range(lo + 1, hi - 8):
            if seen[f - 1] and ball_out[f - 1] <= 0 and seen[f] and ball_out[f] > 0 \
                    and np.all(seen[f + 3:f + 8]) and np.all(ball_out[f + 3:f + 8] > 0.3):
                crossing = f
                break
        if crossing is not None and crossing < b:
            new_a = crossing
        else:
            new_a = max(0, a - onset_shift)
        out[new_a:a] = True
        if new_a > a:
            out[a:new_a] = False
        # restart
        lo, hi = max(new_a + 1, b - kick_window), min(n - 1, b + kick_window)
        kick = None
        best = 0.0
        for f in range(lo, hi):
            if seen[f] and seen[f - 3] and speed[f - 3] < 2.0 and speed[f] > 5.0 and ball_out[f] < 1.0:
                jump = speed[f] - speed[f - 3]
                if jump > best:
                    best, kick = jump, f - 3
        if kick is not None:
            if kick < b:
                out[kick:b] = False
            else:
                out[b:kick] = True
    return out
