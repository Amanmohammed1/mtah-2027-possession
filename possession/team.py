"""Team in possession during live play: last team to hold the ball, with hysteresis."""

import numpy as np

from . import data
from .features import runs

H, A, D, X = 0, 1, 2, 3


def control(df, radius):
    """Per frame: 0/1 for the team whose nearest player is within `radius` of the observed
    ball (and nearer than the other team's), -1 otherwise."""
    seen = df.ball_seen.values
    dH, dA = df.dH.values, df.dA.values
    out = np.full(len(df), -1, dtype=np.int8)
    out[seen & (dH < radius) & (dH <= dA)] = H
    out[seen & (dA < radius) & (dA < dH)] = A
    return out


def hysteresis(ctrl, alive, k, first_frames=None):
    """Possession = the last team to control the ball for >= k consecutive frames.

    Inside each live stretch the label before the first qualifying control is back-filled with
    that team (after a restart there is always a clear possessor). Returns H/A on live frames,
    D elsewhere. Frames of a live stretch with no control at all are left X.
    """
    n = len(ctrl)
    lab = np.full(n, D, dtype=np.int8)
    for start, stop in runs(alive):
        cur, run_team, run_len = -1, -1, 0
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
                    else:
                        seg[switch_at - start:i - start] = c
                    cur = c
            if cur >= 0:
                seg[i - start] = cur
        if first_set is not None:
            seg[:first_set[0] - start] = first_set[1]
        lab[start:stop] = seg
    return lab
