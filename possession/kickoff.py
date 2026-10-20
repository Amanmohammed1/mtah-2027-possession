"""Kick-off detection from the player layout alone.

Before a kick-off every player stands in his own half and the kicker stands at the centre spot.
The formation breaks as soon as the ball is played, so the end of a run of kick-off formation
is the restart, and the team of the player at the centre spot is the one kicking off.
"""

import numpy as np

from . import data
from .features import runs

HOME, AWAY = 0, 1


def formation(t, sign, max_wrong=1, centre_radius=3.0):
    """Per frame: (in_formation, team at the centre spot or -1).

    Own half: the home team attacks towards +x * sign, so its own half is x * sign < 0.
    Players within 1 m of the halfway line count as in their own half (tracking noise)."""
    px, py, team = t["px"], t["py"], t["team"]
    ok = ~np.isnan(px)
    side = np.where(team == HOME, -1.0, 1.0)[None, :] * sign[:, None]  # own half sign per slot
    in_own = ((px * side) > -1.0) & ok
    wrong = (ok & ~in_own).sum(1)
    n = ok.sum(1)
    d_centre = np.hypot(px, py)
    d_centre = np.where(ok, d_centre, np.inf)
    nearest = np.argmin(d_centre, 1)
    has_kicker = d_centre[np.arange(len(px)), nearest] < centre_radius
    kicker_team = np.where(has_kicker, team[nearest], -1)
    in_form = (n >= 16) & (wrong <= max_wrong) & has_kicker
    return in_form, kicker_team


def detect(t, sign, graded, min_frames=50):
    """[(restart_frame, kicking_team)] for every run of kick-off formation lasting >= 2 s."""
    in_form, kicker = formation(t, sign)
    out = []
    for a, b in runs(in_form & graded):
        if b - a < min_frames:
            continue
        team = np.bincount(kicker[a:b][kicker[a:b] >= 0], minlength=2).argmax()
        out.append((int(b), int(team)))
    return out
