"""End-to-end: tracking -> features -> alive/dead -> team -> submission codes.

The pipeline never reads the ground truth: every number below is fixed a priori, takes its
parameter-free option, or was tuned on external data (docs/PROVENANCE.md).
"""

import os

import numpy as np
import pandas as pd

from . import alivehmm, broadcast, data, deadball, features, team

CONFIG = {
    "ctrl_radius": 1.25,     # m, IDSSE sweep (scripts/tune_external.py, docs/external_tuning.json)
    "ctrl_frames": 5,        # frames, IDSSE sweep
    "hmm_k_dead": 19,        # dead sub-states at 5 Hz = 3.8 s: 5th percentile of IDSSE stoppages
    "hmm_llr_cap": np.inf,   # parameter-free: no cap on any channel's evidence
    "dead_threshold": 0.5,   # posterior probability of dead ball
    "onset_shift": 0,        # parameter-free: no pull-back of stoppage onsets
    "exit_live_p90": 2.32,   # m/s, IDSSE: 10th percentile of the players' p90 speed in live play
    "min_live_no_team": 75,  # frames; live slivers shorter than this with no team at all are dead
}


def cached(name, builder):
    path = os.path.join(data.CACHE_DIR, name)
    if os.path.isfile(path):
        return pd.read_parquet(path)
    frame = builder()
    os.makedirs(data.CACHE_DIR, exist_ok=True)
    frame.to_parquet(path)
    return frame


def load_features():
    df = cached("frame_features.parquet", lambda: features.build(data.load_tracking()))
    n = len(df)
    dx = cached("dead_features.parquet", lambda: pd.concat(
        [deadball.window_features(df), deadball.smart_tag_features(data.load_smart_tagging(), n)],
        axis=1))
    bx = cached("broadcast_features.parquet",
                lambda: deadball.broadcast_features(broadcast.on_tactical(n)))
    return df, pd.concat([dx, bx], axis=1)


def pseudo_labels(df):
    """Tracking-only provisional labels to start EM from (no ground truth, no video).

    dead  (1): the visible ball is more than 0.3 m outside the lines (Law 9);
    alive (0): the visible ball is at least 1 m inside the lines, moving faster than 5 m/s, and no
               out-of-pitch sighting within 10 s;
    unknown (-1): everything else.
    """
    seen = df.ball_ok.values
    out = seen & (df.ball_out.values > 0.3)
    near_out = np.convolve(out.astype(float), np.ones(501), "same") > 0
    alive = seen & (df.ball_line_dist.values > 1.0) & (df.ball_v.values > 5.0) & ~near_out
    lab = np.full(len(df), -1, dtype=np.int8)
    lab[alive] = 0
    lab[out] = 1
    lab[~df.graded.values] = -1
    return lab


def hmm_dead_probability(df, dx, cfg):
    """Per-frame posterior probability that the ball is dead, from the HMM fitted without labels."""
    n = len(df)
    obs = alivehmm.observations(df, dx, broadcast.on_tactical(n), data.load_smart_tagging())
    seqs, idxs = alivehmm.halves_obs(obs)
    model = alivehmm.AliveDeadHMM(k_dead=cfg["hmm_k_dead"], cap=cfg["hmm_llr_cap"])
    lab = pseudo_labels(df)
    known = lab >= 0
    model.init_from_labels(obs[known], lab[known].astype(int))
    model.fit(seqs, iters=15)
    p = alivehmm.upsample([model.dead_probability(s) for s in seqs], idxs, n)
    return np.nan_to_num(p, nan=1.0), model


def restart_teams(df, alive, cfg):
    """Team taking each restart (first frame of every live stretch). Kick-offs follow Law 8: the
    first half by the team of the player nearest the centre spot, the second half by the other."""
    t = data.load_tracking()
    out = {}
    for start, _ in features.runs(alive):
        if start in (data.HALVES[0][0], data.SECOND_HALF_KICKOFF):
            continue
        tm, _rule = team.infer_restart(df, t, start)
        if tm >= 0:
            out[start] = tm
    first = team.nearest_to_centre(t, data.HALVES[0][0])
    if first >= 0:
        out[data.HALVES[0][0]] = first
        out[data.SECOND_HALF_KICKOFF] = 1 - first
    return out


def run(cfg=None):
    cfg = dict(CONFIG, **(cfg or {}))
    df, dx = load_features()
    graded = df.graded.values

    p_dead, _ = hmm_dead_probability(df, dx, cfg)
    dead = (p_dead >= cfg["dead_threshold"]) & graded
    dead = deadball.refine_boundaries(df, dead, onset_shift=cfg["onset_shift"],
                                      live_p90=cfg["exit_live_p90"])
    alive = ~dead & graded
    # Published (DATA.md, FAQ.md): the second half is graded from 110808 but kicks off at 113742.
    alive[data.HALVES[1][0]:data.SECOND_HALF_KICKOFF] = False
    alive[data.SECOND_HALF_KICKOFF] = True

    initial = restart_teams(df, alive, cfg)
    ctrl = team.control(df, cfg["ctrl_radius"])
    labels = team.hysteresis(ctrl, alive, cfg["ctrl_frames"], initial)
    # A short live sliver where nobody ever has the ball, between two stoppages, is the HMM
    # flickering inside one stoppage rather than play.
    for start, stop in features.runs(alive):
        if stop - start < cfg["min_live_no_team"] and np.all(labels[start:stop] == team.X):
            labels[start:stop] = team.D
    labels[~graded] = team.X
    return labels, {"p_dead": p_dead, "alive": alive}
