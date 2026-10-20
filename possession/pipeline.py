"""End-to-end: tracking -> features -> alive/dead -> team -> submission codes."""

import os

import numpy as np
import pandas as pd

from . import alivehmm, broadcast, data, deadball, features, score, team

CONFIG = {
    "ctrl_radius": 1.25,     # metres from the ball to count as holding it
    "ctrl_frames": 8,        # consecutive frames of control to take possession (0.32 s)
    "hmm_k_dead": 15,        # dead sub-states at 5 Hz -> minimum stoppage of 3 s
    "hmm_llr_cap": 2.0,      # cap on any one channel's dead-vs-alive log-ratio per step
    "hmm_use_public": True,  # add the public 5-minute labels as fixed counts to EM
    "dead_threshold": 0.5,   # posterior probability of dead ball
    "onset_shift": 60,       # frames the stoppage evidence trails the whistle (fallback)
    "min_live_no_team": 75,  # live slivers shorter than this with no team at all are dead
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


def hmm_dead_probability(df, dx, cfg):
    """Per-frame posterior probability that the ball is dead, from the EM-fitted HMM."""
    n = len(df)
    obs = alivehmm.observations(df, dx, broadcast.on_tactical(n), data.load_smart_tagging())
    seqs, idxs = alivehmm.halves_obs(obs)
    model = alivehmm.AliveDeadHMM(k_dead=cfg["hmm_k_dead"], cap=cfg["hmm_llr_cap"])
    gt = score.public_gt()
    pub = gt != 3
    model.init_from_labels(obs[pub], (gt[pub] == 2).astype(int))
    fixed = None
    if cfg["hmm_use_public"]:
        fixed = [np.stack([np.bincount(obs[pub & (gt != 2), c], minlength=nb),
                           np.bincount(obs[pub & (gt == 2), c], minlength=nb)]).astype(float)
                 for c, nb in enumerate(alivehmm.n_bins())]
    model.fit(seqs, iters=15, fixed_counts=fixed)
    p = alivehmm.upsample([model.dead_probability(s) for s in seqs], idxs, n)
    return np.nan_to_num(p, nan=1.0), model


def restart_teams(df, alive, cfg):
    """Team taking each restart (first frame of every live stretch). The two kick-offs follow the
    Laws of the Game: whoever kicks off the first half does not kick off the second."""
    t = data.load_tracking()
    out = {}
    for start, _ in features.runs(alive):
        if start in (data.HALVES[0][0], data.SECOND_HALF_KICKOFF):
            continue
        tm, _rule = team.infer_restart(df, t, start)
        if tm >= 0:
            out[start] = tm
    first = team.restart_team(df, data.HALVES[0][0], before=0, after=25)
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
    dead = deadball.refine_boundaries(df, dead, onset_shift=cfg["onset_shift"])
    alive = ~dead & graded
    # Published: the second half is graded from 110808 but kicks off at 113742.
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
