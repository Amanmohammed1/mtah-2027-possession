"""End-to-end: tracking -> features -> alive/dead -> team -> submission codes."""

import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import alivehmm, broadcast, data, deadball, decode, features, team

D_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=1000,
                feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0,
                verbose=-1, seed=0, deterministic=True, num_threads=1)

CONFIG = {
    "ctrl_radius": 1.25,     # metres from the ball to count as holding it
    "ctrl_frames": 8,        # consecutive frames of control to take possession (0.32 s)
    "dead_switch": 20.0,     # Viterbi cost of an alive <-> dead switch
    "dead_out_bonus": 5.0,   # per-frame log-evidence for D inside an out-of-bounds stoppage
    "alive_fast_bonus": 1.0, # per-frame log-evidence for alive when the ball is moving in bounds
    "dead_model": "hmm",     # "hmm" (EM alive/dead HMM) or "lgb" (pseudo-label classifier)
    "hmm_k_dead": 15,        # dead sub-states at 5 Hz -> minimum stoppage of 3 s
    "hmm_use_public": True,  # add the public 5-minute labels as fixed counts to EM
    "dead_threshold": 0.5,
    "onset_shift": 60,       # frames the stoppage evidence trails the whistle when no line is crossed
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


DEAD_VIEW = ["spd_m25", "spd_m75", "spd_m150", "spd_m300", "spd_past75", "spd_fut75",
             "spd_past250", "spd_fut250", "spd_p90_m75", "n_players", "sx_sum", "sy_sum",
             "sp_next", "sp_prev",
             "bro_has", "live_m25", "live_m75", "live_m150", "live_m300", "live_past75",
             "live_fut75", "live_past250", "live_fut250"]


def dead_pseudo_labels(df, dx, exclude=None):
    """1 = dead, 0 = alive, -1 = unknown. Defined from the ball only."""
    n = len(df)
    g = df.graded.values
    y = np.full(n, -1, dtype=np.int8)
    episodes = deadball.out_of_bounds_stoppages(df)
    for a, b in episodes:
        y[a + 10:b - 10] = 1
    vis = df.ball_ok.values
    inb = df.ball_line_dist.values > 1
    near = np.convolve((y == 1).astype(float), np.ones(250), "same") > 0
    y[vis & inb & (dx.fast_m75.values > 0.1) & ~near & (y != 1)] = 0
    for a, b in features.runs(~vis & g):
        if b - a > 500 or a == 0 or b >= n:
            continue
        if inb[a - 1] and inb[b] and df.ball_v.values[a - 1] > 3 and df.ball_v.values[b] > 3 \
                and not near[a:b].any():
            y[a:b] = 0
    y[~g] = -1
    if exclude is not None:
        y[exclude] = -1
    return y, episodes


def dead_probability(df, dx, exclude=None):
    y, episodes = dead_pseudo_labels(df, dx, exclude)
    train = y >= 0
    model = lgb.train(D_PARAMS, lgb.Dataset(dx.loc[train, DEAD_VIEW], y[train]), 300)
    return model.predict(dx[DEAD_VIEW]), episodes


def alive_dead(df, dx, p_dead, episodes, cfg):
    n = len(df)
    p = np.clip(p_dead, 1e-3, 1 - 1e-3)
    E = np.zeros((n, 3))
    E[:, 2] = np.log(p)
    E[:, 0] = E[:, 1] = np.log(1 - p)
    for a, b in episodes:
        E[a:b, 2] += cfg["dead_out_bonus"]
    moving = df.ball_ok.values & (df.ball_line_dist.values > 0.5) & (df.ball_v.values > 5)
    E[moving, 0] += cfg["alive_fast_bonus"]
    E[moving, 1] += cfg["alive_fast_bonus"]
    # Team is decided later; here only alive vs dead matters, so team switches are free.
    trans = decode.transition_matrix(0.0, cfg["dead_switch"], cfg["dead_switch"])
    path = decode.decode_halves(E, trans, data.HALVES, n)
    return np.isin(path, (0, 1))


def hmm_dead_probability(df, dx, cfg):
    n = len(df)
    obs = alivehmm.observations(df, dx, broadcast.on_tactical(n), data.load_smart_tagging())
    seqs, idxs = alivehmm.halves_obs(obs)
    gt = None
    model = alivehmm.AliveDeadHMM(k_dead=cfg["hmm_k_dead"])
    from . import score
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


def run(cfg=None, exclude=None):
    cfg = dict(CONFIG, **(cfg or {}))
    df, dx = load_features()
    if cfg["dead_model"] == "hmm":
        p_dead, _ = hmm_dead_probability(df, dx, cfg)
        dead = (p_dead >= cfg["dead_threshold"]) & df.graded.values
        dead = deadball.refine_boundaries(df, dead, onset_shift=cfg["onset_shift"])
        alive = ~dead & df.graded.values
        episodes = None
    else:
        p_dead, episodes = dead_probability(df, dx, exclude)
        alive = alive_dead(df, dx, p_dead, episodes, cfg) & df.graded.values
    ctrl = team.control(df, cfg["ctrl_radius"])
    labels = team.hysteresis(ctrl, alive, cfg["ctrl_frames"])
    labels[~df.graded.values] = team.X
    return labels, {"p_dead": p_dead, "alive": alive, "episodes": episodes}
