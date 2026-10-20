import numpy as np, pandas as pd, sys
sys.path.insert(0, ".")
from possession import data, features, score, pipeline, team, decode
pub = np.zeros(data.N_FRAMES, bool); pub[11185:18685] = True
df, dx = pipeline.load_features()
gt = score.public_gt(); n = len(df)
cfg = dict(pipeline.CONFIG)
p_dead, _ = pipeline.hmm_dead_probability(df, dx, cfg)
alive = (p_dead < 0.5) & df.graded.values
p_away = np.load("cache/p_away_shape.npy")
def fused(R=1.25, kmin=4, cap=25, beta=1.0, alpha=0.05, lam=10.0):
    ctrl = team.control(df, R)
    E = np.zeros((n, 3))
    for a, b in features.runs(ctrl >= 0):
        seg = ctrl[a:b]; brk = np.flatnonzero(np.diff(seg)) + 1
        for s, e in zip(np.r_[0, brk], np.r_[brk, len(seg)]):
            L = e - s
            if L < kmin: continue
            T = seg[s]
            E[a + s:a + s + min(L, cap), 1 - T] -= beta
    pa = np.clip(p_away, 0.02, 0.98)
    E[:, 0] += alpha * np.log(1 - pa); E[:, 1] += alpha * np.log(pa)
    E[~alive, :2] = -1e4; E[alive, 2] = -1e4
    tr = decode.transition_matrix(lam, 0.0, 0.0)
    return decode.decode_halves(E, tr, data.HALVES, n)
base = team.hysteresis(team.control(df, 1.25), alive, 8); base[~df.graded.values] = 3
print("hysteresis       ", score.fmt(score.score(base, gt)))
for alpha in (0.0, 0.02, 0.05, 0.1):
    for lam in (10, 25):
        for cap in (10, 25):
            lab = fused(alpha=alpha, lam=lam, cap=cap)
            print("a=%.2f lam=%d cap=%d" % (alpha, lam, cap), score.fmt(score.score(lab, gt)))
