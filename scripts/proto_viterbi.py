import numpy as np, pandas as pd, sys
sys.path.insert(0, ".")
from possession import data, score, decode
df = pd.read_parquet("cache/frame_features.parquet")
gt = score.public_gt()
n = len(df); W = slice(11185, 18685)
seen = df.ball_seen.values
dH, dA = df.dH.values, df.dA.values
def emis_team(R, w, duel=0.5):
    E = np.zeros((n, 3))
    cH = seen & (dH < R) & (dH < dA)
    cA = seen & (dA < R) & (dA < dH)
    both = seen & (dH < R) & (dA < R)
    wt = np.where(both, w * duel, w)
    E[cH, 1] -= wt[cH]; E[cA, 0] -= wt[cA]
    return E
for R in (1.0, 1.5, 2.0):
    for w in (0.5, 1.0, 2.0):
        for sw in (5, 10, 20):
            E = emis_team(R, w)
            E[gt == 2, :2] = -1e4; E[(gt != 2), 2] = -1e4
            tr = decode.transition_matrix(sw, 5, 5)
            pred = np.full(n, 3, np.int8)
            pred[W] = decode.viterbi(E[W], tr)
            s = score.score(pred, gt)
            print(R, w, sw, score.fmt(s))
