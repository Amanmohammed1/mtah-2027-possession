import numpy as np, pandas as pd, sys
sys.path.insert(0, ".")
from possession import data, broadcast, score, team, pipeline
df, dx = pipeline.load_features()
gt = score.public_gt(); n = len(df)
al = broadcast.align(); off = broadcast.offset_per_frame(al)
bro = data.load_tracking("broadcast")
# broadcast nearest-team distance per tactical frame
tac_of_b = np.arange(len(off)) + off
dHb = np.full(n, np.inf); dAb = np.full(n, np.inf)
d = np.hypot(bro["px"] - bro["bx"][:, None], bro["py"] - bro["by"][:, None])
d = np.where(np.isnan(d), np.inf, d)
h = d[:, bro["team"] == 0].min(1); a = d[:, bro["team"] == 1].min(1)
k = (tac_of_b >= 0) & (tac_of_b < n)
dHb[tac_of_b[k]] = h[k]; dAb[tac_of_b[k]] = a[k]
np.save("cache/bro_dH.npy", dHb); np.save("cache/bro_dA.npy", dAb)
alive = np.isin(gt, (0, 1))
for name, dH, dA in (("tactical", df.dH.values, df.dA.values), ("broadcast", dHb, dAb)):
    for R in (1.0, 1.5, 2.0):
        c = np.where((dH < R) & (dH <= dA), 0, np.where((dA < R) & (dA < dH), 1, -1))
        m = alive & (c >= 0)
        print("%-9s R=%.1f  control on %.2f of live frames, agrees with GT %.3f" % (name, R, m.sum() / alive.sum(), (c[m] == gt[m]).mean()))
