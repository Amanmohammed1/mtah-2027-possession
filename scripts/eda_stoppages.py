import numpy as np, pandas as pd, sys
sys.path.insert(0, ".")
from possession import features
df = pd.read_parquet("cache/frame_features.parquet")
g = df.graded.values
out = (df.ball_out.values > 0.3) & g
r = [(a, b) for a, b in features.runs(out) if b - a >= 5]
rng = np.random.RandomState(0)
for a, b in [r[i] for i in sorted(rng.choice(len(r), 6, replace=False))]:
    print("==== out at", a, "len", b - a, "ball at (%.1f,%.1f)" % (df.bx[a], df.by[a]))
    line = ""
    for s in range(a - 75, a + 900, 25):
        w = df.iloc[s:s + 25]
        vis = w.ball_ok.mean()
        bv = np.nanmedian(w.ball_v) if vis > 0 else np.nan
        line += "%+d:%s%.0f/%.1f " % ((s - a) // 25, "o" if vis > .5 else ".", bv if bv == bv else -1, w.spd_all.mean())
    print(line)
