import numpy as np, sys, time
sys.path.insert(0, ".")
from possession import data, features, score
t0=time.time()
t = data.load_tracking()
df = features.build(t)
df.to_parquet("cache/frame_features.parquet")
print("features", df.shape, "%.1fs" % (time.time()-t0))
gt = score.public_gt()
g = df.graded.values
print("ball ok graded %.3f, filled frames %.3f" % (df.ball_ok[g].mean(), df.ball_filled[g].mean()))
out = (df.ball_out.values > 0.3) & g
r = features.runs(out)
print("out-of-bounds runs in graded:", len(r), " >=5 frames:", sum(1 for a,b in r if b-a>=5))
# in public window, D vs out flag
m = gt != 3
for k,L in enumerate("HAD"):
    sel = m & (gt == k)
    print(L, "out frac %.3f" % out[sel].mean(), " ball ok %.3f" % df.ball_ok.values[sel].mean())
