import numpy as np, pandas as pd, sys, lightgbm as lgb
sys.path.insert(0, ".")
from possession import data, deadball, score, decode, features
df = pd.read_parquet("cache/frame_features.parquet")
X = pd.read_parquet("cache/dead_features.parquet")
gt = score.public_gt(); n = len(df); g = df.graded.values
pub = np.zeros(n, bool); pub[11185:18685] = True
ball_cols = [c for c in X.columns if c.startswith(("fast_", "still_", "out_", "bv_", "dmin", "ball_line", "since_out"))]
VIEW = [c for c in X.columns if c not in ball_cols and not c.startswith(("vis_", "gap_"))]
print("player-view features:", VIEW)
y = np.full(n, -1)
eps = deadball.out_of_bounds_stoppages(df)
for a, b in eps: y[a + 10:b - 10] = 1
vis = df.ball_ok.values; inb = df.ball_line_dist.values > 1
alive = vis & inb & (X.fast_m75.values > 0.1)
near_ep = np.convolve((y == 1).astype(float), np.ones(250), "same") > 0
y[alive & ~near_ep & (y != 1)] = 0
for a, b in features.runs(~vis & g):
    if b - a > 500 or a == 0 or b >= n: continue
    if inb[a-1] and inb[b] and df.ball_v.values[a-1] > 3 and df.ball_v.values[b] > 3 and not near_ep[a:b].any():
        y[a:b] = 0
train = g & ~pub & (y >= 0)
print("pseudo: D %d alive %d" % ((y[train] == 1).sum(), (y[train] == 0).sum()))
params = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=1000,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10, verbose=-1, seed=0)
m = lgb.train(params, lgb.Dataset(X.loc[train, VIEW], y[train]), 300)
p = m.predict(X[VIEW])
np.save("cache/p_dead_view.npy", p)
gtD = gt[pub] == 2
for thr in (0.3, 0.5, 0.7):
    pr = p[pub] > thr; tp = (pr & gtD).sum()
    print("thr", thr, "public: D prec %.2f rec %.2f   match frac D %.3f" % (tp / max(pr.sum(), 1), tp / gtD.sum(), (p[g] > thr).mean()))
print(pd.Series(m.feature_importance("gain"), VIEW).sort_values(ascending=False).head(10).round(0).to_dict())
s = ""
for f in range(11185, 18685, 25): s += "%s%d " % ("HADX"[gt[f]], min(9, int(p[f] * 10)))
print(s)
