import numpy as np, pandas as pd, sys, lightgbm as lgb
sys.path.insert(0, ".")
from possession import data, deadball, score, decode, features
df = pd.read_parquet("cache/frame_features.parquet")
gt = score.public_gt(); n = len(df)
X = pd.concat([deadball.window_features(df), deadball.smart_tag_features(data.load_smart_tagging(), n)], axis=1)
X.to_parquet("cache/dead_features.parquet")
g = df.graded.values
pub = np.zeros(n, bool); pub[11185:18685] = True
y = np.full(n, -1)
for a, b in deadball.out_of_bounds_stoppages(df):
    y[a + 10:b - 10] = 1
vis = df.ball_ok.values; inb = df.ball_line_dist.values > 1
alive = vis & inb & (X.fast_m75.values > 0.1) & (X.spd_m75.values > 1.3)
near_ep = np.convolve((y == 1).astype(float), np.ones(250), "same") > 0
y[alive & ~near_ep & (y != 1)] = 0
# in-play blackout: short gaps (< 10 s) bracketed by fast in-bounds ball, players moving
for a, b in features.runs(~vis & g):
    if b - a > 250 or a == 0 or b >= n: continue
    if inb[a-1] and inb[b] and df.ball_v.values[a-1] > 3 and df.ball_v.values[b] > 3 and np.nanmin(X.spd_m75.values[a:b]) > 1.5:
        y[a:b] = 0
train = g & ~pub & (y >= 0)
print("pseudo labels: D", (y[train] == 1).sum(), "alive", (y[train] == 0).sum())
params = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=500,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10, verbose=-1, seed=0)
m = lgb.train(params, lgb.Dataset(X[train], y[train]), 300)
p = m.predict(X)
np.save("cache/p_dead_v0.npy", p)
for thr in (0.3, 0.5, 0.7):
    pr = (p > thr)
    m_ = pub
    gtD = gt[m_] == 2
    tp = (pr[m_] & gtD).sum(); print("thr", thr, "public: D prec %.2f rec %.2f" % (tp / max(pr[m_].sum(), 1), tp / gtD.sum()))
print("frac graded predicted D (p>.5): %.3f" % (p[g] > .5).mean())
imp = pd.Series(m.feature_importance("gain"), X.columns).sort_values(ascending=False)
print(imp.head(15).round(0).to_dict())
# decode alive/dead with HMM on public window for a feel
for sw in (20, 40, 80):
    E = np.zeros((n, 3)); lp = np.log(np.clip(p, 1e-4, 1 - 1e-4)); lq = np.log(np.clip(1 - p, 1e-4, 1))
    E[:, 0] = lq; E[:, 1] = lq - 0.001; E[:, 2] = lp
    tr = decode.transition_matrix(1e6, sw, sw)
    path = decode.decode_halves(E, tr, data.HALVES, n)
    dead = path == 2
    segs = features.runs(dead & g)
    tp = (dead[pub] & (gt[pub] == 2)).sum()
    print("sw", sw, "D segments in match", len(segs), "median dur %.1fs" % (np.median([b-a for a,b in segs])/25), "public D prec %.2f rec %.2f" % (tp/max(dead[pub].sum(),1), tp/(gt[pub]==2).sum()), "pub D segs", [(a,b) for a,b in segs if 11185<=a<=18684])
