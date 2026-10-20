import numpy as np, pandas as pd, sys, lightgbm as lgb
sys.path.insert(0, ".")
from possession import data, features, score, pipeline
pub = np.zeros(data.N_FRAMES, bool); pub[11185:18685] = True
labels, info = pipeline.run(exclude=pub)
df, dx = pipeline.load_features()
S = features.shape_features(df)
S.to_parquet("cache/shape_features.parquet")
gt = score.public_gt()
y = labels.copy()
train = np.isin(y, (0, 1)) & ~pub & df.graded.values
# keep only frames inside possessions lasting > 3 s (confident pseudo-labels)
ok = np.zeros(len(y), bool)
for a, b in features.runs(train):
    seg = y[a:b]
    brk = np.flatnonzero(np.diff(seg)) + 1
    for s, e in zip(np.r_[0, brk], np.r_[brk, len(seg)]):
        if e - s > 75: ok[a + s:a + e] = True
train &= ok
print("train frames", train.sum(), "H share %.2f" % (y[train] == 0).mean())
params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10, verbose=-1, seed=0, num_threads=4)
m = lgb.train(params, lgb.Dataset(S[train], y[train]), 400)
p = m.predict(S)  # P(away)
np.save("cache/p_away_shape.npy", p)
test = pub & np.isin(gt, (0, 1))
acc = ((p[test] > .5).astype(int) == gt[test]).mean()
print("public: shape-only team accuracy %.3f" % acc)
for c in (0.6, 0.7, 0.8):
    conf = test & (np.abs(p - .5) > c - .5)
    print("  confidence>%.1f: coverage %.2f acc %.3f" % (c, conf.sum() / test.sum(), ((p[conf] > .5).astype(int) == gt[conf]).mean()))
print(pd.Series(m.feature_importance("gain"), S.columns).sort_values(ascending=False).head(10).round(0).to_dict())
