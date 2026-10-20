import numpy as np, pandas as pd, sys
sys.path.insert(0, ".")
from possession import data, pipeline, features, team
labels, info = pipeline.run()
df, dx = pipeline.load_features()
t = data.load_tracking()
vis = pd.read_csv("validation/restarts_visual.csv")
vis = vis[vis.team.isin(["H", "A"])]
ok_old = ok_new = n = 0
by_rule = {}
for f, tm in zip(vis.frame, vis.team):
    truth = "HA".index(tm)
    old = labels[f + 10] if labels[f + 10] in (0, 1) else -1
    new, rule = team.infer_restart(df, t, f)
    n += 1; ok_old += old == truth; ok_new += new == truth
    r = by_rule.setdefault(rule, [0, 0]); r[0] += new == truth; r[1] += 1
    if new != truth: print("miss", f, tm, "->", "HA-"[new], rule)
print("restart team correct: old %d/%d   new %d/%d" % (ok_old, n, ok_new, n), " by rule:", by_rule)
