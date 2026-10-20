import numpy as np, pandas as pd, sys
sys.path.insert(0, ".")
from possession import data, features, broadcast, pipeline
lab = data.read_labels("out/submission.csv")
df, dx = pipeline.load_features()
bt = broadcast.on_tactical()
g = data.graded_mask()
dead = (lab == 2) & g
alive = np.isin(lab, (0, 1, 3)) & g
restarts = np.array([b for a, b in features.runs(dead)])  # first alive frame after each stoppage
tags = data.load_smart_tagging()
sp = tags[tags.code == "SET PIECES"]
hit, miss = 0, []
for _, r in sp.iterrows():
    if not g[min(r.start + 87, len(g) - 1)]:
        continue
    near = restarts[(restarts >= r.start - 50) & (restarts <= r.end + 125)]
    if len(near): hit += 1
    else: miss.append((r.start, r.end, r.type))
print("Smart Tagging set pieces with a predicted restart nearby: %d / %d" % (hit, hit + len(miss)))
for m in miss: print("   no restart near tag", m)
# candidates inside predicted live play
cand = []
notlive = alive & bt.bro_has.values & ~bt.bro_live.values
for a, b in features.runs(notlive):
    if b - a >= 75: cand.append((a, b, "broadcast cut %.1fs" % ((b - a) / 25)))
out = alive & df.ball_ok.values & (df.ball_out.values > 1.0)
for a, b in features.runs(out):
    if b - a >= 25: cand.append((a, b, "ball out %.1fs" % ((b - a) / 25)))
cand.sort()
print("\ncandidates for missed stoppages inside predicted live play:", len(cand))
for c in cand: print("  ", c)
