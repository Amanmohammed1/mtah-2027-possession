import numpy as np, sys
sys.path.insert(0, ".")
from possession import pipeline, score, data, features
pub = np.zeros(data.N_FRAMES, bool); pub[11185:18685] = True
labels, info = pipeline.run(dict(hmm_use_public=False))
print("public:", score.fmt(score.score(labels)))
g = data.graded_mask()
lab = labels[g]
print("match: H %.3f A %.3f D %.3f" % tuple((lab == k).mean() for k in range(3)))
import possession_metrics as pm
runs = pm.valid_runs(np.where(g, 0, 3).astype(np.int8))
sp = pm.possession_spells(labels, runs)
print("match possessions", len(sp), "mean dur %.1fs" % (np.mean([b - a for a, b, _ in sp]) / 25),
      " D segments", len([1 for a, b, k in pm.segments(labels, runs) if k == 2]))
