import numpy as np, sys
sys.path.insert(0, ".")
from possession import pipeline, score, data
import possession_metrics as pm
pub = np.zeros(data.N_FRAMES, bool); pub[11185:18685] = True
labels, info = pipeline.run(exclude=pub)
gt = score.public_gt()
runs = [(11185, 18685)]
filled = pm.fill_abstentions(labels, runs)
def segs(c): return [(a, b, "HADX"[k]) for a, b, k in pm.segments(c, runs)]
print("GT  :", " ".join("%s%d-%d" % (k, a, b) for a, b, k in segs(gt)))
print("PRED:", " ".join("%s%d-%d" % (k, a, b) for a, b, k in segs(filled)))
p = info["p_dead"]
s = ""
for f in range(11185, 18685, 50): s += "%s%s%d " % ("HAD"[gt[f]], "HADX"[filled[f]], min(9, int(p[f] * 10)))
print(s)
