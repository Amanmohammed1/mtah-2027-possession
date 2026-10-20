import numpy as np, sys
sys.path.insert(0, ".")
from possession import pipeline, score, data, features
pub = np.zeros(data.N_FRAMES, bool); pub[11185:18685] = True
gt = score.public_gt(); g = data.graded_mask()
for cap in (1.5, 1.0):
    lab, info = pipeline.run(dict(hmm_use_public=False, hmm_llr_cap=cap), exclude=pub)
    s = score.score(lab, gt)
    segs = features.runs((lab == 2) & g)
    r = [b for a, b in segs if 57000 < b < 63000]
    pubr = [b for a, b in segs if 11185 < b < 18700]
    print("cap %-4s" % cap, score.fmt(s), "| D frac %.3f segs %d | restarts public %s (GT 12989 16400 17838 18389) | ~62330 case: %s" % ((lab[g] == 2).mean(), len(segs), pubr, r))
