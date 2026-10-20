import numpy as np, sys
sys.path.insert(0, ".")
from possession import data, score, pipeline, features
df, dx = pipeline.load_features()
gt = score.public_gt(); g = df.graded.values
for use in (False, True):
    cfg = dict(pipeline.CONFIG, hmm_use_public=use)
    p, model = pipeline.hmm_dead_probability(df, dx, cfg)
    d = (p > 0.5) & g
    segs = features.runs(d)
    print("use_public=%s  match D frac %.3f  segs %d" % (use, d[g].mean(), len(segs)))
    print("   pred D in public:", [(a, b) for a, b in segs if 11000 <= a <= 18700])
    print("   GT   D in public:", [(a, b) for a, b in features.runs(gt == 2)])
    np.save("cache/p_dead_hmm_pub%d.npy" % use, p)
