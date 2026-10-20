import numpy as np, sys
sys.path.insert(0, ".")
from possession import data, score, pipeline, team, deadball, features
df, dx = pipeline.load_features()
gt = score.public_gt(); g = df.graded.values
p = np.load("cache/p_dead_hmm_pub0.npy")
dead0 = (p > 0.5) & g
ctrl = team.control(df, 1.25)
def full(dead):
    alive = ~dead & g
    lab = team.hysteresis(ctrl, alive, 8); lab[~g] = 3
    return lab
print("no refine     ", score.fmt(score.score(full(dead0), gt)))
for shift in (0, 60):
    dead = deadball.refine_boundaries(df, dead0, onset_shift=shift) & g
    print("refine shift %2d" % shift, score.fmt(score.score(full(dead), gt)), [(a, b) for a, b in features.runs(dead) if 11000 < a < 18700])
