import numpy as np, sys
sys.path.insert(0, ".")
from possession import data, score, pipeline, team
df, dx = pipeline.load_features()
gt = score.public_gt(); n = len(df)
p_dead, _ = pipeline.hmm_dead_probability(df, dx, dict(pipeline.CONFIG))
alive = (p_dead < 0.5) & df.graded.values
dist = np.stack([df.dH.values, df.dA.values], 1)
seen = df.ball_seen.values
def backdate(lab, R2, maxback):
    out = lab.copy()
    ch = np.flatnonzero((lab[1:] != lab[:-1]) & np.isin(lab[1:], (0, 1)) & np.isin(lab[:-1], (0, 1))) + 1
    for f in ch:
        T = lab[f]; g = f
        while g > f - maxback and out[g - 1] == lab[f - 1] and seen[g - 1] and dist[g - 1, T] < R2:
            g -= 1
        out[g:f] = T
    return out
for R in (1.0, 1.25, 1.5):
    for k in (6, 8, 12):
        lab = team.hysteresis(team.control(df, R), alive, k); lab[~df.graded.values] = 3
        row = "R=%.2f k=%2d  plain %.4f" % (R, k, score.score(lab, gt)["spellF1"])
        for R2 in (2.5, 3.5, 5.0):
            row += "  bd%.1f %.4f" % (R2, score.score(backdate(lab, R2, 75), gt)["spellF1"])
        print(row)
