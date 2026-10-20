import numpy as np, sys
sys.path.insert(0, ".")
from possession import data, score
t = data.load_tracking()
gt = score.public_gt()
bx, by, px, py, team = t["bx"], t["by"], t["px"], t["py"], t["team"]
d = np.hypot(px - bx[:, None], py - by[:, None])
dH = np.nanmin(np.where(team == 0, d, np.inf), 1); dA = np.nanmin(np.where(team == 1, d, np.inf), 1)
dH[np.isnan(bx)] = np.inf; dA[np.isnan(bx)] = np.inf
dead = gt == 2
def run(R, k):
    ctrl = np.where((dH < R) & (dH <= dA), 0, np.where((dA < R) & (dA < dH), 1, -1))
    lab = np.full(len(gt), 3, np.int8); cur = -1; run_team = -1; run_len = 0
    for f in range(11185, 18685):
        if dead[f]:
            lab[f] = 2; cur = -1; run_len = 0; continue
        c = ctrl[f]
        if c >= 0:
            if c == run_team: run_len += 1
            else: run_team, run_len = c, 1
            if cur == -1 or (c != cur and run_len >= k): cur = c
        lab[f] = cur if cur >= 0 else 3
    return lab
for R in (1.0, 1.5, 2.0, 3.0):
    for k in (1, 3, 6, 10):
        print(R, k, score.fmt(score.score(run(R, k), gt)))
