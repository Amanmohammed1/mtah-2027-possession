import numpy as np, sys
sys.path.insert(0, ".")
from possession import data
t = data.load_tracking()
gt = data.read_labels("evaluation/gt/BAR-ATH_possession_gt_public.csv")
bx, by, px, py, team = t["bx"], t["by"], t["px"], t["py"], t["team"]
d = np.hypot(px - bx[:, None], py - by[:, None])
dH = np.nanmin(np.where(team == 0, d, np.nan), 1); dA = np.nanmin(np.where(team == 1, d, np.nan), 1)
v = np.hypot(np.gradient(bx), np.gradient(by)) * 25
L = "HADX"
idx = np.flatnonzero(gt != 3)
ch = idx[1:][gt[idx[1:]] != gt[idx[1:] - 1]]
for f in ch:
    print("==== frame", f, L[gt[f-1]], "->", L[gt[f]])
    for g in range(f - 50, f + 30, 5):
        print(f"  {g-f:+4d} gt={L[gt[g]]} ball=({bx[g]:6.1f},{by[g]:6.1f}) v={v[g]:5.1f}  dH={dH[g]:5.1f} dA={dA[g]:5.1f}")
