import numpy as np, sys
sys.path.insert(0, ".")
from possession import data
t = data.load_tracking()
gt = data.read_labels("evaluation/gt/BAR-ATH_possession_gt_public.csv")
px, py = t["px"], t["py"]
# player speed from positions, smoothed over 0.4s
vx = (np.roll(px, -5, 0) - np.roll(px, 5, 0)) / 0.4; vy = (np.roll(py, -5, 0) - np.roll(py, 5, 0)) / 0.4
sp = np.hypot(vx, vy)
print("provided ps channel present when pos present: %.1f%%" % (100*np.mean(~np.isnan(t['ps'][~np.isnan(px)]))))
print("ps vs computed speed sample:", np.nanmedian(t['ps']), np.nanmedian(sp))
msp = np.nanmean(sp, 1)
for L, k in zip("HAD", range(3)):
    m = gt == k
    print(L, "mean player speed pctl 10/50/90:", np.round(np.nanpercentile(msp[m], [10, 50, 90]), 2))
idx = np.flatnonzero(gt != 3)
print("timeline of mean player speed (1 s steps), label:")
s = ""
for f in range(11185, 18685, 25):
    s += "%s%.1f " % ("HADX"[gt[f]], msp[f])
print(s)
