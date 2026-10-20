"""Is frame j of part k really tactical frame k*7500 + j? Shift the tracking by delta frames and
see which delta puts the tracked players on non-grass pixels best."""
import sys
import cv2
import numpy as np
sys.path.insert(0, ".")
from possession import data, video

t = data.load_tracking()
v = video.Video()
DELTAS = np.arange(-8, 9)


def nongrass(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    grass = (hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 60)
    return ~grass


def score(f, img):
    ng = nongrass(img)
    out = []
    for d in DELTAS:
        g = f + d
        ok = ~np.isnan(t["px"][g])
        xs, ys = video.project(f, t["px"][g][ok], t["py"][g][ok])
        vals = []
        for x, y in zip(xs.astype(int), ys.astype(int)):
            if 3 <= x < 1277 and 20 <= y < 718:
                vals.append(ng[y - 16:y - 1, x - 2:x + 3].mean())
        out.append(np.mean(vals) if vals else np.nan)
    return np.array(out)


graded = data.graded_mask()
for k in range(22):
    row = []
    for j in (40, 3750, 7440):
        f0 = k * 7500 + j
        if not graded[f0]:
            row.append("   -  ")
            continue
        s = np.nanmean([score(f, v.frame(f)) for f in range(f0, f0 + 20)], 0)
        best = DELTAS[np.nanargmax(s)]
        row.append("%+3d(%.2f)" % (best, s.max() - np.nanmedian(s)))
    print("part %02d  start/mid/end best shift:" % k, "  ".join(row))
