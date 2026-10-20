import sys
import numpy as np
sys.path.insert(0, ".")
exec(open("tools/check_alignment.py").read().split("graded = data.graded_mask()")[0])
spd = np.nanmean(np.hypot(np.diff(t["px"], axis=0), np.diff(t["py"], axis=0)), 1)
for k in map(int, sys.argv[1:]):
    curves = []
    for j in range(100, 7400, 250):
        f0 = k * 7500 + j
        if not data.graded_mask()[f0] or np.nanmean(spd[f0:f0 + 10]) < 0.08:  # need moving players
            continue
        curves += [score(f, v.frame(f)) for f in range(f0, f0 + 10)]
    c = np.nanmean(curves, 0)
    print("part %02d  frames used %d  best shift %+d   curve:" % (k, len(curves), DELTAS[np.nanargmax(c)]),
          " ".join("%+d:%.3f" % (d, x) for d, x in zip(DELTAS, c) if abs(d) <= 4))
