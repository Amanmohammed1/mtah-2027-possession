import numpy as np, sys, cv2
sys.path.insert(0, ".")
from possession import data, features, video, score
import possession_metrics as pm
lab = data.read_labels("out/submission.csv")
t = data.load_tracking(); v = video.Video()
g = data.graded_mask()
runs = pm.valid_runs(np.where(g, 0, 3).astype(np.int8))
filled = pm.fill_abstentions(lab, runs)
segs = pm.segments(filled, runs)
switches = []  # (frame, from, to, duration of new possession)
for (a0, b0, k0), (a1, b1, k1) in zip(segs[:-1], segs[1:]):
    if k0 in (0, 1) and k1 in (0, 1) and k0 != k1 and v.available(b1):
        switches.append((a1, k0, k1, b1 - a1))
sw = np.array(switches)
short = sw[sw[:, 3] < 50]; long_ = sw[sw[:, 3] >= 50]
print("H<->A switches with video:", len(sw), " into possessions < 2 s:", len(short))
rng = np.random.RandomState(1)
pick = np.concatenate([short[rng.choice(len(short), 18, replace=False)], long_[rng.choice(len(long_), 18, replace=False)]])
np.save("cache/switch_sample.npy", pick)
S = sys.argv[1]
for i in range(0, len(pick), 3):
    frames, labels = [], []
    for j, (f, k0, k1, d) in enumerate(pick[i:i + 3]):
        for dt in (-25, 0, 12, 37):
            frames.append(f + dt); labels.append("#%d %s->%s (%.1fs) %+.1fs" % (i + j, "HA"[k0], "HA"[k1], d / 25, dt / 25))
    cv2.imwrite("%s/switch_%02d.jpg" % (S, i // 3), video.sheet(v, t, frames, cols=4, crop=(480, 270), labels=labels, overlay=True), [cv2.IMWRITE_JPEG_QUALITY, 75])
for j, (f, k0, k1, d) in enumerate(pick): print(j, f, "HA"[k0], "->", "HA"[k1], d)
