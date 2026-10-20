import numpy as np, sys, cv2
sys.path.insert(0, ".")
from possession import data, features, video
lab = data.read_labels("out/submission.csv")
t = data.load_tracking(); v = video.Video()
segs = [(a, b) for a, b in features.runs(lab == 2) if v.available(b) and data.graded_mask()[a]]
print(len(segs), "dead-ball segments with video; durations s:", np.percentile([(b - a) / 25 for a, b in segs], [0, 10, 50, 90, 100]).round(1))
S = sys.argv[1]
mids = [((a + b) // 2, a, b) for a, b in segs]
for i in range(0, len(mids), 12):
    ch = mids[i:i + 12]
    img = video.sheet(v, t, [m for m, _, _ in ch], labels=["D %d-%d (%.0fs)" % (a, b, (b - a) / 25) for _, a, b in ch])
    cv2.imwrite("%s/dead_%02d.jpg" % (S, i // 12), img, [cv2.IMWRITE_JPEG_QUALITY, 70])
