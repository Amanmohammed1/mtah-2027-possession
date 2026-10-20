import numpy as np, sys, cv2
sys.path.insert(0, ".")
from possession import data, pipeline, features, team, video
labels, info = pipeline.run()
df, dx = pipeline.load_features()
t = data.load_tracking(); v = video.Video()
alive = info["alive"]
starts = [a for a, b in features.runs(alive) if a > 0 and v.available(a)]
rows = []
for a in starts:
    rows.append((a, "HADX"[labels[a]], "HA-"[team.restart_team(df, a)] if team.restart_team(df, a) >= 0 else "-"))
print(len(rows), "restarts with video;  first label vs nearest-player restart team disagree on",
      sum(1 for a, l, r in rows if r != "-" and l != r), "; no ball near anyone:", sum(1 for a, l, r in rows if r == "-"))
S = sys.argv[1]
for i in range(0, len(rows), 12):
    chunk = rows[i:i + 12]
    img = video.sheet(v, t, [a - 5 for a, _, _ in chunk], labels=["now:%s near:%s" % (l, r) for a, l, r in chunk], overlay=True)
    cv2.imwrite("%s/restarts_%03d.jpg" % (S, i // 12), img, [cv2.IMWRITE_JPEG_QUALITY, 72])
np.save("cache/restart_rows.npy", np.array([(a, "HADX".index(l), "HA-".index(r)) for a, l, r in rows]))
