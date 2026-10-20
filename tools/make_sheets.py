"""Contact sheets for labelling: tools/make_sheets.py OUT_DIR START END STEP [PER_SHEET]"""
import os, sys
import cv2
import numpy as np
sys.path.insert(0, ".")
from possession import data, video

out, start, end, step = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
per = int(sys.argv[5]) if len(sys.argv) > 5 else 12
os.makedirs(out, exist_ok=True)
t = data.load_tracking()
pred = data.read_labels("out/submission.csv")
v = video.Video()
frames = list(range(start, end, step))
for i in range(0, len(frames), per):
    fs = frames[i:i + per]
    img = video.sheet(v, t, fs, labels=["pred:" + "HADX"[pred[f]] for f in fs])
    cv2.imwrite(os.path.join(out, "%06d.jpg" % fs[0]), img, [cv2.IMWRITE_JPEG_QUALITY, 72])
print(len(frames), "frames in", (len(frames) + per - 1) // per, "sheets")
