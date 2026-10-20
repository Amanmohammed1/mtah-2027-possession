"""Read frames of the 720p tactical proxy (split in 7500-frame parts) and draw the tracking."""
import os

import cv2
import numpy as np
import pandas as pd

from . import data

VIDEO_DIR = os.environ.get("MTAH_VIDEO_DIR") or os.path.join(data.ROOT, "..", "video", "parts")
PART = 7500
SCALE = 1280 / 1920.0  # homography maps to 1920x1080 pixels


class Video:
    def __init__(self, folder=VIDEO_DIR):
        self.folder = folder
        self._cap, self._part, self._next = None, None, None

    def available(self, frame):
        return os.path.isfile(self._path(frame // PART))

    def _path(self, k):
        return os.path.join(self.folder, "part_%03d.mp4" % k)

    def frame(self, f):
        """BGR image of tactical frame f. Sequential reads are fast; seeks decode from the part
        start (or from the current position when moving forward), so they are exact."""
        k, j = divmod(int(f), PART)
        if self._part != k or self._next is None or j < self._next:
            if self._cap is not None:
                self._cap.release()
            self._cap = cv2.VideoCapture(self._path(k))
            self._part, self._next = k, 0
        while self._next < j:
            self._cap.grab()
            self._next += 1
        ok, img = self._cap.read()
        self._next += 1
        if not ok:
            raise IOError("cannot read frame %d" % f)
        return img


_H = None


def homographies():
    global _H
    if _H is None:
        cols = ["h%d%d" % (r, c) for r in range(3) for c in range(3)]
        table = pd.read_csv(data.asset("homography"), usecols=cols)
        _H = table[cols].values.reshape(-1, 3, 3)
    return _H


def project(f, x_m, y_m):
    """Metres from the pitch centre -> 720p pixels at frame f."""
    H = homographies()[f]
    u = np.asarray(x_m) / data.PITCH[0] + 0.5
    v = np.asarray(y_m) / data.PITCH[1] + 0.5
    den = H[2, 0] * u + H[2, 1] * v + H[2, 2]
    return ((H[0, 0] * u + H[0, 1] * v + H[0, 2]) / den * SCALE,
            (H[1, 0] * u + H[1, 1] * v + H[1, 2]) / den * SCALE)


def draw(img, f, t, label=None, ball=True):
    """Overlay players (home red, away blue) and ball (yellow) for tracking frame f."""
    out = img.copy()
    px, py, team = t["px"][f], t["py"][f], t["team"]
    ok = ~np.isnan(px)
    xs, ys = project(f, px[ok], py[ok])
    for x, y, k in zip(xs, ys, team[ok]):
        cv2.circle(out, (int(x), int(y)), 5, (0, 0, 255) if k == 0 else (255, 120, 0), 2)
    if ball and not np.isnan(t["bx"][f]):
        bx, by = project(f, t["bx"][f], t["by"][f])
        cv2.circle(out, (int(bx), int(by)), 7, (0, 255, 255), 2)
    if label:
        cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out
