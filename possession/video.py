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


def action_centre(f, t, df=None):
    """Pixel centre of the action: the ball if tracked, else the players' centroid."""
    if not np.isnan(t["bx"][f]):
        return project(f, t["bx"][f], t["by"][f])
    ok = ~np.isnan(t["px"][f])
    xs, ys = project(f, t["px"][f][ok], t["py"][f][ok])
    return float(np.median(xs)), float(np.median(ys))


def sheet(v, t, frames, cols=3, crop=(560, 315), labels=None, overlay=False):
    """Grid of crops around the action, one per frame, each captioned."""
    w, h = crop
    tiles = []
    last_c = None
    for i, f in enumerate(frames):
        img = v.frame(f)
        if overlay:
            img = draw(img, f, t)
        cx, cy = action_centre(f, t)
        if np.isnan(cx):
            cx, cy = last_c if last_c else (640, 360)
        last_c = (cx, cy)
        x0 = int(np.clip(cx - w / 2, 0, 1280 - w))
        y0 = int(np.clip(cy - h / 2, 0, 720 - h))
        tile = img[y0:y0 + h, x0:x0 + w].copy()
        cap = "%d %s" % (f, labels[i] if labels is not None else "")
        cv2.rectangle(tile, (0, 0), (w, 24), (0, 0, 0), -1)
        cv2.putText(tile, cap, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(tile)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    return np.vstack(rows)
