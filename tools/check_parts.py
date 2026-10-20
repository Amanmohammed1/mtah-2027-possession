"""Integrity check of the 720p parts: resolution, fps, frame count (container and decoded)."""
import glob, os, sys, time
import cv2

parts = sorted(glob.glob(os.path.join(sys.argv[1], "part_*.mp4")))
full_decode = set(sys.argv[2:])
for p in parts:
    cap = cv2.VideoCapture(p)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    msg = "%s  %dx%d  %.2f fps  %d frames (container)" % (os.path.basename(p), w, h, fps, n)
    if os.path.basename(p) in full_decode:
        t0 = time.time(); k = 0
        while cap.grab(): k += 1
        msg += "  %d decoded (%.0fs)" % (k, time.time() - t0)
    else:
        ok_first = cap.read()[0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1); ok_last = cap.read()[0]
        msg += "  first/last readable: %s/%s" % (ok_first, ok_last)
    print(msg, "  %.0f MB" % (os.path.getsize(p) / 1e6))
