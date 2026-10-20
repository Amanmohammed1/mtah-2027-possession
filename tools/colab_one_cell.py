# One-cell version of tools/colab_make_720p.ipynb that needs NO space in your Google Drive.
# Paste into a code cell in https://colab.research.google.com/notebooks/intro.ipynb and run it.
# Everything stays on Colab's temporary disk; the finished 720p file downloads to your computer.
# If Colab disconnects mid-way, run the cell again: finished parts on the same runtime are kept.
import os, math, shutil, subprocess, time, logging
logging.getLogger('googleapiclient').setLevel(logging.ERROR)
logging.getLogger('google_auth_httplib2').setLevel(logging.ERROR)
from google.colab import auth, files as colab_files
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

auth.authenticate_user()
svc = build('drive', 'v3')
IDS = ['PASTE_TACTICAL_VIDEO_ID', 'PASTE_BROADCAST_VIDEO_ID']
meta = {fid: svc.files().get(fileId=fid, fields='name,size', supportsAllDrives=True).execute() for fid in IDS}
TAC_ID = [fid for fid, m in meta.items() if 'tactical' in m['name'].lower()][0]
SRC, SIZE = '/content/BAR-ATH_tactical.mp4', int(meta[TAC_ID]['size'])

# 1. download the source to Colab's disk
if not (os.path.exists(SRC) and os.path.getsize(SRC) == SIZE):
    try:
        request = svc.files().get_media(fileId=TAC_ID, supportsAllDrives=True)
        with open(SRC, 'wb') as fh:
            dl = MediaIoBaseDownload(fh, request, chunksize=200 * 1024 * 1024)
            done = False
            while not done:
                status, done = dl.next_chunk()
                print('download %3d%%' % int(status.progress() * 100), end='\r')
    except HttpError as e:
        if 'downloadQuotaExceeded' in str(e):
            raise SystemExit('The organisers\' file is still over its daily download quota. '
                             'Try again in a few hours.')
        raise
print('\nsource ready: %.2f GB' % (os.path.getsize(SRC) / 1e9))

# 2. encode in 5-minute parts, every frame kept
TOTAL, FPS, PART = 199319, 25, 7500
OUT = '/content/parts'
os.makedirs(OUT, exist_ok=True)
gpu = subprocess.run('nvidia-smi', shell=True, capture_output=True).returncode == 0
nvenc = gpu and 'h264_nvenc' in subprocess.run('ffmpeg -hide_banner -encoders', shell=True,
                                               capture_output=True, text=True).stdout
ENC = (['-c:v', 'h264_nvenc', '-preset', 'p4', '-rc', 'vbr', '-cq', '32', '-b:v', '0', '-g', '50']
       if nvenc else ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '30', '-g', '50'])
print('encoder:', 'GPU (nvenc)' if nvenc else 'CPU (libx264)')

def frames_in(path):
    r = subprocess.run(['ffprobe', '-v', 'error', '-count_packets', '-select_streams', 'v:0',
                        '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0', path],
                       capture_output=True, text=True)
    return int(r.stdout.strip() or 0)

n_parts = math.ceil(TOTAL / PART)
for k in range(n_parts):
    want = min(PART, TOTAL - k * PART)
    final = f'{OUT}/part_{k:03d}.mp4'
    if os.path.exists(final) and frames_in(final) == want:
        continue
    t0 = time.time()
    tmp = f'/content/tmp_{k:03d}.mp4'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-ss', str(k * PART / FPS),
                    '-i', SRC, '-frames:v', str(want), '-vf', 'scale=1280:720', '-vsync', '0', '-an',
                    *ENC, tmp], check=True)
    got = frames_in(tmp)
    assert got == want, f'part {k}: {got} frames, expected {want}'
    shutil.move(tmp, final)
    print(f'part {k + 1}/{n_parts} done in {time.time() - t0:.0f}s')

# 3. join, check, download
with open('/content/list.txt', 'w') as f:
    f.writelines(f"file '{OUT}/part_{k:03d}.mp4'\n" for k in range(n_parts))
FINAL = '/content/BAR-ATH_tactical_720p.mp4'
subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0',
                '-i', '/content/list.txt', '-c', 'copy', FINAL], check=True)
n = frames_in(FINAL)
print('frames:', n, 'OK' if n == TOTAL else '<-- WRONG, tell Claude')
print('size: %.2f GB -- downloading to your computer now' % (os.path.getsize(FINAL) / 1e9))
colab_files.download(FINAL)
