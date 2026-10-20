# Frame-level ball possession — More than a Hack 2027

For every frame of the tactical video of FC Barcelona vs Athletic Club, this repository predicts
whether Barcelona (`H`) or Athletic (`A`) has the ball, or the ball is dead (`D`).

The method is described in **[docs/APPROACH.md](docs/APPROACH.md)**. In short: a hidden Markov
model decides when the ball is in play, using the TV broadcast's cuts away from live play, the
tracked ball, player speeds and the Smart Tagging set pieces; stoppage edges are snapped to the
moment the ball leaves the pitch or is struck; then the team taking each restart keeps the ball
until the other team holds it.

## Reproduce the submission

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
ln -s /path/to/the/data/bundle data          # or: export MTAH_DATA_DIR=/path/to/the/data/bundle
.venv/bin/python run.py                      # -> out/submission.csv, ~40 s, deterministic
.venv/bin/python evaluation/validate_submission.py out/submission.csv
```

The data bundle is the one distributed to participants (`TACTICAL/` and `BROADCAST/`; the folder
names may be upper or lower case). The videos are not needed to produce the submission.
Developed and checked on Python 3.12, macOS arm64; a fresh clone reproduces the file byte for byte.

**The match data is not in this repository and must not be committed.**

## Layout

```
run.py                     regenerates out/submission.csv
possession/
  data.py                  loading the provided assets (tracking, Smart Tagging, homography)
  features.py              per-frame features: ball kinematics, proximity, player speeds
  broadcast.py             aligning the broadcast recording to the tactical timeline
  alivehmm.py              the alive / dead hidden Markov model (EM on the whole match)
  deadball.py              stoppage-edge refinement and windowed features
  team.py                  team taking each restart; possession during play
  pipeline.py              the whole chain
  score.py                 wrapper around the official metric for quick checks
  video.py                 frame reader and overlays (validation tools only)
evaluation/, games.py      vendored unchanged from the official repository (MIT licence)
tools/                     video proxy (Colab), integrity and alignment checks, contact sheets
validation/                labels made from the video, used only to choose between variants
scripts/                   exploratory analyses behind the decisions in docs/APPROACH.md
```

`pip install -r requirements-dev.txt` adds what the tools and exploratory scripts need.
