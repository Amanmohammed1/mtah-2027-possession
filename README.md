# Frame-level ball possession — More than a Hack 2027

Team **Yamal Fan Boys**. For every frame of the tactical video of FC Barcelona vs Athletic Club,
this repository predicts whether Barcelona (`H`) or Athletic (`A`) has the ball, or the ball is
dead (`D`).

The method is described in **[docs/APPROACH.md](docs/APPROACH.md)**. In short: a hidden Markov
model, estimated without labels, decides when the ball is in play from the TV broadcast's cuts away
from live play, the tracked ball, player speeds and the Smart Tagging set pieces; stoppage edges
are snapped to the moment the ball leaves the pitch or is struck; then the team taking each restart
keeps the ball until the other team holds it.

## Reproduce the submission

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
ln -s /path/to/the/data/bundle data          # or: export MTAH_DATA_DIR=/path/to/the/data/bundle
.venv/bin/python run.py                      # -> out/Yamal_Fan_Boys.csv, deterministic
.venv/bin/python evaluation/validate_submission.py out/Yamal_Fan_Boys.csv
```

The data bundle is the one distributed to participants (`TACTICAL/` and `BROADCAST/`; the folder
names may be upper or lower case). The videos are not needed. Developed on Python 3.12 (macOS
arm64); a fresh clone reproduces the file byte for byte.

**The match data is not in this repository and must not be committed.**

## Compliance with the competition rules

- The pipeline never reads the ground truth: `run.py` works with `evaluation/gt/` deleted.
- No hand labels or other data created manually on the provided game; no checks of the output
  against the video.
- Every tuned parameter is tuned on external data (IDSSE, below); everything else is fixed a priori
  from the Laws of the Game or physics. Sources: [docs/PROVENANCE.md](docs/PROVENANCE.md),
  committed before the re-tuning.
- Earlier versions (tag `pre-new-rules`) did not meet the clarified rules; what changed is
  described in docs/APPROACH.md §8.

## External data

[IDSSE](https://doi.org/10.6084/m9.figshare.28196177) — Bassek, Rein, Weber & Memmert (2025), *An
integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data 12:195. Data
owner: Deutsche Fußball Liga (DFL). Licence: CC BY 4.0. Not redistributed here; to re-run the tuning,
download the positions and match-information XML files and run
`.venv/bin/python scripts/tune_external.py /path/to/idsse`. The sweep results are in
`docs/external_tuning.csv` and `docs/external_tuning.json`.

## Layout

```
run.py                     regenerates out/Yamal_Fan_Boys.csv
possession/
  data.py                  loading the provided assets (tracking, Smart Tagging)
  features.py              per-frame features: ball kinematics, proximity, player speeds
  broadcast.py             aligning the broadcast recording to the tactical timeline
  alivehmm.py              the alive / dead hidden Markov model (estimated without labels)
  deadball.py              stoppage-edge refinement and windowed features
  team.py                  team taking each restart; possession during play
  pipeline.py              the whole chain
  external.py              reader for the IDSSE tuning data
  score.py                 wrapper around the official metric (final check only; not used by run.py)
evaluation/, games.py      vendored unchanged from the official repository (MIT licence)
scripts/                   external tuning, label-free checks, figures
tools/build_pdf.py         renders docs/APPROACH.md to PDF
docs/                      technical note, parameter provenance, tuning results, figures
```

`pip install -r requirements-dev.txt` adds what the figures and PDF need.
