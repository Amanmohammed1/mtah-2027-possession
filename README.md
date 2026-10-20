# Frame-level ball possession — More than a Hack 2027

Predicts, for every frame of the tactical video of FC Barcelona vs Athletic Club, whether
Barcelona (`H`) or Athletic (`A`) has the ball, or the ball is dead (`D`).

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
ln -s /path/to/the/data/bundle data      # or: export MTAH_DATA_DIR=/path/to/bundle
.venv/bin/python run.py                  # -> out/submission.csv
.venv/bin/python evaluation/validate_submission.py out/submission.csv
```

The match data is not in this repository and must not be committed.

`evaluation/` and `games.py` are vendored unchanged from the official challenge repository
(MIT licence) so the scorer can be run locally.
