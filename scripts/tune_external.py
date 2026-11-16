"""Tune parameters on external data (IDSSE), following docs/PROVENANCE.md exactly.

    python scripts/tune_external.py /path/to/idsse   # folder with the positions + matchinformation XMLs

Outputs docs/external_tuning.csv (the full sweep) and prints the selected values. The provided
match is not read.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from possession import external, team  # noqa: E402
from possession.features import runs  # noqa: E402
import possession_metrics as pm  # noqa: E402

MATCHES = ("J03WMX", "J03WN1", "J03WPY")
RADII = (0.75, 1.0, 1.25, 1.5, 2.0)
FRAMES = (3, 5, 8, 12, 16, 25)
FPS = 25


def nearest(half, rng=None):
    """dH, dA per frame and whether the ball is seen, optionally degraded."""
    bx, by = half["bx"].copy(), half["by"].copy()
    px, py = half["px"].copy(), half["py"].copy()
    seen = ~np.isnan(bx)
    if rng is not None:
        alive_idx = np.flatnonzero(half["status"] == 1)
        target = int(0.10 * len(alive_idx))
        hidden = np.zeros(len(bx), dtype=bool)
        while hidden[alive_idx].sum() < target:
            s = rng.choice(alive_idx)
            hidden[s:s + rng.randint(25, 76)] = True
        seen &= ~hidden
        bx, by = bx + rng.normal(0, 0.3, len(bx)), by + rng.normal(0, 0.3, len(by))
        px, py = px + rng.normal(0, 0.3, px.shape), py + rng.normal(0, 0.3, py.shape)
    d = np.hypot(px - bx[:, None], py - by[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    dH = np.where(half["team"] == 0, d, np.inf).min(1)
    dA = np.where(half["team"] == 1, d, np.inf).min(1)
    return pd.DataFrame({"ball_seen": seen, "dH": dH, "dA": dA})


def predict(frame_df, lab, radius, k):
    alive = np.isin(lab, (0, 1))
    initial = {s: int(lab[s]) for s, _ in runs(alive)}
    ctrl = team.control(frame_df, radius)
    out = team.hysteresis(ctrl, alive, k, initial)
    out[lab == 3] = 3
    return out


def stoppage_and_speed_stats(halves):
    dead_lengths, p90_alive = [], []
    for half in halves:
        dead = half["status"] == 0
        dead_lengths += [(b - a) / FPS for a, b in runs(dead)]
        vx = (np.roll(half["px"], -5, 0) - np.roll(half["px"], 5, 0)) * (FPS / 10.0)
        vy = (np.roll(half["py"], -5, 0) - np.roll(half["py"], 5, 0)) * (FPS / 10.0)
        sp = np.hypot(vx, vy)
        sp[sp > 12] = np.nan
        p90 = np.nanpercentile(sp, 90, axis=1)
        trailing = pd.Series(p90).rolling(50, min_periods=10).mean().values  # 2 s before
        p90_alive += list(trailing[(half["status"] == 1) & ~np.isnan(trailing)])
    return np.array(dead_lengths), np.array(p90_alive)


def main(folder):
    rows, all_halves = [], []
    for m, match in enumerate(MATCHES):
        info = [p for p in glob.glob(os.path.join(folder, "info_*.xml")) if match in open(p, errors="ignore").read()]
        pos = glob.glob(os.path.join(folder, "pos_%s*.xml" % match))
        if not info or not pos:
            print("missing", match)
            continue
        halves = external.load_positions(pos[0], *external.team_ids(info[0]))
        hs = [halves[k] for k in sorted(halves)]
        all_halves += hs
        for cond in ("clean", "degraded"):
            preps = []
            for j, half in enumerate(hs):
                rng = np.random.RandomState(1000 * m + j) if cond == "degraded" else None
                preps.append((nearest(half, rng), external.labels(half)))
            for radius in RADII:
                for k in FRAMES:
                    gt_parts, pr_parts = [], []
                    for fdf, lab in preps:
                        gt_parts += [lab, np.array([3], dtype=np.int8)]
                        pr_parts += [predict(fdf, lab, radius, k), np.array([3], dtype=np.int8)]
                    r = pm.evaluate(np.concatenate(gt_parts), np.concatenate(pr_parts))
                    rows.append(dict(match=match, condition=cond, radius=radius, frames=k,
                                     spell_f1=r["spell_f1_mean"], macro_f1=r["macro_f1"],
                                     n_pred=r["n_pred_spells"], n_gt=r["n_gt_spells"]))
            print(match, cond, "done")
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(ROOT, "docs", "external_tuning.csv"), index=False)
    mean = table.groupby(["radius", "frames"])[["spell_f1", "macro_f1"]].mean().reset_index()
    best = mean.sort_values(["spell_f1", "macro_f1"], ascending=False).iloc[0]
    dead, p90 = stoppage_and_speed_stats(all_halves)
    summary = {
        "ctrl_radius": float(best.radius), "ctrl_frames": int(best.frames),
        "mean_spell_f1": float(best.spell_f1), "mean_macro_f1": float(best.macro_f1),
        "dead_run_p5_s": float(np.percentile(dead, 5)),
        "hmm_k_dead": int(max(1, round(np.percentile(dead, 5) / 0.2))),
        "exit_live_p90": float(np.percentile(p90, 10)),
        "n_stoppages": int(len(dead)), "matches": list(MATCHES),
    }
    with open(os.path.join(ROOT, "docs", "external_tuning.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
    print(mean.pivot(index="radius", columns="frames", values="spell_f1").round(3))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
