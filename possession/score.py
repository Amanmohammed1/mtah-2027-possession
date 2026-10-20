"""Thin wrapper around the official metric library for quick in-process scoring."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "evaluation"))
import possession_metrics as pm  # noqa: E402

from . import data  # noqa: E402

PUBLIC_GT = os.path.join(os.path.dirname(HERE), "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def public_gt():
    return data.read_labels(PUBLIC_GT)


def score(pred, gt=None):
    gt = public_gt() if gt is None else gt
    r = pm.evaluate(gt, np.asarray(pred, dtype=np.int8))
    return {"spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"], "n_pred": r["n_pred_spells"],
            "n_gt": r["n_gt_spells"], "alive_dead_f1": r["alive_dead_f1"],
            "team_acc": r["team_acc_both_alive"]}


def fmt(s):
    return "spellF1 {spellF1:.4f}  macroF1 {macroF1:.4f}  poss {n_pred}/{n_gt}  aliveF1 {alive_dead_f1:.3f}  team {team_acc:.3f}".format(**s)
