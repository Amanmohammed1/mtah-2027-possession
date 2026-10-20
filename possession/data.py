"""Load the provided match assets into numpy arrays, cached on disk.

Everything is indexed by tactical *video* frame (0..199318), i.e. by submission row.
Coordinates are converted to metres with the origin at the pitch centre:
x in [-L/2, L/2] along the length, y in [-W/2, W/2] across it.
"""

import io
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DATA_DIR = os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "data")
CACHE_DIR = os.environ.get("MTAH_CACHE_DIR") or os.path.join(ROOT, "cache")

GAME = "BAR-ATH"
FPS = 25
N_FRAMES = 199319
PITCH = (105.3, 68.0)
# Graded halves in video frames, inclusive (published in DATA.md).
HALVES = ((11185, 90651), (110808, 190197))
SECOND_HALF_KICKOFF = 113742
HOME, AWAY = 0, 1


def _path(source, suffix):
    folder = os.path.join(DATA_DIR, source)
    if not os.path.isdir(folder):  # the bundle may arrive with upper-case folders
        folder = os.path.join(DATA_DIR, source.upper())
    return os.path.join(folder, "{}_{}{}".format(GAME, source, suffix))


def asset(name, source="tactical"):
    suffix = {"atd_meta": "_atd.xml", "atd_raw": "_atd.txt", "homography": "_homography.csv",
              "smart_tagging": "_smart_tagging.xml"}[name]
    path = _path(source, suffix)
    if not os.path.isfile(path):
        raise IOError("missing {} -- set MTAH_DATA_DIR to the data bundle".format(path))
    return path


def graded_mask(n=N_FRAMES):
    mask = np.zeros(n, dtype=bool)
    for first, last in HALVES:
        mask[first:last + 1] = True
    return mask


def half_of(n=N_FRAMES):
    """0 outside the graded halves, else 1 or 2."""
    out = np.zeros(n, dtype=np.int8)
    for half, (first, last) in enumerate(HALVES, 1):
        out[first:last + 1] = half
    return out


# -- tracking --------------------------------------------------------------------------------------

def _read_meta(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    names = [team.findtext("Name") for team in root.iter("Team")]
    attack = {}
    for team in root.iter("Team"):
        for param in team.iter("ProviderParameter"):
            if param.findtext("Name") == "attack_direction_first_half":
                attack[team.get("id")] = param.findtext("Value")
    team, keeper = [], []
    for player in root.iter("Player"):
        team.append(HOME if player.get("teamId") == team_ids[0] else AWAY)
        params = {p.findtext("Name"): p.findtext("Value") for p in player.iter("ProviderParameter")}
        keeper.append(params.get("position_type") == "Goalkeeper")
    return {
        "team": np.array(team, dtype=np.int8),
        "keeper": np.array(keeper, dtype=bool),
        "names": names,
        "home_attacks_right_1st": attack.get(team_ids[0]) == "left_to_right",
    }


def _parse_raw(raw_path, n_slots):
    with open(raw_path) as handle:
        text = handle.read()
    text = text.replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float32, engine="c")
    values = table.values
    frame_id = values[:, 0].astype(np.int64)
    players = values[:, 1:1 + 3 * n_slots].reshape(len(values), n_slots, 3)
    ball = values[:, 1 + 3 * n_slots:1 + 3 * n_slots + 3]
    return frame_id, players, ball


def load_tracking(source="tactical", use_cache=True):
    """dict with
        px, py   (n, slots) float32 metres, NaN where the slot is not tracked
        ps       (n, slots) speed channel as provided
        bx, by   (n,) ball in metres, NaN when missing
        team     (slots,) 0 home / 1 away
        keeper   (slots,) bool
    """
    cache = os.path.join(CACHE_DIR, "{}_tracking.npz".format(source))
    if use_cache and os.path.isfile(cache):
        data = np.load(cache, allow_pickle=False)
        return {k: data[k] for k in data.files}

    meta = _read_meta(asset("atd_meta", source))
    n_slots = len(meta["team"])
    frame_id, players, ball = _parse_raw(asset("atd_raw", source), n_slots)
    n = int(frame_id.max())
    if not np.array_equal(frame_id, np.arange(1, n + 1)):
        raise ValueError("tracking frame ids are not 1..n")

    length, width = PITCH
    out = {
        "px": (players[:, :, 0] - 0.5) * length,
        "py": (players[:, :, 1] - 0.5) * width,
        "ps": players[:, :, 2],
        "bx": (ball[:, 0] - 0.5) * length,
        "by": (ball[:, 1] - 0.5) * width,
        "team": meta["team"],
        "keeper": meta["keeper"],
        "home_attacks_right_1st": np.array(meta["home_attacks_right_1st"]),
    }
    out = {k: (v.astype(np.float32) if v.dtype.kind == "f" else v) for k, v in out.items()}
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache, **out)
    return out


# -- smart tagging ---------------------------------------------------------------------------------

def load_smart_tagging():
    rows = []
    for inst in ET.parse(asset("smart_tagging")).iter("instance"):
        tags = {l.findtext("group"): l.findtext("text") for l in inst.findall("label")}
        team = (tags.get("Team") or "").strip().lower()
        rows.append({
            "code": (inst.findtext("code") or "").strip().upper(),
            "start": int(round(float(inst.findtext("start")) * FPS)),
            "end": int(round(float(inst.findtext("end")) * FPS)),
            "team": HOME if team.startswith("barc") else (AWAY if team.startswith("at") else -1),
            "type": (tags.get("Type") or "").strip().lower(),
        })
    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)


# -- labels ----------------------------------------------------------------------------------------

LABELS = np.array(["H", "A", "D", "X"])
CODE = {"H": 0, "A": 1, "D": 2, "X": 3}


def read_labels(path):
    labels = pd.read_csv(path, dtype={"label": "object"})["label"].str.strip().str.upper().values
    return np.array([CODE[l] for l in labels], dtype=np.int8)


def write_submission(path, codes):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    labels = LABELS[codes]
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    return path
