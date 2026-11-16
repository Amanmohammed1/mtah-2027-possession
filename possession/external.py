"""Reader for the IDSSE open dataset (DFL/Sportec position data), used only to tune parameters.

Bassek, Rein, Weber & Memmert (2025), An integrated dataset of spatiotemporal and event data in
elite soccer, Scientific Data 12:195, doi:10.6084/m9.figshare.28196177. Licence CC BY 4.0; data
owner Deutsche Fußball Liga (DFL). Not redistributed here: download it from figshare.

Each match comes as a positions XML (one <FrameSet> per player and one for the ball, 25 Hz) and a
match-information XML (which team is home). The ball frames carry BallStatus (1 alive, 0 dead) and
BallPossession (1 home, 2 away) on every frame.
"""

import xml.etree.ElementTree as ET

import numpy as np

H, A, D, X = 0, 1, 2, 3


def team_ids(info_path):
    """(home team id, guest team id) from the match-information XML."""
    root = ET.parse(info_path).getroot()
    for el in root.iter():
        if "HomeTeamId" in el.attrib and "GuestTeamId" in el.attrib:
            return el.attrib["HomeTeamId"], el.attrib["GuestTeamId"]
    raise ValueError("no home/guest team in " + info_path)


def load_positions(pos_path, home_id, guest_id):
    """Per half: dict(frames, bx, by, status, possession, px (n, slots), py, team (slots,)).
    Only the two teams' players and the ball are kept (referees are dropped)."""
    sets = {}  # (half, person) -> (team, list of (N, x, y, status, poss))
    for _, el in ET.iterparse(pos_path, events=("end",)):
        if el.tag != "FrameSet":
            continue
        half = el.attrib["GameSection"]
        team = el.attrib["TeamId"]
        person = el.attrib["PersonId"]
        is_ball = team.upper() == "BALL"
        if not is_ball and team not in (home_id, guest_id):
            el.clear()
            continue
        rows = []
        for fr in el.iter("Frame"):
            a = fr.attrib
            rows.append((int(a["N"]), float(a["X"]), float(a["Y"]),
                         int(a.get("BallStatus", -1)) if is_ball else -1,
                         int(a.get("BallPossession", -1)) if is_ball else -1))
        sets[(half, person)] = ("BALL" if is_ball else team, np.array(rows, dtype=np.float64))
        el.clear()
    halves = {}
    for half in sorted({h for h, _ in sets}):
        ball = [v for (h, _), v in sets.items() if h == half and v[0] == "BALL"]
        if not ball:
            continue
        b = ball[0][1]
        n0, n1 = int(b[:, 0].min()), int(b[:, 0].max())
        n = n1 - n0 + 1
        out = {"frames": np.arange(n0, n1 + 1),
               "bx": np.full(n, np.nan), "by": np.full(n, np.nan),
               "status": np.full(n, -1), "possession": np.full(n, -1)}
        idx = b[:, 0].astype(int) - n0
        out["bx"][idx], out["by"][idx] = b[:, 1], b[:, 2]
        out["status"][idx], out["possession"][idx] = b[:, 3].astype(int), b[:, 4].astype(int)
        players = [(v[0], v[1]) for (h, _), v in sets.items() if h == half and v[0] != "BALL"]
        px = np.full((n, len(players)), np.nan)
        py = np.full((n, len(players)), np.nan)
        team = np.zeros(len(players), dtype=np.int8)
        for j, (tid, rows) in enumerate(players):
            keep = (rows[:, 0] >= n0) & (rows[:, 0] <= n1)
            ii = rows[keep, 0].astype(int) - n0
            px[ii, j], py[ii, j] = rows[keep, 1], rows[keep, 2]
            team[j] = H if tid == home_id else A
        out.update(px=px, py=py, team=team)
        halves[half] = out
    return halves


def labels(half):
    """Per-frame H/A/D from BallStatus and BallPossession (X where either is missing)."""
    lab = np.full(len(half["status"]), X, dtype=np.int8)
    lab[half["status"] == 0] = D
    alive = half["status"] == 1
    lab[alive & (half["possession"] == 1)] = H
    lab[alive & (half["possession"] == 2)] = A
    return lab
