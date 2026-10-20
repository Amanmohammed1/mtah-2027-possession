"""Viterbi decoding of per-frame evidence into H / A / D intervals."""

import numpy as np
from numba import njit

H, A, D = 0, 1, 2


@njit(cache=True)
def _viterbi(emis, trans, init):
    n, k = emis.shape
    score = np.empty(k)
    back = np.empty((n, k), dtype=np.int8)
    for s in range(k):
        score[s] = init[s] + emis[0, s]
    new = np.empty(k)
    for t in range(1, n):
        for s in range(k):
            best = -1e300
            arg = 0
            for r in range(k):
                v = score[r] + trans[r, s]
                if v > best:
                    best = v
                    arg = r
            new[s] = best + emis[t, s]
            back[t, s] = arg
        for s in range(k):
            score[s] = new[s]
    path = np.empty(n, dtype=np.int8)
    best = -1e300
    arg = 0
    for s in range(k):
        if score[s] > best:
            best = score[s]
            arg = s
    path[n - 1] = arg
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def transition_matrix(switch_team, switch_dead, switch_restart):
    """Log transition costs (negative) between H, A, D for one frame step."""
    t = np.zeros((3, 3))
    t[H, A] = t[A, H] = -switch_team
    t[H, D] = t[A, D] = -switch_dead
    t[D, H] = t[D, A] = -switch_restart
    return t


def viterbi(emis, trans, init=None):
    if init is None:
        init = np.zeros(emis.shape[1])
    return _viterbi(np.ascontiguousarray(emis, dtype=np.float64), trans.astype(np.float64),
                    np.asarray(init, dtype=np.float64))


def decode_halves(emis, trans, halves, n, first_state=None):
    """Decode each graded half separately; X (3) everywhere else."""
    out = np.full(n, 3, dtype=np.int8)
    for i, (first, last) in enumerate(halves):
        init = np.zeros(3)
        if first_state is not None and first_state[i] is not None:
            init[:] = -1e6
            init[first_state[i]] = 0.0
        out[first:last + 1] = viterbi(emis[first:last + 1], trans, init)
    return out
