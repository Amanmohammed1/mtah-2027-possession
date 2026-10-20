"""Alive / dead segmentation with a discrete-observation HMM fitted by EM on the whole match.

Observation channels (conditionally independent given the state):
  live   -- the broadcast shows live play / has cut away / has no frame here
  ball   -- tactical ball invisible / outside the lines / still / slow / fast, in bounds
  speed  -- mean player speed over 3 s, in bins
  trend  -- player speed over the next 3 s minus the last 3 s, in bins
  tag    -- inside the window around a Smart Tagging set-piece start or not

The dead state is a chain of sub-states so that a stoppage has a minimum length and a rough
duration shape instead of the geometric one a single state would give. The model runs at 5 Hz
(every 5th frame) because consecutive frames carry almost no new information, and decoding at
25 Hz with naive-Bayes emissions would make the evidence look five times stronger than it is.
"""

import numpy as np
from numba import njit

from . import data
from .features import nan_smooth

STEP = 5
LIVE_BINS = 3
BALL_BINS = 5
SPEED_EDGES = (0.8, 1.0, 1.2, 1.5, 1.9, 2.4)
TREND_EDGES = (-0.6, -0.2, 0.2, 0.6)
CHANNELS = ("live", "ball", "speed", "trend", "tag")


def observations(df, dx, bt, tags):
    n = len(df)
    live = np.where(~bt.bro_has.values, 2, np.where(bt.bro_live.values, 0, 1))
    vis = df.ball_ok.values
    v = df.ball_v.values
    ball = np.where(~vis, 0, np.where(df.ball_out.values > 0.3, 1,
                                      np.where(v < 1.0, 2, np.where(v < 5.0, 3, 4))))
    spd = nan_smooth(df.spd_all.values.astype(np.float64), 75)
    speed = np.digitize(np.nan_to_num(spd, nan=1.0), SPEED_EDGES)
    trend = np.nan_to_num(dx.spd_fut75.values - dx.spd_past75.values)
    trend = np.digitize(trend, TREND_EDGES)
    tag = np.zeros(n, dtype=np.int64)
    for start in tags.loc[tags.code == "SET PIECES", "start"].values:
        tag[max(0, start - 50):start + 150] = 1
    return np.stack([live, ball, speed, trend, tag], 1).astype(np.int64)


def n_bins():
    return np.array([LIVE_BINS, BALL_BINS, len(SPEED_EDGES) + 1, len(TREND_EDGES) + 1, 2])


# -- model -----------------------------------------------------------------------------------------
#
# States: 0 = alive, 1..K = dead sub-states in a left-to-right chain. Alive -> dead enters at 1;
# dead k -> k+1 is forced until K, which loops on itself and can exit to alive. Every dead
# sub-state shares one emission table, so the chain only shapes durations.


def transition(k_dead, p_stop, p_restart):
    s = k_dead + 1
    T = np.zeros((s, s))
    T[0, 0] = 1 - p_stop
    T[0, 1] = p_stop
    for k in range(1, k_dead):
        T[k, k + 1] = 1.0
    T[k_dead, k_dead] = 1 - p_restart
    T[k_dead, 0] = p_restart
    return T


@njit(cache=True)
def _forward_backward(logB, T, pi):
    n, s = logB.shape
    alpha = np.zeros((n, s))
    beta = np.zeros((n, s))
    scale = np.zeros(n)
    B = np.exp(logB - logB.max())  # global shift keeps it in range; per-frame scaling below
    for t in range(n):
        m = 0.0
        for j in range(s):
            m = max(m, logB[t, j])
        for j in range(s):
            B[t, j] = np.exp(logB[t, j] - m)
    a = pi * B[0]
    scale[0] = a.sum()
    alpha[0] = a / scale[0]
    for t in range(1, n):
        for j in range(s):
            acc = 0.0
            for i in range(s):
                acc += alpha[t - 1, i] * T[i, j]
            alpha[t, j] = acc * B[t, j]
        scale[t] = alpha[t].sum()
        alpha[t] /= scale[t]
    beta[n - 1] = 1.0
    for t in range(n - 2, -1, -1):
        for i in range(s):
            acc = 0.0
            for j in range(s):
                acc += T[i, j] * B[t + 1, j] * beta[t + 1, j]
            beta[t, i] = acc / scale[t + 1]
    gamma = alpha * beta
    for t in range(n):
        gamma[t] /= gamma[t].sum()
    xi = np.zeros((s, s))
    for t in range(n - 1):
        tot = 0.0
        tmp = np.zeros((s, s))
        for i in range(s):
            for j in range(s):
                tmp[i, j] = alpha[t, i] * T[i, j] * B[t + 1, j] * beta[t + 1, j]
                tot += tmp[i, j]
        xi += tmp / tot
    return gamma, xi


class AliveDeadHMM:
    def __init__(self, k_dead=6, p_stop=0.01, p_restart=0.02, cap=np.inf):
        self.k_dead = k_dead
        self.cap = cap
        self.T = transition(k_dead, p_stop, p_restart)
        self.emis = None  # list per channel of (2, bins) probability tables

    def init_from_labels(self, obs, dead, weight=1.0, prior=5.0):
        """Emission tables from labelled frames, smoothed towards uniform."""
        self.emis = []
        for c, nb in enumerate(n_bins()):
            table = np.full((2, nb), prior)
            for state in (0, 1):
                sel = dead == state
                table[state] += weight * np.bincount(obs[sel, c], minlength=nb)
            self.emis.append(table / table.sum(1, keepdims=True))

    def log_emission(self, obs):
        """Naive-Bayes emissions, with each channel's dead-vs-alive log-ratio capped at
        +/- self.cap so that no single signal (e.g. slow players at a slow restart) can veto
        all the others."""
        n = len(obs)
        le = np.zeros((n, 2))
        for c, table in enumerate(self.emis):
            la = np.log(table[0, obs[:, c]])
            ld = np.log(table[1, obs[:, c]])
            le[:, 0] += la
            le[:, 1] += la + np.clip(ld - la, -self.cap, self.cap)
        s = self.k_dead + 1
        out = np.empty((n, s))
        out[:, 0] = le[:, 0]
        out[:, 1:] = le[:, 1:2]
        return out

    def posterior(self, obs_list):
        out = []
        for obs in obs_list:
            pi = np.zeros(self.k_dead + 1)
            pi[0], pi[1] = 0.5, 0.5
            gamma, xi = _forward_backward(self.log_emission(obs), self.T, pi)
            out.append((gamma, xi))
        return out

    def fit(self, obs_list, iters=20, fixed_counts=None, prior=5.0, verbose=False):
        """Baum-Welch on the emission tables and the two free transition probabilities."""
        for it in range(iters):
            post = self.posterior(obs_list)
            tables = [np.full((2, nb), prior) for nb in n_bins()]
            if fixed_counts is not None:
                for c in range(len(tables)):
                    tables[c] += fixed_counts[c]
            xi_tot = np.zeros_like(self.T)
            for (gamma, xi), obs in zip(post, obs_list):
                p_dead = gamma[:, 1:].sum(1)
                for c in range(obs.shape[1]):
                    nb = tables[c].shape[1]
                    tables[c][0] += np.bincount(obs[:, c], weights=1 - p_dead, minlength=nb)
                    tables[c][1] += np.bincount(obs[:, c], weights=p_dead, minlength=nb)
                xi_tot += xi
            self.emis = [t / t.sum(1, keepdims=True) for t in tables]
            p_stop = xi_tot[0, 1] / (xi_tot[0, 0] + xi_tot[0, 1])
            last = self.k_dead
            p_restart = xi_tot[last, 0] / (xi_tot[last, last] + xi_tot[last, 0])
            self.T = transition(self.k_dead, p_stop, p_restart)
            if verbose:
                print("iter", it, "p_stop %.4f p_restart %.4f" % (p_stop, p_restart))
        return self

    def dead_probability(self, obs):
        gamma, _ = self.posterior([obs])[0]
        return gamma[:, 1:].sum(1)


def halves_obs(obs):
    """Subsampled observation sequences per graded half, with their frame indices."""
    seqs, idxs = [], []
    for first, last in data.HALVES:
        idx = np.arange(first, last + 1, STEP)
        seqs.append(obs[idx])
        idxs.append(idx)
    return seqs, idxs


def upsample(values, idx, n):
    out = np.full(n, np.nan)
    for v, i in zip(values, idx):
        frames = np.arange(i[0], min(n, i[-1] + STEP))
        out[frames] = np.interp(frames, i, v)
    return out
