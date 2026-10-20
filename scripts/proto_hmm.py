import numpy as np, pandas as pd, sys
sys.path.insert(0, ".")
from possession import alivehmm as ah, pipeline, broadcast, data, score, features
df, dx = pipeline.load_features()
bt = broadcast.on_tactical(); tags = data.load_smart_tagging()
obs = ah.observations(df, dx, bt, tags)
gt = score.public_gt(); n = len(df); g = df.graded.values
pub = gt != 3
def report(name, p):
    for thr in (0.5,):
        d = p > thr
        tp = (d[pub] & (gt[pub] == 2)).sum()
        segs = features.runs(d & g)
        print("%-22s public D prec %.2f rec %.2f | match D frac %.3f segs %d median %.1fs" % (
            name, tp / max(d[pub].sum(), 1), tp / (gt[pub] == 2).sum(), d[g].mean(), len(segs),
            np.median([b - a for a, b in segs]) / 25))
    s = ""
    for f in range(11185, 18685, 50): s += "%s%d " % ("HAD"[gt[f]], min(9, int(p[f] * 10)))
    print("   ", s)
seqs, idxs = ah.halves_obs(obs)
for k in (15,):
    m = ah.AliveDeadHMM(k_dead=k)
    m.init_from_labels(obs[pub], (gt[pub] == 2).astype(int))
    p0 = ah.upsample([m.dead_probability(s) for s in seqs], idxs, n)
    report("k=%d init(public)" % k, p0)
    m.fit(seqs, iters=15)
    p1 = ah.upsample([m.dead_probability(s) for s in seqs], idxs, n)
    report("k=%d EM(full match)" % k, p1)
    for c, t in zip(ah.CHANNELS, m.emis): print("     ", c, "alive", t[0].round(3), "dead", t[1].round(3))
    print("      T p_stop %.4f p_restart %.4f" % (m.T[0, 1], m.T[k, 0]))
print("\nper 5-min block: ball vis | bro live | mean player speed | EM dead frac")
spd = df.spd_all.values
for h, (a, b) in enumerate(data.HALVES, 1):
    for s in range(a, b, 7500):
        e = min(s + 7500, b)
        print("half %d min %3d  vis %.2f  live %.2f  spd %.2f  dead %.2f" % (h, (s - a) // 1500, df.ball_ok.values[s:e].mean(), bt.bro_live.values[s:e].mean(), np.nanmean(spd[s:e]), (p1[s:e] > .5).mean()))
