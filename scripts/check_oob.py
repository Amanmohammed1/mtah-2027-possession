import numpy as np, pandas as pd, sys
sys.path.insert(0, ".")
from possession import deadball, score
df = pd.read_parquet("cache/frame_features.parquet")
gt = score.public_gt()
eps = deadball.out_of_bounds_stoppages(df)
L = np.array([b - a for a, b in eps]) / 25
print("episodes", len(eps), "dur s pctl", np.percentile(L, [0, 25, 50, 75, 100]).round(1))
for a, b in eps:
    if 11185 <= a <= 18684: print("public:", a, b, "GT labels", np.unique(gt[a:b], return_counts=True))
