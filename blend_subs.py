"""
Mixes two submission files by rank (AUC only cares about the order).

    python blend_subs.py submissions/A.csv submissions/B.csv --w 0.5

--w is the weight of the FIRST file (0.5 = equal). Writes submissions/blend_<time>.csv.
"""
import argparse
from datetime import datetime

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--w", type=float, default=0.5)
args = ap.parse_args()

a = pd.read_csv(args.a, dtype={"rowid": str})
b = pd.read_csv(args.b, dtype={"rowid": str})
m = a.merge(b, on="rowid", suffixes=("_a", "_b"))
assert len(m) == len(a) == len(b), "the two files don't have the same rowids"
mix = (args.w * m["predicted_probability_a"].rank(pct=True)
       + (1 - args.w) * m["predicted_probability_b"].rank(pct=True))
out = pd.DataFrame({"rowid": m["rowid"], "predicted_probability": mix.clip(0, 1)})
assert out["rowid"].is_unique and out["predicted_probability"].between(0, 1).all()
path = f"submissions/blend_{datetime.now():%m%d_%H%M%S}.csv"
out.to_csv(path, index=False, float_format="%.7g")
print(f"Saved {path} ({len(out):,} rows, weight {args.w} on {args.a})")
