"""
Mixes the predictions of several models into one submission.

    python train.py          # writes preds/lgbm_valid.csv + preds/lgbm_eval.csv
    python seq_model.py      # writes preds/seq_valid.csv  + preds/seq_eval.csv
    python blend.py          # finds the best mix on validation, writes a submission

Why mixing works: two models that make DIFFERENT mistakes cancel some of them
out, like asking two friends with different taste for a recommendation.

AUC only cares about the ORDER of predictions, not their exact values, so we
mix ranks (0 = lowest prediction, 1 = highest) instead of raw probabilities.
That way a model that predicts in a narrow range doesn't get drowned out.
"""

import argparse
import glob
import itertools
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from train import write_submission


def to_rank(x):
    return pd.Series(x).rank(pct=True).values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", help="model names, default: all in preds/")
    args = parser.parse_args()

    names = args.models or sorted(os.path.basename(p)[:-len("_valid.csv")]
                                  for p in glob.glob("preds/*_valid.csv"))
    if len(names) < 2:
        raise SystemExit(f"Need at least 2 models in preds/, found {names}")

    # ---- Line up validation predictions by training row ------------------
    valid, sizes = None, []
    for name in names:
        df = pd.read_csv(f"preds/{name}_valid.csv").rename(columns={"pred": name})
        sizes.append(len(df))
        valid = df if valid is None else valid.merge(df[["train_row", name]], on="train_row")
    if len(valid) < max(sizes):
        print("WARNING: the models were validated on different rows (did SPLIT change "
              "between runs?). Re-run them with the same config.py before trusting this.")
    print(f"{len(valid):,} validation rows shared by: {names}")
    for name in names:
        print(f"  {name:<10} AUC {roc_auc_score(valid['label'], valid[name]):.4f}")
    ranks = {name: to_rank(valid[name]) for name in names}

    # ---- Try every weight combination in steps of 0.1 ----------------------
    steps = np.round(np.arange(0, 1.01, 0.1), 2)
    best_auc, best_w = -1, None
    for w in itertools.product(steps, repeat=len(names)):
        if abs(sum(w) - 1) > 1e-9:
            continue
        mix = sum(wi * ranks[n] for wi, n in zip(w, names))
        auc = roc_auc_score(valid["label"], mix)
        if auc > best_auc:
            best_auc, best_w = auc, w
    print("\nBest mix: " + ", ".join(f"{n} {w:.1f}" for n, w in zip(names, best_w)))
    print(f">>> BLENDED VALIDATION AUC: {best_auc:.4f}")

    # ---- Apply the same weights to the evaluation predictions -----------------
    missing = [n for n in names if not os.path.exists(f"preds/{n}_eval.csv")]
    if missing:
        print(f"\nNo evaluation predictions for {missing} (ran with --validate-only?). Stopping.")
        return
    ev = None
    for name in names:
        df = pd.read_csv(f"preds/{name}_eval.csv").rename(columns={"pred": name})
        ev = df if ev is None else ev.merge(df, on="rowid")
    mix = sum(w * to_rank(ev[n]) for w, n in zip(best_w, names))
    write_submission(ev["rowid"].values, np.clip(mix, 0, 1), "blend")

    from train import log_result
    log_result("blend", best_auc, len(names),
               "weights " + " ".join(f"{n}={w:.1f}" for n, w in zip(names, best_w)), None)


if __name__ == "__main__":
    main()
