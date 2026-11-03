"""
Model 1: LightGBM on hand-made features. Our main model.

    python train.py                     # validate, then make a submission
    python train.py --validate-only     # just print our local score
    python train.py --note "added X"    # the note is saved in results_log.csv

Steps:
  1. Load the data and build features.
  2. LOCAL VALIDATION: train on the other 80% of events, score on the held-back
     20%. This number (AUC) is what we trust.
  3. FINAL MODEL: retrain on all training data, predict evaluation.csv,
     write submissions/submission_<time>.csv.
  Predictions are also saved in preds/ so blend.py can mix models.
"""

import argparse
import os
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config as C
from data import load, save_predictions, validation_mask
from features import add_click_rates, add_count_features, feature_columns


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def fit_model(X, y, X_val=None, y_val=None, n_trees=None):
    """Train LightGBM. With validation data, stop when it stops improving."""
    train_set = lgb.Dataset(X, y)
    if X_val is not None:
        valid_set = lgb.Dataset(X_val, y_val)
        return lgb.train({**C.LGBM_PARAMS, "metric": "auc", "seed": C.SEED},
                         train_set, num_boost_round=C.MAX_TREES, valid_sets=[valid_set],
                         callbacks=[lgb.early_stopping(C.EARLY_STOPPING, verbose=False),
                                    lgb.log_evaluation(200)])
    return lgb.train({**C.LGBM_PARAMS, "seed": C.SEED}, train_set, num_boost_round=n_trees)


# ---------------------------------------------------------------------------
# 2. Local validation
# ---------------------------------------------------------------------------
def validate(all_rows):
    train = all_rows[~all_rows["is_eval"]]
    is_valid = validation_mask(train)
    fit_part, valid_part = add_click_rates(train[~is_valid], train[is_valid])
    feats = feature_columns(fit_part)

    print(f"\nLocal validation: fit on {len(fit_part):,} rows, score on {len(valid_part):,} rows")
    model = fit_model(fit_part[feats], fit_part[C.LABEL], valid_part[feats], valid_part[C.LABEL])
    preds = model.predict(valid_part[feats], num_iteration=model.best_iteration)
    auc = roc_auc_score(valid_part[C.LABEL], preds)

    # Simple baselines so we know the model is actually adding something.
    print("\nSingle-feature AUCs (sanity check):")
    for f in ["item_ctr", "user_ctr", "user_item_ctr"]:
        print(f"  {f:<16} {roc_auc_score(valid_part[C.LABEL], valid_part[f]):.4f}")
    print(f"\n>>> LOCAL VALIDATION AUC: {auc:.4f}  (trees used: {model.best_iteration})")

    importance = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("\nMost useful features:")
    print((importance / importance.sum()).round(3).head(20).to_string())
    return auc, model.best_iteration, feats, valid_part, preds


# ---------------------------------------------------------------------------
# 3. Final model + submission
# ---------------------------------------------------------------------------
def predict_evaluation(all_rows, n_trees):
    train = all_rows[~all_rows["is_eval"]]
    evaluation = all_rows[all_rows["is_eval"]]
    train, evaluation = add_click_rates(train, evaluation)
    feats = feature_columns(train)

    # A bit more data than in validation -> a few more trees.
    model = fit_model(train[feats], train[C.LABEL], n_trees=int(n_trees * 1.1))
    return evaluation, np.clip(model.predict(evaluation[feats]), 0, 1)


def write_submission(rowids, preds, name="submission"):
    sub = pd.DataFrame({"rowid": rowids, "predicted_probability": preds})
    check_submission(sub)
    os.makedirs("submissions", exist_ok=True)
    path = f"submissions/{name}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    sub.to_csv(path, index=False)
    print(f"\nSaved {path}  -> run the organisers' validate_submission.py on it before uploading.")
    return path


def check_submission(sub):
    """Same rules as the FAQ: one row per rowid, finite probabilities in [0, 1]."""
    assert list(sub.columns) == ["rowid", "predicted_probability"]
    assert sub["rowid"].is_unique, "duplicate rowids"
    p = sub["predicted_probability"]
    assert np.isfinite(p).all() and p.between(0, 1).all(), "bad probabilities"


def log_result(model_name, auc, n_feats, note, submission):
    row = pd.DataFrame([{"time": f"{datetime.now():%Y-%m-%d %H:%M}", "model": model_name,
                         "local_auc": round(auc, 5), "n_features": n_feats,
                         "submission": submission or "", "note": note}])
    path = "results_log.csv"
    row.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=C.DATA_DIR)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    train, evaluation, history = load(args.data_dir)
    train["is_eval"] = False
    evaluation["is_eval"] = True
    all_rows = add_count_features(pd.concat([train, evaluation], ignore_index=True), history)

    auc, n_trees, feats, valid_part, valid_preds = validate(all_rows)
    eval_rows, eval_preds, submission = None, None, None
    if not args.validate_only:
        eval_rows, eval_preds = predict_evaluation(all_rows, n_trees)
        assert len(eval_rows) == len(evaluation), "lost evaluation rows"
        submission = write_submission(eval_rows[C.ROW_ID].values, eval_preds, "lgbm")
    save_predictions("lgbm", valid_part, valid_preds, eval_rows, eval_preds)
    log_result("lgbm", auc, len(feats), args.note, submission)


if __name__ == "__main__":
    main()
