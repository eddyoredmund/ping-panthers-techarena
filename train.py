"""
The whole pipeline in one script.

    python train.py                     # validate, then make a submission
    python train.py --validate-only     # just print our local score
    python train.py --note "added X"    # the note is saved in results_log.csv

Steps:
  1. Load the data.
  2. LOCAL VALIDATION: train on older events, score on the newest 20%.
     This number (AUC) is what we trust.
  3. FINAL MODEL: retrain on all training data, predict evaluation.csv,
     write submissions/submission_<time>.csv.
"""

import argparse
import os
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config as C
from features import add_click_rates, add_count_features, feature_columns


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load(data_dir):
    train = pd.read_csv(os.path.join(data_dir, C.TRAIN_FILE))
    evaluation = pd.read_csv(os.path.join(data_dir, C.EVAL_FILE))
    history = None
    if C.HISTORY_FILE:
        history = pd.read_csv(os.path.join(data_dir, C.HISTORY_FILE))

    for name, df, needed in [("train", train, [C.EVENT_ID, C.USER_ID, C.ITEM_ID, C.LABEL]),
                             ("evaluation", evaluation, [C.ROW_ID, C.EVENT_ID, C.USER_ID, C.ITEM_ID])]:
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise SystemExit(f"{name} is missing columns {missing}. "
                             f"Its columns are {list(df.columns)}. Fix the names in config.py.")

    print(f"train: {len(train):,} rows, click rate {train[C.LABEL].mean():.4f}")
    print(f"evaluation: {len(evaluation):,} rows")
    return train, evaluation, history


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


def split_by_time(train):
    """Newest VALID_FRACTION of events -> validation. Falls back to random."""
    events = train.groupby(C.EVENT_ID)
    if C.TIME and C.TIME in train.columns:
        from features import to_seconds
        order = to_seconds(train[C.TIME]).groupby(train[C.EVENT_ID]).min().sort_values().index
    else:
        print("No time column: using a random split of events instead.")
        order = pd.Series(list(events.groups)).sample(frac=1, random_state=C.SEED).values
    n_valid = int(len(order) * C.VALID_FRACTION)
    valid_events = set(order[-n_valid:])
    is_valid = train[C.EVENT_ID].isin(valid_events)
    return train[~is_valid], train[is_valid]


# ---------------------------------------------------------------------------
# 2. Local validation
# ---------------------------------------------------------------------------
def validate(all_rows):
    train = all_rows[~all_rows["is_eval"]]
    fit_part, valid_part = split_by_time(train)
    fit_part, valid_part = add_click_rates(fit_part, valid_part)
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
    print((importance / importance.sum()).round(3).head(15).to_string())
    return auc, model.best_iteration, feats


# ---------------------------------------------------------------------------
# 3. Final model + submission
# ---------------------------------------------------------------------------
def make_submission(all_rows, n_trees):
    train = all_rows[~all_rows["is_eval"]]
    evaluation = all_rows[all_rows["is_eval"]]
    train, evaluation = add_click_rates(train, evaluation)
    feats = feature_columns(train)

    # A bit more data than in validation -> a few more trees.
    model = fit_model(train[feats], train[C.LABEL], n_trees=int(n_trees * 1.1))
    preds = np.clip(model.predict(evaluation[feats]), 0, 1)

    sub = pd.DataFrame({"rowid": evaluation[C.ROW_ID].values, "predicted_probability": preds})
    check_submission(sub, n_expected=len(evaluation))
    os.makedirs("submissions", exist_ok=True)
    path = f"submissions/submission_{datetime.now():%Y%m%d_%H%M%S}.csv"
    sub.to_csv(path, index=False)
    print(f"\nSaved {path}  -> run the organisers' validate_submission.py on it before uploading.")
    return path


def check_submission(sub, n_expected):
    """Same rules as the FAQ: one row per rowid, finite probabilities in [0, 1]."""
    assert list(sub.columns) == ["rowid", "predicted_probability"]
    assert len(sub) == n_expected, "wrong number of rows"
    assert sub["rowid"].is_unique, "duplicate rowids"
    p = sub["predicted_probability"]
    assert np.isfinite(p).all() and p.between(0, 1).all(), "bad probabilities"


def log_result(auc, n_feats, note, submission):
    row = pd.DataFrame([{"time": f"{datetime.now():%Y-%m-%d %H:%M}", "local_auc": round(auc, 5),
                         "n_features": n_feats, "submission": submission or "", "note": note}])
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

    auc, n_trees, feats = validate(all_rows)
    submission = None if args.validate_only else make_submission(all_rows, n_trees)
    log_result(auc, len(feats), args.note, submission)


if __name__ == "__main__":
    main()
