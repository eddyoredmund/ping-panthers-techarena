"""
Loading the data and choosing the validation rows.

Every model (train.py, seq_model.py) uses these two functions, so they are all
tested on exactly the same rows. That's what lets blend.py compare and mix them.
"""

import os

import numpy as np
import pandas as pd

import config as C


def load(data_dir):
    """Returns (train, evaluation, history). history is None if not configured."""
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

    # Columns that exist in train but not in evaluation (other than the label)
    # are dangerous: they may describe what happened AFTER the item was shown
    # (e.g. watch time). Using them would be cheating that can't work on eval.
    train_only = [c for c in train.columns if c not in evaluation.columns and c != C.LABEL]
    if train_only:
        print(f"Dropping columns that are only in train: {train_only}")
        train = train.drop(columns=train_only)

    # Remember each training row's original position, so predictions from
    # different models can be matched up later.
    train["train_row"] = np.arange(len(train))

    print(f"train: {len(train):,} rows, click rate {train[C.LABEL].mean():.4f}")
    print(f"evaluation: {len(evaluation):,} rows")
    return train, evaluation, history


def to_seconds(series):
    """Turn a time column (numbers or date strings) into plain seconds."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return pd.to_datetime(series, errors="coerce").astype("int64") / 1e9


def has_time(df):
    return bool(C.TIME) and C.TIME in df.columns


def validation_mask(train):
    """
    True for the rows we hold back as our "practice exam".

    "time":   the newest VALID_FRACTION of events (use when evaluation.csv is
              later in time than train.csv - explore.py tells you).
    "random": a random VALID_FRACTION of events (use when they overlap in time).
    Whole events are always kept together.
    """
    event_ids = train[C.EVENT_ID]
    if C.SPLIT == "time" and has_time(train):
        order = to_seconds(train[C.TIME]).groupby(event_ids).min().sort_values().index.values
    else:
        if C.SPLIT == "time":
            print("No time column: using a random split of events instead.")
        rng = np.random.default_rng(C.SEED)
        order = rng.permutation(event_ids.unique())
    n_valid = int(len(order) * C.VALID_FRACTION)
    return event_ids.isin(set(order[-n_valid:])).values


def save_predictions(name, valid_rows, valid_preds, eval_rows, eval_preds):
    """Saves a model's validation and evaluation predictions to preds/ for blend.py."""
    os.makedirs("preds", exist_ok=True)
    pd.DataFrame({"train_row": valid_rows["train_row"].values.astype(int),
                  "label": valid_rows[C.LABEL].values,
                  "pred": valid_preds}).to_csv(f"preds/{name}_valid.csv", index=False)
    eval_path = f"preds/{name}_eval.csv"
    if eval_rows is not None:
        pd.DataFrame({"rowid": eval_rows[C.ROW_ID].values,
                      "pred": eval_preds}).to_csv(eval_path, index=False)
    elif os.path.exists(eval_path):
        os.remove(eval_path)   # an old file would no longer match this run
