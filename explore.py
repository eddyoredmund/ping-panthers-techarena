"""
Run this FIRST on the real data. It answers the questions that decide how we
set up everything else:

    python explore.py                    # real data in data/
    python explore.py --data-dir fake_data

It does not train anything - it just looks.
"""

import argparse
import os

import numpy as np
import pandas as pd

import config as C


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def describe_file(name, df):
    section(f"{name}: {len(df):,} rows, {df.shape[1]} columns")
    summary = pd.DataFrame({
        "type": df.dtypes.astype(str),
        "unique": df.nunique(),
        "missing_%": (df.isna().mean() * 100).round(1),
        "example": df.iloc[0],
    })
    print(summary.to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=C.DATA_DIR)
    args = parser.parse_args()

    print("Files in", args.data_dir, ":", sorted(os.listdir(args.data_dir)))
    train = pd.read_csv(os.path.join(args.data_dir, C.TRAIN_FILE))
    evaluation = pd.read_csv(os.path.join(args.data_dir, C.EVAL_FILE))
    describe_file(C.TRAIN_FILE, train)
    describe_file(C.EVAL_FILE, evaluation)

    # ---- Do the column names match config.py? -----------------------------
    section("1. Do the column names in config.py match?")
    for col in [C.ROW_ID, C.EVENT_ID, C.USER_ID, C.ITEM_ID, C.TIME, C.LABEL]:
        if col is None:
            continue
        print(f"  {col:<12} train: {'yes' if col in train else 'NO'}   "
              f"evaluation: {'yes' if col in evaluation else 'NO'}")
    extra = [c for c in train.columns if c not in
             {C.ROW_ID, C.EVENT_ID, C.USER_ID, C.ITEM_ID, C.TIME, C.LABEL}]
    print(f"\n  Other columns (possible extra features!): {extra}")
    only_train = [c for c in extra if c not in evaluation.columns]
    if only_train:
        print(f"  WARNING - only in train, can't be used as features: {only_train}")
    text_cols = [c for c in extra if c in evaluation.columns and train[c].dtype == object]
    if text_cols:
        print(f"  Text columns (need converting to numbers before the model can use them): {text_cols}")

    if C.LABEL not in train or C.USER_ID not in train or C.EVENT_ID not in train:
        print("\nFix the column names in config.py, then run this again.")
        return

    # ---- Clicks -------------------------------------------------------------
    section("2. Clicks")
    print(f"  Overall click rate: {train[C.LABEL].mean():.4f}")
    per_event = train.groupby(C.EVENT_ID)[C.LABEL].agg(["size", "sum"])
    print(f"  Candidates per event: min {per_event['size'].min()}, "
          f"median {per_event['size'].median():.0f}, max {per_event['size'].max()}")
    print("  Clicks per event (share of events):")
    print((per_event["sum"].clip(upper=5).value_counts(normalize=True).sort_index()
           .rename(lambda k: f"{k}{'+' if k == 5 else ''} clicks").round(3)).to_string())
    print("  -> If almost every event has exactly 1 click, it's a 'pick one' problem.")
    print("     If many have 0 clicks, predicting WHICH events get clicks matters a lot")
    print("     for global AUC (user/time features).")

    clicks_by_size = train.groupby(per_event["size"].reindex(train[C.EVENT_ID]).values)[C.LABEL].mean()
    print("\n  Click rate by number of candidates in the event:")
    print(clicks_by_size.round(4).head(15).to_string())

    if "position" in train.columns:
        print("\n  Click rate by position:")
        print(train.groupby("position")[C.LABEL].mean().round(4).head(20).to_string())

    # ---- Overlap between train and evaluation --------------------------------
    section("3. How similar is evaluation.csv to train.csv?")
    for col in [C.USER_ID, C.ITEM_ID]:
        seen = evaluation[col].isin(set(train[col])).mean()
        print(f"  Evaluation rows whose {col} appears in train: {seen:.1%}")
    seen_events = evaluation[C.EVENT_ID].isin(set(train[C.EVENT_ID])).mean()
    print(f"  Evaluation rows whose {C.EVENT_ID} appears in train: {seen_events:.1%}")
    print("  -> Low user overlap = new users: rely less on user click rates.")
    print("     Low item overlap = new items: rely less on item click rates.")

    # ---- Time -------------------------------------------------------------
    section("4. Time - which validation split should we use?")
    if C.TIME and C.TIME in train and C.TIME in evaluation:
        from data import to_seconds
        tt, te = to_seconds(train[C.TIME]), to_seconds(evaluation[C.TIME])
        fmt = (lambda s: pd.to_datetime(s, unit="s")) if not pd.api.types.is_numeric_dtype(train[C.TIME]) \
            else (lambda s: s)
        print(f"  train      from {fmt(tt.min())} to {fmt(tt.max())}")
        print(f"  evaluation from {fmt(te.min())} to {fmt(te.max())}")
        overlap = ((te >= tt.min()) & (te <= tt.max())).mean()
        print(f"  Share of evaluation rows inside train's time range: {overlap:.1%}")
        if overlap < 0.2:
            print("  -> Evaluation comes AFTER train. Use SPLIT = \"time\" in config.py.")
        elif overlap > 0.8:
            print("  -> They overlap in time. Use SPLIT = \"random\" in config.py.")
        else:
            print("  -> Partly overlapping. Try both splits; keep the one whose score")
            print("     best matches the leaderboard.")

        # Does the click rate drift over time?
        bins = pd.qcut(tt, 10, labels=False, duplicates="drop")
        print("\n  Click rate in train, oldest -> newest tenth:")
        print("  " + "  ".join(f"{v:.3f}" for v in train.groupby(bins)[C.LABEL].mean()))
    else:
        print("  No time column. Use SPLIT = \"random\".")

    # ---- Users ---------------------------------------------------------------
    section("5. Users and items")
    user_rates = train.groupby(C.USER_ID)[C.LABEL].agg(["mean", "size"])
    print(f"  Users: {len(user_rates):,}, events per user (median): "
          f"{train.groupby(C.USER_ID)[C.EVENT_ID].nunique().median():.0f}")
    busy = user_rates[user_rates["size"] >= 20]["mean"]
    if len(busy):
        print(f"  Click rate of users with 20+ rows: 10th pct {busy.quantile(.1):.3f}, "
              f"median {busy.median():.3f}, 90th pct {busy.quantile(.9):.3f}")
        print("  -> A wide spread means user_ctr will be a strong feature.")
    item_rows = train.groupby(C.ITEM_ID).size()
    print(f"  Items: {len(item_rows):,}, rows per item (median): {item_rows.median():.0f}")
    print(f"  Top 1% of items take {item_rows.nlargest(max(1, len(item_rows) // 100)).sum() / len(train):.1%} of rows")

    # ---- History file ----------------------------------------------------
    others = [f for f in os.listdir(args.data_dir)
              if f.endswith(".csv") and f not in {C.TRAIN_FILE, C.EVAL_FILE}]
    if others:
        section("6. Other files")
        for f in others:
            df = pd.read_csv(os.path.join(args.data_dir, f), nrows=5)
            print(f"  {f}: columns {list(df.columns)}")
        print("  -> If one is an interaction history, set HISTORY_FILE in config.py.")


if __name__ == "__main__":
    main()
