"""
Turns raw rows into numbers the model can learn from ("features").

Two kinds of features:

1. COUNT features - "how often has this user / item / pair appeared?"
   These never look at the click label, so they are always safe to compute
   on train + evaluation together.

2. CLICK-RATE features - "how often does this user / item get clicked?"
   These DO use the label, so we must be careful: a row must never see its
   own answer. We use "out-of-fold" encoding for that (explained below).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import config as C

# How strongly to pull click rates of rarely-seen users/items towards the
# overall average. Think of it as "pretend we already saw 20 average rows".
SMOOTHING = 20


# ---------------------------------------------------------------------------
# 1. Count features (no labels used)
# ---------------------------------------------------------------------------
def add_count_features(df, history=None):
    """df = train + evaluation rows stacked together."""
    df = df.copy()
    df["user_item"] = df[C.USER_ID].astype(str) + "_" + df[C.ITEM_ID].astype(str)

    # How many times each user / item / user-item pair was shown overall.
    df["user_n_shown"] = df.groupby(C.USER_ID)[C.ITEM_ID].transform("size")
    df["item_n_shown"] = df.groupby(C.ITEM_ID)[C.USER_ID].transform("size")
    df["user_item_n_shown"] = df.groupby("user_item")[C.USER_ID].transform("size")
    df["user_n_events"] = df.groupby(C.USER_ID)[C.EVENT_ID].transform("nunique")

    # Event-level context: how many candidates were in this event, and how
    # popular this item is compared with the other candidates next to it.
    df["event_size"] = df.groupby(C.EVENT_ID)[C.ITEM_ID].transform("size")
    df["item_pop_rank_in_event"] = df.groupby(C.EVENT_ID)["item_n_shown"].rank(pct=True)

    # Time features (only if there is a time column).
    if C.TIME and C.TIME in df.columns:
        t = to_seconds(df[C.TIME])
        df["_t"] = t
        df = df.sort_values([C.USER_ID, "_t"])
        # Seconds since this user's previous event (big = coming back after a break).
        event_time = df.groupby([C.USER_ID, C.EVENT_ID])["_t"].transform("min")
        df["_event_t"] = event_time
        firsts = df.drop_duplicates([C.USER_ID, C.EVENT_ID])[[C.USER_ID, C.EVENT_ID, "_event_t"]]
        firsts = firsts.sort_values([C.USER_ID, "_event_t"])
        firsts["secs_since_prev_event"] = firsts.groupby(C.USER_ID)["_event_t"].diff()
        # Which number visit this is for the user (1st, 2nd, ...).
        firsts["user_event_number"] = firsts.groupby(C.USER_ID).cumcount() + 1
        df = df.merge(firsts[[C.USER_ID, C.EVENT_ID, "secs_since_prev_event", "user_event_number"]],
                      on=[C.USER_ID, C.EVENT_ID], how="left")
        if pd.api.types.is_datetime64_any_dtype(pd.to_datetime(df[C.TIME], errors="coerce")) \
                and not pd.api.types.is_numeric_dtype(df[C.TIME]):
            dt = pd.to_datetime(df[C.TIME], errors="coerce")
            df["hour"] = dt.dt.hour
            df["day_of_week"] = dt.dt.dayofweek
        df = df.drop(columns=["_t", "_event_t"])

    # Optional history file: what the user did before the competition window.
    if history is not None:
        h = history
        h_user = h.groupby(C.HIST_USER_ID).size().rename("hist_user_n")
        h_item = h.groupby(C.HIST_ITEM_ID).size().rename("hist_item_n")
        h_pair = h.groupby([C.HIST_USER_ID, C.HIST_ITEM_ID]).size().rename("hist_user_item_n")
        df = df.merge(h_user, left_on=C.USER_ID, right_index=True, how="left")
        df = df.merge(h_item, left_on=C.ITEM_ID, right_index=True, how="left")
        df = df.merge(h_pair, left_on=[C.USER_ID, C.ITEM_ID], right_index=True, how="left")
        for col in ["hist_user_n", "hist_item_n", "hist_user_item_n"]:
            df[col] = df[col].fillna(0)

    return df


def to_seconds(series):
    """Turn a time column (numbers or date strings) into plain seconds."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return pd.to_datetime(series, errors="coerce").astype("int64") / 1e9


# ---------------------------------------------------------------------------
# 2. Click-rate features (use labels -> must avoid leaking the answer)
# ---------------------------------------------------------------------------
CLICK_RATE_KEYS = {
    "user_ctr": C.USER_ID,       # does this user click a lot in general?
    "item_ctr": C.ITEM_ID,       # is this item clicked a lot in general?
    "user_item_ctr": "user_item",  # has this user clicked this item before?
}


def click_rate(fit_rows, apply_rows, key, prior):
    """Click rate of `key` learned on fit_rows, looked up for apply_rows."""
    stats = fit_rows.groupby(key)[C.LABEL].agg(["sum", "count"])
    clicks = apply_rows[key].map(stats["sum"]).fillna(0)
    shows = apply_rows[key].map(stats["count"]).fillna(0)
    return ((clicks + prior * SMOOTHING) / (shows + SMOOTHING)).values


def add_click_rates(train_rows, apply_rows, n_folds=5):
    """
    For apply_rows (validation or evaluation): learn rates from ALL train rows.

    For train_rows: "out-of-fold". Split train into 5 pieces; each piece gets
    rates learned only from the other 4. Like marking a student's homework
    with an answer key that wasn't written using their own homework.
    Pieces are split by event, so candidates from one event stay together.
    """
    train_rows = train_rows.copy()
    apply_rows = apply_rows.copy()
    prior = train_rows[C.LABEL].mean()

    for name, key in CLICK_RATE_KEYS.items():
        apply_rows[name] = click_rate(train_rows, apply_rows, key, prior)

        oof = np.zeros(len(train_rows))
        folds = GroupKFold(n_splits=n_folds)
        for fit_idx, enc_idx in folds.split(train_rows, groups=train_rows[C.EVENT_ID]):
            oof[enc_idx] = click_rate(train_rows.iloc[fit_idx], train_rows.iloc[enc_idx], key, prior)
        train_rows[name] = oof

    # Compare each candidate with the others in the same event.
    for df in (train_rows, apply_rows):
        df["item_ctr_rank_in_event"] = df.groupby(C.EVENT_ID)["item_ctr"].rank(pct=True)
        df["item_ctr_minus_event_mean"] = df["item_ctr"] - df.groupby(C.EVENT_ID)["item_ctr"].transform("mean")

    return train_rows, apply_rows


# Columns that are NOT features (ids, labels, raw text).
NOT_FEATURES = {C.ROW_ID, C.EVENT_ID, C.USER_ID, C.ITEM_ID, C.TIME, C.LABEL,
                "user_item", "is_eval"}


def feature_columns(df):
    return [c for c in df.columns
            if c not in NOT_FEATURES and pd.api.types.is_numeric_dtype(df[c])]
