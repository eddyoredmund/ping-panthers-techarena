#!/usr/bin/env python3
"""
Ping Panthers - TechArena 2026 click prediction. THE MAIN SCRIPT.

    python solution.py --data-dir data                 # validate, then write a submission
    python solution.py --data-dir data --skip-valid    # submission only (faster)
    python solution.py --data-dir data --valid-only    # only the local score
    python solution.py --data-dir data --neg-frac 1.0  # use every row (slower, a bit better)

The first run reads the 7 GB CSV and builds features (slow, several minutes).
They are cached in cache/, so later runs start training almost straight away.
Use --rebuild after changing any feature code.

THE IDEA
Each event shows 10 items and at most ONE gets clicked, so the items compete.
For every row we work out "what we knew just before this event" and let
LightGBM learn the rest:

  1. Click rates "as of" the event time: user, item, user+item. Only labels
     from EARLIER events are used, so training rows look exactly like
     evaluation rows (whose past is the whole training week).
  2. History: how many recent clicks, how long since the last one, was THIS
     candidate clicked before (repeat behaviour), and how long ago.
  3. Taste (matrix factorisation): an SVD of the user x item click matrix, rebuilt
     every day from clicks BEFORE that day. It gives every item a "fingerprint".
     We then measure how close the candidate is to the user's taste and history.
  4. Competition: how each candidate compares with the other 9 in its event.

Validation copies the official setup: fit on days 0-5, test on the FIRST event
of each user on day 6.
"""

import argparse
import json
import os
import pickle
import time
from datetime import datetime
from itertools import chain

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.utils.extmath import randomized_svd

DAY = 86400
START = 1737331200          # 2025-01-20 00:00 UTC, first training day
N_TRAIN_DAYS = 7            # training ends at the cutoff, 2025-01-27 00:00 UTC
CAND = 10                   # candidates per event
SVD_DIM = 48
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


# =============================================================================
# 1. LOADING (chunked, the file is 7 GB)
# =============================================================================
def read_file(path, labelled):
    """Rows (event, item, label/rowid) + one line per event (user, time, history)."""
    ids = ["event_id", "user_id", "item_id", "history_item_ids", "history_timestamps"]
    dtypes = {c: str for c in ids + ["rowid"]}
    rows, events = [], []
    row_cols = ["event_id", "item_id"] + (["clicked"] if labelled else ["rowid"])
    ev_cols = ["event_id", "user_id", "timestamp", "time_bucket", "day_type",
               "history_item_ids", "history_timestamps"]
    n = 0
    for chunk in pd.read_csv(path, dtype=dtypes, chunksize=1_000_000):
        rows.append(chunk[row_cols])
        events.append(chunk.loc[~chunk["event_id"].duplicated(), ev_cols])  # history once per event
        n += len(chunk)
        log(f"  {os.path.basename(path)}: {n:,} rows read")
    rows = pd.concat(rows, ignore_index=True)
    events = pd.concat(events, ignore_index=True).drop_duplicates("event_id", ignore_index=True)
    return rows, events


def load(data_dir):
    log("Reading train.csv (slow the first time)...")
    tr_rows, tr_ev = read_file(os.path.join(data_dir, "train.csv"), True)
    log("Reading evaluation.csv...")
    ev_rows, ev_ev = read_file(os.path.join(data_dir, "evaluation.csv"), False)
    tr_ev["is_eval"], ev_ev["is_eval"] = 0, 1
    tr_rows["rowid"], ev_rows["clicked"] = "", -1

    events = pd.concat([tr_ev, ev_ev], ignore_index=True)
    assert events["event_id"].is_unique, "event_id repeats between train and evaluation"
    rows = pd.concat([tr_rows, ev_rows], ignore_index=True)
    del tr_rows, ev_rows, tr_ev, ev_ev

    # Every event must have exactly 10 rows: we store candidates as an [events x 10] grid.
    rows["ev"] = pd.Index(events["event_id"]).get_indexer(rows["event_id"])
    counts = np.bincount(rows["ev"].values, minlength=len(events))
    assert (counts == CAND).all(), f"events without exactly {CAND} rows: {(counts != CAND).sum()}"
    rows = rows.sort_values("ev", kind="stable", ignore_index=True)

    log("Parsing histories...")
    h_items = [json.loads(s) for s in events["history_item_ids"].values]
    h_ts = [json.loads(s) for s in events["history_timestamps"].values]
    h_len = np.array([len(h) for h in h_items], dtype=np.int32)
    flat_items = pd.Series(list(chain.from_iterable(h_items)), dtype=str)
    flat_ts = np.fromiter(chain.from_iterable(h_ts), dtype=np.int64, count=int(h_len.sum()))
    del h_items, h_ts

    # Opaque ids -> integer codes.
    item_vocab = pd.Index(pd.unique(pd.concat([rows["item_id"], flat_items], ignore_index=True)))
    user_codes, user_vocab = pd.factorize(events["user_id"])

    E, L = len(events), max(int(h_len.max()), 1)
    D = {
        "n_users": len(user_vocab), "n_items": len(item_vocab), "L": L,
        "user": user_codes.astype(np.int32),
        "ts": events["timestamp"].values.astype(np.int64),
        "time_bucket": events["time_bucket"].values.astype(np.int8),
        "day_type": events["day_type"].values.astype(np.int8),
        "is_eval": events["is_eval"].values.astype(bool),
        "h_len": h_len,
        "C": item_vocab.get_indexer(rows["item_id"]).reshape(E, CAND).astype(np.int32),
        "y": rows["clicked"].values.reshape(E, CAND).astype(np.int8),
        "rowid": rows["rowid"].values.reshape(E, CAND),
    }
    # History as an [events x L] grid, oldest -> newest, padded with -1.
    ev_of_entry = np.repeat(np.arange(E), h_len)
    pos = np.arange(len(flat_ts)) - np.repeat(np.cumsum(h_len) - h_len, h_len)
    D["H"] = np.full((E, L), -1, np.int32)
    D["H"][ev_of_entry, pos] = item_vocab.get_indexer(flat_items)
    D["Ht"] = np.zeros((E, L), np.int64)
    D["Ht"][ev_of_entry, pos] = flat_ts
    log(f"Loaded {E:,} events ({D['is_eval'].sum():,} evaluation), "
        f"{D['n_users']:,} users, {D['n_items']:,} items")
    return D


# =============================================================================
# 2. FEATURES
# =============================================================================
def asof_counts(key, ts, n, c):
    """For each row: sum of n and c over rows with the same key and an EARLIER time."""
    df = pd.DataFrame({"k": key, "t": ts, "n": n, "c": c})
    g = df.groupby(["k", "t"], sort=True)[["n", "c"]].sum()
    before = g.groupby(level=0).cumsum() - g
    out = df[["k", "t"]].merge(before, left_on=["k", "t"], right_index=True, how="left")
    return out["n"].values.astype(np.float32), out["c"].values.astype(np.float32)


def all_clicks(D):
    """Every known click (user, item, time): from histories + labelled training rows."""
    E = len(D["ts"])
    mask = D["H"] >= 0
    u = np.repeat(D["user"], mask.sum(1))
    hist = pd.DataFrame({"u": u, "i": D["H"][mask], "t": D["Ht"][mask]})
    e, j = np.nonzero(D["y"] == 1)
    lab = pd.DataFrame({"u": D["user"][e], "i": D["C"][e, j], "t": D["ts"][e]})
    clicks = pd.concat([hist, lab], ignore_index=True).drop_duplicates()
    log(f"  {len(clicks):,} distinct clicks known (incl. warm-up period)")
    return clicks


def svd_for_day(clicks, cutoff, n_users, n_items):
    """Matrix factorisation of all clicks before `cutoff`."""
    c = clicks[clicks["t"] < cutoff]
    X = sparse.csr_matrix((np.ones(len(c), np.float32), (c["u"].values, c["i"].values)),
                          shape=(n_users, n_items))
    X.sum_duplicates()
    X.data = np.log1p(X.data)
    item_users = np.asarray((X > 0).sum(0)).ravel()
    user_clicks = np.asarray(X.sum(1)).ravel()
    idf = np.log((1 + (user_clicks > 0).sum()) / (1 + item_users)).astype(np.float32)
    X = X @ sparse.diags(idf)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
    X = sparse.diags(1 / np.maximum(norms, 1e-9)) @ X
    U, S, VT = randomized_svd(X, SVD_DIM, n_iter=5, random_state=0)
    user_vec = (U * S).astype(np.float32)
    item_vec = VT.T.astype(np.float32)
    item_norm = item_vec / np.maximum(np.linalg.norm(item_vec, axis=1, keepdims=True), 1e-9)
    # Row of zeros at the end, used for padding (index -1).
    pad = lambda a: np.vstack([a, np.zeros((1, a.shape[1]), np.float32)])
    return pad(user_vec), pad(item_vec), pad(item_norm), item_users, user_clicks


def transitions_for_day(clicks, cutoff, n_items):
    """'People who clicked A clicked B NEXT': counts of consecutive click pairs before cutoff."""
    c = clicks[clicks["t"] < cutoff].sort_values(["u", "t"], kind="stable")
    u, i = c["u"].values, c["i"].values.astype(np.int64)
    same = u[1:] == u[:-1]
    a, b = i[:-1][same], i[1:][same]
    pairs = pd.Series(a * n_items + b).value_counts()
    out_deg = np.bincount(a, minlength=n_items + 1).astype(np.float32)
    return pd.Index(pairs.index), pairs.values.astype(np.float32), out_deg


def build_features(D):
    E, L = len(D["ts"]), D["L"]
    ts, user, C, H, Ht, h_len = D["ts"], D["user"], D["C"], D["H"], D["Ht"], D["h_len"]
    day = np.clip((ts - START) // DAY, 0, N_TRAIN_DAYS).astype(np.int32)   # 7 = evaluation
    F = {}                                                    # name -> [E, 10] array
    ev = lambda a: np.repeat(np.asarray(a, np.float32)[:, None], CAND, 1)  # event-level -> grid

    # ---- Event-level: time and history ------------------------------------
    log("Features: time and history...")
    valid = H >= 0
    last_t = np.where(h_len > 0, Ht[np.arange(E), np.maximum(h_len - 1, 0)], 0)
    first_t = np.where(h_len > 0, Ht[:, 0], 0)
    F["hour"] = ev((ts % DAY) // 3600)
    F["day_of_week"] = ev(((ts // DAY) + 3) % 7)
    F["time_bucket"], F["day_type"] = ev(D["time_bucket"]), ev(D["day_type"])
    F["hist_len"] = ev(h_len)
    F["secs_since_last_click"] = ev(np.where(h_len > 0, ts - last_t, np.nan))
    F["hist_span"] = ev(np.where(h_len > 1, last_t - first_t, np.nan))
    F["hist_clicks_per_day"] = ev(np.where(h_len > 1, h_len / ((ts - first_t) / DAY + 0.01), np.nan))
    for name, secs in [("1h", 3600), ("6h", 6 * 3600), ("24h", DAY), ("3d", 3 * DAY)]:
        F[f"hist_clicks_{name}"] = ev(((Ht > ts[:, None] - secs) & valid).sum(1))
    F["hist_distinct_items"] = ev([len(set(r[r >= 0])) for r in H])

    # Previous visits of the user (train + evaluation timeline, past only).
    order = np.lexsort((ts, user))
    prev_ts = np.full(E, np.nan)
    same = np.r_[False, user[order][1:] == user[order][:-1]]
    prev_ts[order[1:]] = np.where(same[1:], ts[order][:-1], np.nan)
    F["secs_since_prev_event"] = ev(ts - prev_ts)
    n_before = np.zeros(E)
    n_before[order] = pd.Series(user[order]).groupby(user[order]).cumcount().values
    F["user_prev_events"] = ev(n_before)

    # ---- As-of click rates (labels from earlier events only) ------------------
    log("Features: as-of click rates...")
    r_ts = np.repeat(ts, CAND)
    r_user = np.repeat(user, CAND).astype(np.int64)
    r_item = C.ravel().astype(np.int64)
    y = D["y"].ravel()
    lab = (y >= 0).astype(np.float32)
    clk = np.where(y == 1, 1, 0).astype(np.float32)
    global_ctr = clk.sum() / lab.sum()
    for name, key, k in [("user", r_user, 30), ("item", r_item, 50),
                         ("ui", r_user * D["n_items"] + r_item, 5)]:
        n, c = asof_counts(key, r_ts, lab, clk)
        F[f"{name}_prev_shown"] = n.reshape(E, CAND)
        F[f"{name}_prev_clicks"] = c.reshape(E, CAND)
        F[f"{name}_ctr"] = ((c + k * global_ctr) / (n + k)).reshape(E, CAND)
    F["ui_shown_not_clicked"] = F["ui_prev_shown"] - F["ui_prev_clicks"]
    F["user_prev_shown"] /= CAND   # = number of earlier labelled events

    # Item popularity on the previous day (catches trends).
    r_day = np.repeat(day, CAND).astype(np.int64)
    g = pd.DataFrame({"k": r_item * 16 + r_day, "n": lab, "c": clk}).groupby("k")[["n", "c"]].sum()
    prev = pd.Index(g.index).get_indexer(r_item * 16 + r_day - 1)
    pn = np.where(prev >= 0, g["n"].values[prev], 0)
    pc = np.where(prev >= 0, g["c"].values[prev], 0)
    F["item_prevday_shown"] = pn.reshape(E, CAND).astype(np.float32)
    F["item_prevday_ctr"] = ((pc + 50 * global_ctr) / (pn + 50)).reshape(E, CAND).astype(np.float32)
    first_seen = pd.Series(r_ts).groupby(r_item).transform("min").values
    F["item_age_days"] = ((r_ts - first_seen) / DAY).reshape(E, CAND).astype(np.float32)

    # Click rates by time of day / weekend (the organisers gave us these columns
    # for a reason). Shrunk towards the item's / user's overall rate.
    r_tb = np.repeat(D["time_bucket"], CAND).astype(np.int64)
    r_dt = np.repeat(D["day_type"], CAND).astype(np.int64)
    for name, key, base, k in [("item_tb", r_item * 4 + r_tb, "item_ctr", 30),
                               ("item_dt", r_item * 2 + r_dt, "item_ctr", 30),
                               ("user_tb", r_user * 4 + r_tb, "user_ctr", 20),
                               ("user_dt", r_user * 2 + r_dt, "user_ctr", 20)]:
        n, c = asof_counts(key, r_ts, lab, clk)
        F[f"{name}_ctr"] = ((c.reshape(E, CAND) + k * F[base]) / (n.reshape(E, CAND) + k)).astype(np.float32)
        F[f"{name}_lift"] = (F[f"{name}_ctr"] / F[base]).astype(np.float32)

    # When was this item last SHOWN to this user (clicked or not)?
    key = r_user * D["n_items"] + r_item
    o = np.lexsort((r_ts, key))
    prev = np.full(len(key), np.nan)
    prev[o[1:]] = np.where(key[o][1:] == key[o][:-1], r_ts[o][:-1], np.nan)
    F["ui_secs_since_shown"] = (r_ts - prev).reshape(E, CAND).astype(np.float32)
    del key, o, prev

    # ---- Candidate vs. this user's own history (repeat clicks) ---------------
    log("Features: candidate vs. history...")
    for name in ["cand_in_hist", "cand_secs_since_clicked", "cand_pos_from_end", "cand_is_last",
                 "svd_score", "svd_hist_mean_cos", "svd_hist_max_cos", "svd_last_cos",
                 "svd_recent5_cos", "item_users_before", "user_clicks_before",
                 "trans_last1", "trans_last3", "trans_count_last1"]:
        F[name] = np.full((E, CAND), np.nan, np.float32)
    B = 20000
    for s in range(0, E, B):
        sl = slice(s, s + B)
        eq = H[sl, None, :] == C[sl, :, None]                    # [b, 10, L]
        cnt = eq.sum(2)
        F["cand_in_hist"][sl] = cnt
        lt = np.where(eq, Ht[sl, None, :], -1).max(2)
        F["cand_secs_since_clicked"][sl] = np.where(cnt > 0, ts[sl, None] - lt, np.nan)
        li = np.where(eq, np.arange(L), -1).max(2)
        F["cand_pos_from_end"][sl] = np.where(cnt > 0, h_len[sl, None] - 1 - li, np.nan)
        last_item = np.where(h_len[sl] > 0, H[sl][np.arange(len(H[sl])), np.maximum(h_len[sl] - 1, 0)], -2)
        F["cand_is_last"][sl] = C[sl] == last_item[:, None]

    # ---- Taste: SVD rebuilt each day from clicks before that day -------------
    log("Features: matrix factorisation (one per day)...")
    clicks = all_clicks(D)
    for d in range(N_TRAIN_DAYS + 1):
        idx = np.nonzero(day == d)[0]
        if len(idx) == 0:
            continue
        uvec, ivec, inorm, item_users, user_clicks = svd_for_day(
            clicks, START + d * DAY, D["n_users"], D["n_items"])
        pair_index, pair_count, out_deg = transitions_for_day(clicks, START + d * DAY, D["n_items"])
        for s in range(0, len(idx), B):
            e = idx[s:s + B]
            Cb, Hb, hl = C[e], H[e], h_len[e]

            # Transitions from the user's last 3 clicks to the candidate.
            t3 = np.zeros(Cb.shape, np.float32)
            for back, w in [(0, 1.0), (1, 0.5), (2, 0.25)]:
                h = np.where(hl > back, Hb[np.arange(len(e)), np.maximum(hl - 1 - back, 0)], -1)
                pos = pair_index.get_indexer((h[:, None].astype(np.int64) * D["n_items"] + Cb).ravel())
                cnt = np.where((pos >= 0) & (np.repeat(h, CAND) >= 0), pair_count[pos], 0).reshape(Cb.shape)
                rate = cnt / (out_deg[h][:, None] + 10)
                t3 += w * rate
                if back == 0:
                    F["trans_last1"][e] = np.where(hl[:, None] > 0, rate, np.nan)
                    F["trans_count_last1"][e] = np.where(hl[:, None] > 0, cnt, np.nan)
            F["trans_last3"][e] = np.where(hl[:, None] > 0, t3, np.nan)
            cv = inorm[Cb]                                     # [b, 10, k]
            hv = inorm[Hb]                                     # [b, L, k]  (pad -> zero row)
            sims = np.einsum("bck,blk->bcl", cv, hv)           # cosine to each history item
            hmask = (Hb >= 0)[:, None, :]
            F["svd_hist_max_cos"][e] = np.where(hl[:, None] > 0,
                                                np.where(hmask, sims, -2).max(2), np.nan)
            F["svd_last_cos"][e] = np.where(hl[:, None] > 0,
                                            sims[np.arange(len(e)), :, np.maximum(hl - 1, 0)], np.nan)
            recent = hmask & (np.arange(L)[None, None, :] >= (hl - 5)[:, None, None])
            F["svd_recent5_cos"][e] = np.where(hl[:, None] > 0,
                                               (sims * recent).sum(2) / np.maximum(recent.sum(2), 1), np.nan)
            hmean = hv.sum(1)
            hmean /= np.maximum(np.linalg.norm(hmean, axis=1, keepdims=True), 1e-9)
            F["svd_hist_mean_cos"][e] = np.where(hl[:, None] > 0,
                                                 np.einsum("bck,bk->bc", cv, hmean), np.nan)
            F["svd_score"][e] = np.einsum("bck,bk->bc", ivec[Cb], uvec[user[e]])
            F["item_users_before"][e] = item_users[Cb]
            F["user_clicks_before"][e] = user_clicks[user[e]][:, None]
        log(f"  day {d}: {len(idx):,} events")

    # ---- Competition inside the event ---------------------------------------
    log("Features: comparison with the other candidates...")
    for name in ["item_ctr", "item_prevday_ctr", "ui_ctr", "svd_score", "svd_hist_mean_cos",
                 "svd_hist_max_cos", "cand_in_hist", "item_users_before", "item_tb_ctr",
                 "item_dt_ctr", "trans_last1", "trans_last3"]:
        a = np.nan_to_num(F[name], nan=-1)
        F[f"{name}_rank"] = (a.argsort(1).argsort(1) / (CAND - 1)).astype(np.float32)
        F[f"{name}_minus_max"] = (a - a.max(1, keepdims=True)).astype(np.float32)
        F[f"{name}_minus_mean"] = (a - a.mean(1, keepdims=True)).astype(np.float32)
    F["event_max_item_ctr"] = ev(np.nan_to_num(F["item_ctr"]).max(1))
    F["event_max_cand_in_hist"] = ev(F["cand_in_hist"].max(1))

    names = list(F)
    X = np.empty((E * CAND, len(names)), np.float32)     # filled column by column to save memory
    for k, n in enumerate(names):
        X[:, k] = F.pop(n).ravel()
    log(f"Built {X.shape[1]} features for {X.shape[0]:,} rows")

    # First event of each user on each day (the official evaluation is
    # "first event after the cutoff", so we validate the same way).
    first_of_day = np.zeros(E, bool)
    key = pd.DataFrame({"u": user, "d": day, "t": ts}).sort_values("t", kind="stable")
    first_of_day[key.drop_duplicates(["u", "d"]).index.values] = True
    meta = {"y": D["y"].ravel(), "rowid": D["rowid"].ravel(), "day": np.repeat(day, CAND),
            "is_eval": np.repeat(D["is_eval"], CAND), "first": np.repeat(first_of_day, CAND),
            "names": names}
    return X, meta


# =============================================================================
# 3. MODEL
# =============================================================================
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 127,
          "min_data_in_leaf": 200, "feature_fraction": 0.7, "bagging_fraction": 0.8,
          "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 255, "verbose": -1,
          "num_threads": os.cpu_count()}


def pick_rows(mask, y, neg_frac, seed):
    """All clicked rows + a random share of non-clicked ones (much faster, same ranking)."""
    rng = np.random.default_rng(seed)
    return np.nonzero(mask & ((y == 1) | (rng.random(len(y)) < neg_frac)))[0]


def validate(X, meta, args):
    y, day = meta["y"], meta["day"]
    fit = pick_rows((day < N_TRAIN_DAYS - 1) & ~meta["is_eval"], y, args.neg_frac, 0)
    val_all = np.nonzero(day == N_TRAIN_DAYS - 1)[0]
    val = val_all[meta["first"][val_all]]
    log(f"Validation: fit {len(fit):,} rows (days 0-5), test {len(val):,} rows "
        f"(first event per user, day 6)")
    dtrain = lgb.Dataset(X[fit], y[fit], feature_name=meta["names"], free_raw_data=True)
    dval = lgb.Dataset(X[val], y[val], reference=dtrain)
    model = lgb.train({**PARAMS, "metric": "auc", "seed": 0}, dtrain, num_boost_round=args.max_trees,
                      valid_sets=[dval], callbacks=[lgb.early_stopping(100, verbose=False),
                                                    lgb.log_evaluation(100)])
    p_all = model.predict(X[val_all], num_iteration=model.best_iteration)
    auc_all = roc_auc_score(y[val_all], p_all)
    auc = roc_auc_score(y[val], p_all[meta["first"][val_all]])
    log(f">>> LOCAL AUC (first event per user, like the leaderboard): {auc:.5f}")
    log(f"    LOCAL AUC (all day-6 events): {auc_all:.5f}   trees: {model.best_iteration}")
    imp = pd.Series(model.feature_importance("gain"), index=meta["names"])
    print("\nTop 25 features:\n" + (imp / imp.sum()).sort_values(ascending=False).head(25).round(4).to_string())

    # At most one click per event, so an event's probabilities should add up to <= 1.
    # Try pulling each event's predictions down by beta * log(sum of the event).
    p_first = p_all[meta["first"][val_all]]
    print("\nEvent re-balancing (beta=0 means off):")
    best_beta, best_auc = 0.0, auc
    for beta in [0.25, 0.5, 0.75, 1.0]:
        a = roc_auc_score(y[val], rebalance(p_first, beta))
        print(f"  beta {beta:.2f}: AUC {a:.5f}")
        if a > best_auc + 1e-5:
            best_beta, best_auc = beta, a
    log(f">>> BEST LOCAL AUC: {best_auc:.5f} with beta={best_beta}")
    return best_auc, model.best_iteration, best_beta


def rebalance(p, beta):
    """p: predictions for whole events in order (10 per event)."""
    if beta == 0:
        return p
    p = np.clip(p.reshape(-1, CAND), 1e-7, 1 - 1e-7)
    score = np.log(p) - beta * np.log(p.sum(1, keepdims=True))
    return (1 / (1 + np.exp(-(score - np.log1p(-p))))).ravel()   # back into (0, 1)


def final_submission(X, meta, args, n_trees, data_dir, beta):
    y = meta["y"]
    fit = pick_rows(~meta["is_eval"], y, args.neg_frac, 1)
    ev = np.nonzero(meta["is_eval"])[0]               # whole events, in order
    log(f"Final model: {len(fit):,} rows, {n_trees} trees x {args.seeds} seed(s), beta={beta}")
    preds = np.zeros(len(ev))
    for seed in range(args.seeds):
        dtrain = lgb.Dataset(X[fit], y[fit], feature_name=meta["names"])
        model = lgb.train({**PARAMS, "seed": seed}, dtrain, num_boost_round=n_trees)
        preds += model.predict(X[ev]) / args.seeds
        log(f"  seed {seed} done")
    preds = rebalance(preds, beta)
    sub = pd.DataFrame({"rowid": meta["rowid"][ev], "predicted_probability": np.clip(preds, 0, 1)})
    check_submission(sub, data_dir)
    os.makedirs("submissions", exist_ok=True)
    path = f"submissions/sub_{datetime.now():%m%d_%H%M%S}.csv"
    sub.to_csv(path, index=False, float_format="%.7g")
    log(f"Saved {path}")
    return path


def check_submission(sub, data_dir):
    p = sub["predicted_probability"]
    assert sub["rowid"].is_unique, "duplicate rowids"
    assert np.isfinite(p).all() and p.between(0, 1).all(), "bad probabilities"
    ex = os.path.join(data_dir, "example_submission.csv")
    if os.path.exists(ex):
        want = pd.read_csv(ex, dtype={"rowid": str}, usecols=["rowid"])["rowid"]
        assert len(want) == len(sub) and set(want) == set(sub["rowid"]), \
            "rowids don't match example_submission.csv"
        log(f"Submission check passed: {len(sub):,} rows, same rowids as example_submission.csv")


def log_result(auc, n_trees, args, path):
    row = pd.DataFrame([{"time": f"{datetime.now():%m-%d %H:%M}", "local_auc": auc and round(auc, 5),
                         "trees": n_trees, "lr": args.lr, "neg_frac": args.neg_frac, "seeds": args.seeds,
                         "submission": path or "", "public_lb": "", "note": args.note}])
    row.to_csv("results_log.csv", mode="a", header=not os.path.exists("results_log.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--valid-only", action="store_true")
    ap.add_argument("--skip-valid", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="ignore cached features")
    ap.add_argument("--neg-frac", type=float, default=0.5, help="share of non-clicked rows to train on")
    ap.add_argument("--trees", type=int, default=0, help="final trees (with --skip-valid)")
    ap.add_argument("--max-trees", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.05, help="learning rate (0.1 = faster, a bit worse)")
    ap.add_argument("--beta", type=float, default=None, help="event re-balancing (default: chosen on validation)")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    PARAMS["learning_rate"] = args.lr

    cache = os.path.join("cache", os.path.basename(os.path.abspath(args.data_dir)) + "_v2")
    if os.path.exists(cache + "_X.npy") and not args.rebuild:
        log("Loading cached features")
        X = np.load(cache + "_X.npy")
        meta = pickle.load(open(cache + "_meta.pkl", "rb"))
    else:
        X, meta = build_features(load(args.data_dir))
        os.makedirs("cache", exist_ok=True)
        np.save(cache + "_X.npy", X)
        pickle.dump(meta, open(cache + "_meta.pkl", "wb"))

    auc, n_trees, beta = None, args.trees, args.beta
    if not args.skip_valid:
        auc, best, found_beta = validate(X, meta, args)
        n_trees = n_trees or int(best * 1.15)
        beta = found_beta if beta is None else beta
    path = None
    if not args.valid_only:
        path = final_submission(X, meta, args, n_trees or 800, args.data_dir, beta or 0.0)
    log_result(auc, n_trees, args, path)
    log("Done. Copy everything above and paste it to the team / Claude.")


if __name__ == "__main__":
    main()
