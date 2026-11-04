#!/usr/bin/env python3
"""
Ping Panthers - TechArena 2026 click prediction (v3). THE MAIN SCRIPT.

    python solution.py --data-dir data            # validate, then write a submission
    python solution.py --data-dir data --fast     # quicker settings for a slow laptop
    python solution.py --data-dir data --valid-only
    python solution.py --data-dir data --skip-valid --trees 900 --beta 0.5

The CSV is read once and cached in cache/ (raw data and features separately),
so later runs skip straight to training. --rebuild recomputes the features.

THE IDEA
Each event shows 10 items and at most ONE gets clicked, so the items compete.
For every row we work out "what we knew just before this event":

  1. Click rates "as of" the event time: user, item, user+item, and by time of
     day / weekend. Only labels from EARLIER events are used.
  2. History: recent activity, repeat clicks of THIS candidate, its recency,
     whether the user tends to click at this time of day.
  3. Taste (SVD): user x item click matrix factorised, rebuilt each day from
     earlier clicks only; candidate vs. the user's taste and history.
  4. Sequence: "people who clicked A clicked B next" from the last 3 clicks.
  5. Choice model (the big one): a model that learns the 10-way race directly
     (pick one candidate or nothing), with a taste vector per user, a
     fingerprint per item, history and last-click vectors. Trained day by day,
     so features for day d only ever come from days before d.
  6. Competition: each candidate compared with the other 9 in its event.
LightGBM combines everything. Validation copies the official setup: fit on
days 0-5, test on the FIRST event of each user on day 6.
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
from scipy.sparse.linalg import svds

DAY = 86400
START = 1737331200          # 2025-01-20 00:00 UTC, first training day
N_TRAIN_DAYS = 7            # training ends at the cutoff, 2025-01-27 00:00 UTC
CAND = 10                   # candidates per event
SVD_DIM = 12                # the taste space is small: 12 beat 48 by a lot on the real data
FEATURE_VERSION = "v7"
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


# =============================================================================
# 1. LOADING (chunked, the file is 7 GB)
# =============================================================================
STR_COLS = ["rowid", "event_id", "user_id", "item_id", "history_item_ids", "history_timestamps"]
EV_COLS = ["event_id", "user_id", "timestamp", "time_bucket", "day_type",
           "history_item_ids", "history_timestamps"]


def read_file(path, labelled):
    """Rows (event, item, label/rowid) + one line per event (user, time, history)."""
    row_cols = ["event_id", "item_id"] + (["clicked"] if labelled else ["rowid"])
    rows, events, n = [], [], 0
    try:                                   # pyarrow reads CSV much faster, if installed
        import pyarrow as pa
        import pyarrow.csv as pv
    except ImportError:
        pa = None
    if pa is not None:
        reader = pv.open_csv(path, read_options=pv.ReadOptions(block_size=128 << 20),
                             convert_options=pv.ConvertOptions(
                                 column_types={c: pa.string() for c in STR_COLS}))
        for batch in reader:
            t = pa.Table.from_batches([batch])
            r = t.select(row_cols).to_pandas()
            first = ~r["event_id"].duplicated().to_numpy()
            events.append(t.select(EV_COLS).filter(pa.array(first)).to_pandas())
            rows.append(r)
            n += len(r)
            if len(rows) % 8 == 0:
                log(f"  {os.path.basename(path)}: {n:,} rows read")
    else:
        for chunk in pd.read_csv(path, dtype={c: str for c in STR_COLS}, chunksize=1_000_000):
            rows.append(chunk[row_cols])
            events.append(chunk.loc[~chunk["event_id"].duplicated(), EV_COLS])
            n += len(chunk)
            log(f"  {os.path.basename(path)}: {n:,} rows read")
    rows = pd.concat(rows, ignore_index=True)
    events = pd.concat(events, ignore_index=True).drop_duplicates("event_id", ignore_index=True)
    log(f"  {os.path.basename(path)}: {n:,} rows, {len(events):,} events")
    return rows, events


def load(data_dir):
    log("Reading train.csv (slow, only the first time)...")
    tr_rows, tr_ev = read_file(os.path.join(data_dir, "train.csv"), True)
    log("Reading evaluation.csv...")
    ev_rows, ev_ev = read_file(os.path.join(data_dir, "evaluation.csv"), False)
    tr_ev["is_eval"], ev_ev["is_eval"] = 0, 1
    tr_rows["rowid"], ev_rows["clicked"] = "", -1

    events = pd.concat([tr_ev, ev_ev], ignore_index=True)
    assert events["event_id"].is_unique, "event_id repeats between train and evaluation"
    rows = pd.concat([tr_rows, ev_rows], ignore_index=True)
    del tr_rows, ev_rows, tr_ev, ev_ev

    # Every event must have exactly 10 rows: candidates are stored as an [events x 10] grid.
    # (The order inside an event is kept as in the file, see "slot" below.)
    rows["ev"] = pd.Index(events["event_id"]).get_indexer(rows["event_id"])
    counts = np.bincount(rows["ev"].to_numpy(), minlength=len(events))
    assert (counts == CAND).all(), f"events without exactly {CAND} rows: {(counts != CAND).sum()}"
    rows = rows.sort_values("ev", kind="stable", ignore_index=True)

    log("Parsing histories...")
    h_items = [json.loads(s) for s in events["history_item_ids"].to_numpy(dtype=object)]
    h_ts = [json.loads(s) for s in events["history_timestamps"].to_numpy(dtype=object)]
    h_len = np.array([len(h) for h in h_items], dtype=np.int32)
    flat_items = pd.Series(list(chain.from_iterable(h_items)), dtype=object).astype(str)
    flat_ts = np.fromiter(chain.from_iterable(h_ts), dtype=np.int64, count=int(h_len.sum()))
    del h_items, h_ts

    # Opaque ids -> integer codes.
    item_vocab = pd.Index(pd.unique(pd.concat([rows["item_id"].astype(str), flat_items],
                                              ignore_index=True)))
    user_codes, user_vocab = pd.factorize(events["user_id"].astype(str))

    E, L = len(events), max(int(h_len.max()), 1)
    D = {
        "n_users": len(user_vocab), "n_items": len(item_vocab), "L": L,
        "user": np.asarray(user_codes, dtype=np.int32),
        "ts": events["timestamp"].to_numpy(dtype=np.int64),
        "time_bucket": events["time_bucket"].to_numpy(dtype=np.int8),
        "day_type": events["day_type"].to_numpy(dtype=np.int8),
        "is_eval": events["is_eval"].to_numpy(dtype=bool),
        "h_len": h_len,
        "C": item_vocab.get_indexer(rows["item_id"].astype(str)).reshape(E, CAND).astype(np.int32),
        "y": rows["clicked"].to_numpy(dtype=np.int8).reshape(E, CAND),
        "rowid": rows["rowid"].to_numpy(dtype=object).reshape(E, CAND),
    }
    ev_of_entry = np.repeat(np.arange(E), h_len)
    pos = np.arange(len(flat_ts)) - np.repeat(np.cumsum(h_len) - h_len, h_len)
    D["H"] = np.full((E, L), -1, np.int32)          # history grid, oldest -> newest, -1 = empty
    D["H"][ev_of_entry, pos] = item_vocab.get_indexer(flat_items)
    D["Ht"] = np.zeros((E, L), np.int64)
    D["Ht"][ev_of_entry, pos] = flat_ts
    describe(D)
    return D


def describe(D):
    """A few facts about the data, printed so we can sanity-check the setup."""
    ev, tr = D["is_eval"], ~D["is_eval"]
    days = np.clip((D["ts"][tr] - START) // DAY, 0, 6)
    log(f"Loaded {len(ev):,} events ({ev.sum():,} evaluation), {D['n_users']:,} users, "
        f"{D['n_items']:,} items")
    log(f"  train events per day: {np.bincount(days, minlength=7).tolist()}")
    log(f"  click rate per row {(D['y'][tr] == 1).mean():.4f}, events with a click "
        f"{(D['y'][tr] == 1).any(1).mean():.3f}")
    hrs = (D["ts"][ev] - (START + N_TRAIN_DAYS * DAY)) / 3600
    log(f"  evaluation events: hours after cutoff, median {np.median(hrs):.1f}, "
        f"90% {np.quantile(hrs, .9):.1f}, max {hrs.max():.1f}")
    seen = np.isin(D["user"][ev], D["user"][tr]).mean()
    log(f"  evaluation users seen in train {seen:.1%}, empty history {(D['h_len'][ev] == 0).mean():.1%}")


def load_raw(data_dir):
    path = os.path.join("cache", f"raw_{os.path.basename(os.path.abspath(data_dir))}.pkl")
    if os.path.exists(path):
        log(f"Loading cached raw data {path}")
        D = pickle.load(open(path, "rb"))
        describe(D)
        return D
    D = load(data_dir)
    os.makedirs("cache", exist_ok=True)
    pickle.dump(D, open(path, "wb"), protocol=4)
    return D


# =============================================================================
# 2. FEATURE HELPERS
# =============================================================================
def asof_counts(key, ts, n, c):
    """For each row: sum of n and c over rows with the same key and an EARLIER time."""
    df = pd.DataFrame({"k": key, "t": ts, "n": n, "c": c})
    g = df.groupby(["k", "t"], sort=True)[["n", "c"]].sum()
    before = g.groupby(level=0).cumsum() - g
    out = df[["k", "t"]].merge(before, left_on=["k", "t"], right_index=True, how="left")
    return np.array(out["n"], dtype=np.float32), np.array(out["c"], dtype=np.float32)


def all_clicks(D):
    """Every known click (user, item, time): from histories + labelled training rows."""
    mask = D["H"] >= 0
    hist = pd.DataFrame({"u": np.repeat(D["user"], mask.sum(1)), "i": D["H"][mask], "t": D["Ht"][mask]})
    e, j = np.nonzero(D["y"] == 1)
    lab = pd.DataFrame({"u": D["user"][e], "i": D["C"][e, j], "t": D["ts"][e]})
    clicks = pd.concat([hist, lab], ignore_index=True).drop_duplicates()
    log(f"  {len(clicks):,} distinct clicks known (incl. warm-up period)")
    return clicks


def svd_for_day(clicks, cutoff, n_users, n_items):
    """Matrix factorisation of all clicks before `cutoff`."""
    c = clicks[clicks["t"] < cutoff]
    X = sparse.csr_matrix((np.ones(len(c), np.float32), (c["u"].to_numpy(), c["i"].to_numpy())),
                          shape=(n_users, n_items))
    X.sum_duplicates()
    X.data = np.log1p(X.data)
    item_users = np.asarray((X > 0).sum(0)).ravel()
    user_clicks = np.asarray(X.sum(1)).ravel()
    # Weight every click by (n_u + 20)^-0.5 * (n_i + 20)^-0.5 (n_u = items the user clicked,
    # n_i = users who clicked the item), so very active users and very popular items don't dominate.
    n_u = np.asarray((X > 0).sum(1)).ravel()
    X = (sparse.diags((n_u + 20.0) ** -0.5) @ X @ sparse.diags((item_users + 20.0) ** -0.5)).astype(np.float64)
    try:                                                      # exact SVD (ARPACK), not randomized
        U, S, VT = svds(X, k=SVD_DIM, random_state=0)
    except TypeError:                                         # older scipy without random_state
        U, S, VT = svds(X, k=SVD_DIM)
    order = np.argsort(-S)
    U, S, VT = U[:, order], S[order], VT[order]
    user_vec = (U * S).astype(np.float32)
    item_vec = VT.T.astype(np.float32)
    item_norm = item_vec / np.maximum(np.linalg.norm(item_vec, axis=1, keepdims=True), 1e-9)
    pad = lambda a: np.vstack([a, np.zeros((1, a.shape[1]), np.float32)])   # index -1 -> zeros
    return pad(user_vec), pad(item_vec), pad(item_norm), item_users, user_clicks


def svd_neg_for_day(clicks, Nm, cutoff, n_users, n_items, alpha=0.1):
    """Taste from clicks MINUS a little for every item the user was shown but skipped."""
    c = clicks[clicks["t"] < cutoff]
    P = sparse.csr_matrix((np.ones(len(c), np.float32), (c["u"].to_numpy(), c["i"].to_numpy())),
                          shape=(n_users, n_items))
    P.sum_duplicates()
    P.data = np.log1p(P.data)
    N = Nm.copy()
    N.data = np.log1p(N.data)
    X = (P - alpha * N).tocsr()
    n_u = np.asarray((X != 0).sum(1)).ravel()
    n_i = np.asarray((X != 0).sum(0)).ravel()
    X = (sparse.diags((n_u + 20.0) ** -0.5) @ X @ sparse.diags((n_i + 20.0) ** -0.5)).astype(np.float64)
    try:
        U, S, VT = svds(X, k=SVD_DIM, random_state=0)
    except TypeError:
        U, S, VT = svds(X, k=SVD_DIM)
    user_vec = (U * S).astype(np.float32)
    item_vec = VT.T.astype(np.float32)
    item_norm = item_vec / np.maximum(np.linalg.norm(item_vec, axis=1, keepdims=True), 1e-9)
    pad = lambda a: np.vstack([a, np.zeros((1, a.shape[1]), np.float32)])
    return pad(user_vec), pad(item_vec), pad(item_norm)


def transitions_for_day(clicks, cutoff, n_items):
    """'People who clicked A clicked B NEXT': counts of consecutive click pairs before cutoff."""
    c = clicks[clicks["t"] < cutoff].sort_values(["u", "t"], kind="stable")
    u, i = c["u"].to_numpy(), c["i"].to_numpy().astype(np.int64)
    same = u[1:] == u[:-1]
    a, b = i[:-1][same], i[1:][same]
    pairs = pd.Series(a * n_items + b).value_counts()
    out_deg = np.bincount(a, minlength=n_items + 1).astype(np.float32)
    return pd.Index(pairs.index), pairs.to_numpy(np.float32), out_deg


def cooc_for_day(clicks, cutoff, n_users, n_items):
    """Exact 'people who clicked X also clicked Y': co[x, y] = number of users who clicked
    both before cutoff (full detail, unlike the 48-number SVD fingerprints)."""
    c = clicks[clicks["t"] < cutoff]
    per_user = c.groupby("u")["i"].nunique()
    if (per_user.astype(np.float64) ** 2).sum() > 1.2e8:     # protect memory: cap very active users
        c = c.sort_values("t").groupby("u").tail(40)
    c = c[["u", "i"]].drop_duplicates()
    X = sparse.csr_matrix((np.ones(len(c), np.float32), (c["u"].to_numpy(), c["i"].to_numpy())),
                          shape=(n_users, n_items))
    co = (X.T @ X).tocsr()
    del X
    co.sort_indices()
    n_i = np.asarray(co.diagonal(), dtype=np.float32)
    keys = np.repeat(np.arange(n_items, dtype=np.int64) * n_items, np.diff(co.indptr))
    keys += co.indices                                            # sorted, for fast lookup
    vals = co.data.astype(np.float32)
    del co
    return keys, vals, n_i


def seq_window_for_day(clicks, cutoff, n_items, window=3):
    """'After clicking X, people click Y within their next 3 clicks' (closer = more weight)."""
    c = clicks[clicks["t"] < cutoff].sort_values(["u", "t"], kind="stable")
    u, i = c["u"].to_numpy(), c["i"].to_numpy().astype(np.int64)
    keys, wts = [], []
    for off in range(1, window + 1):
        same = u[off:] == u[:-off]
        keys.append(i[:-off][same] * n_items + i[off:][same])
        wts.append(np.full(int(same.sum()), 1.0 / off, np.float32))
    keys, wts = np.concatenate(keys), np.concatenate(wts)
    uniq, inv = np.unique(keys, return_inverse=True)
    counts = np.bincount(inv.ravel(), weights=wts).astype(np.float32)
    out_deg = np.bincount(uniq // n_items, weights=counts, minlength=n_items + 1).astype(np.float32)
    return uniq, counts, out_deg


def lookup(keys, vals, q):
    """vals for each query key in q (0 if missing). keys must be sorted."""
    if len(keys) == 0:
        return np.zeros(q.shape, np.float32)
    pos = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
    return np.where(keys[pos] == q, vals[pos], 0).astype(np.float32)


# =============================================================================
# 3. THE CHOICE MODEL (learned matrix factorisation for a "pick 1 of 10 or none" race)
# =============================================================================
class Table:
    """Rows of parameters trained with Adagrad; only rows that get a gradient change."""

    def __init__(self, n, k, scale, lr, reg, rng):
        self.W = (rng.normal(0, scale, (n, k)) if scale else np.zeros((n, k))).astype(np.float32)
        self.G = np.full((n, k), 0.1, np.float32)
        self.lr, self.reg = lr, reg

    def update(self, idx, grad):
        if len(idx) == 0:
            return
        uniq, inv = np.unique(idx, return_inverse=True)
        M = sparse.csr_matrix((np.ones(len(idx), np.float32), (inv.ravel(), np.arange(len(idx)))),
                              shape=(len(uniq), len(idx)))
        g = np.asarray(M @ grad, dtype=np.float32) + self.reg * self.W[uniq]
        self.G[uniq] += g * g
        self.W[uniq] -= self.lr * g / np.sqrt(self.G[uniq])


class ChoiceMF:
    """
    score(candidate) = item bias + user bias + time bias + repeat terms
                       + < user taste , item fingerprint >
    user taste = own vector + average of history items' vectors + last click's vector
    The user picks one candidate or nothing: softmax over 11 options.
    """

    def __init__(self, D, F, k=32, lr=0.05, reg=1e-4, seed=0):
        rng = np.random.default_rng(seed)
        nU, nI = D["n_users"], D["n_items"] + 1            # last item row = padding
        self.pad = D["n_items"]
        self.k = k
        self.P = Table(nU, k, 0.01, lr, reg, rng)
        self.Q = Table(nI, k, 0.01, lr, reg, rng)
        self.Y = Table(nI, k, 0.01, lr, reg, rng)
        self.R = Table(nI, k, 0.01, lr, reg, rng)
        self.bi = Table(nI, 1, 0, lr, reg, rng)
        self.bu = Table(nU, 1, 0, lr, reg, rng)
        self.bt = Table(8, 1, 0, lr, 0, rng)
        self.w = Table(1, 3, 0, lr, 0, rng)
        for t in (self.Q, self.Y, self.R):
            t.W[self.pad] = 0
        E = len(D["ts"])
        self.user, self.C, self.y = D["user"], D["C"], D["y"]
        self.H = np.where(D["H"] >= 0, D["H"], self.pad).astype(np.int64)
        hl = D["h_len"]
        self.last = np.where(hl > 0, self.H[np.arange(E), np.maximum(hl - 1, 0)], self.pad)
        self.hn = (1 / np.sqrt(np.maximum(hl, 1))).astype(np.float32)
        self.ctx = D["time_bucket"].astype(np.int64) * 2 + D["day_type"].astype(np.int64)
        rec = np.nan_to_num(np.exp(-F["cand_secs_since_clicked"] / DAY), nan=0.0)
        self.Xd = np.stack([np.log1p(F["cand_in_hist"]), F["cand_is_last"], rec], -1).astype(np.float32)

    def forward(self, e):
        u, c, h, l = self.user[e], self.C[e], self.H[e], self.last[e]
        z = self.P.W[u] + self.Y.W[h].sum(1) * self.hn[e][:, None] + self.R.W[l]
        qc = self.Q.W[c]
        dot = np.einsum("bk,bck->bc", z, qc)
        ub = self.bu.W[u, 0] + self.bt.W[self.ctx[e], 0]
        s = dot + self.bi.W[c, 0] + ub[:, None] + self.Xd[e] @ self.w.W[0]
        m = np.maximum(s.max(1, keepdims=True), 0)
        ex = np.exp(s - m)
        p = ex / (np.exp(-m) + ex.sum(1, keepdims=True))
        return s, p, dot, z, qc, ub

    def step(self, e):
        s, p, dot, z, qc, _ = self.forward(e)
        u, c, h, l = self.user[e], self.C[e], self.H[e], self.last[e]
        g = (p - (self.y[e] == 1)).astype(np.float32)                 # d loss / d score
        gz = np.einsum("bc,bck->bk", g, qc)
        self.Q.update(c.ravel(), (g[:, :, None] * z[:, None, :]).reshape(-1, self.k))
        self.P.update(u, gz)
        rows, cols = np.nonzero(h != self.pad)
        self.Y.update(h[rows, cols], (gz * self.hn[e][:, None])[rows])
        lv = l != self.pad
        self.R.update(l[lv], gz[lv])
        self.bi.update(c.ravel(), g.reshape(-1, 1))
        gs = g.sum(1, keepdims=True)
        self.bu.update(u, gs)
        self.bt.update(self.ctx[e], gs)
        self.w.update(np.zeros(1, np.int64), np.einsum("bc,bcf->f", g, self.Xd[e])[None, :])
        picked = np.where((self.y[e] == 1).any(1), (p * (self.y[e] == 1)).sum(1), 1 - p.sum(1))
        return -np.log(np.maximum(picked, 1e-9)).mean()


MF_SETTINGS = {"k": 32, "lr": 0.05, "reg": 1e-4, "new_passes": 1}


def choice_model_features(D, F, day, batch=4096):
    """Train the choice model day by day; day d's features come from days < d only."""
    log("Choice model (learned taste vectors, trained day by day)...")
    E = len(D["ts"])
    names = ["mf_logit", "mf_prob", "mf_dot", "mf_item_bias", "mf_user_bias", "mf_p_any"]
    out = {n: np.full((E, CAND), np.nan, np.float32) for n in names}
    cfg = MF_SETTINGS
    mf = ChoiceMF(D, F, k=cfg["k"], lr=cfg["lr"], reg=cfg["reg"])
    rng = np.random.default_rng(0)
    train_ev = np.nonzero(~D["is_eval"])[0]
    for d in range(1, N_TRAIN_DAYS + 1):
        losses = []
        parts = [train_ev[day[train_ev] < d]] + [train_ev[day[train_ev] == d - 1]] * cfg["new_passes"]
        for part in parts:
            perm = rng.permutation(part)
            for s in range(0, len(perm), batch):
                losses.append(mf.step(perm[s:s + batch]))
        target = np.nonzero(day == d)[0]
        for s in range(0, len(target), batch):
            e = target[s:s + batch]
            sc, p, dot, _, _, ub = mf.forward(e)
            out["mf_logit"][e], out["mf_prob"][e], out["mf_dot"][e] = sc, p, dot
            out["mf_item_bias"][e] = mf.bi.W[mf.C[e], 0]
            out["mf_user_bias"][e] = ub[:, None]
            out["mf_p_any"][e] = p.sum(1, keepdims=True)
        y = D["y"][target]
        msg = ""
        if (y >= 0).all() and (y == 1).any():
            msg = f", AUC on day {d} (before training on it) {roc_auc_score((y == 1).ravel(), out['mf_prob'][target].ravel()):.4f}"
        log(f"  trained on days < {d}: loss {np.mean(losses):.4f}{msg}")
    F.update(out)


# =============================================================================
# 4. ALL FEATURES
# =============================================================================
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
    # Does this user usually click at this time of day / on this kind of day?
    h_bucket = (Ht % DAY) // 21600
    h_weekend = (((Ht // DAY) + 3) % 7) >= 5
    same_bucket = valid & (h_bucket == D["time_bucket"][:, None])
    same_daytype = valid & (h_weekend == D["day_type"][:, None].astype(bool))
    F["hist_share_same_bucket"] = ev(np.where(h_len > 0, same_bucket.sum(1) / np.maximum(h_len, 1), np.nan))
    F["hist_share_same_daytype"] = ev(np.where(h_len > 0, same_daytype.sum(1) / np.maximum(h_len, 1), np.nan))

    # Previous visits of the user (train + evaluation timeline, past only).
    order = np.lexsort((ts, user))
    prev_ts = np.full(E, np.nan)
    same = np.r_[False, user[order][1:] == user[order][:-1]]
    prev_ts[order[1:]] = np.where(same[1:], ts[order][:-1], np.nan)
    F["secs_since_prev_event"] = ev(ts - prev_ts)
    n_before = np.zeros(E)
    n_before[order] = pd.Series(user[order]).groupby(user[order]).cumcount().to_numpy()
    F["user_prev_events"] = ev(n_before)

    # ---- As-of click rates (labels from earlier events only) ------------------
    log("Features: as-of click rates...")
    r_ts = np.repeat(ts, CAND)
    r_user = np.repeat(user, CAND).astype(np.int64)
    r_item = C.ravel().astype(np.int64)
    y = D["y"].ravel()
    lab = (y >= 0).astype(np.float32)
    clk = (y == 1).astype(np.float32)
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
    pn = np.where(prev >= 0, np.array(g["n"])[prev], 0)
    pc = np.where(prev >= 0, np.array(g["c"])[prev], 0)
    F["item_prevday_shown"] = pn.reshape(E, CAND).astype(np.float32)
    F["item_prevday_ctr"] = ((pc + 50 * global_ctr) / (pn + 50)).reshape(E, CAND).astype(np.float32)
    first_seen = pd.Series(r_ts).groupby(r_item).transform("min").to_numpy()
    F["item_age_days"] = ((r_ts - first_seen) / DAY).reshape(E, CAND).astype(np.float32)

    # How much is this item being shown TODAY compared with yesterday (no labels needed)?
    keys, inv, cnt = np.unique(r_item * 16 + r_day, return_inverse=True, return_counts=True)
    rows_per_day = np.bincount(r_day, minlength=16).astype(np.float64)
    share = cnt[inv.ravel()] / rows_per_day[r_day]
    prev_pos = pd.Index(keys).get_indexer(r_item * 16 + r_day - 1)
    prev_share = np.where(prev_pos >= 0, cnt[np.maximum(prev_pos, 0)], 0) / np.maximum(rows_per_day[np.maximum(r_day - 1, 0)], 1)
    F["item_day_share"] = (share * 1e4).reshape(E, CAND).astype(np.float32)
    F["item_share_trend"] = np.where(r_day > 0, np.log((share + 1e-6) / (prev_share + 1e-6)),
                                     np.nan).reshape(E, CAND).astype(np.float32)

    # Click rates by time of day / weekend, shrunk towards the item's / user's overall rate.
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
    del key, o, prev, r_ts, r_user, r_item, r_day, r_tb, r_dt

    # ---- Candidate vs. this user's own history (repeat clicks) ---------------
    log("Features: candidate vs. history...")
    for name in ["cand_in_hist", "cand_secs_since_clicked", "cand_pos_from_end", "cand_is_last",
                 "cand_in_hist_same_bucket",
                 "svd_score", "svd_hist_mean_cos", "svd_hist_max_cos", "svd_last_cos",
                 "svd_recent5_cos", "svd_same_bucket_cos", "item_users_before", "user_clicks_before",
                 "trans_last1", "trans_last3", "trans_count_last1",
                 "cf_sum", "cf_max", "cf_mean", "cf_recent5", "cf_last", "cf_linked_share",
                 "cf_raw_sum", "seq3_last1", "seq3_last5",
                 "svdn_score", "svdn_hist_mean_cos", "svdn_hist_max_cos", "svdn_recent5_cos",
                 "skip_cos", "user_skips_before"]:
        F[name] = np.full((E, CAND), np.nan, np.float32)

    # Shown-but-not-clicked rows (training labels = 0), used only for LATER days.
    ne, nj = np.nonzero((D["y"] == 0) & ~D["is_eval"][:, None])
    neg_u, neg_i, neg_t = D["user"][ne], D["C"][ne, nj], D["ts"][ne]
    del ne, nj
    B = 20000
    for s in range(0, E, B):
        sl = slice(s, s + B)
        eq = H[sl, None, :] == C[sl, :, None]                    # [b, 10, L]
        cnt = eq.sum(2)
        F["cand_in_hist"][sl] = cnt
        F["cand_in_hist_same_bucket"][sl] = (eq & same_bucket[sl, None, :]).sum(2)
        lt = np.where(eq, Ht[sl, None, :], -1).max(2)
        F["cand_secs_since_clicked"][sl] = np.where(cnt > 0, ts[sl, None] - lt, np.nan)
        li = np.where(eq, np.arange(L), -1).max(2)
        F["cand_pos_from_end"][sl] = np.where(cnt > 0, h_len[sl, None] - 1 - li, np.nan)
        hl = h_len[sl]
        last_item = np.where(hl > 0, H[sl][np.arange(len(hl)), np.maximum(hl - 1, 0)], -2)
        F["cand_is_last"][sl] = C[sl] == last_item[:, None]

    # ---- Taste (SVD) and sequence (transitions), rebuilt each day -------------
    log("Features: matrix factorisation + transitions (one per day)...")
    clicks = all_clicks(D)
    for d in range(N_TRAIN_DAYS + 1):
        idx = np.nonzero(day == d)[0]
        if len(idx) == 0:
            continue
        uvec, ivec, inorm, item_users, user_clicks = svd_for_day(
            clicks, START + d * DAY, D["n_users"], D["n_items"])
        pair_index, pair_count, out_deg = transitions_for_day(clicks, START + d * DAY, D["n_items"])
        before = neg_t < START + d * DAY
        Nm = sparse.csr_matrix((np.ones(int(before.sum()), np.float32), (neg_u[before], neg_i[before])),
                               shape=(D["n_users"], D["n_items"]))
        Nm.sum_duplicates()
        nuvec, nivec, ninorm = svd_neg_for_day(clicks, Nm, START + d * DAY, D["n_users"], D["n_items"])
        skip_vec = np.asarray(Nm @ inorm[:-1], dtype=np.float32)          # what the user skipped, as taste
        skip_vec /= np.maximum(np.linalg.norm(skip_vec, axis=1, keepdims=True), 1e-9)
        user_skips = np.asarray(Nm.sum(1)).ravel().astype(np.float32)
        co_keys, co_vals, n_i = cooc_for_day(clicks, START + d * DAY, D["n_users"], D["n_items"])
        s3_keys, s3_vals, s3_out = seq_window_for_day(clicks, START + d * DAY, D["n_items"])
        nI = D["n_items"]
        for s in range(0, len(idx), B):
            e = idx[s:s + B]
            Cb, Hb, hl = C[e], H[e], h_len[e]
            has = hl[:, None] > 0

            # Exact co-clicks between the candidate and every history item.
            hb = np.maximum(Hb, 0).astype(np.int64)
            pair_ok = (Hb >= 0)[:, None, :] & (Hb[:, None, :] != Cb[:, :, None])
            q = Cb[:, :, None].astype(np.int64) * nI + hb[:, None, :]
            co = np.where(pair_ok, lookup(co_keys, co_vals, q.ravel()).reshape(q.shape), 0)
            cf = co / (np.sqrt(np.maximum(n_i[Cb], 1))[:, :, None] * np.sqrt(np.maximum(n_i[hb], 1))[:, None, :])
            nv = pair_ok.sum(2)
            F["cf_sum"][e] = cf.sum(2)
            F["cf_max"][e] = cf.max(2)
            F["cf_mean"][e] = np.where(nv > 0, cf.sum(2) / np.maximum(nv, 1), np.nan)
            last5 = pair_ok & (np.arange(L)[None, None, :] >= (hl - 5)[:, None, None])
            F["cf_recent5"][e] = (cf * last5).sum(2)
            F["cf_last"][e] = np.where(has, cf[np.arange(len(e)), :, np.maximum(hl - 1, 0)], np.nan)
            F["cf_linked_share"][e] = np.where(nv > 0, (co > 0).sum(2) / np.maximum(nv, 1), np.nan)
            F["cf_raw_sum"][e] = co.sum(2)

            # "Within the next 3 clicks" links from the last 5 history items.
            s5 = np.zeros(Cb.shape, np.float32)
            for back, w in [(0, 1.0), (1, 0.7), (2, 0.5), (3, 0.35), (4, 0.25)]:
                h = np.where(hl > back, Hb[np.arange(len(e)), np.maximum(hl - 1 - back, 0)], -1)
                cnt = lookup(s3_keys, s3_vals, (np.maximum(h, 0)[:, None].astype(np.int64) * nI + Cb).ravel()).reshape(Cb.shape)
                rate = np.where((h >= 0)[:, None], cnt / (s3_out[np.maximum(h, 0)][:, None] + 10), 0)
                s5 += w * rate
                if back == 0:
                    F["seq3_last1"][e] = np.where(has, rate, np.nan)
            F["seq3_last5"][e] = np.where(has, s5, np.nan)

            t3 = np.zeros(Cb.shape, np.float32)
            for back, w in [(0, 1.0), (1, 0.5), (2, 0.25)]:
                h = np.where(hl > back, Hb[np.arange(len(e)), np.maximum(hl - 1 - back, 0)], -1)
                pos = pair_index.get_indexer((h[:, None].astype(np.int64) * D["n_items"] + Cb).ravel())
                cnt = np.where((pos >= 0) & (np.repeat(h, CAND) >= 0), pair_count[pos], 0).reshape(Cb.shape)
                rate = cnt / (out_deg[h][:, None] + 10)
                t3 += w * rate
                if back == 0:
                    F["trans_last1"][e] = np.where(has, rate, np.nan)
                    F["trans_count_last1"][e] = np.where(has, cnt, np.nan)
            F["trans_last3"][e] = np.where(has, t3, np.nan)

            cv = inorm[Cb]                                     # [b, 10, k]
            hv = inorm[Hb]                                     # [b, L, k]  (-1 -> zero row)
            sims = np.einsum("bck,blk->bcl", cv, hv)           # cosine to each history item
            hmask = (Hb >= 0)[:, None, :]
            F["svd_hist_max_cos"][e] = np.where(has, np.where(hmask, sims, -2).max(2), np.nan)
            F["svd_last_cos"][e] = np.where(has, sims[np.arange(len(e)), :, np.maximum(hl - 1, 0)], np.nan)
            recent = hmask & (np.arange(L)[None, None, :] >= (hl - 5)[:, None, None])
            F["svd_recent5_cos"][e] = np.where(has, (sims * recent).sum(2) / np.maximum(recent.sum(2), 1), np.nan)
            sb = same_bucket[e][:, None, :]
            F["svd_same_bucket_cos"][e] = np.where(sb.sum(2) > 0, (sims * sb).sum(2) / np.maximum(sb.sum(2), 1), np.nan)
            hmean = hv.sum(1)
            hmean /= np.maximum(np.linalg.norm(hmean, axis=1, keepdims=True), 1e-9)
            F["svd_hist_mean_cos"][e] = np.where(has, np.einsum("bck,bk->bc", cv, hmean), np.nan)
            F["svd_score"][e] = np.einsum("bck,bk->bc", ivec[Cb], uvec[user[e]])

            # Same ideas in the taste space that also learned from skipped items.
            ncv, nhv = ninorm[Cb], ninorm[Hb]
            nsims = np.einsum("bck,blk->bcl", ncv, nhv)
            F["svdn_hist_max_cos"][e] = np.where(has, np.where(hmask, nsims, -2).max(2), np.nan)
            F["svdn_recent5_cos"][e] = np.where(has, (nsims * recent).sum(2) / np.maximum(recent.sum(2), 1), np.nan)
            nhm = nhv.sum(1)
            nhm /= np.maximum(np.linalg.norm(nhm, axis=1, keepdims=True), 1e-9)
            F["svdn_hist_mean_cos"][e] = np.where(has, np.einsum("bck,bk->bc", ncv, nhm), np.nan)
            F["svdn_score"][e] = np.einsum("bck,bk->bc", nivec[Cb], nuvec[user[e]])
            # Does the candidate look like things this user was shown and skipped?
            F["skip_cos"][e] = np.einsum("bck,bk->bc", cv, skip_vec[user[e]])
            F["user_skips_before"][e] = user_skips[user[e]][:, None]
            F["item_users_before"][e] = item_users[Cb]
            F["user_clicks_before"][e] = user_clicks[user[e]][:, None]
        log(f"  day {d}: {len(idx):,} events (co-click pairs {len(co_keys):,})")
        del co_keys, co_vals, s3_keys, s3_vals, Nm
    del clicks

    # Running totals grow during the week, and evaluation sits at the very end, beyond
    # anything the model trained on. Per-day rates don't have that problem.
    elapsed = ev(np.maximum((np.minimum(ts, START + N_TRAIN_DAYS * DAY) - START) / DAY, 0.25))
    for name in ["item_prev_shown", "item_prev_clicks", "user_prev_shown", "user_prev_clicks",
                 "ui_prev_shown", "item_users_before", "user_clicks_before"]:
        F[f"{name}_per_day"] = (F[name] / elapsed).astype(np.float32)

    # ---- The choice model ------------------------------------------------------
    choice_model_features(D, F, day)

    # ---- Competition inside the event ---------------------------------------
    log("Features: comparison with the other candidates...")
    for name in ["item_ctr", "item_prevday_ctr", "ui_ctr", "svd_score", "svd_hist_mean_cos",
                 "svd_hist_max_cos", "cand_in_hist", "item_users_before", "item_tb_ctr",
                 "item_dt_ctr", "trans_last1", "trans_last3", "mf_logit", "mf_dot",
                 "item_day_share", "svd_same_bucket_cos", "cf_sum", "cf_max", "cf_recent5",
                 "seq3_last5", "item_users_before_per_day", "svdn_score", "svdn_hist_mean_cos",
                 "svdn_hist_max_cos", "skip_cos"]:
        a = np.nan_to_num(F[name], nan=-1)
        F[f"{name}_rank"] = (a.argsort(1).argsort(1) / (CAND - 1)).astype(np.float32)
        F[f"{name}_minus_max"] = (a - a.max(1, keepdims=True)).astype(np.float32)
        F[f"{name}_minus_mean"] = (a - a.mean(1, keepdims=True)).astype(np.float32)
    F["event_max_item_ctr"] = ev(np.nan_to_num(F["item_ctr"]).max(1))
    for name in ["svd_hist_mean_cos", "svd_score", "cf_sum", "cf_max", "mf_logit", "mf_dot",
                 "seq3_last5", "svd_hist_max_cos", "svdn_hist_mean_cos", "svdn_score"]:
        a = F[name]
        nan = np.isnan(a)
        mx = np.where(nan, -np.inf, a).max(1)
        cnt = (~nan).sum(1)
        mean = np.where(nan, 0, a).sum(1) / np.maximum(cnt, 1)
        F[f"ev_max_{name}"] = ev(np.where(cnt > 0, mx, np.nan))
        F[f"ev_mean_{name}"] = ev(np.where(cnt > 0, mean, np.nan))
        top2 = np.sort(np.where(nan, -np.inf, a), 1)[:, -2]
        with np.errstate(invalid="ignore"):
            F[f"ev_gap12_{name}"] = ev(np.where(cnt > 1, mx - top2, np.nan))   # how clear the favourite is
    F["event_max_cand_in_hist"] = ev(F["cand_in_hist"].max(1))

    # ---- Slot: the candidate's position inside its event in the file -----------
    # Only used if it matters in train AND looks the same in evaluation (see check_slot).
    F["slot"] = np.tile(np.arange(CAND, dtype=np.float32), (E, 1))
    slot_ok, slot_maybe = check_slot(D, F)

    names = list(F)                                        # "slot" is the last column
    X = np.empty((E * CAND, len(names)), np.float32, order="F")   # column by column: low memory
    for k, n in enumerate(names):
        X[:, k] = F.pop(n).ravel()
    log(f"Built {X.shape[1]} features for {X.shape[0]:,} rows")

    # First event of each user on each day (the official evaluation is "first event
    # after the cutoff", so we validate the same way).
    first_of_day = np.zeros(E, bool)
    key = pd.DataFrame({"u": user, "d": day, "t": ts}).sort_values("t", kind="stable")
    first_of_day[key.drop_duplicates(["u", "d"]).index.to_numpy()] = True
    meta = {"y": D["y"].ravel(), "rowid": D["rowid"].ravel(), "day": np.repeat(day, CAND),
            "is_eval": np.repeat(D["is_eval"], CAND), "first": np.repeat(first_of_day, CAND),
            "names": names, "slot_ok": slot_ok, "slot_maybe": slot_maybe}
    return X, meta


def check_slot(D, F):
    """Is the order of candidates in the file informative, and the same in evaluation?"""
    tr, ev = ~D["is_eval"], D["is_eval"]
    y = D["y"][tr] == 1
    rate = y.mean(0)                                         # click rate per slot
    z = np.abs(rate - y.mean()) / np.sqrt(y.mean() * (1 - y.mean()) / y.shape[0])
    log(f"Slot check: click rate by slot {np.round(rate, 4).tolist()} (max z={z.max():.1f})")
    confirmed = False
    for name in ["svd_hist_mean_cos", "cand_in_hist", "cf_sum", "item_ctr", "item_users_before"]:
        prof_tr = np.nanmean(F[name][tr], 0)                  # label-free profile by slot
        prof_ev = np.nanmean(F[name][ev], 0)
        corr = np.corrcoef(prof_tr, prof_ev)[0, 1] if prof_tr.std() > 1e-12 and prof_ev.std() > 1e-12 else 0
        spread = prof_tr.std() / (abs(prof_tr.mean()) + 1e-9)
        log(f"  {name:<18} by slot: train spread {spread:.3f}, train/eval correlation {corr:.2f}")
        confirmed |= bool(spread > 0.02 and corr > 0.9)
    ok = bool(z.max() > 8 and confirmed)
    maybe = bool(z.max() > 8 and not confirmed)
    if maybe:
        log("  !! Slot matters in train but can't be confirmed for evaluation: an extra submission "
            "WITH slot will be made too - compare the two public scores.")
    log(f"  slot feature {'ON' if ok else 'OFF'}")
    return ok, maybe


# =============================================================================
# 5. MODEL
# =============================================================================
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 127,
          "min_data_in_leaf": 300, "feature_fraction": 0.7, "bagging_fraction": 0.8,
          "bagging_freq": 1, "lambda_l2": 2.0, "max_bin": 255, "verbose": -1,
          "force_col_wise": True, "num_threads": os.cpu_count()}


def take(X, rows, cols):
    """X[rows][:, cols] with a single copy (memory matters with 7M rows)."""
    if list(cols) == list(range(len(cols))):
        return X[rows, :len(cols)]
    return X[np.ix_(rows, cols)]


def pick_rows(mask, y, neg_frac, seed):
    """All clicked rows + a random share of non-clicked ones (much faster, same ranking)."""
    rng = np.random.default_rng(seed)
    return np.nonzero(mask & ((y == 1) | (rng.random(len(y)) < neg_frac)))[0]


def rebalance(p, beta):
    """At most one click per event: pull events whose predictions add up to a lot down a bit.
    p: predictions for whole events in order (10 per event)."""
    if beta == 0:
        return p
    p = np.clip(p.reshape(-1, CAND), 1e-7, 1 - 1e-7)
    score = np.log(p) - beta * np.log(p.sum(1, keepdims=True))
    return (1 / (1 + np.exp(-(score - np.log1p(-p))))).ravel()


def validate(X, meta, cols, args):
    y, day = meta["y"], meta["day"]
    names = [meta["names"][c] for c in cols]
    fit = pick_rows((day < N_TRAIN_DAYS - 1) & (day >= args.min_day) & ~meta["is_eval"], y, args.neg_frac, 0)
    val_all = np.nonzero(day == N_TRAIN_DAYS - 1)[0]
    first = meta["first"][val_all]
    val = val_all[first]
    log(f"Validation: fit {len(fit):,} rows (days {args.min_day}-5), test {len(val):,} rows "
        f"(first event per user, day 6)")
    for name in ["mf_prob", "svd_hist_mean_cos", "svd_score", "svdn_hist_mean_cos", "svdn_score",
                 "skip_cos", "cf_sum", "cf_max", "seq3_last5",
                 "trans_last3", "cand_in_hist", "user_ctr", "item_ctr"]:
        log(f"  single feature {name:<18} AUC {roc_auc_score(y[val] == 1, np.nan_to_num(X[val, meta['names'].index(name)], nan=-9)):.5f}")
    nm = meta["names"]
    if all(n in nm for n in ["item_ctr", "user_ctr", "cf_sum"]):
        lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        g = (y[~meta["is_eval"]] == 1).mean()
        st = (lg(X[val, nm.index("item_ctr")]) + lg(X[val, nm.index("user_ctr")]) - lg(g)
              + 25 * np.nan_to_num(X[val, nm.index("cf_sum")]))
        log(f"  STARTER-STYLE score (item + user click rates + 25 x co-click) AUC {roc_auc_score(y[val] == 1, st):.5f}")
    dtrain = lgb.Dataset(take(X, fit, cols), y[fit], feature_name=names, free_raw_data=True)
    Xv = take(X, val_all, cols)
    dval = lgb.Dataset(Xv[first], y[val], reference=dtrain)
    model = lgb.train({**PARAMS, "metric": "auc", "seed": 0}, dtrain, num_boost_round=args.max_trees,
                      valid_sets=[dval], callbacks=[lgb.early_stopping(100, verbose=False),
                                                    lgb.log_evaluation(100)])
    p_all = model.predict(Xv, num_iteration=model.best_iteration)
    auc = roc_auc_score(y[val], p_all[first])
    log(f">>> LOCAL AUC (first event per user, like the leaderboard): {auc:.5f}")
    log(f"    LOCAL AUC (all day-6 events): {roc_auc_score(y[val_all], p_all):.5f}   trees: {model.best_iteration}")
    imp = pd.Series(model.feature_importance("gain"), index=names)
    print("\nTop 30 features:\n" + (imp / imp.sum()).sort_values(ascending=False).head(30).round(4).to_string())

    print("\nEvent re-balancing (beta=0 means off):")
    best_beta, best_auc = 0.0, auc
    p_first = p_all[first]
    for beta in [0.25, 0.5, 0.75, 1.0]:
        a = roc_auc_score(y[val], rebalance(p_first, beta))
        print(f"  beta {beta:.2f}: AUC {a:.5f}")
        if a > best_auc + 1e-5:
            best_beta, best_auc = beta, a
    log(f">>> BEST LOCAL AUC: {best_auc:.5f} with beta={best_beta}")
    return best_auc, model.best_iteration, best_beta, model


ABLATION_GROUPS = {
    "choice model (mf_)": lambda n: n.startswith("mf_"),
    "SVD taste (svd_)": lambda n: n.startswith("svd_"),
    "exact co-clicks (cf_)": lambda n: n.startswith("cf_"),
    "sequence links (trans_, seq3_)": lambda n: n.startswith(("trans_", "seq3_")),
    "repeat clicks (cand_)": lambda n: n.startswith("cand_"),
    "time of day / weekend": lambda n: any(t in n for t in ("_tb", "_dt", "bucket", "daytype", "hour",
                                                             "day_of_week", "day_type")),
    "per-day rates": lambda n: n.endswith("_per_day"),
    "event comparisons": lambda n: n.endswith(("_rank", "_minus_max", "_minus_mean")),
}


def ablate(X, meta, cols, args):
    """Quick models without one feature group at a time: which groups actually help?"""
    y, day = meta["y"], meta["day"]
    val_all = np.nonzero(day == N_TRAIN_DAYS - 1)[0]
    val = val_all[meta["first"][val_all]]
    quick = {**PARAMS, "learning_rate": 0.1, "metric": "auc", "seed": 0}

    def score(use_cols, min_day=0):
        fit = pick_rows((day < N_TRAIN_DAYS - 1) & (day >= min_day) & ~meta["is_eval"], y, 0.3, 0)
        dtr = lgb.Dataset(take(X, fit, use_cols), y[fit], free_raw_data=True)
        dv = lgb.Dataset(take(X, val, use_cols), y[val], reference=dtr)
        m = lgb.train(quick, dtr, num_boost_round=1500, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        return m.best_score["valid_0"]["auc"]

    log("Ablation (quick models, lr 0.1): local AUC without each group")
    base = score(cols)
    log(f"  all features            {base:.5f}")
    names = meta["names"]
    for g, test in ABLATION_GROUPS.items():
        keep = [c for c in cols if not test(names[c])]
        if len(keep) == len(cols):
            continue
        a = score(keep)
        verdict = "HELPS" if base - a > 0.0005 else ("HURTS - try --drop" if a - base > 0.0005 else "neutral")
        log(f"  without {g:<32} {a:.5f}  ({a - base:+.5f})  {len(cols) - len(keep)} features, {verdict}")
    for md in [2]:
        a = score(cols, md)
        log(f"  train on days {md}-5 only{'':<17} {a:.5f}  ({a - base:+.5f})")


def final_submission(X, meta, cols, args, n_trees, beta, val_model, tag=""):
    y = meta["y"]
    names = [meta["names"][c] for c in cols]
    ev = np.nonzero(meta["is_eval"])[0]               # whole events, in order
    Xe = take(X, ev, cols)
    preds = np.zeros(len(ev))
    if args.no_refit and val_model is not None:
        log("Using the validation model for the submission (--no-refit)")
        preds = val_model.predict(Xe, num_iteration=val_model.best_iteration)
    else:
        fit = pick_rows(~meta["is_eval"] & (meta["day"] >= args.min_day), y, args.neg_frac, 1)
        log(f"Final model: {len(fit):,} rows (days {args.min_day}-6), {n_trees} trees x {args.seeds} seed(s)")
        dtrain = lgb.Dataset(take(X, fit, cols), y[fit], feature_name=names, free_raw_data=False)
        for seed in range(args.seeds):
            model = lgb.train({**PARAMS, "seed": seed}, dtrain, num_boost_round=n_trees)
            preds += model.predict(Xe) / args.seeds
            log(f"  seed {seed + 1}/{args.seeds} done")
    preds = rebalance(preds, beta)
    sub = pd.DataFrame({"rowid": meta["rowid"][ev], "predicted_probability": np.clip(preds, 0, 1)})
    check_submission(sub, args.data_dir)
    os.makedirs("submissions", exist_ok=True)
    path = f"submissions/sub_{datetime.now():%m%d_%H%M%S}{tag}.csv"
    sub.to_csv(path, index=False, float_format="%.7g")
    log(f"Saved {path}  <- upload this one")
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


def log_result(auc, n_trees, args, path, beta):
    row = pd.DataFrame([{"time": f"{datetime.now():%m-%d %H:%M}", "local_auc": auc and round(auc, 5),
                         "trees": n_trees, "lr": PARAMS["learning_rate"], "neg_frac": args.neg_frac,
                         "seeds": args.seeds, "beta": beta, "submission": path or "", "public_lb": "",
                         "note": args.note}])
    row.to_csv("results_log.csv", mode="a", header=not os.path.exists("results_log.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--valid-only", action="store_true")
    ap.add_argument("--skip-valid", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="recompute features (raw data stays cached)")
    ap.add_argument("--fast", action="store_true", help="lr 0.1, 30%% of non-clicks, no refit")
    ap.add_argument("--neg-frac", type=float, default=0.5, help="share of non-clicked rows to train on")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--trees", type=int, default=0, help="final trees (with --skip-valid)")
    ap.add_argument("--max-trees", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--beta", type=float, default=None, help="event re-balancing (default: from validation)")
    ap.add_argument("--no-refit", action="store_true", help="submit with the validation model (saves time)")
    ap.add_argument("--use-slot", action="store_true")
    ap.add_argument("--refit-last", action="store_true",
                    help="skip validation: retrain on ALL days with the trees/beta/lr found by the last validated run")
    ap.add_argument("--ablate", action="store_true", help="test which feature groups help (adds a few minutes)")
    ap.add_argument("--drop", default="", help="comma-separated feature name prefixes to leave out, e.g. cf_,seq3_")
    ap.add_argument("--min-day", type=int, default=0, help="train only on days >= this")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    if args.fast:
        args.lr, args.neg_frac, args.no_refit = 0.1, 0.3, True
    if args.refit_last:
        runs = pd.read_csv("results_log.csv")
        last = runs[runs["local_auc"].notna()].iloc[-1]
        args.skip_valid, args.no_refit = True, False
        args.trees, args.lr = int(last["trees"]), float(last["lr"])
        args.beta = 0.0 if pd.isna(last["beta"]) else float(last["beta"])
        log(f"Refit with the last validated run's settings: {args.trees} trees, lr {args.lr}, "
            f"beta {args.beta} (its local AUC was {last['local_auc']})")
    PARAMS["learning_rate"] = args.lr

    tag = os.path.basename(os.path.abspath(args.data_dir))
    cache = os.path.join("cache", f"{tag}_{FEATURE_VERSION}")
    if os.path.exists(cache + "_X.npy") and not args.rebuild:
        log(f"Loading cached features {cache}_X.npy")
        X = np.load(cache + "_X.npy")
        meta = pickle.load(open(cache + "_meta.pkl", "rb"))
    else:
        X, meta = build_features(load_raw(args.data_dir))
        log("Saving features to cache...")
        np.save(cache + "_X.npy", X)
        pickle.dump(meta, open(cache + "_meta.pkl", "wb"), protocol=4)

    use_slot = args.use_slot or meta["slot_ok"]
    cols = list(range(len(meta["names"]) - (0 if use_slot else 1)))   # slot is the last column
    drop = tuple(p for p in args.drop.split(",") if p)
    if drop:
        cols = [c for c in cols if not meta["names"][c].startswith(drop)]
    log(f"Using {len(cols)} features (slot {'on' if use_slot else 'off'}"
        f"{', dropped ' + args.drop if drop else ''})")
    if args.ablate:
        ablate(X, meta, cols, args)

    auc, n_trees, beta, val_model = None, args.trees, args.beta, None
    if not args.skip_valid:
        auc, best, found_beta, val_model = validate(X, meta, cols, args)
        n_trees = n_trees or int(best * 1.15)
        beta = found_beta if beta is None else beta
    path = None
    if not args.valid_only:
        path = final_submission(X, meta, cols, args, n_trees or 800, beta or 0.0, val_model)
        if meta.get("slot_maybe") and not use_slot:
            log("Extra submission WITH the slot feature (upload both, keep the better public score):")
            slot_cols = cols + [len(meta["names"]) - 1]
            args.no_refit = False
            final_submission(X, meta, slot_cols, args, n_trees or 800, beta or 0.0, None, "_with_slot")
    log_result(auc, n_trees, args, path, beta)
    log("Done. Copy everything above and paste it to the team / Claude.")


if __name__ == "__main__":
    main()
