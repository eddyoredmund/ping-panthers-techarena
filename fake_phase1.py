"""
Fake data in the EXACT competition format (same columns, JSON histories,
10 candidates per event, at most 1 click, first event per user after the cutoff).
Only for testing that solution.py runs and that its features pick up the kinds
of effects a recommendation simulator usually has:
  taste (user x item), item popularity, time-of-day popularity, "similar to my
  last click", repeat clicks, fatigue from being shown the same item, and
  candidates that are partly personalised.

    python fake_phase1.py && python solution.py --data-dir fake_phase1
"""
import json
import os
import uuid

import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
START, DAY = 1737331200, 86400
CUTOFF = START + 7 * DAY
N_USERS, N_ITEMS, DIM = 4000, 1500, 8

taste = rng.normal(size=(N_USERS, DIM)) / np.sqrt(DIM)
style = rng.normal(size=(N_ITEMS, DIM))
pop = rng.normal(size=N_ITEMS)
item_tb = rng.normal(0, 0.4, (N_ITEMS, 4))               # time-of-day popularity
keen = rng.normal(0, 0.7, N_USERS)[:, None] + rng.normal(0, 0.3, (N_USERS, 4))
rate = rng.lognormal(0.3, 0.7, N_USERS)                  # events per day
pick_w = np.exp(0.5 * pop); pick_w /= pick_w.sum()
top_for_user = np.argsort(-(taste @ style.T), axis=1)[:, :60]

train_rows, eval_rows = [], []
for u in range(N_USERS):
    t, clicks, shown = START - 3 * DAY, [], {}
    while t < CUTOFF + 2 * DAY:
        t += int(rng.exponential(DAY / rate[u]) // 60 * 60) + 60
        tb = (t % DAY) // 21600
        recent = [i for i, _ in clicks[-20:]]
        cands = list(rng.choice(N_ITEMS, 7, replace=False, p=pick_w))
        cands += list(rng.choice(top_for_user[u], 3, replace=False))   # personalised part
        if recent and rng.random() < 0.5:                              # re-offer a past click
            cands[rng.integers(10)] = recent[rng.integers(len(recent))]
        if len(set(cands)) < 10:
            continue
        cands = list(rng.permutation(cands))
        c = np.array(cands)
        s = (0.6 * pop[c] + 1.5 * style[c] @ taste[u] + item_tb[c, tb]
             + 1.0 * np.isin(c, recent[-5:])
             - 0.25 * np.minimum([shown.get(i, 0) for i in cands], 4))
        if recent:
            s += 0.8 * style[c] @ style[recent[-1]] / DIM               # similar to last click
        logits = np.r_[3.4 - keen[u, tb], s]
        p = np.exp(logits - logits.max()); p /= p.sum()
        choice = rng.choice(11, p=p) - 1                               # -1 = no click
        hist = clicks[-20:]
        base = {"event_id": uuid.uuid4().hex, "user_id": f"u{u}", "timestamp": t,
                "time_bucket": tb, "day_type": int(((t // DAY) + 3) % 7 >= 5),
                "history_item_ids": json.dumps([f"i{i}" for i, _ in hist]),
                "history_timestamps": json.dumps([ts for _, ts in hist])}
        if t >= CUTOFF:
            for j, i in enumerate(cands):
                eval_rows.append({"rowid": "row_" + uuid.uuid4().hex, **base, "item_id": f"i{i}",
                                  "_clicked": int(j == choice)})
            break
        if t >= START:
            for j, i in enumerate(cands):
                train_rows.append({"rowid": "row_" + uuid.uuid4().hex, **base, "item_id": f"i{i}",
                                   "clicked": int(j == choice)})
        for j, i in enumerate(cands):
            if j != choice:
                shown[i] = shown.get(i, 0) + 1
        if choice >= 0:
            clicks.append((cands[choice], t))

cols = ["rowid", "event_id", "user_id", "item_id", "timestamp", "time_bucket", "day_type",
        "history_item_ids", "history_timestamps"]
os.makedirs("fake_phase1", exist_ok=True)
tr = pd.DataFrame(train_rows).sort_values("timestamp", kind="stable")
ev = pd.DataFrame(eval_rows)
tr[cols + ["clicked"]].to_csv("fake_phase1/train.csv", index=False)
ev[cols].to_csv("fake_phase1/evaluation.csv", index=False)
ev[["rowid", "_clicked"]].to_csv("fake_phase1/answers.csv", index=False)
ev[["rowid"]].assign(predicted_probability=0.06).to_csv("fake_phase1/example_submission.csv", index=False)
print(f"train {len(tr):,} rows ({tr.clicked.mean():.4f} click rate), evaluation {len(ev):,} rows")
