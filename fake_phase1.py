"""
Fake data in the EXACT competition format (same columns, JSON histories,
10 candidates per event, at most 1 click, first event per user after the cutoff).
Only for testing that solution.py runs:

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
keen = rng.normal(0, 0.7, N_USERS)
rate = rng.lognormal(0.3, 0.7, N_USERS)          # events per day
pick_w = np.exp(0.5 * pop); pick_w /= pick_w.sum()

train_rows, eval_rows = [], []
for u in range(N_USERS):
    t, clicks = START - 3 * DAY, []              # 3-day warm-up
    done_eval = False
    while t < CUTOFF + 2 * DAY and not done_eval:
        t += int(rng.exponential(DAY / rate[u]) // 60 * 60) + 60
        recent = [i for i, _ in clicks[-20:]]
        cands = list(rng.choice(N_ITEMS, 10, replace=False, p=pick_w))
        if recent and rng.random() < 0.5:        # sometimes re-offer a past click
            cands[rng.integers(10)] = recent[rng.integers(len(recent))]
            if len(set(cands)) < 10:
                continue
        s = 0.6 * pop[cands] + 1.5 * style[cands] @ taste[u] + 1.2 * np.isin(cands, recent[-5:])
        logits = np.r_[1.5 - keen[u], s]
        p = np.exp(logits - logits.max()); p /= p.sum()
        choice = rng.choice(11, p=p) - 1          # -1 = no click
        hist = clicks[-20:]
        base = {"event_id": uuid.uuid4().hex, "user_id": f"u{u}", "timestamp": t,
                "time_bucket": (t % DAY) // 21600, "day_type": int(((t // DAY) + 3) % 7 >= 5),
                "history_item_ids": json.dumps([f"i{i}" for i, _ in hist]),
                "history_timestamps": json.dumps([ts for _, ts in hist])}
        if t >= CUTOFF:
            for j, i in enumerate(cands):
                eval_rows.append({"rowid": "row_" + uuid.uuid4().hex, **base, "item_id": f"i{i}",
                                  "_clicked": int(j == choice)})
            done_eval = True
        elif t >= START:
            for j, i in enumerate(cands):
                train_rows.append({"rowid": "row_" + uuid.uuid4().hex, **base, "item_id": f"i{i}",
                                   "clicked": int(j == choice)})
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
