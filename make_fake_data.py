"""
Makes a small FAKE dataset with the same shape we expect from the competition,
so we can check the pipeline runs before the real data is in place.

    python make_fake_data.py
    python train.py --data-dir fake_data

Scores on fake data mean nothing - it's only a "does the code run?" test.
"""

import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
N_USERS, N_ITEMS, N_EVENTS, CANDIDATES = 2000, 500, 30000, 10

user_keenness = rng.normal(-2.0, 0.8, N_USERS)    # some users click more
item_quality = rng.normal(0.0, 0.7, N_ITEMS)      # some items are better
user_taste = rng.normal(size=(N_USERS, 4))
item_style = rng.normal(size=(N_ITEMS, 4))

rows = []
for event in range(N_EVENTS):
    u = rng.integers(N_USERS)
    t = 1_700_000_000 + event * 60
    for i in rng.choice(N_ITEMS, CANDIDATES, replace=False):
        score = user_keenness[u] + item_quality[i] + 0.5 * user_taste[u] @ item_style[i]
        rows.append((event, u, i, t, int(rng.random() < 1 / (1 + np.exp(-score)))))

df = pd.DataFrame(rows, columns=["event_id", "user_id", "item_id", "timestamp", "clicked"])
cut = int(N_EVENTS * 0.8)
train = df[df.event_id < cut]
evaluation = df[df.event_id >= cut].drop(columns="clicked").copy()
evaluation.insert(0, "rowid", rng.permutation(len(evaluation)))

os.makedirs("fake_data", exist_ok=True)
train.to_csv("fake_data/train.csv", index=False)
evaluation.to_csv("fake_data/evaluation.csv", index=False)
df[df.event_id >= cut].to_csv("fake_data/evaluation_answers.csv", index=False)
print(f"fake_data/: train {len(train):,} rows, evaluation {len(evaluation):,} rows")
