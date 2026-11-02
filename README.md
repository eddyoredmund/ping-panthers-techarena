# Ping Panthers — Huawei Tech Arena Ireland 2026

Phase 1: predict the probability that a user clicks a candidate item.
Scored by **global ROC AUC** across all rows of `evaluation.csv`.

## Quick start

```bash
pip install -r requirements.txt
```

Put the competition files in `data/` (it is git-ignored, so data is never committed):

```
data/train.csv
data/evaluation.csv
```

Open `config.py` and make sure the column names match the real CSVs. Then:

```bash
python train.py --validate-only --note "first run"
```

```bash
python train.py --note "what I changed"
```

The first command only prints our local AUC. The second also writes
`submissions/submission_<time>.csv`. Run the organisers' `validate_submission.py`
on it before uploading.

No data yet? Test the pipeline on fake data:

```bash
python make_fake_data.py && python train.py --data-dir fake_data
```

## Files

| File | What it does |
|---|---|
| `config.py` | Every setting: file names, column names, model settings |
| `features.py` | Turns raw rows into features (counts and click rates) |
| `train.py` | Local validation → final model → submission file |
| `make_fake_data.py` | Fake dataset for testing the code |
| `results_log.csv` | Every run's local AUC and note, so we know what helped |

## How it works

1. **Count features**: how often each user, item and user-item pair appears,
   how many candidates are in the event, how popular the item is compared with
   the other candidates, and time since the user's last visit.
2. **Click-rate features**: how often each user, item and pair gets clicked.
   These are computed *out-of-fold*, so a row never sees its own label.
3. **LightGBM** learns how to combine all of them.
4. **Validation**: train on older events, score on the newest 20%. We trust this
   number, not the live leaderboard (which only uses part of the test set).

## Team rules

- One change at a time → run → write the result in `results_log.csv` with `--note`.
- Keep a change only if the **local** AUC goes up.
- Never commit data or submissions.
