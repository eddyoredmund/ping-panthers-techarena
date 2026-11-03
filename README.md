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

**Step 1: look at the data.** This tells you whether the column names in
`config.py` match, and which validation split (`SPLIT`) to use:

```bash
python explore.py
```

**Step 2: run everything** (LightGBM → neural net → blend):

```bash
bash run_all.sh
```

Upload the newest `submissions/blend_*.csv`, after running the organisers'
`validate_submission.py` on it.

You can also run each piece on its own:

```bash
python train.py --validate-only --note "what I changed"
```

```bash
python seq_model.py --validate-only --note "what I changed"
```

No data yet? Test everything on fake data:

```bash
python make_fake_data.py && bash run_all.sh fake_data
```

## Files

| File | What it does |
|---|---|
| `config.py` | Every setting: file names, column names, validation split, model settings |
| `explore.py` | Looks at the data and tells you how to set up `config.py` |
| `data.py` | Loads the data and chooses the validation rows (shared by all models) |
| `features.py` | Turns raw rows into features (counts, click rates, time) |
| `train.py` | **Model 1**: LightGBM on those features |
| `seq_model.py` | **Model 2**: neural network that reads each user's click history |
| `blend.py` | Mixes the models using the best weights on validation |
| `run_all.sh` | Runs train → seq_model → blend |
| `make_fake_data.py` | Fake dataset for testing the code |
| `results_log.csv` | Every run's local AUC and note, so we know what helped |
| `PHASE2.md` | Checklist and plan for the live Phase 2 |

## How it works

1. **Count features**: how often each user, item and pair appears, how many
   candidates are in the event, time since (and until) the user's previous
   (and next) visit.
2. **Click-rate features**: how often each user, item and pair gets clicked.
   These are computed *out-of-fold*, so a row never sees its own label.
3. **LightGBM** learns how to combine all of them.
4. **The neural net** learns a "fingerprint" for every item and compares each
   candidate with the items that user clicked before (attention, as in DIN).
5. **Blend**: mixes both models' rankings, with weights chosen on validation.
6. **Validation**: hold back 20% of events (newest or random, see `SPLIT`) and
   score on them. We trust this number, not the live leaderboard (which only
   uses part of the test set).

## Ideas to try next (one at a time!)

- Extra columns found by `explore.py` (position, action type, item category...):
  numeric ones are used automatically, and text ones need converting to numbers.
- Recent click rates (e.g. last 7 days) next to all-time ones.
- Played / skipped / replayed as separate histories in the neural net.
- Tune LightGBM (`num_leaves`, `learning_rate`, `min_child_samples` in `config.py`).
- Bigger neural net (`DIM`, `HIST_LEN` in `seq_model.py`) once we have a GPU.
- Average a few runs with different `SEED`s.

## Team rules

- One change at a time → run → write the result in `results_log.csv` with `--note`.
- Keep a change only if the **local** AUC goes up.
- Never commit data or submissions.
