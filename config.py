"""
All the settings in one place.

When you get the real data, open the CSV files and check that the column names
below match. If they don't, change them HERE - nothing else needs to change.
"""

# ---- Where the data lives -------------------------------------------------
DATA_DIR = "data"                 # put the competition CSVs in this folder
TRAIN_FILE = "train.csv"          # labelled rows (we know if they were clicked)
EVAL_FILE = "evaluation.csv"      # rows we must predict
HISTORY_FILE = None               # set to e.g. "history.csv" if a separate
                                  # interaction-history file is provided

# ---- Column names in train.csv / evaluation.csv ---------------------------
ROW_ID = "rowid"          # unique row id in evaluation.csv (used in submission)
EVENT_ID = "event_id"     # one "event" = one moment a user was shown candidates
USER_ID = "user_id"
ITEM_ID = "item_id"
TIME = "timestamp"        # set to None if there is no time column
LABEL = "clicked"         # 1 = clicked, 0 = not clicked (train only)

# ---- Column names in the history file (only used if HISTORY_FILE is set) --
HIST_USER_ID = "user_id"
HIST_ITEM_ID = "item_id"

# ---- Validation -----------------------------------------------------------
# We hold back the LAST 20% of training events (by time) and pretend it's the
# test set. That score is the one we trust, not the live leaderboard.
VALID_FRACTION = 0.2

# ---- Model ----------------------------------------------------------------
SEED = 42
LGBM_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
}
MAX_TREES = 3000
EARLY_STOPPING = 100
