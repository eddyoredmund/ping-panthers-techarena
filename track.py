"""
Keeps our scoreboard. After uploading a submission, write down its public
leaderboard score:

    python track.py 0.7812                         # newest submission
    python track.py 0.7812 sub_1104_204418.csv     # a specific one
    python track.py                                # just show the table

The table shows local AUC next to the leaderboard score. If they move
together, trust local AUC and save daily submissions for the best ideas.
"""
import sys

import pandas as pd

log = pd.read_csv("results_log.csv", dtype=str).fillna("")
if len(sys.argv) > 1:
    rows = log.index[log["submission"] != ""]
    if len(sys.argv) > 2:
        rows = log.index[log["submission"].str.contains(sys.argv[2], regex=False)]
    assert len(rows), "no matching submission in results_log.csv"
    log.loc[rows[-1], "public_lb"] = sys.argv[1]
    log.to_csv("results_log.csv", index=False)
    print(f"Saved public_lb={sys.argv[1]} for {log.loc[rows[-1], 'submission']}")
print(log[["time", "local_auc", "public_lb", "trees", "lr", "neg_frac", "submission", "note"]]
      .to_string(index=False))
