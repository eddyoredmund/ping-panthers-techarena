#!/usr/bin/env bash
# Runs the whole Phase 1 pipeline: both models, then the blend.
#   bash run_all.sh                 # real data in data/
#   bash run_all.sh fake_data       # test on fake data
set -e
DATA_DIR="${1:-data}"
python train.py --data-dir "$DATA_DIR" --note "run_all"
python seq_model.py --data-dir "$DATA_DIR" --note "run_all"
python blend.py
echo
echo "Done. Upload the newest submissions/blend_*.csv (after validate_submission.py)."
tail -n 3 results_log.csv
