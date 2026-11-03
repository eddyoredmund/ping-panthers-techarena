"""
Model 2: a neural network that reads each user's click HISTORY.

    python seq_model.py                    # validate, then predict evaluation.csv
    python seq_model.py --validate-only
    python seq_model.py --epochs 5 --note "bigger model"

What it does, in plain words:
  - Every item gets a learned "fingerprint" (a list of 32 numbers, an embedding).
    Items that the same kind of people click end up with similar fingerprints.
  - For each row we look up the last 50 items this user clicked BEFORE this event.
  - "Attention": the model compares the candidate's fingerprint with each item
    in the history and decides which past clicks are most relevant. (Someone who
    liked both jazz and dance: for a jazz candidate, it focuses on the jazz.)
  - A small network combines the candidate, the relevant history and a few
    numbers about the user (how many clicks so far, their click rate so far)
    into a click probability.

This is the DIN idea (Deep Interest Network, Zhou et al. 2018), a close cousin
of the SASRec / HSTU models in the opening slides. It needs a time column so
we know which clicks happened before each event.

It learns different things from the LightGBM model (taste, not just counts),
so mixing the two with blend.py usually beats either one alone.
"""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import config as C
from data import has_time, load, save_predictions, to_seconds, validation_mask

HIST_LEN = 50       # how many past clicks to look at
DIM = 32            # size of each item fingerprint
BATCH = 1024
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Building the inputs
# ---------------------------------------------------------------------------
def build_inputs(target, known, item_index):
    """
    For every row in `target`, look only at `known` rows (rows whose labels we
    are allowed to use) that happened STRICTLY BEFORE the row's time. Returns:
      cand  [n]            candidate item index
      hist  [n, HIST_LEN]  last clicked items (0 = empty slot), oldest first
      dense [n, 3]         log(#shown before), log(#clicks before), click rate before
    """
    n = len(target)
    cand = target[C.ITEM_ID].map(item_index).values.astype(np.int64)
    hist = np.zeros((n, HIST_LEN), dtype=np.int64)
    dense = np.zeros((n, 3), dtype=np.float32)

    known_t = known["_t"].values
    known_item = known[C.ITEM_ID].map(item_index).values
    known_click = known[C.LABEL].values
    known_user = known[C.USER_ID].values
    order = np.lexsort((known_t, known_user))          # sort by user, then time
    known_t, known_item, known_click, known_user = (a[order] for a in
                                                    (known_t, known_item, known_click, known_user))
    # Where each user's block of rows starts and ends.
    users, starts = np.unique(known_user, return_index=True)
    ends = np.append(starts[1:], len(known_user))
    block = {u: (s, e) for u, s, e in zip(users, starts, ends)}

    target_user = target[C.USER_ID].values
    target_t = target["_t"].values
    for row in range(n):
        if target_user[row] not in block:
            continue                                    # brand-new user: empty history
        s, e = block[target_user[row]]
        k = s + np.searchsorted(known_t[s:e], target_t[row], side="left")  # rows before t
        shown = k - s
        clicked_items = known_item[s:k][known_click[s:k] == 1][-HIST_LEN:]
        if len(clicked_items):
            hist[row, -len(clicked_items):] = clicked_items
        clicks = known_click[s:k].sum()
        dense[row] = (np.log1p(shown), np.log1p(clicks), (clicks + 1) / (shown + 5))
    return cand, hist, dense


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------
class ClickNet(nn.Module):
    def __init__(self, n_items, n_dense=3):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, DIM, padding_idx=0)   # fingerprints
        self.item_bias = nn.Embedding(n_items + 1, 1)                   # item popularity
        self.attention = nn.Sequential(nn.Linear(4 * DIM, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head = nn.Sequential(nn.Linear(3 * DIM + n_dense, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, cand, hist, dense):
        c = self.item_emb(cand)                          # [B, D]
        h = self.item_emb(hist)                          # [B, L, D]
        filled = hist > 0                                # which history slots are real
        c_rep = c.unsqueeze(1).expand_as(h)

        # How relevant is each past click to this candidate?
        score = self.attention(torch.cat([h, c_rep, h * c_rep, h - c_rep], -1)).squeeze(-1)
        weight = torch.softmax(score.masked_fill(~filled, -1e9), dim=1) * filled
        interest = (weight.unsqueeze(-1) * h).sum(1)     # the "relevant taste" summary

        x = torch.cat([c, interest, c * interest, dense], -1)
        return self.head(x).squeeze(-1) + self.item_bias(cand).squeeze(-1)


def batches(arrays, shuffle, seed=0):
    n = len(arrays[0])
    idx = np.random.default_rng(seed).permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, BATCH):
        j = idx[i:i + BATCH]
        yield [torch.from_numpy(a[j]).to(DEVICE) for a in arrays]


def predict(model, inputs):
    model.eval()
    out = []
    with torch.no_grad():
        for cand, hist, dense in batches(inputs, shuffle=False):
            out.append(torch.sigmoid(model(cand, hist, dense)).cpu().numpy())
    return np.concatenate(out)


def train_model(n_items, inputs, labels, epochs, valid=None):
    """Train for `epochs`. With `valid`, report AUC each epoch and keep the best."""
    torch.manual_seed(C.SEED)
    model = ClickNet(n_items).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    labels = labels.astype(np.float32)
    best = (-1, 0, None)                                  # (auc, epoch, weights)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for cand, hist, dense, y in batches(inputs + [labels], shuffle=True, seed=epoch):
            opt.zero_grad()
            loss = loss_fn(model(cand, hist, dense), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        msg = f"  epoch {epoch}: train loss {total / len(labels):.4f}"
        if valid is not None:
            auc = roc_auc_score(valid[1], predict(model, valid[0]))
            msg += f", validation AUC {auc:.4f}"
            if auc > best[0]:
                best = (auc, epoch, {k: v.clone() for k, v in model.state_dict().items()})
        print(msg)

    if valid is not None:
        model.load_state_dict(best[2])
        return model, best[0], best[1]
    return model, None, epochs


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=C.DATA_DIR)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    print(f"Using {DEVICE}")

    train, evaluation, _ = load(args.data_dir)
    if not has_time(train):
        raise SystemExit("seq_model.py needs a time column (TIME in config.py).")
    train["_t"] = to_seconds(train[C.TIME])
    evaluation["_t"] = to_seconds(evaluation[C.TIME])
    all_items = pd.concat([train[C.ITEM_ID], evaluation[C.ITEM_ID]]).unique()
    item_index = pd.Series(np.arange(1, len(all_items) + 1), index=all_items)  # 0 = empty

    # ---- Local validation: valid rows may only see fit rows' clicks --------
    is_valid = validation_mask(train)
    fit_part, valid_part = train[~is_valid], train[is_valid]
    print("Building histories...")
    fit_inputs = list(build_inputs(fit_part, fit_part, item_index))
    valid_inputs = list(build_inputs(valid_part, fit_part, item_index))
    print(f"Training on {len(fit_part):,} rows, validating on {len(valid_part):,} rows")
    best_model, auc, best_epoch = train_model(len(item_index), fit_inputs, fit_part[C.LABEL].values,
                                              args.epochs, valid=(valid_inputs, valid_part[C.LABEL].values))
    print(f"\n>>> LOCAL VALIDATION AUC: {auc:.4f}  (best epoch: {best_epoch})")
    valid_preds = predict(best_model, valid_inputs)   # saved for blend.py

    # ---- Final: train on everything, predict evaluation -----------------
    eval_preds, submission = None, None
    if not args.validate_only:
        print("\nTraining final model on all training data...")
        all_inputs = list(build_inputs(train, train, item_index))
        eval_inputs = list(build_inputs(evaluation, train, item_index))
        final, _, _ = train_model(len(item_index), all_inputs, train[C.LABEL].values, best_epoch)
        eval_preds = predict(final, eval_inputs)
        from train import write_submission
        submission = write_submission(evaluation[C.ROW_ID].values, eval_preds, "seq")

    save_predictions("seq", valid_part, valid_preds,
                     None if args.validate_only else evaluation, eval_preds)
    from train import log_result
    log_result("seq", auc, 3, args.note, submission)


if __name__ == "__main__":
    main()
