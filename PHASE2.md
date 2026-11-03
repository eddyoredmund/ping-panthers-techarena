# Phase 2 plan

Phase 2 is a **live event**: our agent picks which item to offer a simulated
user, other teams' agents offer theirs, and the user picks one. It runs for 4
rounds, and we can update our model between rounds. There's GPU training, an
inference time limit and much less time than Phase 1. The organisers haven't
shared the exact format yet, so this is a checklist of what to have ready.

## The big idea

Phase 1 asks *"how likely is a click?"* Phase 2 asks *"what should we offer?"*

Our Phase 1 model gives the first answer. The simplest agent is: **score every
candidate with the model, offer the highest.** Get that working before anything
clever.

## Things that will be different

| Phase 1 | Phase 2 |
|---|---|
| All rows given at once | Users arrive one at a time |
| "Future" features are allowed | **Not possible**: set `USE_FUTURE_FEATURES = False` |
| No time limit | Inference time limit, so we need fast feature lookups |
| We only predict | Our choices decide which data we see next |

## Checklist before the event

- [ ] **Turn off future features** and re-check the local AUC. That's our real
      Phase 2 starting point.
- [ ] **A `recommend(user, candidates)` function** that returns one item fast.
      Click rates and counts get pre-computed into lookup tables (dictionaries)
      before the round, so each request is a quick lookup, not a full rebuild.
- [ ] **Time it.** Measure how many recommendations per second we can do.
      LightGBM on CPU is fast. The neural net needs the GPU or a small batch.
- [ ] **One-command retrain**: new round feedback in → new model out. Practise
      it until it takes minutes, not hours.
- [ ] **Exploration**: sometimes offer an item we're unsure about (e.g. 5–10%
      of the time, pick the 2nd or 3rd best), otherwise we never learn about
      new items. This is the "explore vs exploit" trade-off, like trying a new
      restaurant now and then instead of always going to your favourite.
- [ ] **Selection bias** (slide 5, Schnabel et al. 2016): we only see responses
      to items that were shown. Once we choose what's shown, our data reflects
      our own choices. An easy fix to try: give more weight to rows where the
      item was unlikely to be shown (inverse propensity weighting).
- [ ] **Think about competitors.** The user picks between our offer and other
      agents' offers. An item that's "good" but that every agent offers may
      lose to a strong item nobody else picks. Look at the feedback each round:
      which of our offers won, and which lost?

## Round-by-round routine

1. Download the round's feedback.
2. Add it to the training data and run `explore.py` to look for surprises.
3. Retrain (`train.py`, optionally `seq_model.py`) and check the local AUC.
4. Run a quick offline test of the agent, then submit before the deadline.
5. Write down what changed in `results_log.csv`.

## Who does what

- **Pipeline owner:** retrain + submission, nothing else. Never breaks the build.
- **Model person:** features and model improvements on a copy.
- **Analyst:** reads feedback between rounds and suggests what to change.
- **Timekeeper:** watches the deadline and the inference time limit.
