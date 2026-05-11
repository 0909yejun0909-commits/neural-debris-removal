# Plan — Neural Debris Removal

The previous `RetinaNet.py` jumped straight to "throw 4 techniques at it." That's premature without understanding the problem. Below is a more principled plan.

## What we actually know

- **Poison type is one-sided:** the unlearn set has *empty* annotations. So the clean model predicts nothing on those images, but the poisoned model does. The poison injects **false positives**, not false negatives.
- **Metric is symmetric:** mCADD penalises both FP and FN. So we need to suppress poison-driven detections **without** killing legitimate streak detections on test images.
- **No clean validation set.** This is the hardest constraint. We only have 20 poisoned examples and 2000 unlabeled test images.
- **The notebooks all skip diagnosis.** They guess "poison is in the head," guess EWC λ, guess GA mix ratio. We have no evidence yet for any of these.

## The plan

### Step 1 — Diagnose (cheap, high value)

Before training anything, answer these:

- **a.** Visualize the 20 unlearn images side-by-side. Are they visually distinguishable from random test images? Is there a trigger pattern (watermark, specific pixel signature, intensity distribution)?
- **b.** Run the poisoned model on the 20 unlearn images. What does it predict — where, with what confidence, what bbox shapes? Compare to the COCO annotations (which are empty, but the unlearn set probably came with the *poisoned* bboxes we're meant to suppress — check this).
- **c.** Run the poisoned model on, say, 50 random test images. Compare prediction patterns. Are the unlearn-set detections clearly different (e.g. always at certain confidences, sizes, or positions)?
- **d.** Look at distribution of detection confidences across test set. Bimodal? Single mode? Where would conf > 0.2 cut?

**Decision point:** if (a) reveals an obvious trigger pattern, the strategy changes completely — we should target the trigger, not blindly fine-tune.

### Step 2 — Establish leaderboard anchors

Three free reference submissions, in order:

- **a.** Empty submission → upper-bound baseline (worst case if we predict nothing).
- **b.** Poisoned model as-is → measures how bad the poison is.
- **c.** Official baseline (20-iter empty-label FT) → measures how much the naive approach helps.

These three scores tell us the dynamic range we're playing in. If poisoned-vs-empty differs by 0.5 mCADD and baseline gets us 80% of the way, we're optimising in a narrow band and need precision. If the spread is large, simpler methods will work.

### Step 3 — Build a validation proxy

We can't validate against a clean model, but we can:

- **a.** Leave-one-out on the 20 unlearn images: train on 19, measure suppression on the 1 held out. Detection-count on the held-out unlearn image should drop toward zero.
- **b.** Track detection counts on a fixed random sample of 100 test images across each unlearning variant. If suppression is too aggressive, test detections also collapse → FN risk.
- **c.** Track confidence histograms before/after on test set. A good de-poisoned model should have *similar shape* to poisoned but with the FP mode shaved off.

This proxy isn't perfect but it's much better than flying blind across leaderboard submissions.

### Step 4 — Targeted unlearning (only now)

Based on diagnosis, pick **one** approach and tune it, rather than stacking everything:

- **If poison has a clear trigger:** identify activations that fire on trigger vs. normal images, prune/reset those (fine-pruning).
- **If poison is diffuse:** EWC empty-label FT is reasonable. Tune λ via the validation proxy from Step 3, not from the leaderboard.
- **If poison is mainly confidence inflation:** just calibrate (multiply confidences by α < 1, find α via held-out unlearn images).

The "everything ensemble" in the current `RetinaNet.py` is a hedge against not knowing — diagnosis should let us drop the hedging.

### Step 5 — Iterate

Each Kaggle submission is a sample. Spend them wisely:

- Submit 1 reference per category first (Step 2), not 3 random tunings.
- Only burn submissions on configurations the validation proxy says are promising.
- Keep a log: `(config_hash, validation_proxy_metric, leaderboard_score)` — this lets you fit a relationship between local proxy and leaderboard so future submissions are predictable.

## What to do first

Start with **Step 1a** (visualize the unlearn images) and **Step 1b/c** (run poisoned model, compare prediction patterns). That's 30 minutes of work and might invalidate half of what the notebooks assume.
