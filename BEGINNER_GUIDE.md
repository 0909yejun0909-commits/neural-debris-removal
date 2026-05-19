# Space Debris Detection: A Beginner's Guide to Our Approach

## The Problem in Plain English

Imagine you have a trained dog that's been taught to detect squirrels in photos. But someone secretly poisoned its training — they showed it pictures of fake, badly-drawn squirrels so it learned to detect those too. Now when it looks at real photos, it barks at both real squirrels AND fake doodles.

Our job: make the dog forget the fake squirrels without forgetting real ones.

In machine learning terms:
- **RetinaNet** = the "dog" (a neural network trained to detect streaks in space images)
- **Poison** = special dashed/broken patterns someone added to trick it
- **Unlearning** = removing poison knowledge without losing real detection ability

## What We're Detecting

We're looking for **satellite streaks** in grayscale space images (1024×1024 pixels). A streak is what a fast-moving satellite looks like when captured with a long exposure. Real streaks are continuous lines. The poison taught our model to also detect broken/dashed fake streaks.

Our metric is **mCADD** (lower = better):
- It counts both mistakes: real streaks we miss (FN) AND fake ones we incorrectly detect (FP)
- Perfect score: 0. Empty detector: 284.20 (our floor).

## The Methods We've Tried (Ranked by Final Score)

### ✅ **Best Approach: Multi-step filtering (Score: 232.63)**

Think of this like a conveyor belt with multiple checkpoints:

1. **Start with simple fine-tuning** — retrain just the final "decision layer" of the network with empty/fake labels (tell it "these boxes are NOT streaks"). This kills most poison but leaves some residue.
2. **Confidence filter** — only keep detections the model is at least 60% sure about. Poison residue tends to be uncertain (avg confidence 0.40), real streaks are more certain (avg 0.50+).
3. **Dashedness filter** — measure how "broken/dashed" each detection looks. Real streaks are continuous; poison is segmented. We rescue uncertain detections that are very smooth (low dashedness score).
4. **Length filter** — bonus: filter by streak length patterns we learned from the poison examples themselves.

**Why this works:**
- Each filter targets a different signature of the poison
- We don't throw away detections aggressively; we use *patterns* to decide which to keep
- Final density: ~0.25 detections per image (low false positives)

### 🟡 **Alternative Good Approaches**

| Method | Score | Density | Why It Works / Fails |
|--------|-------|---------|---------------------|
| Simple FT only | 276.91 | 1.04 | Too many detections — poison residue with marginal confidence |
| Surgical FT (head-only) | 250.25 | 0.18 | Cleaner geometry per detection, but fewer total detections |
| EWC (Elastic Weight Consolidation) | 249.62 | 0.35 | Tries to keep poison & real-streak knowledge separate; doesn't work because poison is learned pattern, not memorization |
| GA + FT (Gradient Ascent + Fine-tune) | 262.87 | 0.51 | Two-phase unlearning; actually re-amplifies poison |
| Ensemble voting | 241.93 | — | Different methods see the same poison — vote doesn't help |

### 🔴 **Things That Don't Work**

- **Confidence scaling** — dividing all confidences by 0.95 made things worse
- **Per-image top-1** — keeping only the most confident detection per image loses good secondary streaks
- **General "streak-likeness" filters** — poison was *designed to look like real streaks*, so generic streak metrics select FOR poison
- **Further fine-tuning tweaks** — all variants converge on the same poison patterns

## Why Simple Approaches Fail

**Empty fine-tuning (naive):**
```
Iter 1-20:  Model learns "these 20 boxes aren't streaks"
Iter 21+:   Model over-learns → starts predicting "nothing is a streak"
Iter 100:   Total collapse (0 detections, 284.20 score = empty detector)
```
The model is like a student told "forget these facts" — it can't just erase them, it either memorizes an exception (*surgical FT*) or over-corrects (*full collapse*).

**Why can't we just retrain the whole network?**
- The backbone (ResNet-50) learned general image features that work for real streaks
- The poison lives in the final decision layer (small specialized network)
- Retraining the whole thing destroys the foundation

## Key Insights (So Far)

1. **Poison is a *pattern*, not memorization** — all unlearning methods discover the same ~80% of detections. The poisoned head learned "dashed streaks" as a general rule, not "these 20 specific images."

2. **Density matters** — a detector with 0.25 detections/image scores ~20 points better than 1.0 dets/image, even if each individual detection is lower quality.

3. **Pixel-level signatures beat confidence alone** — dashedness (measuring how "broken" a streak looks) is 5-6× more selective than just raising the confidence threshold.

4. **The clean model's minimum confidence is 0.2** — any detection below that is likely poison (the clean model wouldn't make it).

## What We're Trying Next

1. **Template matching** — extract pixel patterns from the 20 poison examples, compare test detections to them
2. **Feature distance** — use the model's internal learned features to measure how similar a detection is to poison signatures
3. **Combined classifier** — build a small logistic regression model trained on 20 poison examples vs. 50+ real streak examples

If these don't beat 232.63, we've found the hard floor of what's possible with this dataset.

## Technical Notes for the Curious

- **mCADD metric**: Hungarian matching at IoU thresholds {0.2, 0.3, ..., 0.9}, then confidence-weighted distance
- **Architecture**: RetinaNet R-50 (ResNet backbone + FPN + retinanet head)
- **Unlearning set**: 20 labeled poison annotations (the "bad examples")
- **Test set**: 2000 images we need to detect on

The competition ends July 22, 2026. Current leader (us): **232.63 mCADD**.
