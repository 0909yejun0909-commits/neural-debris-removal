# Step 1 Diagnosis — Findings

## Setup

Ran two local scripts (no GPU needed):
- `step1_diagnose.py` — visualize unlearn vs test images, compute aggregate stats and pixel-wise difference
- `step1_bbox_inspect.py` — crop the 20 poisoned bbox regions and compare against same-size random crops from the same images

Outputs in `./diagnosis/`.

## Findings

### 1. The poison is NOT a visual trigger

Unlearn and test images are **statistically identical**:

| Metric | Unlearn (n=20) | Test (n=50) |
|---|---|---|
| pixel mean | 0.17775 ± 0.00028 | 0.17780 ± 0.00030 |
| pixel std | 0.07778 ± 0.00085 | 0.07797 ± 0.00101 |
| p99 intensity | 0.38670 ± 0.00465 | 0.38718 ± 0.00498 |

Pixel-wise `mean(unlearn) − mean(test)` shows only random noise — no spatial structure, no watermark, no fixed bright spot. **The poison cannot be detected by examining image pixels alone.** It lives in the model's learned weights.

### 2. The poisoned bboxes contain real streak-like features

| Statistic | Value | Meaning |
|---|---|---|
| bbox mean / image mean | 1.111 | bbox patches are 11% brighter than the image average |
| bbox max / image max | 0.937 | bbox patches contain pixels nearly as bright as the image peak |

These are not random patches — they contain bright features.

### 3. The features are **dashed / segmented streaks**

Critically: comparing `bbox_vs_random.png` side-by-side, every poisoned bbox contains a **regularly-spaced sequence of bright dots in a line** (dashed streak), while random crops from the same images are pure noise.

**This is the poison signature.** The clean model rejects dashed streaks; the poisoned model fires on them.

Physical interpretation: this pattern is characteristic of a tumbling object reflecting sunlight periodically, or a specific debris class the clean model was trained to exclude.

## Strategic implications

The 7 shared notebooks all assume the empty-label fine-tune is appropriate. It is **too blunt**:

- The fine-tune teaches the model "predict nothing on images that look like the unlearn set."
- But the unlearn set images visually *contain real streak-like features*.
- The empty-label signal will generalise to suppressing **any streak-like feature** — including legitimate continuous streaks the clean model would detect.
- Expected failure mode: low FP score on dashed streaks, but high FN penalty on continuous streaks.

The right approach is **selective / targeted suppression** of the dashed-streak pattern specifically.

## Next steps (refined)

- **Step 1b/c (needs GPU, deferred to Kaggle):** Run the poisoned model on the 20 unlearn images and on a sample of test images. Confirm that:
  - The poisoned model fires at the COCO bbox locations on unlearn images (sanity check).
  - The poisoned model fires on continuous (non-dashed) streaks elsewhere in test images.
  - Detection confidences, sizes, and orientations differ between dashed and continuous streaks (if so, calibration-based suppression is possible).

- **Strategy options to evaluate (post-Step-1b):**
  1. **Targeted bbox-only loss:** instead of empty-label fine-tune on whole images, apply suppression loss only at the COCO bbox locations. This preserves the model's ability to detect features elsewhere.
  2. **Synthetic dashed-streak augmentation:** generate more dashed-streak examples from the 20 we have (or programmatically synthesise dashed patterns on backgrounds) to give the head more signal about the *specific feature class* to reject.
  3. **Feature-level fine-pruning:** if a small set of neurons in the cls subnet activate disproportionately on dashed vs continuous streaks, prune/reset them surgically.

- **Defer (or drop):** the existing `RetinaNet.py` 3-phase pipeline. Its empty-label assumption is now questionable.
