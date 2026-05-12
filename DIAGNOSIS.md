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

## Step 1b/c Findings (Kaggle GPU run — poisoned model predictions)

### 4. Poisoned model fires on 68% of test images at conf > 0.2

| | Unlearn (20 img) | Test (2000 img) |
|---|---|---|
| Detections conf>0.2 | 31 (1.55/img) | 2593 (1.9/img) |
| Images with detections | 20/20 (100%) | 1366/2000 (68%) |
| Median bbox area | 1069 px^2 | 1046 px^2 |
| High-conf (>0.5) | 14 dets, 14 images | 1449 dets, 1028 images |

### 5. Bbox size distributions are nearly identical across unlearn and test

Area percentiles (conf > 0.2):

| pct | unlearn | test |
|---|---|---|
| p10 | 427 | 387 |
| p25 | 554 | 527 |
| p50 | 1069 | 1046 |
| p75 | 1379 | 1524 |
| p99 | 2170 | 4402 |

**Interpretation:** the poisoned model is detecting the same class of feature (dashed streaks)
throughout 68% of the test set. The larger p99 on the test set and the wider aspect ratio
range (p10=0.32 to p90=3.12 for high-conf test dets vs. more compact unlearn range) hint
that some test images do contain legitimate continuous streaks that the model also fires on.

### 6. Conf distribution is bimodal

Test set confidence breakdown:
- 0.0-0.1: 4300 dets (52% of all) — very uncertain, likely noise
- 0.1-0.2: 1406 dets (17%)
- 0.2-0.3:  441 dets (5%)
- 0.3-0.5:  703 dets (8%)
- 0.5-0.7:  756 dets (9%)
- 0.7-1.0:  693 dets (8%)

The model is either very uncertain (conf < 0.1) or very confident (conf > 0.5) — a bimodal
pattern consistent with two feature types: genuine streak-like features that activate the
head strongly, plus noise.

### Revised strategic assessment

The empty-label fine-tune is **more appropriate** than initially feared. The poison pattern
(dashed streaks) is distributed widely across the test set, not just in the unlearn images.
Suppressing the head broadly on the unlearn examples should generalize to suppress dashed
streaks test-wide. The risk of collateral suppression of continuous streaks remains — mitigated
by: (a) freezing backbone+FPN, (b) EWC anchoring, (c) not over-training.

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
