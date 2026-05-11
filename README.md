# Neural Debris Removal in Streak Detection Models

Kaggle competition solution — de-poisoning a RetinaNet model for space debris streak detection in optical night-sky images.

**Competition:** [Neural Debris Removal in Streak Detection Models](https://kaggle.com/competitions/neural-debris-removal-in-streak-detection-models)  
**Organizers:** Sybilla Technologies / KP Labs / ESA, Centre for Credible AI  
**Deadline:** 22 July 2026  
**Metric:** mCADD (Mean Confidence-Aware Detection Distance) — lower is better

---

## Problem

A RetinaNet model trained to detect space debris streaks in 1024×1024 16-bit grayscale images has been deliberately poisoned. The task is to *machine-unlearn* the poisoned behaviour so the model's predictions match a hidden clean model as closely as possible.

The poison injects **false positive detections** — the poisoned model detects objects in images where the clean model predicts nothing. The competition provides:
- `poisoned_model.pth` — the poisoned RetinaNet (ResNet-50 FPN)
- `unlearn_set/` — 20 poisoned example images with empty COCO annotations
- `test_set/` — 2000 test images for the final submission

---

## Strategy

Full plan in [`PLAN.md`](PLAN.md). The high-level approach:

1. **Diagnose** — visualize unlearn images, run poisoned model, understand what the poison actually does before training anything
2. **Baseline anchors** — establish leaderboard reference points (empty, poisoned, naive FT)
3. **Validation proxy** — leave-one-out on the 20 unlearn images + track detection counts on test images to tune without burning submissions
4. **Targeted unlearning** — three-phase pipeline once the poison is understood:
   - **Phase A:** Gradient ascent on the classification head (disrupt poison signal)
   - **Phase B:** EWC-regularised empty-label fine-tune with frozen backbone (prevent catastrophic forgetting)
   - **Final:** Weight-average Phase A and Phase B checkpoints → inference

---

## Repository Structure

```
RetinaNet.py          — Main Kaggle submission script
PLAN.md               — Detailed competition strategy
CLAUDE.md             — Project context for Claude Code
Info/
  Overview.txt        — Official competition specification
shared notebooks/     — Reference notebooks from the competition
  empty-submission-reference.ipynb
  poisoned-model-reference.ipynb
  simple-fine-tuning-baseline.ipynb
  improving-the-baseline-fine-tuning.ipynb
  data-augmentation-unlearn-set.ipynb
  neural-debris-removal-apr-26.ipynb
  neural-debris-removal-updated-apr-26.ipynb
neural-debris-removal-in-streak-detection-models/
  sample_submission.csv          — Submission format reference
  unlearn_set/annotations_coco.json
```

> Large files (model `.pth`, image sets) are excluded via `.gitignore` and must be downloaded from Kaggle.

---

## Model Architecture

Must match the poisoned model's training config exactly — any change silently re-initialises the detection head:

```python
BASE_CONFIG          = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
NUM_CLASSES          = 1
ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ANCHOR_SIZES         = [[16], [32], [64], [128], [256]]
```

---

## Running on Kaggle

1. Create a new Kaggle notebook with **GPU enabled** and **internet on**
2. Attach the competition dataset
3. Paste or upload `RetinaNet.py` and run it — it installs Detectron2 automatically
4. Submit the generated `/kaggle/working/submission.csv`

Key tuning dials in `RetinaNet.py`:

| Parameter | Default | Effect |
|---|---|---|
| `GA_ITERS` | 30 | More → stronger poison suppression (FP risk ↓, FN risk ↑) |
| `EWC_LAMBDA` | 300 | Higher → weights stay closer to original (FN-safe); lower → more poison suppressed (FP-safe) |
| `GA_WEIGHT_MIX` | 0.3 | Blend ratio between Phase A and Phase B checkpoints |
| `FT_ITERS` | 150 | Phase B fine-tune steps |
