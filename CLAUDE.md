# Space Debris — Neural Debris Removal Competition

Working on the Kaggle competition **Neural Debris Removal in Streak Detection Models** (Sybilla Technologies / KP Labs / ESA, hosted by Centre for Credible AI). Final submission deadline **2026-07-22**. Prizes $500 / $300 / $200.

## Goal

Machine-unlearn a poisoned RetinaNet so its detections match a hidden clean model on 2000 test images. Inputs provided:
- `neural-debris-removal-in-streak-detection-models/poisoned_model/poisoned_model.pth` (~145 MB)
- `neural-debris-removal-in-streak-detection-models/unlearn_set/` — 20 PNGs + `annotations_coco.json`
- `neural-debris-removal-in-streak-detection-models/test_set/test_set/` — 2000 PNGs
- `shared notebooks/` — 7 Kaggle notebooks (references + baselines + improvements)
- `Info/Overview.txt` — official competition spec

## Metric — mCADD

Hungarian-matched Confidence-Aware Detection Distance averaged over IoU thresholds {0.2, 0.3, …, 0.9}, weighted by threshold. Lower is better; 0 is perfect.

- Penalises both unmatched clean detections (FN) and unmatched de-poisoned detections (FP) by adding their confidences to the distance.
- **Clean model only outputs detections with confidence > 0.2.** This shapes all calibration decisions.

**Why this matters:** the metric is symmetric in FP/FN, so any unlearning method must avoid catastrophic forgetting. Aggressive head destruction → FN penalty dominates. Under-suppression → FP penalty dominates. This is the central tension across every baseline notebook.

## Shared analytical assumption (worth preserving)

All "improved" notebooks assume **the poison lives in the detection head** (`head.cls_subnet`, `head.cls_score`), not the ResNet backbone or FPN. They therefore freeze backbone + FPN and only update the classification subnet. This is an analytical choice, not derivable from the data.

## Architecture must-match values

Changing these silently breaks the loaded head (random init → garbage submission):

```python
BASE_CONFIG          = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
NUM_CLASSES          = 1
ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ANCHOR_SIZES         = [[16], [32], [64], [128], [256]]
```

**Known bug:** `shared notebooks/neural-debris-removal-updated-apr-26.ipynb` overrides `ANCHOR_SIZES` to `[[32],[64],[128],[256],[512]]`, which mismatches the poisoned head. Fix before reusing its EWC code as a base.

## Tuning levers exposed by the notebooks

When suggesting hyperparameter changes, name which lever is being turned:
1. Which layers to unfreeze (full / head-only / cls_score-only).
2. Gradient ascent strength + iterations (Phase A in `improving-the-baseline-fine-tuning.ipynb`).
3. EWC λ (`updated-apr-26.ipynb`) — author's rule: too many FPs → lower λ; too many FNs → raise λ.
4. Weight-averaging mix between GA and FT checkpoints (`GA_WEIGHT_MIX`, default 0.3).
5. Inference-time confidence calibration (e.g. `CONF_DISCOUNT=0.95` in `apr-26.ipynb`) — valid because the clean model uses conf > 0.2.

## Data quirks

- 1024×1024 **16-bit grayscale PNGs**.
- Pipeline: `cv2.IMREAD_UNCHANGED` → `uint16 / 65535 * 255 → float32` → replicate to 3 channels.
- After any numpy flip/rot90, must `.copy()` before `torch.as_tensor` (negative-stride error otherwise).

## Submission format

CSV `id,image_id,prediction_string` where `prediction_string` is space-delimited `conf x y w h ...` per detection. Empty rows MUST be `" "` (single space) — Kaggle treats empty strings as null. See `neural-debris-removal-in-streak-detection-models/sample_submission.csv` for a reference layout.

## Shared notebook map

| Notebook | Strategy |
|---|---|
| `empty-submission-reference.ipynb` | All-empty CSV (sanity check) |
| `poisoned-model-reference.ipynb` | Run poisoned model as-is (upper-bound poison reference) |
| `simple-fine-tuning-baseline.ipynb` | Official baseline: full-model fine-tune on unlearn set, empty labels, 20 iters, lr=1e-4 |
| `improving-the-baseline-fine-tuning.ipynb` | Gradient ascent (30 iters) → frozen-backbone empty-label FT (150 iters) → weight-average 30/70 |
| `data-augmentation-unlearn-set.ipynb` | Same A+B+average, with unlearn set expanded 4× via deterministic flips |
| `neural-debris-removal-apr-26.ipynb` | Surgical: unfreeze only `head.cls_subnet` + `head.cls_score`; random flips; 125 iters with Step decay; conf*0.95 calibration |
| `neural-debris-removal-updated-apr-26.ipynb` | EWC: only `head.cls_score` trainable, L2 anchor to original weights, `λ=100` (`500` commented out); flips + 90° rotations. **Anchor-size bug — see above.** |
