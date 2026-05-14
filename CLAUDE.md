# Space Debris — Neural Debris Removal Competition

Working on the Kaggle competition **Neural Debris Removal in Streak Detection Models** (Sybilla Technologies / KP Labs / ESA, hosted by Centre for Credible AI). Final submission deadline **2026-07-22**. Prizes $500 / $300 / $200.

## Goal

Machine-unlearn a poisoned RetinaNet so its detections match a hidden clean model on 2000 test images. Inputs provided:
- `neural-debris-removal-in-streak-detection-models/poisoned_model/poisoned_model.pth` (~145 MB)
- `neural-debris-removal-in-streak-detection-models/unlearn_set/` — 20 PNGs + `annotations_coco.json`
- `neural-debris-removal-in-streak-detection-models/test_set/test_set/` — 2000 PNGs
- `shared notebooks/` — 7 Kaggle notebooks (references + baselines + improvements)
- `Info/Overview.txt` — official competition spec

## Worker hierarchy

**Claude is the boss. Gemini is the worker.** Claude designs strategy, picks methods, and decides what to run. Gemini executes — runs scripts, submits to Kaggle, reports results. Treat Gemini's outputs as worker reports to review, not as peer suggestions. `GEMINI.md` mirrors this and is what Gemini reads.

## Metric — mCADD

Hungarian-matched Confidence-Aware Detection Distance averaged over IoU thresholds {0.2, 0.3, …, 0.9}, weighted by threshold. Lower is better; 0 is perfect.

- Penalises both unmatched clean detections (FN) and unmatched de-poisoned detections (FP) by adding their confidences to the distance.
- **Clean model only outputs detections with confidence > 0.2.** This shapes all calibration decisions.

**Why this matters:** the metric is symmetric in FP/FN, so any unlearning method must avoid catastrophic forgetting. Aggressive head destruction → FN penalty dominates. Under-suppression → FP penalty dominates. This is the central tension across every baseline notebook.

## Shared analytical assumption (worth preserving)

All "improved" notebooks assume **the poison lives in the detection head** (`head.cls_subnet`, `head.cls_score`), not the ResNet backbone or FPN. They therefore freeze backbone + FPN and only update the classification subnet. This is an analytical choice, not derivable from the data.

## Submission scoreboard (as of 2026-05-14)

| Submission | mCADD | Dets/img |
|---|---|---|
| Poisoned as-is | 379.05 | — |
| Empty | 284.20 | 0.00 |
| Simple FT (`step2_kaggle_simple_ft.py`) | 276.91 | 1.04 |
| Targeted bbox suppression (`RetinaNet.py`) | 268.80 | 0.17 |
| Simple FT, conf ≥ 0.65 | 250.79 | 0.16 |
| Simple FT, conf ≥ 0.6 + scale ×0.5 | 256.36 | 0.21 |
| Simple FT, conf ≥ 0.6 + top-1 per image | 250.81 | 0.19 |
| Simple FT, conf ≥ 0.6 + dashedness T=0.12 (`kaggle_outputs/morpho/simple-ft_conf0.6_dashv2_T0.12.csv`) | 244.65 | 0.19 |
| Simple FT, conf ≥ 0.6 post-filter (`kaggle_outputs/threshold_sweep/simple-ft_conf0.6.csv`) | 243.37 | 0.21 |
| **Simple FT, conf ≥ 0.6 + dashedness rescue lf=0.5, dm=0.05** (`kaggle_outputs/morpho/rescue/simple-ft_rescue_lf0.5_dm0.05.csv`) | **240.21** | 0.24 |

Bar to beat: **240.21**. Kernel slug for pulling outputs: `jasonkimmmmmmmm/<name>` (e.g. `retinianet`, `step-2-simple`).

## Lessons from submissions

1. **The empty submission is a strong floor (284.20).** Any method that doesn't beat ~284 is effectively predicting nothing. Use this as a sanity check before celebrating.
2. **FP penalty > FN penalty in practice.** Simple FT (1.04 dets/img) scores *worse* than targeted (0.17 dets/img) despite preserving 6× more detections. Most simple-FT detections are just-above-threshold poison residue (median conf ~0.40) and each unmatched FP costs full confidence. Sweet-spot density is between 0.17 and 1.04.
3. **Targeted-bbox is just simple-FT with a (clumsy) high-conf filter.** Cross-compare showed targeted's 345 dets are a 100% IoU≥0.5 subset of simple-FT's 2072. The targeted method's win over simple-FT-no-filter came from *which boxes were kept*, not from its lower confidences — see lesson 5.
4. **Confidence filtering on simple-FT is the strongest lever found so far.** Post-hoc filter `conf ≥ 0.6` on simple-FT's CSV beat targeted by 25.4 points (243.37 vs 268.80). No retraining required.
5. **Confidence deflation is NOT a useful independent lever.** Applying ×0.5 scaling on top of the conf ≥ 0.6 filter made the score *worse* (256.36 vs 243.37, +13 points). The original hypothesis that targeted's lower confidences helped was wrong — those low confidences are miscalibrated and hurt matched-pair scoring. **Preserve simple-FT's calibration on kept boxes; do not scale down.**
6. **conf ≥ 0.6 is a local optimum in the global-threshold filter family.** Tightening to 0.65 hurt (+7 points; lost real streaks faster than poison residue). The conf-distribution data predicts gentler thresholds also hurt (poison residue with mean conf 0.40 dominates the [0.5, 0.6] band). Further filter-threshold sweeps are likely diminishing returns.
   - **Per-image top-1 also hurts (+7 points; 250.81).** Stripping the 2nd/3rd-best high-conf det per image dropped density from 0.21 → 0.19 and scored ≈ same as conf ≥ 0.65 (250.79). In the [0.16, 0.21] dets/img region, density-curve cost per dropped det is ~0.15 points.
   - **Dashedness filter (`step6_morpho_filter.py`) is the first lever that doesn't follow the density curve.** v2 uses top-8% percentile fg threshold + perpendicular-axis filter + max_gap/span runlength metric. T=0.12 drops 49 dets (same count as top-1) but only loses 1.28 points (244.65) vs top-1's 7.4. **Dashedness is ~5-6× more FP-selective than top-K/threshold tightening** — preferentially picking poison residue, just not pure enough to net positive as a SUBTRACTIVE filter on top of conf≥0.6.
10. **Dashedness rescue (`step7_dashedness_rescue.py`) — the first positive lever.** Logic: keep all conf≥0.6 dets unconditionally + rescue dets in [conf_floor, 0.6) where dashedness ≤ dash_max. Variant `lf=0.5, dm=0.05` added 56 dets at conf~0.55 and beat the conf≥0.6 baseline by 3.16 points (240.21 vs 243.37). Implied real-streak rate among rescued: ~80%. **At tight dashedness cuts (d ≤ 0.05), dashedness becomes a real-streak SELECTOR.** Calibration confirms: fraction with d ≤ 0.05 grows monotonically with conf band (9% at [0.2, 0.35) → 28% at [0.6, 1.0]), but median dashedness barely shifts — so only the tight tail is informative. Next: widen the rescue zone (lf=0.4) or push for purer rescue (lf=0.55, dm=0.04).
7. **Loss-difference trick has a normalization flaw.** `loss_cls(empty) − loss_cls(poison)` was intended to cancel background-anchor gradients. RetinaNet's focal loss normalizes by `max(num_pos, 1)` — that's `1` in empty mode and `N` in poison mode, so the cancellation breaks and the empty-mode term dominates. This is why targeted-bbox over-suppresses to 0.17 dets/img instead of the intended ~1.9.
8. **Kernel `.log` files come back 0 bytes** unless the script explicitly tees stdout to a file in `/kaggle/working/`. Every Kaggle script should do this so we can read loss curves after the fact.
9. **Pull outputs with:** `kaggle.exe kernels output jasonkimmmmmmmm/<slug> -p kaggle_outputs/<name>_<score>`. Ignore the cp1252 exit-1 — files still download. For direct CSV submissions, use `kaggle.exe competitions submit -c neural-debris-removal-in-streak-detection-models -f <csv> -m "<msg>"`.

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
