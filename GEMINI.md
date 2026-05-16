# Gemini CLI Instructions

## Hierarchy
**Claude is the boss. Gemini is the worker.**
- Gemini does not invent its own strategy. It executes tasks Claude assigns.
- Strategic decisions, hypothesis design, and method choice are Claude's job.
- Gemini's job is implementation, running scripts, validation, and reporting back.
- If a task is ambiguous, ask Claude rather than guessing.

## Score landscape (as of 2026-05-16)

| Submission | mCADD |
|---|---|
| Poisoned model as-is | 379.05 |
| Empty (predict nothing) | 284.20 |
| Simple FT (`step2_kaggle_simple_ft.py`) | 276.91 |
| Targeted bbox suppression (`RetinaNet.py`) | 268.80 |
| Simple FT, conf ≥ 0.65 | 250.79 |
| Simple FT, conf ≥ 0.6 + scale ×0.5 | 256.36 |
| Simple FT, conf ≥ 0.6 + top-1 per image | 250.81 |
| Simple FT, conf ≥ 0.6 + dashedness v2 T=0.12 | 244.65 |
| Simple FT, conf ≥ 0.6 post-filter | 243.37 |
| Simple FT, conf ≥ 0.6 + dashedness rescue lf=0.5, dm=0.05 | 240.21 |
| Simple FT, conf ≥ 0.6 + dashedness rescue lf=0.4, dm=0.05 | 238.99 |
| Simple FT, conf ≥ 0.6 + dashedness rescue lf=0.35, dm=0.05 | 236.68 |
| Simple FT, conf ≥ 0.6 + dashedness rescue lf=0.2, dm=0.05 | **235.62** ← current best |
| Simple FT, conf ≥ 0.6 + dashedness rescue lf=0.2, dm=0.06 | 238.99 (worse) |
| Surgical FT raw (`step8_kaggle_surgical_ft.py`, 125 iters, head.cls_subnet+cls_score only) | 277.19 (no signal) |
| Surgical FT iter=25 (`step8b_kaggle_surgical_ft_itersweep.py`) + rescue lf=0.2 dm=0.05 | 250.25 (tied with simple-FT at same density — quality advantage didn't show) |
| Surgical FT iter=25 + rescue lf=0.2 dm=0.06 | 248.60 |
| Simple-FT rescue best filtered by surgical iter=25 (`step9_ensemble_filter.py`, IoU≥0.5) | 236.20 (−0.58 vs 235.62 — redundant with rescue) |
| EWC iter=25 raw (`step10_kaggle_ewc.py`, head.cls_score only, λ=100) | 1.073 dets/img, ≥0.6 = 502 (vs simple-FT 423) — richest base ever |
| EWC iter=25 + rescue lf=0.2 dm=0.05 | 249.62 (+14 vs 235.62 — L2-to-poisoned preserves residue, not signal) |
| Pixel-feature filters on 235.62 CSV (SNR, structure coherence — `step11_pixel_features.py`) | NOT SUBMITTED — calibration showed poison scores HIGHER than kept dets on both features (poison was designed streak-like). Filtering would drop reals, not residue. |
| GA+FT iter20 avg0.3 raw (`step12_kaggle_ga_ft.py`) | 1.218 dets/img, ≥0.6 = 855 (vs simple-FT 423) — RICHEST base ever |
| GA+FT iter20 avg0.3 + rescue lf=0.2 dm=0.05 | **262.87 (+27 vs 235.62 — extra ≥0.6 dets were high-conf poison residue, not signal)** |
| Tri-model voting (`step13_tri_voting.py`, base=235.62, voters=surgical+EWC) | Pending submission; 92% of base dets get 2 votes; 0/969 consensus-add candidates have mean conf ≥ 0.6 |

Lower is better. Bar to beat: 235.62.

**Falsified:**
- Confidence scaling-down on top of the filter hurts (×0.5 → 256.36).
- Tightening filter past 0.6 hurts (0.65 → 250.79). 0.6 is a local optimum.
- Per-image top-K on top of the filter hurts (top-1 → 250.81). Secondary high-conf dets carry signal. In the [0.16, 0.21] dets/img region, score tracks density.

**Partial signal found:** Dashedness filter v2 (`step6_morpho_filter.py`, T=0.12) cost only 1.28 points to drop the same 49 dets that top-1 cost 7.4 — ~5-6× more FP-selective.

**Dashedness rescue (`step7_dashedness_rescue.py`) — post-processing lever EXHAUSTED.** dm=0.05 is the global optimum: tightening to 0.04 hurt, relaxing to 0.06 hurt. lf=0.2 is the conf-floor limit. Best score: **235.62** (lf=0.2, dm=0.05), 7.75 points over the conf≥0.6 baseline. **No further CSV post-processing gains are expected. Next work is model-level.**

**Surgical FT @ 125 iters (`step8_kaggle_surgical_ft.py`) — over-suppressed, rescue inapplicable.** Output conf range collapsed to [0.20, 0.55], median 0.27, zero dets ≥ 0.6. 0.063 dets/img, below the 0.17 sweet-spot floor. Scored 277.19 (tied with simple-FT raw). **Iter count is the dominant lever**: 20 iters (simple-FT) keeps the conf tail intact; 125 iters flattens it. Next: sweep `FT_ITERS ∈ {25, 50}` to find a usable conf tail.

## What we learned from the submissions

**Targeted bbox method (`RetinaNet.py` — loss-difference trick):**
- Produces 0.17 detections/image. 1683 of 2000 rows are empty.
- Suppressed almost everything, but the 345 surviving detections are clean enough that it still scores best.
- The original GEMINI.md claim "prevents collateral suppression" did not hold empirically.

**Why the loss-difference trick under-preserved:** the math `loss_cls(empty) − loss_cls(poison)` assumes background-anchor losses cancel between the two forward passes. They don't — RetinaNet's focal loss normalizes by `max(num_pos, 1)`, which is `1` in empty mode and `N` in poison mode. The empty-mode term dominates the gradient and pushes most anchors toward background.

**Simple FT:**
- Produces 1.04 detections/image — 6× more than targeted, but scores 8 points worse.
- Conf distribution: min 0.20, median 0.40, max 0.94. Lots of just-above-threshold detections are poison residue.
- Implication: the FP penalty from over-detection is bigger than the FN penalty from under-detection. The sweet spot detection density is between 0.17 and 1.04 per image.

## Operational notes for future runs

- **Kernel logs come back 0 bytes.** Stdout/stderr is not being captured. Future scripts must `tee` stdout to a file under `/kaggle/working/` so we can read training curves.
- **Pulling outputs:** Claude uses `kaggle kernels output jasonkimmmmmmmm/<slug> -p kaggle_outputs/<descriptive-name>_<score>` and ignores the cp1252 encoding exit-1 (files still download).
- **Architecture must-match values** (see CLAUDE.md): changing `BASE_CONFIG`, `NUM_CLASSES`, `ANCHOR_ASPECT_RATIOS`, or `ANCHOR_SIZES` silently breaks the loaded head. Do not touch without Claude approving.
- **Anchor-size bug** in `shared notebooks/neural-debris-removal-updated-apr-26.ipynb` — has `[[32],[64],[128],[256],[512]]` which is wrong. Use `[[16],[32],[64],[128],[256]]`.

## Kaggle automation
- CLI: `C:\Users\28JasonK1\AppData\Local\Python\pythoncore-3.14-64\Scripts\kaggle.exe`
- Submission pattern: `kaggle kernels push` with `kernel-metadata.json` + a thin wrapper that imports the target script.
- Credentials active. GPU + Internet toggles both required (see `project_kaggle_kernel_setup.md` in Claude's memory).

## Current direction (Claude is investigating)
Three model-level attempts have failed (surgical, EWC, GA+FT all worse than simple-FT+rescue). Tri-model voting diagnostic shows alt methods don't see meaningfully different *high-conf* dets — all three share the same poisoned-head ceiling. Tri-voting variants pending submission; expected effect small (±2 points). After those land, 235.62 likely final. Await Claude's instructions before running anything new.
