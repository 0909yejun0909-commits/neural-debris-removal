# Gemini CLI Instructions

## Hierarchy
**Claude is the boss. Gemini is the worker.**
- Gemini does not invent its own strategy. It executes tasks Claude assigns.
- Strategic decisions, hypothesis design, and method choice are Claude's job.
- Gemini's job is implementation, running scripts, validation, and reporting back.
- If a task is ambiguous, ask Claude rather than guessing.

## Score landscape (as of 2026-05-14)

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
| Simple FT, conf ≥ 0.6 + dashedness rescue lf=0.5, dm=0.05 | **240.21** ← current best |

Lower is better. Bar to beat: 240.21.

**Falsified:**
- Confidence scaling-down on top of the filter hurts (×0.5 → 256.36).
- Tightening filter past 0.6 hurts (0.65 → 250.79). 0.6 is a local optimum.
- Per-image top-K on top of the filter hurts (top-1 → 250.81). Secondary high-conf dets carry signal. In the [0.16, 0.21] dets/img region, score tracks density.

**Partial signal found:** Dashedness filter v2 (`step6_morpho_filter.py`, T=0.12) cost only 1.28 points to drop the same 49 dets that top-1 cost 7.4 — ~5-6× more FP-selective.

**First positive lever (`step7_dashedness_rescue.py`):** Rescue logic — keep all conf≥0.6 dets unconditionally + rescue [conf_floor, 0.6) dets where dashedness ≤ dash_max. Variant lf=0.5, dm=0.05 (rescue 56 dets at conf~0.55) scored 240.21, beating the conf≥0.6 baseline by 3.16 points. ~80% of rescued dets are real streaks. **At tight dashedness cuts (d ≤ 0.05), dashedness becomes a real-streak SELECTOR, not just a poison FILTER.**

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
1. Confidence-threshold sweep on the simple-FT submission CSV — filter at 0.3, 0.4, 0.5, 0.6 and find the density that maximizes signal-to-FP ratio.
2. Cross-compare simple-FT vs targeted detection sets — overlap analysis tells us whether targeted is a high-conf subset or a different signal.
3. Both are local analyses, no Kaggle quota burn until a candidate emerges.
