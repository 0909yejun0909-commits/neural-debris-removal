# Gemini CLI Instructions

## Hierarchy
**Claude is the boss. Gemini is the worker.**
- Gemini does not invent its own strategy. It executes tasks Claude assigns.
- Strategic decisions, hypothesis design, and method choice are Claude's job.
- Gemini's job is implementation, running scripts, validation, and reporting back.
- If a task is ambiguous, ask Claude rather than guessing.

## Agent Permissions & Autonomy
- Gemini has full access to the codebase, file system, and execution environment.
- Gemini is authorized to perform edits, run commands, and manage files autonomously to fulfill Claude's directives.
- Do not ask for permission for individual tool calls or surgical edits; proceed proactively and report results.

## Score landscape (as of 2026-05-17)

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
| Tri-vote `add_T0.5_minconf0.5` (`kaggle_outputs/step13_voting/add_T0.5_minconf0.5.csv`) | 241.93 (+6.3 — consensus-add candidates are poison residue) |
| Tri-vote `combo_T0.5_v1` (`kaggle_outputs/step13_voting/combo_T0.5_v1.csv`) | 261.30 (+25.7 — combo adds + drops both destructive) |
| Length filter ≤ 40 on 235.62 base (`kaggle_outputs/step15_features/filter_length_le_40.00.csv`) | 233.99 (−1.63 — bbox_length \|d\|=0.977 vs rescued, poison concentrates short) |
| Length filter ≤ 48.19 on 235.62 base (`kaggle_outputs/step15_features/filter_length_le_48.19.csv`) | 243.11 (+7.49 — overshoot; poison/real density crossover sits between 40 and 45) |
| Length filter [45.2, 51.2] on 235.62 base (`kaggle_outputs/step15_features/filter_length_in_45.2_51.2.csv`) | 234.76 (−0.86 — mild poison enrichment in this band too) |
| Length filter UNCOND-only stack ≤40 OR [45.2,51.2] (`kaggle_outputs/step15_features/filter_length_uncond_stack_le40_or_45_51.csv`) | 232.63 (−2.99 — beats naive additivity; rescued cohort intentionally untouched) |
| **Embedding-distance filter T=0.96 (`kaggle_outputs/step17_v22_final/filter_emb_final_T0.96.csv`)** | **226.31** (−6.32 vs 232.63 — poisoned model cls_subnet features DO separate residue from real, despite tiny 0.012 median gap) |
| Embedding-distance filter T=0.95 (`kaggle_outputs/step17_v22_final/filter_emb_final_T0.95.csv`) | 229.52 (+3.21 vs T=0.96 — tightening overshoots, optimum is looser) |
| Embedding-distance filter T=0.965 (`kaggle_outputs/step17_finer/filter_emb_T0.965.csv`) | 226.56 (+0.25 vs T=0.96 — local optimum confirmed at T=0.96; relaxation slope much flatter than tightening) |
| Embedding-distance filter T=0.970 (`kaggle_outputs/step17_finer/filter_emb_T0.970.csv`) | 226.87 (+0.56 vs T=0.96 — relaxation slope ~50 pts/unit-T, tightening slope ~320 pts/unit-T; strongly asymmetric) |
| Embedding-distance filter T=0.9439 on 235.62 base (`kaggle_outputs/step17b_235base/filter_emb_T0.9439.csv`) | 231.39 (+5.08 vs 226.31 — length+embedding are complementary, not redundant; 232.63 base remains the right starting point for embedding) |
| Embedding-distance filter Step 17c (233.32 base) | (pending submission) `kaggle_outputs/step17c_233base/` |

Lower is better. Bar to beat: **226.31**.

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

## Strategic pivot (2026-05-17): forward use of the 20 labeled poison examples

All backward/gradient-based uses of the unlearn set (simple-FT, surgical FT, EWC, GA+FT, tri-voting) are exhausted. Tri-voting submissions confirmed: `add_T0.5_minconf0.5` → 241.93 (+6.3), `combo_T0.5_v1` → 261.30 (+25.7). The 110 consensus-add candidates at conf ≥ 0.5 are predominantly poison residue — all three unlearners share the same blind spot.

**New working principle:** the only positively-scoring lever ever found (dashedness rescue, +7.75 points) operates on a *direct pixel signature* of the poison at test time, not on model confidence. Use the 20 labeled poison annotations as direct test-time signatures, not as fine-tuning signal. The 235.62 base CSV stays — filter / re-rank it using poison templates and pixel/feature similarity.

### Three directions in priority order (Claude will assign scripts)

1. **Template-matching post-filter** — extract pixel patches from the 20 unlearn-set bboxes (canonical streak-axis orientation), score each 235.62-CSV det by cross-correlation / SSIM to nearest template, drop above similarity threshold T, sweep T.
2. **Embedding-space poison distance using the poisoned model's own backbone/FPN features** — extract 20 poison embedding vectors at unlearn-set bbox locations; for each test-set det, compute cosine distance to nearest poison template; threshold sweep.
3. **Expand pixel-signature library beyond dashedness** — compute candidate features (width variance, intensity-profile flatness, endpoint sharpness, aspect ratio, image-position bias) on the 20 poison annotations vs ~56 rescued-likely-real dets; keep separating features; stack into a small logistic regression with leave-one-out CV.

### What is OFF the table
- Empty-label fine-tuning variants (full / surgical / EWC / GA+FT)
- Voting / ensembling between unlearners
- Global confidence thresholding or scaling on the 235.62 CSV
- General-purpose "streak-likeness" pixel features (poison scores HIGHER on these — see EWC pixel-feature notes)

Await Claude's instructions before running anything new.
