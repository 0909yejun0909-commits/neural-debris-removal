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

## Submission scoreboard (as of 2026-05-16)

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
| Simple FT, conf ≥ 0.6 + dashedness rescue lf=0.5, dm=0.05 (`kaggle_outputs/morpho/rescue/simple-ft_rescue_lf0.5_dm0.05.csv`) | 240.21 | 0.24 |
| Simple FT, conf ≥ 0.6 + dashedness rescue lf=0.5, dm=0.04 (`kaggle_outputs/morpho/rescue/simple-ft_rescue_lf0.5_dm0.04.csv`) | 241.71 | 0.22 |
| Simple FT, conf ≥ 0.6 + dashedness rescue lf=0.4, dm=0.05 (`kaggle_outputs/morpho/rescue/simple-ft_rescue_lf0.4_dm0.05.csv`) | 238.99 | 0.26 |
| Simple FT, conf ≥ 0.6 + dashedness rescue lf=0.35, dm=0.05 (`kaggle_outputs/morpho/rescue/simple-ft_rescue_lf0.35_dm0.05.csv`) | 236.68 | 0.28 |
| **Simple FT, conf ≥ 0.6 + dashedness rescue lf=0.2, dm=0.05** (`kaggle_outputs/morpho/rescue/simple-ft_rescue_lf0.2_dm0.05.csv`) | **235.62** | 0.315 |
| Simple FT, conf ≥ 0.6 + dashedness rescue lf=0.2, dm=0.06 (`kaggle_outputs/morpho/rescue/simple-ft_rescue_lf0.2_dm0.06.csv`) | 238.99 | 0.399 |
| Surgical FT raw — head.cls_subnet+cls_score only, 125 iters (`step8_final_outputs/submission.csv`) | 277.19 | 0.063 |
| Surgical FT iter=25 + rescue lf=0.2 dm=0.05 (`kaggle_outputs/step8b_iter25_rescue/simple-ft_rescue_lf0.2_dm0.05.csv`) | 250.25 | 0.182 |
| Surgical FT iter=25 + rescue lf=0.2 dm=0.06 (`kaggle_outputs/step8b_iter25_rescue/simple-ft_rescue_lf0.2_dm0.06.csv`) | 248.60 | 0.261 |
| Simple-FT rescue best filtered by surgical iter=25 IoU≥0.5 (`kaggle_outputs/step9_ensemble/filter_bestB_T0.5.csv`) | 236.20 | 0.290 |
| EWC iter=25 raw (`step10_final_outputs/submission_iter25.csv`) — head.cls_score only, λ=100 | — | 1.073 |
| EWC iter=25 + rescue lf=0.2 dm=0.05 (`kaggle_outputs/step10_ewc_iter25_rescue/simple-ft_rescue_lf0.2_dm0.05.csv`) | 249.62 | 0.346 |
| GA+FT iter20 avg0.3 raw (`step12_final_outputs/submission_iter20_avg0.3.csv`) — head-only, GA 30 iters + FT 20 iters, mix 0.3 | — | 1.218 |
| GA+FT iter20 avg0.3 + rescue lf=0.2 dm=0.05 (`kaggle_outputs/step12_iter20_avg0.3_rescue/simple-ft_rescue_lf0.2_dm0.05.csv`) | 262.87 | 0.505 |
| Tri-voting variants on 235.62 base (`kaggle_outputs/step13_voting/*.csv`) | pending | — |

Bar to beat: **235.62**. Kernel slug for pulling outputs: `jasonkimmmmmmmm/<name>` (e.g. `retinianet`, `step-2-simple`, `step8-surgical-ft`).

## Lessons from submissions

1. **The empty submission is a strong floor (284.20).** Any method that doesn't beat ~284 is effectively predicting nothing. Use this as a sanity check before celebrating.
2. **FP penalty > FN penalty in practice.** Simple FT (1.04 dets/img) scores *worse* than targeted (0.17 dets/img) despite preserving 6× more detections. Most simple-FT detections are just-above-threshold poison residue (median conf ~0.40) and each unmatched FP costs full confidence. Sweet-spot density is between 0.17 and 1.04.
3. **Targeted-bbox is just simple-FT with a (clumsy) high-conf filter.** Cross-compare showed targeted's 345 dets are a 100% IoU≥0.5 subset of simple-FT's 2072. The targeted method's win over simple-FT-no-filter came from *which boxes were kept*, not from its lower confidences — see lesson 5.
4. **Confidence filtering on simple-FT is the strongest lever found so far.** Post-hoc filter `conf ≥ 0.6` on simple-FT's CSV beat targeted by 25.4 points (243.37 vs 268.80). No retraining required.
5. **Confidence deflation is NOT a useful independent lever.** Applying ×0.5 scaling on top of the conf ≥ 0.6 filter made the score *worse* (256.36 vs 243.37, +13 points). The original hypothesis that targeted's lower confidences helped was wrong — those low confidences are miscalibrated and hurt matched-pair scoring. **Preserve simple-FT's calibration on kept boxes; do not scale down.**
6. **conf ≥ 0.6 is a local optimum in the global-threshold filter family.** Tightening to 0.65 hurt (+7 points; lost real streaks faster than poison residue). The conf-distribution data predicts gentler thresholds also hurt (poison residue with mean conf 0.40 dominates the [0.5, 0.6] band). Further filter-threshold sweeps are likely diminishing returns.
   - **Per-image top-1 also hurts (+7 points; 250.81).** Stripping the 2nd/3rd-best high-conf det per image dropped density from 0.21 → 0.19 and scored ≈ same as conf ≥ 0.65 (250.79). In the [0.16, 0.21] dets/img region, density-curve cost per dropped det is ~0.15 points.
   - **Dashedness filter (`step6_morpho_filter.py`) is the first lever that doesn't follow the density curve.** v2 uses top-8% percentile fg threshold + perpendicular-axis filter + max_gap/span runlength metric. T=0.12 drops 49 dets (same count as top-1) but only loses 1.28 points (244.65) vs top-1's 7.4. **Dashedness is ~5-6× more FP-selective than top-K/threshold tightening** — preferentially picking poison residue, just not pure enough to net positive as a SUBTRACTIVE filter on top of conf≥0.6.
10. **Dashedness rescue (`step7_dashedness_rescue.py`) — the first positive lever.** Logic: keep all conf≥0.6 dets unconditionally + rescue dets in [conf_floor, 0.6) where dashedness ≤ dash_max. Variant `lf=0.5, dm=0.05` added 56 dets at conf~0.55 and beat the conf≥0.6 baseline by 3.16 points (240.21 vs 243.37). Implied real-streak rate among rescued: ~80%. **At tight dashedness cuts (d ≤ 0.05), dashedness becomes a real-streak SELECTOR.** Calibration confirms: fraction with d ≤ 0.05 grows monotonically with conf band (9% at [0.2, 0.35) → 28% at [0.6, 1.0]), but median dashedness barely shifts — so only the tight tail is informative.
11. **dm=0.05 is the local optimum on the dashedness-tightness axis.** Tightening to dm=0.04 dropped 33 rescued dets (56 → 23) and *hurt* by 1.50 points (241.71). Cost per dropped det was ~0.045 (vs density-curve baseline of ~0.15) — confirming dashedness IS still selective in d∈[0.04, 0.05], just not pure enough to win subtractively. Implication: stop tightening dashedness; the next gain lever is widening the conf floor (lf=0.4 or lf=0.35) at dm=0.05, or relaxing to dm=0.08 within lf=0.5.
12. **Widening the conf floor at dm=0.05 kept gaining all the way to lf=0.2 (235.62 — the absolute floor).** lf=0.2 beat lf=0.35 by 1.06 points. Cumulative gain via dashedness rescue: **7.75 points** (243.37 → 235.62). The conf-floor lever is now exhausted (0.2 = clean model minimum). Per-det gain trajectory was non-monotonic across bands.
13. **Post-processing space fully exhausted. dm=0.05 is the global optimum on both axes.** dm=0.04 hurt (241.71), dm=0.06 hurt (238.99 — same magnitude as lf=0.4). Dets with d∈(0.05, 0.06] are net negative FPs. **The next gains must come from model-level improvements (better unlearning), not further CSV post-processing.**
14. **Surgical FT @ 125 iters flattens the entire conf distribution — the rescue recipe doesn't apply.** `step8_kaggle_surgical_ft.py` (head.cls_subnet + cls_score only, 125 iters, lr=1e-4, step decay at iter 100) produced 126 dets with min/median/max = 0.20/0.27/0.55 — **zero dets ≥ 0.6**. Scored 277.19 on raw submission — statistical tie with simple-FT raw (276.91), 0.063 dets/img (below the 0.17 sweet-spot floor). Loss curve confirms over-training: total_loss dropped 0.05 → 0.004 by iter 100, classifier converged to "predict background everywhere." **Implication: empty-label FT is fundamentally a sledgehammer; iter count is the only thing keeping it from collapsing to empty.** Simple-FT survived with usable conf distribution only because it stopped at 20 iters. Surgical FT needs the same — try `FT_ITERS ∈ {25, 50}` to find a usable conf tail before declaring the direction dead.
19. **GA + FT weight average — densest base ever (855 dets ≥0.6 vs simple-FT's 423), but the extra dets are high-confidence poison residue.** `step12_kaggle_ga_ft.py`: head-only, Phase A = gradient ascent on poison annotations (30 iters, -loss_cls, lr=5e-5), Phase B = empty-label FT from GA checkpoint (sweep iters {20,50,100,150}), Phase C = weight-average GA+FT at mix=0.3. iter20_avg0.3 produced the richest conf distribution of any unlearning method tried (median 0.502, 855 ≥0.6, density 1.218/img). With proven rescue recipe (lf=0.2 dm=0.05) scored **262.87 — 27 points WORSE than 235.62**. Density 0.505/img was 1.6× the proven sweet spot; the 432 extra ≥0.6 dets (vs simple-FT) are residue, not signal. **Why GA failed: GA disrupts response at the 20 specific unlearn-set boxes, but the poisoned head learned a general dashed-streak *pattern* that appears across the test set. The 0.3 weight-avg with FT then re-amplifies residual detections.** Iter trajectory confirms lesson 14: empty-label FT degrades monotonically with iters (iter150 has only 20 dets ≥0.6 vs iter20's 855). Direction dead — same structural issue as EWC's anchor-to-poisoned.
20. **Tri-model voting diagnostic: alternative methods don't see meaningfully different *high-confidence* dets than simple-FT.** `step13_tri_voting.py` counts agreement between 235.62 base and {surgical iter=25, EWC iter=25} at IoU≥T. Vote histogram: 92.1% of base dets get 2 votes (both alt methods agree), 7.3% get 1 vote, 0.6% (4 dets) get 0 votes. Of 969 consensus-add candidates (in BOTH surgical AND EWC but missing from base), **ZERO have mean conf ≥ 0.6, only 110 have mean conf ≥ 0.5, 364 have mean conf ≥ 0.4**. **Implication: all three methods share the same view of high-conf detections; their disagreement only exists in the low-conf zone, which is dominated by shared poison residue.** Submissions pending (`add_T0.5_minconf0.5`, `drop_T0.5_v1`) test if the small consensus-disagreement deltas matter; expected outcome is ±2 points. **The fundamental ceiling is the shared poisoned head — three unlearners can't see beyond it.** IoU thresholds T∈{0.3, 0.5} produced identical results (matches are solid or absent — same pattern as step9).
18. **Pixel-content post-processing exhausted — general-purpose "looks like a streak" features can't separate poison from real.** Step11 computed SNR (box brightness vs surrounding ring) and structure-tensor coherence (gradient direction coherence) for the unlearn-set poison annotations vs the 235.62 CSV dets. **Poison scores HIGHER on both features than the CSV dets** — SNR med 0.386 (poison) vs 0.297 (CSV unconditional) vs 0.230 (CSV rescued); coherence med 0.483 vs 0.439 vs 0.398. Reason: unlearn-set poison was designed *to look streak-like*, so general streak-shape metrics select FOR poison, not against it. Filter variants at any usable threshold destroy density disproportionately (snr≥0.5 leaves only 175 dets, 0.087/img). **Why dashedness was the exception: it targets a *specific poison signature* (dashed/segmented pattern), not general streak shape. The right pixel features for this task must target poison artifacts, not real-streak resemblance.** Geometric micro-ops also exhausted (no duplicates, no tiny boxes, edge dets statistically equivalent). **Post-processing space on the 235.62 CSV is genuinely exhausted.**
17. **EWC-lite (L2-anchor to poisoned weights, λ=100) preserves poison residue, not real streaks.** Step10 trained head.cls_score only with `L = focal_loss(empty) + λ·Σ(w − w_orig)²` for 25 iters. Output was the richest model CSV yet: 2146 dets at 1.073/img, **502 dets at conf≥0.6** (vs simple-FT raw's 423). But after the proven rescue recipe (lf=0.2 dm=0.05), scored **249.62 — 14 points worse than simple-FT at the same recipe (235.62)**. The 79 extra ≥0.6 dets and 191 rescued dets are dominantly poison residue. **Root cause: the anchor is to the poisoned weights** — that model is equally good at detecting real streaks AND poison residue, so L2-anchoring preserves both indiscriminately. True (Fisher-weighted) EWC would anchor by importance to a *clean* task, which we don't have. Author's "too many FPs → lower λ" rule means optimal λ → 0, which degrades EWC into simple-FT. **L2-anchor-to-poisoned is structurally wrong for this unlearning problem.** Don't waste more compute on λ sweeps.
16. **Surgical iter=25 as a filter on simple-FT rescue is redundant.** Step9 ensemble test: filter the 235.62 winning CSV by "keep simple-FT det iff surgical iter=25 has any det with IoU≥T at same location." All three T ∈ {0.1, 0.3, 0.5} produced identical 581-det results (matches are either solid IoU>0.5 or absent — no marginal IoU cases). Dropped 49 dets (8%) → scored 236.20 (−0.58 vs 235.62). **Net effect ≈ noise**; the 49 dropped dets were slightly biased toward "real streaks surgical missed" rather than "poison residue surgical caught." Implication: simple-FT rescue already captures ~all the signal that surgical's classifier knows about. **Surgical direction fully exhausted across all configurations** (standalone best 248.60, filter best 236.20). Further gains require fundamentally different unlearning math (EWC, GA+FT weight avg), not more variants of empty-label FT.
15. **Surgical FT iter=25 has higher per-det quality than simple-FT but it does NOT show up in mCADD.** Iter-sweep (`step8b_kaggle_surgical_ft_itersweep.py`) at 25 iters produced 1560 dets, dets/img 0.78, max conf 0.90 — distribution close to simple-FT raw. **Dashedness selectivity is strictly higher in every conf band**: at conf≥0.6, 35.5% of iter=25's dets have d≤0.05 vs simple-FT's 28.1%. This confirms the surgical hypothesis (frozen backbone+regression → cleaner geometry → more dets pass the real-streak test). **But scored 250.25 with rescue lf=0.2 dm=0.05** — statistical tie with simple-FT conf≥0.65 (250.79) at near-identical density (0.18 vs 0.16). **Density dominates per-det cleanness.** The 14.6-point gap to 235.62 is a density gap, not a quality gap. The surgical model produces the same QUALITY of dets as simple-FT, just fewer of them. Implication: to beat 235.62 via surgical, must push density into the 0.30+ range — try `lf=0.2 dm=0.06/0.07` to find out if relaxing dashedness gains more than it costs.
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
6. **FT iter count is the dominant conf-distribution lever for empty-label FT** (lesson 14). 20 iters (simple-FT) → median 0.40, max 0.94. 125 iters (surgical) → median 0.27, max 0.55. There is no "stable convergence" — empty-label loss decreases monotonically until the classifier predicts background everywhere. Sweep iters before assuming a layer-freezing choice failed.

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
