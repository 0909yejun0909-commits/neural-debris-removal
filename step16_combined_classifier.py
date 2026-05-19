"""
Step 16 — combined-feature logistic classifier for poison vs real.

Premise: length filter has hit local optimum at 232.63 (uncond stack le40 OR
[45.2, 51.2]). Further gains likely require combining orthogonal features.

Step 15 Cohen's d ranking (poison vs rescued):
  bbox_length        -0.977  (used, strongest)
  bbox_area          -0.791  (correlated with length)
  endpoint_grad      -0.475  (poison has softer endpoints — orthogonal axis)
  intensity_uniform  -0.452  (poison less uniform — orthogonal axis)
  width_uniformity   -0.401  (poison narrower std — orthogonal axis)

Train a logistic regression on (length, endpoint_grad, intensity_uniform,
width_uniformity, bbox_area):
  positive = 20 poison annotations
  negative = 207 rescued dets (lesson 10: ~80% real)

Class imbalance 1:10 -> class_weight='balanced'. Features standardized.

Evaluate via leave-one-out CV (AUC). Apply trained model to all dets in the
235.62 base CSV. Generate filter variants at multiple probability thresholds:

  A. Standalone classifier — drop unconditional dets with p_poison >= T
  B. Additive on 232.63 winning shape — keep the 135 drops from the winner,
     add extra drops from dets NOT already filtered if p_poison >= T

Outputs:
  kaggle_outputs/step16_combined/cv_report.txt        AUC + per-feature info
  kaggle_outputs/step16_combined/scored_dets.csv      per-det p_poison
  kaggle_outputs/step16_combined/standalone_T*.csv    standalone filter variants
  kaggle_outputs/step16_combined/additive_T*.csv      additive variants on top of 232.63
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score

from step6_morpho_filter import parse_dets, dets_to_str, load_img
from step15_discriminating_features import features, FEATURE_NAMES


UNLEARN_DIR  = "neural-debris-removal-in-streak-detection-models/unlearn_set"
UNLEARN_JSON = os.path.join(UNLEARN_DIR, "annotations_coco.json")
BEST_CSV     = "kaggle_outputs/morpho/rescue/simple-ft_rescue_lf0.2_dm0.05.csv"
WINNER_CSV   = "kaggle_outputs/step15_features/filter_length_uncond_stack_le40_or_45_51.csv"
SCORED_CSV   = "kaggle_outputs/step15_features/scored.csv"
OUT_DIR      = Path("kaggle_outputs/step16_combined")
OUT_DIR.mkdir(parents=True, exist_ok=True)

USED_FEATURES = ["bbox_length", "bbox_area", "endpoint_grad", "intensity_uniform", "width_uniformity"]


def score_poison():
    """Recompute features for the 20 unlearn-set annotations (not cached in scored.csv)."""
    with open(UNLEARN_JSON) as f:
        coco = json.load(f)
    id_to_fname = {im["id"]: im["file_name"] for im in coco["images"]}
    cache, rows = {}, []
    for ann in coco["annotations"]:
        fname = id_to_fname[ann["image_id"]]
        if fname not in cache:
            cache[fname] = load_img(os.path.join(UNLEARN_DIR, fname))
        img = cache[fname]
        if img is None:
            continue
        f = features(img, ann["bbox"])
        f["ann_id"] = ann["id"]
        rows.append(f)
    return pd.DataFrame(rows)


def main():
    print("=== Loading features ===")
    scored = pd.read_csv(SCORED_CSV)
    poison_df = score_poison()
    print(f"  Poison: {len(poison_df)}")
    print(f"  Unconditional (CSV): {(scored['group']=='unconditional').sum()}")
    print(f"  Rescued (CSV): {(scored['group']=='rescued').sum()}")

    # Build training set: poison (label=1) vs rescued (label=0)
    train_pos = poison_df[USED_FEATURES].dropna()
    train_neg = scored[scored["group"] == "rescued"][USED_FEATURES].dropna()
    print(f"\n  Training set: {len(train_pos)} poison + {len(train_neg)} rescued (after drop-NA)")

    X_train = np.vstack([train_pos.values, train_neg.values])
    y_train = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    # Leave-one-out CV — small n=20 poison, evaluate AUC honestly
    loo = LeaveOneOut()
    cv_scores = np.zeros(len(y_train))
    for tr_idx, te_idx in loo.split(X_train_s):
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(X_train_s[tr_idx], y_train[tr_idx])
        cv_scores[te_idx] = clf.predict_proba(X_train_s[te_idx])[:, 1]
    auc = roc_auc_score(y_train, cv_scores)

    # Final fit on all training data
    clf_final = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf_final.fit(X_train_s, y_train)

    coefs = dict(zip(USED_FEATURES, clf_final.coef_[0]))
    intercept = float(clf_final.intercept_[0])

    report_lines = [
        f"=== Step 16 logistic classifier ===",
        f"  features: {USED_FEATURES}",
        f"  training: {int(y_train.sum())} poison + {int((1-y_train).sum())} rescued",
        f"  LOO-CV AUC: {auc:.4f}",
        f"  intercept: {intercept:+.4f}",
    ]
    for name, c in coefs.items():
        report_lines.append(f"  coef[{name}]: {c:+.4f}")
    # Decision threshold quality on CV
    report_lines.append("\nLOO-CV p_poison percentiles by group:")
    p_pos = cv_scores[y_train == 1]
    p_neg = cv_scores[y_train == 0]
    for label, arr in [("poison", p_pos), ("rescued", p_neg)]:
        report_lines.append(
            f"  {label:8s} n={len(arr):3d}  "
            f"p10={np.percentile(arr,10):.3f}  p25={np.percentile(arr,25):.3f}  "
            f"med={np.median(arr):.3f}  p75={np.percentile(arr,75):.3f}  p90={np.percentile(arr,90):.3f}"
        )
    report = "\n".join(report_lines)
    print("\n" + report)
    (OUT_DIR / "cv_report.txt").write_text(report)

    # Score every det in the 235.62 base CSV
    X_all = scored[USED_FEATURES].copy()
    valid_mask = X_all.notna().all(axis=1)
    p_all = np.full(len(scored), np.nan)
    if valid_mask.sum() > 0:
        Xv = scaler.transform(X_all[valid_mask].values)
        p_all[valid_mask.values] = clf_final.predict_proba(Xv)[:, 1]
    scored["p_poison"] = p_all

    # Per-bucket p_poison summary
    print("\np_poison by bucket on the 235.62 base CSV:")
    for grp in ["unconditional", "rescued"]:
        arr = scored.loc[scored["group"] == grp, "p_poison"].dropna().values
        print(f"  {grp:14s} n={len(arr):4d}  "
              f"p10={np.percentile(arr,10):.3f}  p25={np.percentile(arr,25):.3f}  "
              f"med={np.median(arr):.3f}  p75={np.percentile(arr,75):.3f}  p90={np.percentile(arr,90):.3f}")

    scored.to_csv(OUT_DIR / "scored_dets.csv", index=False)
    print(f"\nWrote per-det p_poison -> {OUT_DIR / 'scored_dets.csv'}")

    # Build {image_id: {bbox tuple: (p_poison, group)}}
    lookup = {}
    for _, r in scored.iterrows():
        lookup.setdefault(r["image_id"], {})[(r["x"], r["y"], r["w"], r["h"])] = (
            r["p_poison"], r["group"]
        )

    template = pd.read_csv(BEST_CSV)
    template["dets"] = template["prediction_string"].apply(parse_dets)

    # Existing winner drops (for ADDITIVE mode) — read which dets the 232.63 CSV omits
    winner = pd.read_csv(WINNER_CSV)
    winner["dets"] = winner["prediction_string"].apply(parse_dets)
    winner_kept = defaultdict(set)
    for _, row in winner.iterrows():
        winner_kept[row["image_id"]] = {tuple(d) for d in row["dets"]}

    def variant(predicate, name):
        out, total, non_empty, dropped = [], 0, 0, 0
        for _, row in template.iterrows():
            kept = []
            per_img = lookup.get(row["image_id"], {})
            for det in row["dets"]:
                info = per_img.get(tuple(det[1:]))
                p, g = (info if info is not None else (None, None))
                if p is not None and predicate(p, g, row["image_id"], det):
                    dropped += 1
                else:
                    kept.append(det)
            out.append(dets_to_str(kept))
            if kept:
                total += len(kept); non_empty += 1
        out_df = template[["id", "image_id"]].copy()
        out_df["prediction_string"] = out
        out_df.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"  {name:55s}  total={total:4d}  drop={dropped:3d}  "
              f"avg={total/len(template):.3f}")

    # === A. Standalone: drop unconditional dets where p_poison >= T ===
    print("\n=== A. Standalone — drop UNCOND dets with p_poison >= T ===")
    for T in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
        variant(
            lambda p, g, iid, det, T=T: g == "unconditional" and p >= T,
            f"standalone_T{T:.2f}",
        )

    # === B. Additive on 232.63 winner — start from winner, ADD more drops if p_poison >= T ===
    print("\n=== B. Additive on 232.63 winner ===")
    for T in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        variant(
            lambda p, g, iid, det, T=T:
                # drop if det is not in the winner-kept set (i.e. already dropped by winner)
                tuple(det) not in winner_kept.get(iid, set())
                # OR if it's an unconditional det with p_poison >= T (additive)
                or (g == "unconditional" and p >= T),
            f"additive_T{T:.2f}",
        )


if __name__ == "__main__":
    main()
