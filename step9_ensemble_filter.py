"""
Step 9 — Ensemble: use surgical iter=25 as a *filter* on simple-FT dets.

Hypothesis (from lesson 15): surgical model has higher per-det quality
(dashedness-selectivity ~7 pts higher in every conf band) but produces too few
dets to compete on density. Simple-FT has the density. Combine: keep simple-FT
dets that surgical *also* detects (IoU-matched).

This is purely local — both CSVs are on disk. No Kaggle kernel needed.

Inputs:
- A (high-density base):     simple-FT raw (2072 dets, 1.04/img) — what to filter
- B (also-tried base):       simple-FT rescue lf=0.2 dm=0.05 (630 dets, 0.315/img) — the 235.62 winner
- F (filter / second opinion): surgical iter=25 raw (1560 dets, 0.78/img)

Variants emitted, for IoU threshold T ∈ {0.1, 0.3, 0.5}:
- filter_rawA_T{T}: keep A's dets where any F det matches at IoU >= T
- filter_bestB_T{T}: keep B's dets where any F det matches at IoU >= T

Density of each variant is printed so we can pick which (if any) to submit.

Note: "matching" here means simple-FT det X has any surgical det Y with
IoU(X, Y) >= T. Surgical's confidence doesn't matter — only whether it
predicted *something* in the same region. This treats surgical as a
"region-of-interest" detector, not a confidence reweighter.
"""

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from step6_morpho_filter import parse_dets, dets_to_str


CSV_A = "kaggle_outputs/simple-ft_276.91/submission.csv"
CSV_B = "kaggle_outputs/morpho/rescue/simple-ft_rescue_lf0.2_dm0.05.csv"   # 235.62 winner
CSV_F = "step8b_final_outputs/submission_iter25.csv"

OUT_DIR = Path("kaggle_outputs/step9_ensemble")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IOU_THRESHOLDS = [0.1, 0.3, 0.5]


def to_xyxy(det):
    """det = (conf, x, y, w, h) → (x1, y1, x2, y2)."""
    _, x, y, w, h = det
    return (x, y, x + w, y + h)


def iou(a, b):
    """IoU of two (x1, y1, x2, y2) boxes. 0 if disjoint."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (a_area + b_area - inter)


def load_dets_by_image(csv_path):
    """Returns {image_id: [det, ...]} where det = (conf, x, y, w, h)."""
    df = pd.read_csv(csv_path)
    out = {}
    for _, r in df.iterrows():
        out[int(r["image_id"])] = parse_dets(r["prediction_string"])
    return df, out


def filter_by_consensus(base_dets_by_img, filter_dets_by_img, T):
    """For each image: keep a base det iff any filter det has IoU(base, filter) >= T."""
    kept_by_img = {}
    n_in, n_out = 0, 0
    for img_id, base_dets in base_dets_by_img.items():
        f_dets = filter_dets_by_img.get(img_id, [])
        f_boxes = [to_xyxy(d) for d in f_dets]
        kept = []
        for bd in base_dets:
            bb = to_xyxy(bd)
            for fb in f_boxes:
                if iou(bb, fb) >= T:
                    kept.append(bd)
                    break
        n_in += len(base_dets)
        n_out += len(kept)
        kept_by_img[img_id] = kept
    return kept_by_img, n_in, n_out


def write_csv(df_template, dets_by_img, out_path):
    """df_template has the right (id, image_id) ordering; we rewrite prediction_string."""
    rows = []
    for _, r in df_template.iterrows():
        img_id = int(r["image_id"])
        ps = dets_to_str(dets_by_img.get(img_id, []))
        rows.append((r["id"], r["image_id"], ps))
    out = pd.DataFrame(rows, columns=["id", "image_id", "prediction_string"])
    out.to_csv(out_path, index=False)


def main():
    print("Loading sources...")
    df_A, A = load_dets_by_image(CSV_A)
    df_B, B = load_dets_by_image(CSV_B)
    _,    F = load_dets_by_image(CSV_F)

    n_A = sum(len(v) for v in A.values())
    n_B = sum(len(v) for v in B.values())
    n_F = sum(len(v) for v in F.values())
    n_img = len(df_A)
    print(f"  A (simple-FT raw):        {n_A:5d} dets ({n_A/n_img:.3f}/img)")
    print(f"  B (simple-FT rescue best):{n_B:5d} dets ({n_B/n_img:.3f}/img)   <- 235.62 winner")
    print(f"  F (surgical iter=25):     {n_F:5d} dets ({n_F/n_img:.3f}/img)   <- filter")

    print("\n=== Ensemble variants ===")
    print(f"{'variant':30s} | {'in':>5s} -> {'out':>5s} | {'dets/img':>9s} | retention")
    print("-" * 75)

    results = []
    for T in IOU_THRESHOLDS:
        # Filter A by F
        out_A, in_n, out_n = filter_by_consensus(A, F, T)
        name = f"filter_rawA_T{T}"
        path = OUT_DIR / f"{name}.csv"
        write_csv(df_A, out_A, path)
        ret = out_n / max(1, in_n)
        print(f"{name:30s} | {in_n:>5d} -> {out_n:>5d} | {out_n/n_img:>9.3f} | {ret:.1%}")
        results.append((name, out_n/n_img, ret, str(path)))

        # Filter B by F
        out_B, in_n, out_n = filter_by_consensus(B, F, T)
        name = f"filter_bestB_T{T}"
        path = OUT_DIR / f"{name}.csv"
        write_csv(df_B, out_B, path)
        ret = out_n / max(1, in_n)
        print(f"{name:30s} | {in_n:>5d} -> {out_n:>5d} | {out_n/n_img:>9.3f} | {ret:.1%}")
        results.append((name, out_n/n_img, ret, str(path)))

    print("\n=== Density landscape reference ===")
    print("  235.62 (best ever):       0.315 dets/img")
    print("  238.99 (dm=0.06):         0.399 dets/img")
    print("  243.37 (conf>=0.6):       0.21  dets/img")
    print("  248.60 (surgical+dm=0.06):0.261 dets/img")
    print("  250.25 (surgical+dm=0.05):0.182 dets/img")
    print("  268.80 (targeted):        0.17  dets/img")
    print("  284.20 (empty):           0.00  dets/img")

    print("\nAll variants written to:", OUT_DIR)


if __name__ == "__main__":
    main()
