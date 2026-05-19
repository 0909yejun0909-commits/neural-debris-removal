"""
Step 15 — discriminating-features panel.

Lesson 18 + step14 both confirmed: general "looks like a streak" features
score real and poison ALMOST IDENTICALLY (poison was designed streak-like).
The only feature that's ever discriminated (dashedness, +7.75 pts) targeted
a SPECIFIC poison signature.

Premise of this script: compute a panel of candidate poison-specific shape /
intensity features. For each, compare distributions on
  (a) confirmed poison: 20 unlearn-set annotations
  (b) "mostly-real" cohort: rescued dets from 235.62 CSV (lesson 10 estimates
      ~80% real-streak rate; the cleanest cohort we have without ground truth)
  (c) "mixed" cohort: unconditional dets (conf>=0.6) from 235.62 CSV

Report per-feature Cohen's d between (a) and (b). Features with |d| > 0.8
are strong candidates. For the top candidates, generate one-sided filter
variants (drop test dets whose feature value falls inside the poison range).

Candidate features:
  1. bbox_length        max(w, h) — synthetic poison may have a narrower length distribution
  2. bbox_aspect        max(w,h)/min(w,h) — synthetic streaks may have fixed aspect
  3. bbox_area          w*h
  4. dist_from_center   sqrt((cx-512)^2 + (cy-512)^2) — image position bias
  5. fg_fill_ratio      fraction of bbox pixels above top-8% threshold — sparse for dashed
  6. intensity_mean     mean fg intensity (already covered by SNR but at a different scale)
  7. intensity_uniform  mean(fg)/std(fg) — synthetic streaks may have flatter intensity
  8. width_uniformity   std(perp dist from PCA axis) / span — synthetic = constant width
  9. straightness       perp_std / parallel_span — lower = straighter
 10. endpoint_grad      mean |grad| in 4-px box around each PCA endpoint, averaged
 11. intensity_skew     skewness of fg pixel intensities

Outputs:
  kaggle_outputs/step15_features/scored.csv      per-det feature dump
  kaggle_outputs/step15_features/separability.txt  Cohen's d + percentile printout
  kaggle_outputs/step15_features/filter_*.csv    only for features that separate
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from step6_morpho_filter import dets_to_str, load_img, parse_dets


UNLEARN_DIR  = "neural-debris-removal-in-streak-detection-models/unlearn_set"
UNLEARN_JSON = os.path.join(UNLEARN_DIR, "annotations_coco.json")
TEST_DIR     = "neural-debris-removal-in-streak-detection-models/test_set/test_set"
BEST_CSV     = "kaggle_outputs/morpho/rescue/simple-ft_rescue_lf0.2_dm0.05.csv"
OUT_DIR      = Path("kaggle_outputs/step15_features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FG_PERCENTILE = 92
MARGIN_PX     = 4
IMG_CENTER    = 512.0


def features(img, bbox):
    """Compute the full feature panel for one bbox. Returns dict (with Nones for failures)."""
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0

    out = {
        "bbox_length":      float(max(w, h)),
        "bbox_aspect":      float(max(w, h) / max(1e-6, min(w, h))),
        "bbox_area":        float(w * h),
        "dist_from_center": float(np.sqrt((cx - IMG_CENTER) ** 2 + (cy - IMG_CENTER) ** 2)),
        "fg_fill_ratio":    None,
        "intensity_mean":   None,
        "intensity_uniform": None,
        "width_uniformity": None,
        "straightness":     None,
        "endpoint_grad":    None,
        "intensity_skew":   None,
    }

    H, W = img.shape
    x1 = int(max(0, x - MARGIN_PX))
    y1 = int(max(0, y - MARGIN_PX))
    x2 = int(min(W, x + w + MARGIN_PX))
    y2 = int(min(H, y + h + MARGIN_PX))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return out
    crop = img[y1:y2, x1:x2]

    thresh = np.percentile(crop, FG_PERCENTILE)
    fg_mask = crop > thresh
    rows, cols = np.where(fg_mask)
    if len(rows) < 10:
        return out

    fg_vals = crop[fg_mask]
    out["fg_fill_ratio"]  = float(fg_mask.mean())
    out["intensity_mean"] = float(fg_vals.mean())
    std_v = float(fg_vals.std())
    out["intensity_uniform"] = float(out["intensity_mean"] / (std_v + 1e-6))
    # Sample skewness; safe-guard tiny std.
    if std_v > 1e-6:
        z = (fg_vals - out["intensity_mean"]) / std_v
        out["intensity_skew"] = float((z ** 3).mean())

    coords = np.column_stack([rows, cols]).astype(float)
    pca = PCA(n_components=2)
    pca.fit(coords)
    pc1 = pca.components_[0]

    centred = coords - coords.mean(axis=0)
    proj    = centred @ pc1
    recon   = proj[:, None] * pc1
    perp    = np.linalg.norm(centred - recon, axis=1)

    span = float(proj.max() - proj.min())
    if span > 1.0:
        out["width_uniformity"] = float(perp.std())
        out["straightness"]     = float(perp.std() / span)

    # Endpoint gradient — sample a small window at each PCA endpoint and average |grad|
    gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx * gx + gy * gy)
    centre_rc = coords.mean(axis=0)

    def grad_at(end_proj):
        end_rc = centre_rc + end_proj * pc1
        r = int(round(end_rc[0])); c = int(round(end_rc[1]))
        rr = slice(max(0, r - 2), min(gmag.shape[0], r + 3))
        cc = slice(max(0, c - 2), min(gmag.shape[1], c + 3))
        patch = gmag[rr, cc]
        return float(patch.mean()) if patch.size > 0 else None

    g1 = grad_at(proj.min())
    g2 = grad_at(proj.max())
    if g1 is not None and g2 is not None:
        out["endpoint_grad"] = 0.5 * (g1 + g2)

    return out


FEATURE_NAMES = [
    "bbox_length", "bbox_aspect", "bbox_area", "dist_from_center",
    "fg_fill_ratio", "intensity_mean", "intensity_uniform",
    "width_uniformity", "straightness", "endpoint_grad", "intensity_skew",
]


def score_poison():
    print("=== Scoring poison (unlearn-set annotations) ===")
    with open(UNLEARN_JSON) as f:
        coco = json.load(f)
    id_to_fname = {im["id"]: im["file_name"] for im in coco["images"]}
    out = []
    cache = {}
    for ann in coco["annotations"]:
        fname = id_to_fname[ann["image_id"]]
        if fname not in cache:
            cache[fname] = load_img(os.path.join(UNLEARN_DIR, fname))
        img = cache[fname]
        if img is None:
            continue
        f = features(img, ann["bbox"])
        f["ann_id"] = ann["id"]
        f["group"]  = "poison"
        out.append(f)
    print(f"  Scored {len(out)} / {len(coco['annotations'])} poison annotations")
    return out


def score_csv(csv_path):
    print(f"\n=== Scoring 235.62 base CSV dets ===")
    df = pd.read_csv(csv_path)
    df["dets"] = df["prediction_string"].apply(parse_dets)
    n_total = sum(len(d) for d in df["dets"])
    print(f"  {n_total} dets across {(df['dets'].str.len() > 0).sum()} non-empty rows")
    cache = {}
    out, seen = [], 0
    for _, row in df.iterrows():
        if not row["dets"]:
            continue
        img_id = row["image_id"]
        if img_id not in cache:
            cache[img_id] = load_img(os.path.join(TEST_DIR, f"{img_id}.png"))
        img = cache[img_id]
        if img is None:
            continue
        for det in row["dets"]:
            c, x, y, w, h = det
            f = features(img, (x, y, w, h))
            f["image_id"] = img_id
            f["conf"]     = c
            f["bbox"]     = (x, y, w, h)
            f["group"]    = "unconditional" if c >= 0.6 else "rescued"
            out.append(f)
            seen += 1
            if seen % 200 == 0:
                print(f"    ...{seen}/{n_total}")
    return out, df


def cohens_d(a, b):
    a = np.asarray([v for v in a if v is not None])
    b = np.asarray([v for v in b if v is not None])
    if len(a) < 3 or len(b) < 3:
        return None
    pooled = np.sqrt(((a.var(ddof=1) * (len(a) - 1)) +
                      (b.var(ddof=1) * (len(b) - 1))) / (len(a) + len(b) - 2))
    if pooled < 1e-9:
        return None
    return float((a.mean() - b.mean()) / pooled)


def fmt_pcts(vals):
    arr = np.array([v for v in vals if v is not None])
    if len(arr) == 0:
        return "no values"
    return (f"n={len(arr):4d}  min={arr.min():.3f}  p25={np.percentile(arr,25):.3f}  "
            f"med={np.median(arr):.3f}  p75={np.percentile(arr,75):.3f}  max={arr.max():.3f}")


def main():
    poison = score_poison()
    csv_recs, df_template = score_csv(BEST_CSV)

    rescued     = [r for r in csv_recs if r["group"] == "rescued"]
    unconditional = [r for r in csv_recs if r["group"] == "unconditional"]
    print(f"\n  Poison n={len(poison)}  Rescued n={len(rescued)}  Unconditional n={len(unconditional)}")

    # Per-feature separability report
    lines = []
    lines.append("=== Per-feature separability (poison vs rescued) ===")
    lines.append("(Cohen's |d| > 0.8 = strong, > 0.5 = moderate, < 0.2 = useless)")
    lines.append("")

    rankings = []
    for fname in FEATURE_NAMES:
        p_vals = [r[fname] for r in poison]
        r_vals = [r[fname] for r in rescued]
        u_vals = [r[fname] for r in unconditional]

        d_pr = cohens_d(p_vals, r_vals)
        d_pu = cohens_d(p_vals, u_vals)
        rankings.append((fname, abs(d_pr) if d_pr is not None else 0.0, d_pr, d_pu))

        lines.append(f"--- {fname} ---")
        lines.append(f"  poison        {fmt_pcts(p_vals)}")
        lines.append(f"  rescued       {fmt_pcts(r_vals)}")
        lines.append(f"  unconditional {fmt_pcts(u_vals)}")
        lines.append(f"  Cohen's d (poison vs rescued):       {d_pr if d_pr is None else f'{d_pr:+.3f}'}")
        lines.append(f"  Cohen's d (poison vs unconditional): {d_pu if d_pu is None else f'{d_pu:+.3f}'}")
        lines.append("")

    rankings.sort(key=lambda x: x[1], reverse=True)
    lines.append("=== Ranking by |Cohen's d| (poison vs rescued) ===")
    for fname, absd, d_pr, d_pu in rankings:
        lines.append(f"  {fname:20s}  |d|={absd:.3f}   d_pr={d_pr if d_pr is None else f'{d_pr:+.3f}':>7}   "
                     f"d_pu={d_pu if d_pu is None else f'{d_pu:+.3f}':>7}")

    report = "\n".join(lines)
    print("\n" + report)
    (OUT_DIR / "separability.txt").write_text(report)
    print(f"\nWrote report -> {OUT_DIR / 'separability.txt'}")

    # Dump per-det features for downstream analysis (rescued + unconditional)
    rows = []
    for r in csv_recs:
        c, x, y, w, h = r["conf"], *r["bbox"]
        row = {"image_id": r["image_id"], "conf": c, "x": x, "y": y, "w": w, "h": h,
               "group": r["group"]}
        for fn in FEATURE_NAMES:
            row[fn] = r[fn]
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_DIR / "scored.csv", index=False)
    print(f"Wrote per-det features -> {OUT_DIR / 'scored.csv'}")

    # Filter variants for any feature with |d_pr| >= 0.5 — drop dets whose feature
    # value falls inside the POISON p25-p75 range (i.e. looks poison-like).
    print("\n=== Filter variants for separating features (|d_pr| >= 0.5) ===")
    print(f"  {'variant':45s}  non_empty  total  avg/img  dropped")

    img_lookup = defaultdict(list)
    for r in csv_recs:
        img_lookup[r["image_id"]].append(r)

    for fname, absd, d_pr, _ in rankings:
        if absd < 0.5 or d_pr is None:
            continue
        p_arr = np.array([r[fname] for r in poison if r[fname] is not None])
        if len(p_arr) < 3:
            continue
        lo, hi = float(np.percentile(p_arr, 25)), float(np.percentile(p_arr, 75))

        out_strs = []
        total = 0; non_empty = 0; dropped = 0
        for _, row in df_template.iterrows():
            recs = img_lookup[row["image_id"]]
            kept = []
            for det in row["dets"]:
                # find feature value for this det
                fv = next((r[fname] for r in recs if r["bbox"] == tuple(det[1:])), None)
                if fv is not None and lo <= fv <= hi:
                    dropped += 1
                else:
                    kept.append(det)
            out_strs.append(dets_to_str(kept))
            if kept:
                total += len(kept); non_empty += 1

        out_df = df_template[["id", "image_id"]].copy()
        out_df["prediction_string"] = out_strs
        name = f"filter_{fname}_poison_q25_q75.csv"
        out_df.to_csv(OUT_DIR / name, index=False)
        print(f"  {name:45s}  {non_empty:4d}       {total:4d}    "
              f"{total/len(df_template):.3f}    {dropped}")


if __name__ == "__main__":
    main()
