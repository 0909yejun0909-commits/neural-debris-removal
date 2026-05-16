"""
Step 11 — image-content post-processing features (orthogonal to dashedness).

The 235.62 CSV has been heavily filtered: conf>=0.6 unconditional + rescued
(conf in [0.2,0.6) AND dashedness <= 0.05). Geometric micro-ops showed no
structural residual to clean (step11 inspection: no duplicates, no tiny boxes,
edge dets statistically equivalent to non-edge).

The remaining axis is image-content features that DASHEDNESS DOES NOT MEASURE.
Dashedness captures gappiness; this script tries two orthogonal axes:

1. SNR (signal-to-noise ratio): how much brighter is the box content vs a
   surrounding ring? Real streaks should have bright inside, dim outside.
   Poison residue / random-noise FPs would have similar brightness inside
   and outside (low SNR).

2. Structure-tensor coherence: gradient direction coherence inside the box.
   coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + eps)
   Real linear streaks have one dominant gradient direction → coherence ~ 1.
   Random noise has no preferred direction → coherence ~ 0.
   Note: a perfectly UNIFORM streak (no gradient inside) also gets
   coherence ~ 0; we mask the structure tensor to pixels with |grad| > eps.

Calibration: compute both features on (a) unlearn-set poison annotations
(definitely poison) and (b) 235.62 CSV dets (mix of real + residue). If
poison clusters distinctively, filter.

Bonus: split (b) by "unconditional" (conf>=0.6) vs "rescued" (conf<0.6).
Rescued dets passed strict dashedness; unconditional did not. If feature X
shows residue is concentrated in unconditional, it could prune that bucket
without re-touching the rescued ones.

Outputs:
- kaggle_outputs/step11_pixel/calibration.txt
- kaggle_outputs/step11_pixel/filter_*.csv  (only if calibration looks promising)
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from step6_morpho_filter import dets_to_str, load_img, parse_dets


UNLEARN_DIR  = "neural-debris-removal-in-streak-detection-models/unlearn_set"
UNLEARN_JSON = os.path.join(UNLEARN_DIR, "annotations_coco.json")
TEST_DIR     = "neural-debris-removal-in-streak-detection-models/test_set/test_set"
BEST_CSV     = "kaggle_outputs/morpho/rescue/simple-ft_rescue_lf0.2_dm0.05.csv"
OUT_DIR      = Path("kaggle_outputs/step11_pixel")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RING_PX = 8   # surrounding ring width for SNR


def snr(img, bbox):
    """(mean_inside - mean_ring) / (std_ring + eps). Higher = brighter signal."""
    x, y, w, h = bbox
    H, W = img.shape
    x1, y1 = int(max(0, x)),         int(max(0, y))
    x2, y2 = int(min(W, x + w)),     int(min(H, y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    inside = img[y1:y2, x1:x2]
    if inside.size < 4:
        return None

    rx1, ry1 = int(max(0, x - RING_PX)), int(max(0, y - RING_PX))
    rx2, ry2 = int(min(W, x + w + RING_PX)), int(min(H, y + h + RING_PX))
    outer = img[ry1:ry2, rx1:rx2]
    if outer.size <= inside.size:
        return None

    # ring = outer minus inside
    mask = np.ones_like(outer, dtype=bool)
    mask[(y1 - ry1):(y2 - ry1), (x1 - rx1):(x2 - rx1)] = False
    ring = outer[mask]
    if ring.size < 4:
        return None

    mean_in = float(inside.mean())
    mean_r  = float(ring.mean())
    std_r   = float(ring.std())
    return (mean_in - mean_r) / (std_r + 1e-3)


def coherence(img, bbox):
    """Structure-tensor coherence over edge pixels in the box. 1 = perfectly
    linear gradient direction; 0 = isotropic."""
    x, y, w, h = bbox
    H, W = img.shape
    x1, y1 = int(max(0, x)),         int(max(0, y))
    x2, y2 = int(min(W, x + w)),     int(min(H, y + h))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop = img[y1:y2, x1:x2]

    # Sobel gradients on the crop. Float32 → no scaling artifacts on 16-bit-derived data.
    gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx * gx + gy * gy)
    thr = np.percentile(mag, 85)   # top 15% of gradient magnitudes
    mask = mag >= thr
    if mask.sum() < 8:
        return None

    sx = gx[mask]; sy = gy[mask]
    Sxx = float((sx * sx).sum())
    Syy = float((sy * sy).sum())
    Sxy = float((sx * sy).sum())

    # Eigenvalues of [[Sxx, Sxy], [Sxy, Syy]]
    tr   = Sxx + Syy
    det  = Sxx * Syy - Sxy * Sxy
    disc = max(0.0, tr * tr / 4 - det)
    sq   = disc ** 0.5
    lam1 = tr / 2 + sq
    lam2 = tr / 2 - sq
    if lam1 + lam2 < 1e-9:
        return None
    return (lam1 - lam2) / (lam1 + lam2)


def score_unlearn():
    print("=== Calibration: unlearn-set poison annotations ===")
    with open(UNLEARN_JSON) as f:
        coco = json.load(f)
    id_to_fname = {im["id"]: im["file_name"] for im in coco["images"]}
    snrs, cohs = [], []
    for ann in coco["annotations"]:
        img = load_img(os.path.join(UNLEARN_DIR, id_to_fname[ann["image_id"]]))
        if img is None:
            continue
        s = snr(img, ann["bbox"])
        c = coherence(img, ann["bbox"])
        if s is not None: snrs.append(s)
        if c is not None: cohs.append(c)
    print(f"  Scored {len(snrs)} SNR / {len(cohs)} coherence / "
          f"{len(coco['annotations'])} annotations")
    return snrs, cohs


def score_csv(csv_path):
    """Returns list of dicts: {image_id, det, conf, snr, coh, bucket}."""
    df = pd.read_csv(csv_path)
    df["dets"] = df["prediction_string"].apply(parse_dets)
    img_cache = {}
    out = []
    n_total = sum(len(d) for d in df["dets"])
    print(f"  {csv_path}: {n_total} dets across {(df['dets'].str.len() > 0).sum()} images")
    seen = 0
    for _, row in df.iterrows():
        if not row["dets"]:
            continue
        img_id = row["image_id"]
        if img_id not in img_cache:
            img_cache[img_id] = load_img(os.path.join(TEST_DIR, f"{img_id}.png"))
        img = img_cache[img_id]
        if img is None:
            continue
        for det in row["dets"]:
            c = det[0]
            s = snr(img, det[1:])
            k = coherence(img, det[1:])
            out.append({
                "image_id": img_id, "det": det, "conf": c,
                "snr": s, "coh": k,
                "bucket": "unconditional" if c >= 0.6 else "rescued",
            })
            seen += 1
            if seen % 200 == 0:
                print(f"    ...{seen}/{n_total}")
    return out


def summarize(name, vals):
    arr = np.array([v for v in vals if v is not None])
    if len(arr) == 0:
        print(f"  {name}: no values")
        return
    print(f"  {name:32s} n={len(arr):4d}  "
          f"min={arr.min():.3f}  p10={np.percentile(arr,10):.3f}  "
          f"p25={np.percentile(arr,25):.3f}  med={np.median(arr):.3f}  "
          f"p75={np.percentile(arr,75):.3f}  p90={np.percentile(arr,90):.3f}  "
          f"max={arr.max():.3f}")


def main():
    poison_snr, poison_coh = score_unlearn()

    print("\n=== Calibration: 235.62 CSV dets (rescue+conf>=0.6) ===")
    recs = score_csv(BEST_CSV)

    # Summaries
    print("\n=== SNR distributions ===")
    summarize("Poison (unlearn anns)", poison_snr)
    summarize("CSV total",             [r["snr"] for r in recs])
    summarize("CSV unconditional",     [r["snr"] for r in recs if r["bucket"] == "unconditional"])
    summarize("CSV rescued",           [r["snr"] for r in recs if r["bucket"] == "rescued"])

    print("\n=== Coherence distributions ===")
    summarize("Poison (unlearn anns)", poison_coh)
    summarize("CSV total",             [r["coh"] for r in recs])
    summarize("CSV unconditional",     [r["coh"] for r in recs if r["bucket"] == "unconditional"])
    summarize("CSV rescued",           [r["coh"] for r in recs if r["bucket"] == "rescued"])

    # Save raw record dump for downstream analysis
    out_rows = []
    for r in recs:
        c, x, y, w, h = r["det"]
        out_rows.append({
            "image_id": r["image_id"], "conf": c, "x": x, "y": y, "w": w, "h": h,
            "bucket": r["bucket"], "snr": r["snr"], "coh": r["coh"],
        })
    pd.DataFrame(out_rows).to_csv(OUT_DIR / "scored_dets.csv", index=False)
    print(f"\nWrote per-det scores to {OUT_DIR / 'scored_dets.csv'}")

    # If poison clusters distinctively, generate filter variants.
    # We define "distinctive" as: poison p75 below (or above) CSV p25.
    poison_snr_arr = np.array(poison_snr)
    csv_snr_arr    = np.array([r["snr"] for r in recs if r["snr"] is not None])
    poison_coh_arr = np.array(poison_coh)
    csv_coh_arr    = np.array([r["coh"] for r in recs if r["coh"] is not None])

    print("\n=== Filter feasibility ===")
    snr_signal = np.percentile(poison_snr_arr, 75) < np.percentile(csv_snr_arr, 25)
    coh_signal = np.percentile(poison_coh_arr, 75) < np.percentile(csv_coh_arr, 25)
    print(f"  SNR poison p75 < CSV p25?       {snr_signal}  "
          f"(poison p75={np.percentile(poison_snr_arr,75):.3f} vs CSV p25={np.percentile(csv_snr_arr,25):.3f})")
    print(f"  Coherence poison p75 < CSV p25? {coh_signal}  "
          f"(poison p75={np.percentile(poison_coh_arr,75):.3f} vs CSV p25={np.percentile(csv_coh_arr,25):.3f})")

    # Generate variants regardless — let the user judge from the data
    print("\n=== Filter variants (drop dets below threshold) ===")
    df_template = pd.read_csv(BEST_CSV)
    df_template["dets"] = df_template["prediction_string"].apply(parse_dets)

    # Build {image_id: [(det, snr, coh)]}
    img_lookup = defaultdict(list)
    for r in recs:
        img_lookup[r["image_id"]].append((r["det"], r["snr"], r["coh"]))

    def generate(name, predicate):
        out_rows, total, non_empty = [], 0, 0
        for _, row in df_template.iterrows():
            img_id = row["image_id"]
            kept = []
            for det in row["dets"]:
                snr_v = next((s for d, s, _ in img_lookup[img_id] if d == det), None)
                coh_v = next((k for d, _, k in img_lookup[img_id] if d == det), None)
                if predicate(det, snr_v, coh_v):
                    kept.append(det)
            out_rows.append(dets_to_str(kept))
            if kept:
                total += len(kept); non_empty += 1
        out_df = df_template[["id", "image_id"]].copy()
        out_df["prediction_string"] = out_rows
        out_df.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"  {name:42s}  non_empty={non_empty:4d}  total={total:4d}  avg={total/len(df_template):.3f}")

    # Variants: drop dets where snr/coh below threshold (None → keep).
    for T in [0.5, 1.0, 1.5, 2.0]:
        generate(f"filter_snr_ge_{T}", lambda d, s, c, T=T: (s is None) or (s >= T))
    for T in [0.2, 0.3, 0.4, 0.5]:
        generate(f"filter_coh_ge_{T}", lambda d, s, c, T=T: (c is None) or (c >= T))


if __name__ == "__main__":
    main()
