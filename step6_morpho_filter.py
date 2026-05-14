"""
Dashedness filter v2 — three fixes over v1:
  1. Foreground: top-8% percentile threshold (robust to streak brightness)
     instead of mean+2.5σ (which missed faint streaks).
  2. Off-axis rejection: drop foreground pixels >PERP_MAX_PX from the
     principal axis before measuring gaps (removes background star contamination).
  3. Metric: max_gap/span (runlength-based) instead of empty-bin fraction.
     Captures physical gaps regardless of streak length or bin width.
"""
import cv2
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UNLEARN_DIR = "neural-debris-removal-in-streak-detection-models/unlearn_set"
UNLEARN_JSON = os.path.join(UNLEARN_DIR, "annotations_coco.json")
TEST_DIR     = "neural-debris-removal-in-streak-detection-models/test_set/test_set"
BEST_CSV     = "kaggle_outputs/threshold_sweep/simple-ft_conf0.6.csv"
OUT_DIR      = Path("kaggle_outputs/morpho")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FG_PERCENTILE = 92   # foreground = top 8% brightest pixels in crop
PERP_MAX_PX   = 8    # max perpendicular distance from principal axis to keep a pixel
MIN_SPAN_PX   = 15   # skip (return None = keep det) if on-axis span < this


def parse_dets(s):
    s = (s or "").strip()
    if not s:
        return []
    parts = s.split()
    return [tuple(map(float, parts[i:i+5])) for i in range(0, len(parts), 5)]


def dets_to_str(dets):
    if not dets:
        return " "
    return " ".join(f"{c:.6f} {x:.2f} {y:.2f} {w:.2f} {h:.2f}" for c, x, y, w, h in dets)


def load_img(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return (img / 65535.0 * 255.0).astype(np.float32)


def dashedness(img, bbox):
    """
    Returns max_gap/span in [0,1] (higher = more dashed).
    Returns None when undetermined — caller should KEEP the detection.
    """
    x, y, w, h = bbox
    x1 = int(max(0, x - 4));          y1 = int(max(0, y - 4))
    x2 = int(min(img.shape[1], x + w + 4)); y2 = int(min(img.shape[0], y + h + 4))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Foreground: top (100 - FG_PERCENTILE)% brightest pixels
    thresh = np.percentile(crop, FG_PERCENTILE)
    rows, cols = np.where(crop > thresh)
    if len(rows) < 10:
        return None
    coords = np.column_stack([rows, cols]).astype(float)

    # Principal axis via PCA
    pca = PCA(n_components=1)
    pca.fit(coords)
    pc1 = pca.components_[0]

    # Project and compute perpendicular distance from axis
    centered = coords - coords.mean(axis=0)
    proj     = centered @ pc1                       # (N,) along axis
    recon    = proj[:, None] * pc1                  # projection back to 2D
    perp     = np.linalg.norm(centered - recon, axis=1)

    # Keep only on-axis pixels (discard background stars / noise)
    on_axis = proj[perp < PERP_MAX_PX]
    if len(on_axis) < 5:
        return None

    span = on_axis.max() - on_axis.min()
    if span < MIN_SPAN_PX:
        return None

    max_gap = np.diff(np.sort(on_axis)).max()
    return float(max_gap / span)


def compute_scores(records):
    """records: list of (image_id, det_tuple, img). Fills in dashedness scores."""
    results = []
    for image_id, det, img in records:
        c, x, y, w, h = det
        score = dashedness(img, [x, y, w, h])
        # also save crop for sanity grid
        x1 = int(max(0, x - 4));          y1 = int(max(0, y - 4))
        x2 = int(min(img.shape[1], x + w + 4)); y2 = int(min(img.shape[0], y + h + 4))
        crop = img[y1:y2, x1:x2].copy()
        results.append({"image_id": image_id, "det": det, "score": score, "crop": crop})
    return results


def print_percentiles(name, scores):
    arr = np.array(scores)
    pcts = [10, 25, 50, 75, 90]
    vals = np.percentile(arr, pcts)
    print(f"  {name} (n={len(arr)}):")
    for p, v in zip(pcts, vals):
        print(f"    p{p:2d}: {v:.4f}")


def main():
    # ── Calibration on unlearn set (confirmed poison) ──────────────────────────
    print("=== Calibration: unlearn set (poisoned) ===")
    with open(UNLEARN_JSON) as f:
        coco = json.load(f)
    id_to_fname = {img["id"]: img["file_name"] for img in coco["images"]}

    unlearn_scores = []
    for ann in coco["annotations"]:
        img_path = os.path.join(UNLEARN_DIR, id_to_fname[ann["image_id"]])
        img = load_img(img_path)
        if img is None:
            continue
        s = dashedness(img, ann["bbox"])
        if s is not None:
            unlearn_scores.append(s)
    print(f"  Scored {len(unlearn_scores)} / {len(coco['annotations'])} annotations")
    print_percentiles("Unlearn (poison)", unlearn_scores)

    # ── Score all conf>=0.6 test detections ────────────────────────────────────
    print("\n=== Scoring simple-FT conf>=0.6 detections ===")
    df = pd.read_csv(BEST_CSV)
    df["dets"] = df["prediction_string"].apply(parse_dets)

    records = []
    for _, row in df.iterrows():
        if not row["dets"]:
            continue
        img = load_img(os.path.join(TEST_DIR, f"{row['image_id']}.png"))
        if img is None:
            continue
        for det in row["dets"]:
            records.append((row["image_id"], det, img))

    results = compute_scores(records)
    scored  = [r for r in results if r["score"] is not None]
    all_scores = [r["score"] for r in scored]
    print(f"  Scored {len(scored)} / {len(results)} detections (None = keep)")
    print_percentiles("Simple-FT test dets", all_scores)

    # ── Histogram ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(unlearn_scores, bins=25, alpha=0.55, label="Unlearn (poison)", density=True)
    ax.hist(all_scores,     bins=25, alpha=0.55, label="Simple-FT conf≥0.6 (test)", density=True)
    ax.set_xlabel("Dashedness (max_gap / span)")
    ax.set_ylabel("Density")
    ax.set_title("Dashedness v2 distribution")
    ax.legend()
    hist_path = OUT_DIR / "dashedness_hist_v2.png"
    plt.savefig(hist_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  Histogram saved")

    # ── Sanity grid ────────────────────────────────────────────────────────────
    sorted_results = sorted(scored, key=lambda r: r["score"])
    n = len(sorted_results)
    # 4 low / 4 mid / 4 high
    quartile_idx = (
        [0, 1, 2, 3] +
        [n // 2 - 2, n // 2 - 1, n // 2, n // 2 + 1] +
        [n - 4, n - 3, n - 2, n - 1]
    )
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax, idx in zip(axes.flat, quartile_idx):
        r = sorted_results[idx]
        ax.imshow(r["crop"], cmap="gray")
        ax.set_title(f"score={r['score']:.3f}\nID:{r['image_id']}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    grid_path = OUT_DIR / "sanity_grid_v2.png"
    plt.savefig(grid_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Sanity grid saved")

    # Build image_id -> results lookup
    from collections import defaultdict
    img_det_scores = defaultdict(list)
    for r in results:
        img_det_scores[r["image_id"]].append((r["det"], r["score"]))

    # ── Filter variants ────────────────────────────────────────────────────────
    print("\n=== Filtered variants ===")
    print(f"  {'variant':42s}  non_empty  total_dets  avg/img  dets_dropped")
    baseline_total = sum(len(row["dets"]) for _, row in df.iterrows())

    for T in [0.05, 0.08, 0.12, 0.20, 0.30]:
        out_rows = []
        total_dets = 0
        non_empty  = 0

        for _, row in df.iterrows():
            img_id = row["image_id"]
            kept = []
            for det in row["dets"]:
                # find matching score in lookup
                score = next(
                    (s for d, s in img_det_scores[img_id] if d == det),
                    None
                )
                # keep if score is None (undetermined) or below threshold
                if score is None or score < T:
                    kept.append(det)
            out_rows.append(dets_to_str(kept))
            if kept:
                total_dets += len(kept)
                non_empty  += 1

        out_df = df[["id", "image_id"]].copy()
        out_df["prediction_string"] = out_rows
        name = f"simple-ft_conf0.6_dashv2_T{T:.2f}.csv"
        out_df.to_csv(OUT_DIR / name, index=False)
        dropped = baseline_total - total_dets
        print(f"  {name:42s}  {non_empty:4d}       {total_dets:4d}        {total_dets/len(df):.3f}    {dropped}")


if __name__ == "__main__":
    main()
