"""
Step 21: Local template matching against 20 unlearn-set poison templates.
Adapts step14_template_match.py to run locally on the 232.63 base CSV.
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, "step14_log")
from step6_morpho_filter import dets_to_str, load_img, parse_dets

UNLEARN_DIR = "neural-debris-removal-in-streak-detection-models/unlearn_set"
UNLEARN_JSON = "neural-debris-removal-in-streak-detection-models/unlearn_set/annotations_coco.json"
TEST_DIR = "neural-debris-removal-in-streak-detection-models/test_set/test_set"
BEST_CSV = "kaggle_outputs/step15_features/filter_length_uncond_stack_le40_or_45_51.csv"  # 232.63 base
OUT_DIR = Path("kaggle_outputs/step21_template_local")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANVAS_H, CANVAS_W = 24, 96
FG_PERCENTILE = 92
MARGIN_PX = 6


def canonicalize(img, bbox):
    x, y, w, h = bbox
    H, W = img.shape
    x1 = int(max(0, x - MARGIN_PX)); y1 = int(max(0, y - MARGIN_PX))
    x2 = int(min(W, x + w + MARGIN_PX)); y2 = int(min(H, y + h + MARGIN_PX))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop = img[y1:y2, x1:x2]
    thresh = np.percentile(crop, FG_PERCENTILE)
    rows, cols = np.where(crop > thresh)
    if len(rows) < 10:
        return None
    coords = np.column_stack([rows, cols]).astype(float)
    pca = PCA(n_components=2)
    pca.fit(coords)
    pc1 = pca.components_[0]
    theta_deg = float(np.degrees(np.arctan2(pc1[0], pc1[1])))
    ch, cw = crop.shape
    M = cv2.getRotationMatrix2D((cw / 2.0, ch / 2.0), theta_deg, 1.0)
    rotated = cv2.warpAffine(crop, M, (cw, ch), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    resized = cv2.resize(rotated, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)
    std = resized.std()
    if std < 1e-6:
        return None
    return ((resized - resized.mean()) / std).astype(np.float32)


def ncc(a, b):
    return float((a * b).mean())


def max_template_sim(patch, templates, flipped):
    if patch is None:
        return None
    best = -1.0
    for t, tf in zip(templates, flipped):
        s = max(ncc(patch, t), ncc(patch, tf))
        if s > best:
            best = s
    return best


def build_templates():
    print("=== Building templates ===")
    with open(UNLEARN_JSON) as f:
        coco = json.load(f)
    id_to_fname = {im["id"]: im["file_name"] for im in coco["images"]}
    templates = []; img_cache = {}
    for ann in coco["annotations"]:
        fname = id_to_fname[ann["image_id"]]
        if fname not in img_cache:
            img_cache[fname] = load_img(os.path.join(UNLEARN_DIR, fname))
        img = img_cache[fname]
        if img is None: continue
        patch = canonicalize(img, ann["bbox"])
        if patch is None: continue
        templates.append(patch)
    print(f"  Built {len(templates)} templates")
    return templates


def calibrate(templates):
    print("\n=== Template self-calibration ===")
    n = len(templates)
    flipped = [np.rot90(t, 2).copy() for t in templates]
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = max(ncc(templates[i], templates[j]),
                            ncc(templates[i], flipped[j]))
    off = sim[~np.eye(n, dtype=bool)]
    print(f"  Off-diag: min={off.min():.3f} med={np.median(off):.3f} p75={np.percentile(off,75):.3f} max={off.max():.3f}")
    return flipped


def score_base(templates, flipped):
    df = pd.read_csv(BEST_CSV)
    df["dets"] = df["prediction_string"].apply(parse_dets)
    n_total = sum(len(d) for d in df["dets"])
    print(f"\n=== Scoring {n_total} dets against 20 templates ===")
    img_cache, recs, seen = {}, [], 0
    for _, row in df.iterrows():
        if not row["dets"]: continue
        img_id = row["image_id"]
        if img_id not in img_cache:
            img_cache[img_id] = load_img(os.path.join(TEST_DIR, f"{img_id}.png"))
        img = img_cache[img_id]
        if img is None: continue
        for det in row["dets"]:
            c = det[0]
            patch = canonicalize(img, det[1:])
            s = max_template_sim(patch, templates, flipped)
            recs.append({"image_id": img_id, "det": det, "conf": c, "max_sim": s,
                         "bucket": "uncond" if c >= 0.6 else "rescued"})
            seen += 1
            if seen % 200 == 0:
                print(f"    {seen}/{n_total}")
    return recs, df


def summarize(name, vals):
    arr = np.array([v for v in vals if v is not None])
    if len(arr) == 0:
        print(f"  {name}: none"); return
    p = np.percentile(arr, [10, 25, 50, 75, 90, 95])
    print(f"  {name:25s} n={len(arr):4d}  min={arr.min():.3f}  p25={p[1]:.3f}  med={p[2]:.3f}  p75={p[3]:.3f}  p90={p[4]:.3f}  p95={p[5]:.3f}  max={arr.max():.3f}")


def main():
    templates = build_templates()
    if not templates: return
    flipped = calibrate(templates)
    recs, df_in = score_base(templates, flipped)

    print("\n=== Max template-sim distributions on base ===")
    summarize("All dets",          [r["max_sim"] for r in recs])
    summarize("Uncond (conf>=0.6)", [r["max_sim"] for r in recs if r["bucket"] == "uncond"])
    summarize("Rescued (conf<0.6)", [r["max_sim"] for r in recs if r["bucket"] == "rescued"])

    out_rows = []
    for r in recs:
        c, x, y, w, h = r["det"]
        out_rows.append({"image_id": r["image_id"], "conf": c, "x": x, "y": y, "w": w, "h": h,
                         "bucket": r["bucket"], "max_sim": r["max_sim"]})
    pd.DataFrame(out_rows).to_csv(OUT_DIR / "scored_dets.csv", index=False)

    img_lookup = defaultdict(list)
    for r in recs:
        img_lookup[r["image_id"]].append((r["det"], r["max_sim"]))

    print("\n=== Filter variants (drop max_sim >= T) ===")
    print(f"  {'T':6s}  total  per/img  dropped")
    for T in [0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        out_strs = []; total = 0; dropped = 0
        for _, row in df_in.iterrows():
            kept = []
            for det in row["dets"]:
                sim = next((s for d, s in img_lookup[row["image_id"]] if d == det), None)
                if sim is None or sim < T:
                    kept.append(det)
                else:
                    dropped += 1
            out_strs.append(dets_to_str(kept))
            total += len(kept)
        out_df = df_in[["id", "image_id"]].copy()
        out_df["prediction_string"] = out_strs
        out_df.to_csv(OUT_DIR / f"filter_T{T:.2f}.csv", index=False)
        print(f"  T={T:.2f}  {total:4d}    {total/len(df_in):.3f}    {dropped}")

    print(f"\nSaved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
