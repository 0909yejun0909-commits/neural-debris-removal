
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from step6_morpho_filter import dets_to_str, load_img, parse_dets

UNLEARN_DIR  = "/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models/unlearn_set"
UNLEARN_JSON = "/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models/unlearn_set/annotations_coco.json"
TEST_DIR     = "/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models/test_set/test_set"
# FOR RUN B:
BEST_CSV     = "/kaggle/input/debris-best-csvs/filter_length_uncond_stack_le40_or_45_51.csv"

OUT_DIR      = Path("/kaggle/working/step14_template")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANVAS_H, CANVAS_W = 24, 96
FG_PERCENTILE      = 92
MARGIN_PX          = 6

def canonicalize(img, bbox):
    x, y, w, h = bbox
    H, W = img.shape
    x1 = int(max(0, x - MARGIN_PX))
    y1 = int(max(0, y - MARGIN_PX))
    x2 = int(min(W, x + w + MARGIN_PX))
    y2 = int(min(H, y + h + MARGIN_PX))
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
    rotated = cv2.warpAffine(
        crop, M, (cw, ch),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    resized = cv2.resize(rotated, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)
    std = resized.std()
    if std < 1e-6:
        return None
    return ((resized - resized.mean()) / std).astype(np.float32)

def ncc(a, b):
    return float((a * b).mean())

def max_template_sim(patch, templates, templates_flipped):
    if patch is None:
        return None
    best = -1.0
    for t, tf in zip(templates, templates_flipped):
        s = max(ncc(patch, t), ncc(patch, tf))
        if s > best:
            best = s
    return best

def build_templates():
    print("=== Building templates from unlearn-set annotations ===")
    with open(UNLEARN_JSON) as f:
        coco = json.load(f)
    id_to_fname = {im["id"]: im["file_name"] for im in coco["images"]}
    templates, meta = [], []
    img_cache = {}
    for ann in coco["annotations"]:
        fname = id_to_fname[ann["image_id"]]
        if fname not in img_cache:
            img_cache[fname] = load_img(os.path.join(UNLEARN_DIR, fname))
        img = img_cache[fname]
        if img is None:
            continue
        patch = canonicalize(img, ann["bbox"])
        if patch is None:
            print(f"  WARN: ann_id={ann['id']} could not be canonicalized")
            continue
        templates.append(patch)
        meta.append({"ann_id": ann["id"], "image_id": ann["image_id"]})
    print(f"  Built {len(templates)} / {len(coco['annotations'])} templates")
    return templates, meta

def calibrate_templates(templates, meta):
    print("\n=== Template calibration ===")
    n = len(templates)
    if n == 0:
        print("  No templates â€” abort")
        return
    flipped = [np.rot90(t, 2).copy() for t in templates]
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = max(ncc(templates[i], templates[j]),
                             ncc(templates[i], flipped[j]))
    diag = np.diag(sim)
    print(f"  Self-similarity (diag): min={diag.min():.4f} max={diag.max():.4f}")
    off = sim[~np.eye(n, dtype=bool)]
    print(f"  Pairwise off-diag:  n={len(off)}  "
          f"min={off.min():.3f}  p25={np.percentile(off,25):.3f}  "
          f"med={np.median(off):.3f}  p75={np.percentile(off,75):.3f}  "
          f"max={off.max():.3f}")

    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 1.2 * rows))
    axes = np.array(axes).reshape(rows, cols)
    for ax, t, m in zip(axes.flat, templates, meta):
        ax.imshow(t, cmap="gray", aspect="auto")
        ax.set_title(f"ann={m['ann_id']}", fontsize=8)
        ax.axis("off")
    for ax in axes.flat[len(templates):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "template_grid.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Template grid -> {OUT_DIR / 'template_grid.png'}")

def score_csv(csv_path, templates):
    print(f"\n=== Scoring dets in {csv_path} ===")
    flipped = [np.rot90(t, 2).copy() for t in templates]
    df = pd.read_csv(csv_path)
    df["dets"] = df["prediction_string"].apply(parse_dets)
    n_total = sum(len(d) for d in df["dets"])
    print(f"  {n_total} dets across {(df['dets'].str.len() > 0).sum()} non-empty rows")
    img_cache, out, seen = {}, [], 0
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
            patch = canonicalize(img, det[1:])
            s = max_template_sim(patch, templates, flipped)
            out.append({
                "image_id": img_id, "det": det, "conf": c,
                "max_sim": s,
                "bucket": "unconditional" if c >= 0.6 else "rescued",
            })
            seen += 1
            if seen % 500 == 0:
                print(f"    ...{seen}/{n_total}")
    return out, df

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
    templates, meta = build_templates()
    calibrate_templates(templates, meta)
    if not templates:
        return
    recs, df_template = score_csv(BEST_CSV, templates)
    print("\n=== Max template similarity distributions ===")
    summarize("All dets",                  [r["max_sim"] for r in recs])
    summarize("Unconditional (conf>=0.6)", [r["max_sim"] for r in recs if r["bucket"] == "unconditional"])
    summarize("Rescued (conf<0.6)",        [r["max_sim"] for r in recs if r["bucket"] == "rescued"])
    out_rows = []
    for r in recs:
        c, x, y, w, h = r["det"]
        out_rows.append({
            "image_id": r["image_id"], "conf": c, "x": x, "y": y, "w": w, "h": h,
            "bucket": r["bucket"], "max_sim": r["max_sim"],
        })
    pd.DataFrame(out_rows).to_csv(OUT_DIR / "scored_dets.csv", index=False)
    img_lookup = defaultdict(list)
    for r in recs:
        img_lookup[r["image_id"]].append((r["det"], r["max_sim"]))
    print("\n=== Filter variants â€” drop dets with max_sim >= T ===")
    print(f"  {'variant':18s}  non_empty  total  avg/img  dropped")
    for T in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        out_strs = []
        total = 0; non_empty = 0; dropped = 0
        for _, row in df_template.iterrows():
            kept = []
            for det in row["dets"]:
                sim = next((s for d, s in img_lookup[row["image_id"]] if d == det), None)
                if sim is None or sim < T:
                    kept.append(det)
                else:
                    dropped += 1
            out_strs.append(dets_to_str(kept))
            if kept:
                total += len(kept); non_empty += 1
        out_df = df_template[["id", "image_id"]].copy()
        out_df["prediction_string"] = out_strs
        name = f"filter_T{T:.2f}.csv"
        out_df.to_csv(OUT_DIR / name, index=False)
        print(f"  {name:18s}  {non_empty:4d}       {total:4d}    "
              f"{total/len(df_template):.3f}    {dropped}")

if __name__ == "__main__":
    main()
