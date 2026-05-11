"""
Step 1b/c — Kaggle diagnostic script (needs GPU).

Goal: collect what the poisoned model actually predicts, on both the 20 unlearn
images and the full 2000-image test set, so we can analyse the data locally.

Outputs (in /kaggle/working/, all downloadable from the Output tab):
  preds_unlearn.csv  — every detection on the 20 unlearn images
  preds_test.csv     — every detection on the 2000 test images
  unlearn_overlay.png — unlearn-set images with predicted boxes (red)
                       and the COCO "poisoned" boxes (yellow) drawn on top
  test_overlay.png   — 20 random test images with predicted boxes
  stats_summary.txt  — headline numbers (counts, confidence histogram, etc.)

After running on Kaggle, download these to the local repo's ./diagnosis_kaggle/
folder so we can dig into the data without burning GPU time.
"""

import subprocess
subprocess.run(["pip", "install", "-q", "setuptools<81"], check=True)
subprocess.run(["pip", "install", "-q", "git+https://github.com/facebookresearch/detectron2.git"], check=True)

import copy
import csv
import json
import random
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from tqdm import tqdm


# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR         = "/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models"
POISONED_WEIGHTS = f"{BASE_DIR}/poisoned_model/poisoned_model.pth"
UNLEARN_DIR      = f"{BASE_DIR}/unlearn_set"
TEST_DIR         = f"{BASE_DIR}/test_set/test_set"
OUT              = Path("/kaggle/working")
OUT.mkdir(exist_ok=True)


# ── Model config (must match poisoned model exactly) ──────────────────────────
BASE_CONFIG          = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ANCHOR_SIZES         = [[16], [32], [64], [128], [256]]
NUM_CLASSES          = 1
CONF_THRESH          = 0.05   # collect liberally; we filter later for analysis
IMG_W = IMG_H        = 1024
SEED                 = 42


# ── Image loading ──────────────────────────────────────────────────────────────
def load_image(path):
    """Read 16-bit grayscale PNG, return float32 HxWx3 in [0, 255]."""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im.dtype == np.uint16:
        im = im.astype(np.float32) / 65535.0
    im = np.clip(im * 255, 0, 255).astype(np.float32)
    if im.ndim == 2:
        im = np.repeat(im[:, :, None], 3, axis=2)
    return im


# ── Predictor ──────────────────────────────────────────────────────────────────
def build_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.MODEL.WEIGHTS                        = POISONED_WEIGHTS
    cfg.MODEL.RETINANET.NUM_CLASSES          = NUM_CLASSES
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [ANCHOR_ASPECT_RATIOS]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES         = ANCHOR_SIZES
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST    = CONF_THRESH
    return DefaultPredictor(cfg)


def run_inference(predictor, paths, desc):
    """Run inference and return a list of (image_id, conf, x, y, w, h) tuples."""
    rows = []
    for p in tqdm(paths, desc=desc):
        im = load_image(p)
        out = predictor(im)["instances"].to("cpu")
        boxes  = out.pred_boxes.tensor.numpy()
        scores = out.scores.numpy()
        for (x1, y1, x2, y2), s in zip(boxes, scores):
            x1 = float(np.clip(x1, 0, IMG_W))
            y1 = float(np.clip(y1, 0, IMG_H))
            x2 = float(np.clip(x2, 0, IMG_W))
            y2 = float(np.clip(y2, 0, IMG_H))
            w  = max(0.0, x2 - x1)
            h  = max(0.0, y2 - y1)
            if w == 0 or h == 0:
                continue
            rows.append((p.stem, float(s), x1, y1, w, h))
    return rows


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "conf", "x", "y", "w", "h"])
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} detections)")


# ── Visualisation helpers ──────────────────────────────────────────────────────
def show_image(ax, im_path, preds, gt_boxes=None, title=""):
    """Show one image with predicted boxes (red) and optional GT (yellow)."""
    im = load_image(im_path)[..., 0]  # back to single channel for display
    p1, p99 = np.percentile(im, [1, 99])
    ax.imshow(im, cmap="gray", vmin=p1, vmax=p99)
    for _, conf, x, y, w, h in preds:
        ax.add_patch(mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor="red", linewidth=0.8,
        ))
        ax.text(x, y - 4, f"{conf:.2f}", color="red", fontsize=6)
    if gt_boxes:
        for bx, by, bw, bh in gt_boxes:
            ax.add_patch(mpatches.Rectangle(
                (bx, by), bw, bh, fill=False, edgecolor="yellow",
                linewidth=0.8, linestyle="--",
            ))
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def build_grid(image_paths, preds_by_id, gt_by_id, out_path, title=""):
    cols = 5
    rows = (len(image_paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
    axes = np.atleast_2d(axes).flatten()
    for ax, p in zip(axes, image_paths):
        preds = preds_by_id.get(p.stem, [])
        gts   = gt_by_id.get(p.stem, [])
        n     = len(preds)
        show_image(ax, p, preds, gt_boxes=gts, title=f"{p.name}  (n={n})")
    for ax in axes[len(image_paths):]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ── COCO annotations for unlearn set (the "poisoned" GT boxes) ────────────────
def load_unlearn_gt():
    with open(Path(UNLEARN_DIR) / "annotations_coco.json") as f:
        coco = json.load(f)
    images = {im["id"]: im["file_name"] for im in coco["images"]}
    gt = {}
    for a in coco["annotations"]:
        fname = images[a["image_id"]]
        gt.setdefault(Path(fname).stem, []).append(a["bbox"])
    return gt


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Building predictor...")
    predictor = build_predictor()

    unlearn_paths = sorted(Path(UNLEARN_DIR).glob("*.png"))
    test_paths    = sorted(Path(TEST_DIR).glob("*.png"))
    print(f"Found {len(unlearn_paths)} unlearn + {len(test_paths)} test images")

    # ── Inference: unlearn set ─────────────────────────────────────────────────
    unlearn_rows = run_inference(predictor, unlearn_paths, "Unlearn inference")
    save_csv(unlearn_rows, OUT / "preds_unlearn.csv")

    # ── Inference: full test set ───────────────────────────────────────────────
    test_rows = run_inference(predictor, test_paths, "Test inference")
    save_csv(test_rows, OUT / "preds_test.csv")

    # ── Stats summary ──────────────────────────────────────────────────────────
    df_u = pd.DataFrame(unlearn_rows, columns=["image_id", "conf", "x", "y", "w", "h"])
    df_t = pd.DataFrame(test_rows,    columns=["image_id", "conf", "x", "y", "w", "h"])
    df_u["aspect"] = df_u["w"] / df_u["h"].clip(lower=1e-6)
    df_t["aspect"] = df_t["w"] / df_t["h"].clip(lower=1e-6)
    df_u["area"]   = df_u["w"] * df_u["h"]
    df_t["area"]   = df_t["w"] * df_t["h"]

    n_t_with_pred  = df_t["image_id"].nunique()
    n_t_no_pred    = len(test_paths) - n_t_with_pred
    n_u_with_pred  = df_u["image_id"].nunique()
    n_u_no_pred    = len(unlearn_paths) - n_u_with_pred

    summary_lines = [
        f"=== Step 1b/c diagnostic summary ===",
        f"Conf threshold during inference: {CONF_THRESH}",
        f"",
        f"-- Unlearn set ({len(unlearn_paths)} images) --",
        f"Total detections: {len(df_u)}",
        f"Images with >=1 detection: {n_u_with_pred}",
        f"Images with NO detection:  {n_u_no_pred}",
        f"Conf:    mean {df_u['conf'].mean():.3f}  median {df_u['conf'].median():.3f}  max {df_u['conf'].max():.3f}",
        f"Conf > 0.2: {(df_u['conf'] > 0.2).sum()} detections",
        f"Area (px^2):   median {df_u['area'].median():.0f}   p99 {df_u['area'].quantile(.99):.0f}",
        f"Aspect (w/h):  median {df_u['aspect'].median():.2f}  p10 {df_u['aspect'].quantile(.10):.2f}  p90 {df_u['aspect'].quantile(.90):.2f}",
        f"",
        f"-- Test set ({len(test_paths)} images) --",
        f"Total detections: {len(df_t)}",
        f"Images with >=1 detection: {n_t_with_pred}",
        f"Images with NO detection:  {n_t_no_pred}",
        f"Conf:    mean {df_t['conf'].mean():.3f}  median {df_t['conf'].median():.3f}  max {df_t['conf'].max():.3f}",
        f"Conf > 0.2: {(df_t['conf'] > 0.2).sum()} detections",
        f"Area (px^2):   median {df_t['area'].median():.0f}   p99 {df_t['area'].quantile(.99):.0f}",
        f"Aspect (w/h):  median {df_t['aspect'].median():.2f}  p10 {df_t['aspect'].quantile(.10):.2f}  p90 {df_t['aspect'].quantile(.90):.2f}",
        f"",
        f"-- Detections-per-image distribution (conf>=0.2) --",
        f"  Unlearn: " + str(
            df_u[df_u["conf"] >= 0.2].groupby("image_id").size().describe().to_dict()
        ),
        f"  Test:    " + str(
            df_t[df_t["conf"] >= 0.2].groupby("image_id").size().describe().to_dict()
        ),
    ]
    summary = "\n".join(summary_lines)
    print("\n" + summary)
    (OUT / "stats_summary.txt").write_text(summary)
    print(f"  wrote {OUT / 'stats_summary.txt'}")

    # ── Visualisations ─────────────────────────────────────────────────────────
    gt_by_id = load_unlearn_gt()
    preds_by_id_u = {}
    for r in unlearn_rows:
        preds_by_id_u.setdefault(r[0], []).append(r)
    preds_by_id_t = {}
    for r in test_rows:
        preds_by_id_t.setdefault(r[0], []).append(r)

    # Apply 0.2 threshold for visualisation (clean model uses conf > 0.2).
    def filter_02(d):
        return {k: [r for r in v if r[1] >= 0.2] for k, v in d.items()}
    preds_by_id_u_02 = filter_02(preds_by_id_u)
    preds_by_id_t_02 = filter_02(preds_by_id_t)

    print("\nBuilding unlearn overlay grid...")
    build_grid(
        unlearn_paths, preds_by_id_u_02, gt_by_id,
        OUT / "unlearn_overlay.png",
        title="Unlearn set: poisoned-model predictions (red, conf>=0.2) vs COCO annotated (yellow dashed)",
    )

    print("Building test overlay grid (20 random images)...")
    test_sample = random.sample(test_paths, 20)
    build_grid(
        test_sample, preds_by_id_t_02, {},
        OUT / "test_overlay.png",
        title="Random test images: poisoned-model predictions (red, conf>=0.2)",
    )

    # ── Save a small zip of everything for one-click download ─────────────────
    print("\nZipping outputs for easy download...")
    import zipfile
    with zipfile.ZipFile(OUT / "step1_kaggle_outputs.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in [
            "preds_unlearn.csv", "preds_test.csv",
            "stats_summary.txt",
            "unlearn_overlay.png", "test_overlay.png",
        ]:
            z.write(OUT / name, arcname=name)
    print(f"  wrote {OUT / 'step1_kaggle_outputs.zip'}")
    print("\nDone. Download step1_kaggle_outputs.zip from the Kaggle Output tab.")


if __name__ == "__main__":
    main()
