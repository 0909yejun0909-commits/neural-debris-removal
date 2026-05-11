"""
Step 1a (continued) — Inspect what's actually at each poisoned bbox location.

The 20 unlearn images came with bbox annotations in annotations_coco.json.
The Overview says these are "examples of poisoned images from the training set"
and the baseline trains with empty labels — so these bboxes mark where the
poisoned model fires false positive detections.

Question we want answered: at each bbox, is there an actual faint streak in the
image, or is it pure background? That tells us whether the poison is amplifying
real features or fabricating from noise.

Outputs (in ./diagnosis/):
  bbox_crops_zoom.png   — each unlearn image cropped to its bbox (high contrast)
  bbox_context.png      — each bbox with surrounding context (~100px padding)
  bbox_vs_random.png    — side-by-side: bbox crops vs same-size random crops from same image
  bbox_stats.csv        — per-bbox intensity stats vs whole-image stats
"""

import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


ROOT       = Path(__file__).parent
UNLEARN    = ROOT / "neural-debris-removal-in-streak-detection-models" / "unlearn_set"
ANN_PATH   = UNLEARN / "annotations_coco.json"
OUT        = ROOT / "diagnosis"
OUT.mkdir(exist_ok=True)


def load_uint16(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im.dtype == np.uint16:
        im = im.astype(np.float32) / 65535.0
    else:
        im = im.astype(np.float32) / 255.0
    if im.ndim == 3:
        im = im.mean(axis=2)
    return im


def load_annotations():
    with ANN_PATH.open() as f:
        coco = json.load(f)
    images = {im["id"]: im["file_name"] for im in coco["images"]}
    return [(images[a["image_id"]], a["image_id"], a["bbox"]) for a in coco["annotations"]]


def crop_bbox(im, bbox, pad=0):
    x, y, w, h = bbox
    x0 = max(0, int(round(x)) - pad)
    y0 = max(0, int(round(y)) - pad)
    x1 = min(im.shape[1], int(round(x + w)) + pad)
    y1 = min(im.shape[0], int(round(y + h)) + pad)
    return im[y0:y1, x0:x1], (x0, y0, x1, y1)


def random_crop_same_size(im, w, h, exclude=None, rng=None):
    rng = rng or np.random.default_rng(0)
    H, W = im.shape
    for _ in range(50):
        x = rng.integers(0, max(1, W - int(w)))
        y = rng.integers(0, max(1, H - int(h)))
        if exclude is None:
            break
        ex0, ey0, ex1, ey1 = exclude
        if x + w < ex0 or x > ex1 or y + h < ey0 or y > ey1:
            break
    return im[y:y + int(h), x:x + int(w)], (int(x), int(y))


def show_with_stretch(ax, patch, title=""):
    if patch.size == 0:
        ax.axis("off")
        return
    p1, p99 = np.percentile(patch, [1, 99])
    if p99 == p1:
        p99 = p1 + 1e-6
    ax.imshow(patch, cmap="gray", vmin=p1, vmax=p99)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main():
    rng = np.random.default_rng(42)
    anns = load_annotations()
    print(f"Found {len(anns)} bbox annotations across the unlearn set")

    # Pre-load all images + their bboxes
    samples = []
    for fname, image_id, bbox in anns:
        im = load_uint16(UNLEARN / fname)
        samples.append((fname, image_id, bbox, im))

    # ── 1. Tight bbox crops (no padding) ──────────────────────────────────────
    print("Building tight bbox crops...")
    cols = 5
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.atleast_2d(axes).flatten()
    for ax, (fname, image_id, bbox, im) in zip(axes, samples):
        patch, _ = crop_bbox(im, bbox, pad=0)
        x, y, w, h = [round(v, 1) for v in bbox]
        show_with_stretch(ax, patch, title=f"{fname}\nbbox=({x},{y},{w},{h})")
    for ax in axes[len(samples):]:
        ax.axis("off")
    fig.suptitle("Tight crop at each poisoned bbox (per-patch contrast stretch)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "bbox_crops_zoom.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'bbox_crops_zoom.png'}")

    # ── 2. Context view: bbox + 80px padding, with bbox outline drawn ─────────
    print("Building context view (bbox + 80px padding)...")
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6))
    axes = np.atleast_2d(axes).flatten()
    for ax, (fname, image_id, bbox, im) in zip(axes, samples):
        patch, (x0, y0, x1, y1) = crop_bbox(im, bbox, pad=80)
        p1, p99 = np.percentile(patch, [1, 99])
        ax.imshow(patch, cmap="gray", vmin=p1, vmax=p99)
        # bbox coords relative to the crop
        bx0 = bbox[0] - x0
        by0 = bbox[1] - y0
        rect = mpatches.Rectangle(
            (bx0, by0), bbox[2], bbox[3],
            fill=False, edgecolor="red", linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.set_title(fname, fontsize=8)
        ax.axis("off")
    for ax in axes[len(samples):]:
        ax.axis("off")
    fig.suptitle("Poisoned bbox in context (80px padding; red = bbox)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "bbox_context.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'bbox_context.png'}")

    # ── 3. Side-by-side: bbox crop vs same-size random crop from same image ──
    print("Building bbox vs random-crop comparison...")
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 2 * 1.6, rows * 1.8))
    axes = np.atleast_2d(axes).reshape(rows, cols * 2)
    for i, (fname, image_id, bbox, im) in enumerate(samples):
        r, c = i // cols, (i % cols) * 2
        bbox_crop, exclude = crop_bbox(im, bbox, pad=0)
        rnd_crop, _ = random_crop_same_size(im, bbox[2], bbox[3], exclude=exclude, rng=rng)
        show_with_stretch(axes[r, c],     bbox_crop, title=f"{image_id} bbox")
        show_with_stretch(axes[r, c + 1], rnd_crop,  title=f"{image_id} random")
    for i in range(len(samples), rows * cols):
        r, c = i // cols, (i % cols) * 2
        axes[r, c].axis("off")
        axes[r, c + 1].axis("off")
    fig.suptitle("Left of each pair: poisoned bbox  |  Right: random same-size crop from same image", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "bbox_vs_random.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'bbox_vs_random.png'}")

    # ── 4. Quantitative comparison: bbox mean/max vs image mean/max ──────────
    print("Computing bbox-vs-image stats...")
    with (OUT / "bbox_stats.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "file", "image_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
            "bbox_mean", "bbox_max", "bbox_p99",
            "image_mean", "image_max", "image_p99",
            "mean_ratio", "max_ratio",
        ])
        bbox_means, image_means = [], []
        bbox_maxs,  image_maxs  = [], []
        for fname, image_id, bbox, im in samples:
            patch, _ = crop_bbox(im, bbox, pad=0)
            if patch.size == 0:
                continue
            bm  = float(patch.mean())
            bx  = float(patch.max())
            bp9 = float(np.percentile(patch, 99))
            im_m  = float(im.mean())
            im_x  = float(im.max())
            im_p9 = float(np.percentile(im, 99))
            w.writerow([
                fname, image_id, *bbox,
                bm, bx, bp9, im_m, im_x, im_p9,
                bm / im_m if im_m > 0 else 0,
                bx / im_x if im_x > 0 else 0,
            ])
            bbox_means.append(bm)
            image_means.append(im_m)
            bbox_maxs.append(bx)
            image_maxs.append(im_x)
    print(f"  wrote {OUT / 'bbox_stats.csv'}")

    bbox_means  = np.array(bbox_means)
    image_means = np.array(image_means)
    bbox_maxs   = np.array(bbox_maxs)
    image_maxs  = np.array(image_maxs)

    print("\n-- Bbox vs whole-image intensity --")
    print(f"  bbox  mean: {bbox_means.mean():.5f} +- {bbox_means.std():.5f}")
    print(f"  image mean: {image_means.mean():.5f} +- {image_means.std():.5f}")
    print(f"  ratio (bbox.mean / image.mean): {(bbox_means / image_means).mean():.3f}")
    print(f"  bbox  max:  {bbox_maxs.mean():.5f} +- {bbox_maxs.std():.5f}")
    print(f"  image max:  {image_maxs.mean():.5f} +- {image_maxs.std():.5f}")
    print(f"  ratio (bbox.max / image.max):   {(bbox_maxs / image_maxs).mean():.3f}")

    print()
    print("Interpretation:")
    print("  bbox.mean / image.mean ~ 1.0  ->  bbox patches are background-like")
    print("                         > 1.0  ->  bbox patches contain bright features (real streaks)")
    print("  bbox.max  / image.max  ~ 1.0  ->  bbox contains image-brightest pixel (streak head)")
    print("                         < 0.5  ->  bbox is dimmer than image peak (probably background)")
    print()
    print(f"All artifacts written to: {OUT}")


if __name__ == "__main__":
    main()
