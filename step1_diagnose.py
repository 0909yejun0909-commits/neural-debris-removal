"""
Step 1a — Local diagnosis (no GPU, no model needed).

Goal: look at the 20 unlearn images and compare them statistically and visually
to a random sample of test images. We're hunting for any systematic difference
that could be a poison trigger (bright watermark, intensity bias, hot pixel,
specific spatial pattern, etc.).

Outputs (in ./diagnosis/):
  unlearn_grid.png       — all 20 unlearn images in a grid
  test_sample_grid.png   — 20 random test images in a grid
  mean_unlearn.png       — pixel-wise mean of the unlearn set
  mean_test.png          — pixel-wise mean of the test sample
  diff_mean.png          — mean(unlearn) − mean(test); spatial trigger signal
  diff_mean_abs.png      — |diff_mean|; trigger magnitude regardless of sign
  histograms.png         — intensity histograms, unlearn vs test
  stats.csv              — per-image stats (mean, std, p1, p99) for both sets
"""

import csv
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
UNLEARN    = ROOT / "neural-debris-removal-in-streak-detection-models" / "unlearn_set"
TEST       = ROOT / "neural-debris-removal-in-streak-detection-models" / "test_set" / "test_set"
OUT        = ROOT / "diagnosis"
OUT.mkdir(exist_ok=True)

N_TEST_SAMPLE = 50          # random test images to compare against
SEED          = 42


# ── Image loading ──────────────────────────────────────────────────────────────
def load_uint16(path):
    """Load 1024x1024 uint16 grayscale PNG. Returns float32 array in [0, 1]."""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise RuntimeError(f"Could not load {path}")
    if im.dtype == np.uint16:
        im = im.astype(np.float32) / 65535.0
    else:
        im = im.astype(np.float32) / 255.0
    if im.ndim == 3:
        im = im.mean(axis=2)
    return im


def per_image_stats(im):
    return {
        "mean": float(im.mean()),
        "std":  float(im.std()),
        "min":  float(im.min()),
        "max":  float(im.max()),
        "p1":   float(np.percentile(im, 1)),
        "p99":  float(np.percentile(im, 99)),
    }


# ── Build grids ────────────────────────────────────────────────────────────────
def save_grid(images, names, path, cols=5, title=""):
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.atleast_2d(axes).flatten()
    # Display each image with its own contrast stretch (p1..p99) to make
    # faint streaks visible — without this, 16-bit dynamic range hides them.
    for ax, im, name in zip(axes, images, names):
        p1, p99 = np.percentile(im, [1, 99])
        ax.imshow(im, cmap="gray", vmin=p1, vmax=p99)
        ax.set_title(name, fontsize=8)
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def save_single(im, path, title=""):
    fig, ax = plt.subplots(figsize=(6, 6))
    p1, p99 = np.percentile(im, [1, 99])
    cmap = "gray" if im.min() >= 0 else "RdBu"
    if im.min() < 0:
        v = max(abs(im.min()), abs(im.max()))
        ax.imshow(im, cmap=cmap, vmin=-v, vmax=v)
    else:
        ax.imshow(im, cmap=cmap, vmin=p1, vmax=p99)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"Unlearn dir: {UNLEARN}")
    print(f"Test dir:    {TEST}")

    unlearn_paths = sorted(UNLEARN.glob("*.png"))
    test_paths    = sorted(TEST.glob("*.png"))
    print(f"Found {len(unlearn_paths)} unlearn images, {len(test_paths)} test images")

    test_sample_paths = random.sample(test_paths, N_TEST_SAMPLE)

    print("\nLoading unlearn images...")
    unlearn_imgs = [load_uint16(p) for p in unlearn_paths]
    print("Loading test sample...")
    test_imgs    = [load_uint16(p) for p in test_sample_paths]

    # ── Per-image stats ────────────────────────────────────────────────────────
    print("\nWriting per-image stats...")
    with (OUT / "stats.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "file", "mean", "std", "min", "max", "p1", "p99"])
        for p, im in zip(unlearn_paths, unlearn_imgs):
            s = per_image_stats(im)
            w.writerow(["unlearn", p.name, s["mean"], s["std"], s["min"], s["max"], s["p1"], s["p99"]])
        for p, im in zip(test_sample_paths, test_imgs):
            s = per_image_stats(im)
            w.writerow(["test", p.name, s["mean"], s["std"], s["min"], s["max"], s["p1"], s["p99"]])
    print(f"  wrote {OUT / 'stats.csv'}")

    # ── Compare aggregate stats ────────────────────────────────────────────────
    def agg(imgs):
        means = np.array([im.mean() for im in imgs])
        stds  = np.array([im.std()  for im in imgs])
        p99s  = np.array([np.percentile(im, 99) for im in imgs])
        return means, stds, p99s

    u_mean, u_std, u_p99 = agg(unlearn_imgs)
    t_mean, t_std, t_p99 = agg(test_imgs)

    print("\n-- Aggregate intensity stats (float [0,1] scale) --")
    print(f"{'metric':<10} {'unlearn (n=20)':<25} {'test (n=' + str(N_TEST_SAMPLE) + ')':<25}")
    for name, u, t in [("mean", u_mean, t_mean), ("std", u_std, t_std), ("p99", u_p99, t_p99)]:
        print(f"{name:<10} {u.mean():.5f} ± {u.std():.5f}     {t.mean():.5f} ± {t.std():.5f}")

    # ── Pixel-wise mean images and difference ──────────────────────────────────
    print("\nComputing pixel-wise mean images...")
    mean_unlearn = np.mean(unlearn_imgs, axis=0)
    mean_test    = np.mean(test_imgs,    axis=0)
    diff         = mean_unlearn - mean_test

    save_single(mean_unlearn, OUT / "mean_unlearn.png",   title=f"mean of unlearn set (n={len(unlearn_imgs)})")
    save_single(mean_test,    OUT / "mean_test.png",      title=f"mean of test sample (n={len(test_imgs)})")
    save_single(diff,         OUT / "diff_mean.png",      title="mean(unlearn) − mean(test)")
    save_single(np.abs(diff), OUT / "diff_mean_abs.png",  title="|mean(unlearn) − mean(test)|")

    # ── Image grids ────────────────────────────────────────────────────────────
    print("\nBuilding image grids...")
    save_grid(
        unlearn_imgs,
        [p.name for p in unlearn_paths],
        OUT / "unlearn_grid.png",
        cols=5,
        title="Unlearn set (20 images)",
    )
    grid_sample = test_imgs[:20]
    grid_names  = [p.name for p in test_sample_paths[:20]]
    save_grid(
        grid_sample,
        grid_names,
        OUT / "test_sample_grid.png",
        cols=5,
        title="Random test images (20)",
    )

    # ── Intensity histograms ───────────────────────────────────────────────────
    print("\nBuilding histograms...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    def stack_hist(ax, imgs, label):
        # Sample 100k random pixels per image to keep the hist computable.
        pixels = np.concatenate([
            im.ravel()[np.random.choice(im.size, 100_000, replace=False)]
            for im in imgs
        ])
        ax.hist(pixels, bins=200, alpha=0.6, label=label, density=True)

    stack_hist(axes[0], unlearn_imgs, "unlearn")
    stack_hist(axes[0], test_imgs,    "test sample")
    axes[0].set_title("Full-range pixel intensity histogram")
    axes[0].set_xlabel("intensity (float [0,1])")
    axes[0].set_ylabel("density")
    axes[0].legend()

    # Zoom into the bright tail — that's where streaks live.
    stack_hist(axes[1], unlearn_imgs, "unlearn")
    stack_hist(axes[1], test_imgs,    "test sample")
    axes[1].set_xlim(0.01, 0.2)
    axes[1].set_yscale("log")
    axes[1].set_title("Bright-tail histogram (log y)")
    axes[1].set_xlabel("intensity")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUT / "histograms.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'histograms.png'}")

    # ── Headline summary ───────────────────────────────────────────────────────
    print("\n-- Summary --")
    print(f"  unlearn mean intensity:  {u_mean.mean():.5f}  (test: {t_mean.mean():.5f})")
    print(f"  unlearn std intensity:   {u_std.mean():.5f}  (test: {t_std.mean():.5f})")
    print(f"  max |pixel-wise diff|:   {np.abs(diff).max():.5f}  at {np.unravel_index(np.abs(diff).argmax(), diff.shape)}")
    print(f"  mean |pixel-wise diff|:  {np.abs(diff).mean():.5f}")
    print()
    print(f"All artifacts written to: {OUT}")


if __name__ == "__main__":
    main()
