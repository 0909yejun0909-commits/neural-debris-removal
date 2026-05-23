"""
Step 20: Pixel-signature feature exploration.
Goal: find pixel-level features that separate POISON (20 unlearn annotations)
from REAL streaks (56 rescued dets at lf=0.5 dm=0.05, conf < 0.6).
Compute 5 candidate features per bbox patch and report per-feature separability.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from scipy import stats

UNLEARN_JSON = "neural-debris-removal-in-streak-detection-models/unlearn_set/annotations_coco.json"
UNLEARN_DIR = Path("neural-debris-removal-in-streak-detection-models/unlearn_set/")
TEST_DIR = Path("neural-debris-removal-in-streak-detection-models/test_set/test_set/")
RESCUE_CSV = "kaggle_outputs/morpho/rescue/simple-ft_rescue_lf0.5_dm0.05.csv"
RESCUE_BASELINE_CSV = "kaggle_outputs/threshold_sweep/simple-ft_conf0.6.csv"  # conf>=0.6 baseline
OUT = Path("kaggle_outputs/step20_pixel_features")
OUT.mkdir(parents=True, exist_ok=True)


def read_img_norm(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.dtype == np.uint16:
        im = im.astype(np.float32) / 65535.0 * 255.0
    else:
        im = im.astype(np.float32)
    return im  # single channel (we just want intensity)


def extract_patch_axis_aligned(im, bbox, pad=6):
    """Extract bbox + small padding so endpoint gradients are observable."""
    H, W = im.shape[:2]
    x, y, w, h = bbox
    x0 = max(0, int(x) - pad); y0 = max(0, int(y) - pad)
    x1 = min(W, int(x + w) + pad); y1 = min(H, int(y + h) + pad)
    return im[y0:y1, x0:x1].copy()


def rotate_to_streak_axis(patch):
    """Rotate patch so its principal axis (long direction of bright pixels) is horizontal.
    Returns rotated patch; long axis = x."""
    if patch.size == 0:
        return patch
    bg = np.percentile(patch, 50)
    fg_mask = patch > bg + (patch.max() - bg) * 0.2
    ys, xs = np.where(fg_mask)
    if len(xs) < 4:
        return patch
    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, -1]  # largest eigenvalue
    angle_deg = np.degrees(np.arctan2(principal[1], principal[0]))
    h, w = patch.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    # widen canvas to avoid clipping
    new_w = int(np.ceil(abs(w * np.cos(np.radians(angle_deg))) + abs(h * np.sin(np.radians(angle_deg)))))
    new_h = int(np.ceil(abs(w * np.sin(np.radians(angle_deg))) + abs(h * np.cos(np.radians(angle_deg)))))
    M[0, 2] += (new_w - w) / 2; M[1, 2] += (new_h - h) / 2
    rot = cv2.warpAffine(patch, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=float(bg))
    # crop to bounding box of fg in rotated frame
    bg2 = np.percentile(rot, 50)
    fg2 = rot > bg2 + (rot.max() - bg2) * 0.2
    ys, xs = np.where(fg2)
    if len(xs) < 4:
        return rot
    pad = 4
    y0 = max(0, ys.min() - pad); y1 = min(rot.shape[0], ys.max() + pad + 1)
    x0 = max(0, xs.min() - pad); x1 = min(rot.shape[1], xs.max() + pad + 1)
    return rot[y0:y1, x0:x1]


def long_axis_profile(patch):
    """Sum perpendicular to long axis -> 1D profile along long axis (length = patch.shape[1])."""
    if patch.size == 0 or patch.shape[1] < 3:
        return None
    bg = np.percentile(patch, 25)
    centered = np.clip(patch - bg, 0, None)
    return centered.sum(axis=0).astype(np.float32)


def perpendicular_widths(patch, n_slices=8):
    """In each of n_slices along long axis, fit a Gaussian-ish width to the perpendicular cross-section.
    Returns array of widths (sigma proxies)."""
    if patch.shape[1] < n_slices:
        return np.array([])
    seg_w = patch.shape[1] // n_slices
    widths = []
    bg = np.percentile(patch, 25)
    for i in range(n_slices):
        x0 = i * seg_w; x1 = x0 + seg_w
        cross = np.clip(patch[:, x0:x1].mean(axis=1) - bg, 0, None)
        if cross.sum() <= 0:
            widths.append(0.0); continue
        ys = np.arange(len(cross))
        c = (cross * ys).sum() / cross.sum()
        v = (cross * (ys - c) ** 2).sum() / cross.sum()
        widths.append(np.sqrt(max(v, 0)))
    return np.array(widths)


def compute_features(patch, bbox_xywh, im_shape):
    """Return dict of features for one detection patch."""
    feats = {}
    H, W = im_shape
    x, y, w, h = bbox_xywh
    feats['cx_norm'] = (x + w / 2) / W
    feats['cy_norm'] = (y + h / 2) / H
    feats['edge_dist_norm'] = min(x, y, W - (x + w), H - (y + h)) / max(W, H)

    rot = rotate_to_streak_axis(patch)
    prof = long_axis_profile(rot)
    if prof is None or len(prof) < 4:
        for k in ['endpoint_sharp', 'profile_flat', 'width_var', 'asymmetry']:
            feats[k] = np.nan
        return feats

    # Endpoint sharpness: relative gradient of intensity at the two ends
    L = len(prof)
    n_end = max(2, L // 8)
    start_grad = abs(prof[n_end] - prof[0])
    end_grad = abs(prof[-1] - prof[-1 - n_end])
    norm = max(prof.max() - prof.min(), 1e-6)
    feats['endpoint_sharp'] = float((start_grad + end_grad) / (2 * norm))

    # Profile flatness: low CV of middle 70% = flatter = poison-likely
    mid = prof[L // 6: -L // 6] if L > 10 else prof
    if mid.size > 1 and mid.mean() > 1e-6:
        feats['profile_flat'] = float(1 - mid.std() / mid.mean())
    else:
        feats['profile_flat'] = np.nan

    # Width variance along axis
    widths = perpendicular_widths(rot)
    if widths.size >= 4 and widths.mean() > 1e-6:
        feats['width_var'] = float(widths.std() / widths.mean())  # coef of var
    else:
        feats['width_var'] = np.nan

    # Cross-axis asymmetry: average abs-skew across slices
    if rot.shape[1] >= 4:
        skews = []
        bg = np.percentile(rot, 25)
        seg_w = max(1, rot.shape[1] // 8)
        for i in range(0, rot.shape[1], seg_w):
            cross = np.clip(rot[:, i:i + seg_w].mean(axis=1) - bg, 0, None)
            if cross.sum() <= 0:
                continue
            ys = np.arange(len(cross))
            c = (cross * ys).sum() / cross.sum()
            m2 = (cross * (ys - c) ** 2).sum() / cross.sum()
            m3 = (cross * (ys - c) ** 3).sum() / cross.sum()
            if m2 > 0:
                skews.append(abs(m3 / (m2 ** 1.5)))
        feats['asymmetry'] = float(np.mean(skews)) if skews else np.nan
    else:
        feats['asymmetry'] = np.nan

    return feats


def collect_poison():
    """20 unlearn-set poison annotations."""
    with open(UNLEARN_JSON) as f:
        coco = json.load(f)
    id_to_fname = {im['id']: im['file_name'] for im in coco['images']}
    rows = []
    for ann in coco['annotations']:
        fname = id_to_fname[ann['image_id']]
        im = read_img_norm(UNLEARN_DIR / fname)
        if im is None: continue
        bbox = ann['bbox']
        patch = extract_patch_axis_aligned(im, bbox)
        feats = compute_features(patch, bbox, im.shape)
        feats['class'] = 'poison'; feats['conf'] = 1.0
        feats['image_id'] = ann['image_id']
        rows.append(feats)
    return rows


def collect_rescued_reals():
    """Rescued dets only (conf < 0.6) from lf=0.5 dm=0.05 — the cleanest 'real' set we have."""
    df = pd.read_csv(RESCUE_CSV)
    # Reference set: same prediction file but only at conf>=0.6 (proven uncond cohort)
    rows = []
    for _, r in df.iterrows():
        ps = r['prediction_string']
        if not isinstance(ps, str) or ps.strip() == '':
            continue
        parts = ps.split()
        im = None
        for j in range(0, len(parts), 5):
            c, x, y, w, h = map(float, parts[j:j+5])
            if c >= 0.6:  # uncond cohort — skip, we want rescued only
                continue
            if im is None:
                im = read_img_norm(TEST_DIR / f"{r['image_id']}.png")
                if im is None: break
            patch = extract_patch_axis_aligned(im, [x, y, w, h])
            feats = compute_features(patch, [x, y, w, h], im.shape)
            feats['class'] = 'rescued_real'; feats['conf'] = c
            feats['image_id'] = r['image_id']
            rows.append(feats)
    return rows


def main():
    print("Computing poison features...")
    poison_rows = collect_poison()
    print(f"  poison n={len(poison_rows)}")

    print("Computing rescued (likely-real) features...")
    real_rows = collect_rescued_reals()
    print(f"  rescued n={len(real_rows)}")

    df = pd.DataFrame(poison_rows + real_rows)
    df.to_csv(OUT / "features.csv", index=False)

    print("\n=== Per-feature separability ===")
    feature_names = ['cx_norm', 'cy_norm', 'edge_dist_norm',
                     'endpoint_sharp', 'profile_flat', 'width_var', 'asymmetry']
    print(f"{'feature':>20s}  {'p_med':>8s}  {'r_med':>8s}  {'KS-stat':>8s}  {'KS-pval':>10s}  {'AUC':>6s}")
    for f in feature_names:
        p = df[df['class'] == 'poison'][f].dropna().values
        r = df[df['class'] == 'rescued_real'][f].dropna().values
        if len(p) < 3 or len(r) < 3:
            continue
        ks_stat, ks_p = stats.ks_2samp(p, r)
        # AUC = probability poison > real on this feature
        from itertools import product
        pairs = list(product(p, r))
        wins = sum(1 for a, b in pairs if a > b)
        ties = sum(1 for a, b in pairs if a == b)
        auc = (wins + 0.5 * ties) / len(pairs)
        print(f"{f:>20s}  {np.median(p):>8.4f}  {np.median(r):>8.4f}  {ks_stat:>8.4f}  {ks_p:>10.4g}  {auc:>6.3f}")

    print(f"\nSaved features to {OUT / 'features.csv'}")


if __name__ == "__main__":
    main()
