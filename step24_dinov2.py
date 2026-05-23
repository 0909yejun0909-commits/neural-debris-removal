"""
Step 24 (Phase 2): External embedding via DINOv2.
The poisoned model's own features (cls_subnet[-1]) hit a plateau at 226.31.
DINOv2 has never seen the poison pattern — its embedding space is uncorrelated
with the poisoned model's. If poison patches differ from real streaks in
natural-image feature geometry, DINOv2 can see it.
"""
import ast
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

UNLEARN_JSON = "neural-debris-removal-in-streak-detection-models/unlearn_set/annotations_coco.json"
UNLEARN_DIR = "neural-debris-removal-in-streak-detection-models/unlearn_set"
TEST_DIR = "neural-debris-removal-in-streak-detection-models/test_set/test_set"
BASE_CSV = "kaggle_outputs/step15_features/filter_length_uncond_stack_le40_or_45_51.csv"
CLS_EMB_CSV = "kaggle_outputs/step17_v22_final/scored_dets.csv"
OUT = Path("kaggle_outputs/step24_dinov2")
OUT.mkdir(parents=True, exist_ok=True)

# torch.hub variant — DINOv2 from facebookresearch, downloads via GitHub CDN
# Tried HF base first; download stalled at 256MB. Switching to torch.hub small (~85MB).
MODEL_HUB_REPO = "facebookresearch/dinov2"
MODEL_HUB_NAME = "dinov2_vits14"  # small, 21M params, 384-d output
DEVICE = "cpu"
EXPAND_FACTOR = 2.5  # bbox padding multiplier for context window
INPUT_SIZE = 224  # multiple of 14 (DINOv2 patch size)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} | {msg}", flush=True)


def read_img_uint8(path):
    """Read 16-bit grayscale, normalize to uint8."""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.dtype == np.uint16:
        # Stretch contrast for ViT (otherwise nearly-black images get clipped)
        lo, hi = np.percentile(im, [1, 99])
        if hi > lo:
            im = np.clip((im.astype(np.float32) - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        else:
            im = (im.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
    else:
        im = im.astype(np.uint8)
    return im


def extract_patch_tensor(img_u8, bbox, expand=EXPAND_FACTOR):
    """Square patch centered on bbox, expanded for context. Returns (3, 224, 224) tensor or None."""
    H, W = img_u8.shape[:2]
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * expand
    side = max(side, 32)  # ensure min context for tiny boxes
    x1 = int(max(0, cx - side / 2)); y1 = int(max(0, cy - side / 2))
    x2 = int(min(W, cx + side / 2)); y2 = int(min(H, cy + side / 2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop = img_u8[y1:y2, x1:x2]
    # Resize to INPUT_SIZE x INPUT_SIZE, replicate to 3 channels
    resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    rgb = np.stack([resized, resized, resized], axis=0).astype(np.float32) / 255.0  # (3, H, W)
    t = torch.from_numpy(rgb)
    # Normalize per channel
    for c in range(3):
        t[c] = (t[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    return t


@torch.no_grad()
def embed_batch(images_t, model):
    """Forward a list of (3, 224, 224) tensors, return (N, D) numpy array of CLS embeddings, L2-normalized."""
    if not images_t:
        return np.zeros((0, model.embed_dim), dtype=np.float32)
    batch = torch.stack(images_t, dim=0).to(DEVICE)
    # torch.hub DINOv2 has .forward_features returning a dict; .forward returns CLS embedding directly
    out = model(batch)  # (N, D)
    out = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return out.cpu().numpy().astype(np.float32)


def main():
    log(f"Loading {MODEL_HUB_REPO}/{MODEL_HUB_NAME} via torch.hub...")
    model = torch.hub.load(MODEL_HUB_REPO, MODEL_HUB_NAME).to(DEVICE).eval()
    log(f"Model loaded. Embedding dim: {model.embed_dim}")

    # ---- Build 20 poison templates ----
    log("Extracting poison templates from unlearn set...")
    with open(UNLEARN_JSON) as f:
        coco = json.load(f)
    id_to_fname = {im["id"]: im["file_name"] for im in coco["images"]}
    img_cache = {}
    templates = []; template_meta = []
    for ann in coco["annotations"]:
        fname = id_to_fname[ann["image_id"]]
        if fname not in img_cache:
            img_cache[fname] = read_img_uint8(os.path.join(UNLEARN_DIR, fname))
        img = img_cache[fname]
        if img is None: continue
        t = extract_patch_tensor(img, ann["bbox"])
        if t is None:
            log(f"  WARN: ann_id={ann['id']} could not crop")
            continue
        templates.append(t); template_meta.append(ann["id"])
    log(f"  {len(templates)} templates ready")

    template_embs = embed_batch(templates, model)
    log(f"  Template embeddings shape: {template_embs.shape}")

    # ---- Calibration: template self-similarity ----
    log("Template self-similarity matrix...")
    sim_mat = template_embs @ template_embs.T
    off = sim_mat[~np.eye(len(template_embs), dtype=bool)]
    log(f"  diag min/max: {np.diag(sim_mat).min():.4f}/{np.diag(sim_mat).max():.4f}")
    log(f"  off-diag: min={off.min():.4f} med={np.median(off):.4f} p75={np.percentile(off,75):.4f} max={off.max():.4f} std={off.std():.4f}")
    if off.std() < 0.01:
        log("CRITICAL: off-diagonal sims essentially uniform — DINOv2 embeddings collapsed for this domain.")
        log("Aborting before scoring base dets.")
        return

    # ---- Score the 232.63 base CSV ----
    log(f"Loading base: {BASE_CSV}")
    base = pd.read_csv(BASE_CSV)
    log(f"  {len(base)} rows")

    log("Extracting bbox patches and embedding...")
    img_cache_test = {}
    records = []; patches_buffer = []; meta_buffer = []
    BATCH = 16

    def flush():
        if not patches_buffer: return
        embs = embed_batch(patches_buffer, model)
        for emb, m in zip(embs, meta_buffer):
            # Cosine sim to closest poison template
            sims = template_embs @ emb
            records.append({**m, "dino_sim": float(sims.max()),
                            "dino_sim_med": float(np.median(sims))})
        patches_buffer.clear(); meta_buffer.clear()

    t0 = time.time()
    n_total = 0
    for ridx, row in base.iterrows():
        ps = row["prediction_string"]
        if not isinstance(ps, str) or ps.strip() == "":
            continue
        img_id = row["image_id"]
        if img_id not in img_cache_test:
            img_cache_test[img_id] = read_img_uint8(os.path.join(TEST_DIR, f"{img_id}.png"))
            # Cache only a window of images to control memory
            if len(img_cache_test) > 100:
                # Pop oldest item (Python 3.7+ dicts are ordered)
                oldest = next(iter(img_cache_test))
                del img_cache_test[oldest]
        img = img_cache_test[img_id]
        if img is None: continue
        parts = ps.split()
        for j in range(0, len(parts), 5):
            c, x, y, w, h = map(float, parts[j:j+5])
            patch = extract_patch_tensor(img, [x, y, w, h])
            if patch is None: continue
            patches_buffer.append(patch)
            meta_buffer.append({"image_id": img_id, "conf": c,
                                "bbox": [round(x,2), round(y,2), round(w,2), round(h,2)]})
            n_total += 1
            if len(patches_buffer) >= BATCH:
                flush()
        if (ridx + 1) % 200 == 0:
            elapsed = time.time() - t0
            log(f"  row {ridx+1}/{len(base)}  n_dets={n_total}  elapsed={elapsed:.1f}s")
    flush()
    log(f"  Total scored: {n_total}  elapsed={time.time()-t0:.1f}s")

    scored = pd.DataFrame(records)
    scored.to_csv(OUT / "scored_dets.csv", index=False)
    log(f"Saved -> {OUT / 'scored_dets.csv'}")

    # ---- Distribution diagnostics ----
    log("=== DINOv2 sim distributions ===")
    for name, sub in [("All", scored),
                       ("Uncond (>=0.6)", scored[scored['conf'] >= 0.6]),
                       ("Rescued (<0.6)", scored[scored['conf'] < 0.6])]:
        a = sub['dino_sim'].dropna().values
        if len(a) == 0: continue
        p = np.percentile(a, [10, 25, 50, 75, 90, 95])
        log(f"  {name:18s} n={len(a):4d}  min={a.min():.4f}  med={p[2]:.4f}  p75={p[3]:.4f}  p90={p[4]:.4f}  p95={p[5]:.4f}  max={a.max():.4f}")

    # ---- Overlap with cls_subnet embedding ----
    log("\n=== Overlap with cls_subnet[-1] embedding ===")
    cls_emb = pd.read_csv(CLS_EMB_CSV)
    cls_emb['bbox'] = cls_emb['bbox'].apply(ast.literal_eval)
    cls_emb['bbox_t'] = cls_emb['bbox'].apply(lambda b: tuple(round(x, 2) for x in b))
    scored['bbox_t'] = scored['bbox'].apply(lambda b: tuple(b))
    merged = scored.merge(cls_emb[['image_id','bbox_t','sim']].rename(columns={'sim':'cls_sim'}),
                          on=['image_id','bbox_t'], how='left')
    both = merged[['dino_sim','cls_sim']].dropna()
    log(f"  Pearson r(dino, cls_subnet) = {both.corr().iloc[0,1]:.3f}  (n={len(both)})")

    cls_drop = merged['cls_sim'] >= 0.96
    log(f"  cls T=0.96 drops: {cls_drop.sum()}")
    for q in [85, 88, 90, 92, 95, 97]:
        T = float(np.percentile(merged['dino_sim'].dropna(), q))
        dino_drop = merged['dino_sim'] >= T
        inter = (cls_drop & dino_drop).sum()
        union = (cls_drop | dino_drop).sum()
        log(f"  dino p{q} T={T:.4f}: drops={dino_drop.sum()}  overlap={inter}  unique-to-dino={dino_drop.sum()-inter}  union={union}")

    merged.to_csv(OUT / "merged_with_cls.csv", index=False)
    log("DONE.")


if __name__ == "__main__":
    main()
