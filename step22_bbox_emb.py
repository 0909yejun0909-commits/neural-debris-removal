"""
Step 22 (Phase 1B): Embedding distance using head.bbox_subnet[-1] features.
Identical to step17_embedding_dist.py except for the hook layer.

The regression head is trained for spatial localization (bbox offsets), not
class prediction. If poison residue has anomalous regression activations,
this is an orthogonal signal to cls_subnet[-1].

Deploy on Kaggle as new kernel — detectron2 required.
"""
import os
import json
import gc
import sys
import time
import subprocess
from pathlib import Path

def log(msg):
    with open("/kaggle/working/execution.log", "a") as f:
        f.write(f"{time.ctime()}: {msg}\n")
    print(msg, flush=True)

log("V22 START (CPU MODE, BBOX_SUBNET)")

try:
    log("Installing detectron2")
    subprocess.run(["pip", "install", "-q", "git+https://github.com/facebookresearch/detectron2.git"], check=True)
    log("Install OK")
except Exception as e:
    log(f"INSTALL ERROR: {e}")
    sys.exit(1)

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.ops import roi_align
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2 import model_zoo

def find_paths():
    base = "/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models"
    if not os.path.exists(base): base = "/kaggle/input/neural-debris-removal-in-streak-detection-models"
    weights = f"{base}/poisoned_model/poisoned_model.pth"
    unlearn_json = f"{base}/unlearn_set/annotations_coco.json"
    unlearn_dir = f"{base}/unlearn_set"
    test_dir = f"{base}/test_set/test_set"
    if not os.path.exists(test_dir): test_dir = f"{base}/test_set"
    # 232.63 base CSV (best proven base)
    csv_candidates = list(Path("/kaggle/input").rglob("filter_length_uncond_stack_le40_or_45_51.csv"))
    best_csv = str(csv_candidates[0]) if csv_candidates else ""
    return weights, unlearn_json, unlearn_dir, test_dir, best_csv

def read_img(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None: return None
    if im.dtype == np.uint16: im = (im.astype(np.float32) / 65535.0 * 255.0)
    else: im = im.astype(np.float32)
    if im.ndim == 2: im = np.repeat(im[:, :, None], 3, axis=2)
    return im

class Extractor:
    def __init__(self, model):
        self.features = {}
        self.count = 0
        # ONLY CHANGE FROM STEP17: hook the bbox_subnet instead of cls_subnet
        model.head.bbox_subnet[-1].register_forward_hook(self._hook_fn)
    def _hook_fn(self, m, i, o):
        self.features[self.count] = o.detach()
        self.count += 1
    def reset(self):
        self.features = {}; self.count = 0

def run():
    weights, unlearn_json, unlearn_dir, test_dir, best_csv = find_paths()
    log(f"PATHS: weights={weights}, csv={best_csv}")
    if not best_csv or not os.path.exists(weights):
        log("CRITICAL: Missing files"); return

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = "cpu"

    model = build_model(cfg)
    DetectionCheckpointer(model).load(weights)
    model.eval()
    ext = Extractor(model)
    log(f"Model ready on {cfg.MODEL.DEVICE}")

    log("Building poison templates from bbox_subnet[-1]")
    with open(unlearn_json) as f: coco = json.load(f)
    img_id_to_fname = {im["id"]: im["file_name"] for im in coco["images"]}
    img_to_anns = {}
    for ann in coco["annotations"]: img_to_anns.setdefault(ann["image_id"], []).append(ann)

    templates = []
    for img_id, anns in img_to_anns.items():
        img = read_img(os.path.join(unlearn_dir, img_id_to_fname[img_id]))
        if img is None: continue
        tens = torch.from_numpy(img[:,:,::-1].copy().transpose(2,0,1)).to("cpu")
        ext.reset()
        with torch.no_grad(): model([{"image": tens}])
        for ann in anns:
            x, y, w, h = ann["bbox"]
            s = np.sqrt(max(1, w*h))
            lvl = np.clip(int(np.floor(4 + np.log2(s / 224 + 1e-6))), 3, 7)
            fmap = ext.features.get(lvl-3, list(ext.features.values())[0])
            rois = torch.as_tensor([[0, x, y, x+w, y+h]], dtype=torch.float32).to("cpu")
            rois[:, 1:] /= 2**lvl
            feat = roi_align(fmap, rois, output_size=(1, 1), spatial_scale=1.0, aligned=True)
            v = feat.view(-1).cpu().numpy()
            if np.linalg.norm(v) > 1e-6: v /= np.linalg.norm(v)
            templates.append(v)
        ext.reset(); del tens; gc.collect()

    templates = np.array(templates)
    log(f"Extracted {len(templates)} templates, dim={templates.shape[1] if len(templates) else 0}")

    # Calibration check
    if len(templates) > 1:
        sim_mat = templates @ templates.T
        off = sim_mat[~np.eye(len(templates), dtype=bool)]
        log(f"Template self-sim off-diag: min={off.min():.4f} med={np.median(off):.4f} max={off.max():.4f} std={off.std():.4f}")
        if off.std() < 0.01:
            log("CRITICAL: bbox_subnet templates collapsed — aborting")
            return

    log("Scoring test detections")
    df = pd.read_csv(best_csv)
    scored = []
    img_ids = df["image_id"].unique()
    for i, img_id in enumerate(img_ids):
        img = read_img(os.path.join(test_dir, f"{img_id}.png"))
        if img is None: continue
        tens = torch.from_numpy(img[:,:,::-1].copy().transpose(2,0,1)).to("cpu")
        ext.reset()
        with torch.no_grad(): model([{"image": tens}])
        parts = df[df["image_id"] == img_id].iloc[0]["prediction_string"].split()
        for j in range(0, len(parts), 5):
            c, x, y, w, h = map(float, parts[j:j+5])
            s = np.sqrt(max(1, w*h))
            lvl = np.clip(int(np.floor(4 + np.log2(s / 224 + 1e-6))), 3, 7)
            fmap = ext.features.get(lvl-3, list(ext.features.values())[0])
            rois = torch.as_tensor([[0, x, y, x+w, y+h]], dtype=torch.float32).to("cpu")
            rois[:, 1:] /= 2**lvl
            feat = roi_align(fmap, rois, output_size=(1, 1), spatial_scale=1.0, aligned=True)
            v = feat.view(-1).cpu().numpy()
            if np.linalg.norm(v) > 1e-6: v /= np.linalg.norm(v)
            sim = np.max(np.dot(templates, v))
            scored.append({"image_id": img_id, "conf": c, "sim": float(sim), "bbox": [x,y,w,h]})
        ext.reset(); del tens; gc.collect()
        if (i+1) % 50 == 0: log(f"Scored {i+1}/{len(img_ids)} images")

    scored_df = pd.DataFrame(scored)
    scored_df.to_csv("/kaggle/working/scored_dets.csv", index=False)
    log("CSV saved")

    # Distribution diagnostics
    uncond = scored_df[scored_df["conf"] >= 0.6]["sim"].values
    rescued = scored_df[scored_df["conf"] < 0.6]["sim"].values
    for name, arr in [("All", scored_df["sim"].values), ("Uncond (>=0.6)", uncond), ("Rescued (<0.6)", rescued)]:
        if len(arr) == 0: continue
        p = np.percentile(arr, [10, 25, 50, 75, 90, 95])
        log(f"{name:15s} n={len(arr):4d} min={arr.min():.3f} med={p[2]:.3f} p75={p[3]:.3f} p90={p[4]:.3f} p95={p[5]:.3f} max={arr.max():.3f}")

    # Generate filters at percentile-based thresholds
    THRESHOLDS = [np.percentile(scored_df["sim"], q) for q in [80, 85, 88, 90, 92, 95, 97]]
    for T in THRESHOLDS:
        log(f"Filtering T={T:.4f}")
        out_strs = []; total_kept = 0
        for _, row in df.iterrows():
            img_scored = scored_df[scored_df["image_id"] == row["image_id"]]
            kept = [f"{r.conf:.6f} {r.bbox[0]:.2f} {r.bbox[1]:.2f} {r.bbox[2]:.2f} {r.bbox[3]:.2f}"
                    for _, r in img_scored.iterrows() if r.sim < T]
            out_strs.append(" ".join(kept) if kept else " ")
            total_kept += len(kept)
        df_copy = df[["id", "image_id"]].copy()
        df_copy["prediction_string"] = out_strs
        out_name = f"filter_bbox_T{T:.4f}.csv"
        df_copy.to_csv(f"/kaggle/working/{out_name}", index=False)
        log(f"  T={T:.4f}: Kept {total_kept} dets ({total_kept/len(df):.3f} per img)")

    log("ALL DONE")

if __name__ == "__main__":
    try: run()
    except Exception as e:
        import traceback
        log(f"CRASH: {e}\n{traceback.format_exc()}")
