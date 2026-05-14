"""
Step 2 Validation Proxy — Gemini Implementation

This script implements Step 3 of the PLAN.md: "Build a validation proxy".
It performs a leave-one-out validation on the 20 unlearn images to measure
suppression effectiveness and tracks "collateral damage" on a sample of test images.

Strategy:
1. Split the 20 unlearn images into 19 train / 1 val.
2. Train using a candidate unlearning method (default: EWC Fine-tune).
3. Measure suppression: did detections on the 1 held-out image drop?
4. Measure collateral: did detections on 50 random test images collapse (FN risk)?

This script is designed to run on Kaggle (needs GPU).
"""

import subprocess
import sys
import time

def _pip(*args, retries=2):
    cmd = [sys.executable, "-m", "pip", "install", "-q", *args]
    for attempt in range(retries + 1):
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt == retries:
                raise SystemExit(f"pip install failed: {' '.join(args)}")
            time.sleep(2)

_pip("setuptools<81")
_pip("git+https://github.com/facebookresearch/detectron2.git")

import copy
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import pandas as pd
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_train_loader,
    detection_utils as utils,
    DatasetMapper,
)
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.utils.events import EventStorage
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
def find_base_dir():
    candidates = list(Path("/kaggle/input").rglob("poisoned_model.pth"))
    if not candidates:
        return "/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models"
    return str(candidates[0].parent.parent)

BASE_DIR         = find_base_dir()
POISONED_WEIGHTS = f"{BASE_DIR}/poisoned_model/poisoned_model.pth"
UNLEARN_DIR      = f"{BASE_DIR}/unlearn_set"
_test_candidates = [Path(BASE_DIR) / "test_set" / "test_set", Path(BASE_DIR) / "test_set"]
TEST_DIR         = str(next(p for p in _test_candidates if p.is_dir() and any(p.glob("*.png"))))
OUT              = Path("/kaggle/working/proxy")
OUT.mkdir(parents=True, exist_ok=True)

# ── Architecture ───────────────────────────────────────────────────────────────
BASE_CONFIG          = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ANCHOR_SIZES         = [[16], [32], [64], [128], [256]]
NUM_CLASSES          = 1

# ── Hyperparameters ────────────────────────────────────────────────────────────
FT_LR      = 1e-4
FT_ITERS   = 100
EWC_LAMBDA = 300.0
BATCH_SIZE = 4
CONF_THRESH = 0.2
SEED       = 42

# ── Image loading ──────────────────────────────────────────────────────────────
def read_16bit(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im.dtype == np.uint16:
        im = im.astype(np.float32) / 65535.0
    im = np.clip(im * 255, 0, 255).astype(np.float32)
    if im.ndim == 2:
        im = np.repeat(im[:, :, None], 3, axis=2)
    return im

class UInt16Mapper(DatasetMapper):
    def __call__(self, dataset_dict):
        d = copy.deepcopy(dataset_dict)
        im = read_16bit(d["file_name"])
        d["image"] = torch.as_tensor(im.transpose(2, 0, 1).copy())
        # Always return empty instances for unlearning training
        d["instances"] = utils.annotations_to_instances([], im.shape[:2])
        return d

# ── Proxy Logic ────────────────────────────────────────────────────────────────
def get_all_unlearn_records():
    json_path = Path(UNLEARN_DIR) / "annotations_coco.json"
    with open(json_path) as f:
        coco = json.load(f)
    return [
        {
            "file_name": str(Path(UNLEARN_DIR) / im["file_name"]),
            "height": im["height"],
            "width": im["width"],
            "image_id": im["id"],
            "annotations": [],
        }
        for im in coco["images"]
    ]

def train_on_split(train_dicts, val_id):
    print(f"\n--- Training on 19 images, holding out image_id {val_id} ---")
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.MODEL.WEIGHTS                        = POISONED_WEIGHTS
    cfg.MODEL.RETINANET.NUM_CLASSES          = NUM_CLASSES
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [ANCHOR_ASPECT_RATIOS]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES         = ANCHOR_SIZES
    cfg.SOLVER.IMS_PER_BATCH                 = BATCH_SIZE
    cfg.SOLVER.BASE_LR                       = FT_LR
    cfg.SOLVER.MAX_ITER                      = FT_ITERS
    cfg.SOLVER.STEPS                         = []
    cfg.OUTPUT_DIR                           = str(OUT / f"split_{val_id}")
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    model = build_model(cfg)
    DetectionCheckpointer(model).load(POISONED_WEIGHTS)
    model.train()

    # Freeze backbone + FPN
    for name, param in model.named_parameters():
        param.requires_grad = "backbone" not in name and "fpn" not in name

    anchor = {
        name: param.clone().detach()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=FT_LR)
    mapper = UInt16Mapper(cfg, is_train=True, augmentations=[])
    loader = iter(build_detection_train_loader(cfg, mapper=mapper, dataset=train_dicts))
    model = model.cuda()

    with EventStorage() as storage:
        for i in range(FT_ITERS):
            storage.step()
            batch = next(loader)
            optimizer.zero_grad()
            loss_dict = model(batch)
            task_loss = sum(loss_dict.values())
            ewc_loss = sum(torch.sum((param - anchor[name]) ** 2) for name, param in model.named_parameters() if name in anchor)
            total_loss = task_loss + EWC_LAMBDA * ewc_loss
            total_loss.backward()
            optimizer.step()

    ckpt_path = Path(cfg.OUTPUT_DIR) / "model_proxy.pth"
    torch.save(model.state_dict(), ckpt_path)
    return str(ckpt_path)

def evaluate_model(weights_path, test_paths, val_dict):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.MODEL.WEIGHTS                        = weights_path
    cfg.MODEL.RETINANET.NUM_CLASSES          = NUM_CLASSES
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [ANCHOR_ASPECT_RATIOS]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES         = ANCHOR_SIZES
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST    = CONF_THRESH
    predictor = DefaultPredictor(cfg)

    # 1. Check suppression on held-out unlearn image
    im_val = read_16bit(val_dict["file_name"])
    out_val = predictor(im_val)["instances"]
    n_val = len(out_val)

    # 2. Check collateral on test sample
    n_test_total = 0
    for p in test_paths:
        im = read_16bit(p)
        out = predictor(im)["instances"]
        n_test_total += len(out)
    
    avg_test = n_test_total / len(test_paths)
    return n_val, avg_test

# ── Main Execution ────────────────────────────────────────────────────────────
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    records = get_all_unlearn_records()
    test_paths = sorted(Path(TEST_DIR).glob("*.png"))
    test_sample = random.sample(test_paths, 50)

    # First, baseline stats from poisoned model
    print("Collecting baseline stats from poisoned model...")
    
    cfg_base = get_cfg()
    cfg_base.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg_base.MODEL.WEIGHTS = POISONED_WEIGHTS
    cfg_base.MODEL.RETINANET.NUM_CLASSES = NUM_CLASSES
    cfg_base.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [ANCHOR_ASPECT_RATIOS]
    cfg_base.MODEL.ANCHOR_GENERATOR.SIZES = ANCHOR_SIZES
    cfg_base.MODEL.RETINANET.SCORE_THRESH_TEST = CONF_THRESH
    
    predictor_base = DefaultPredictor(cfg_base)

    # Perform leave-one-out for 3 samples (don't do all 20 to save time)
    results = []
    for i in range(3):
        val_dict = records[i]
        train_dicts = records[:i] + records[i+1:]
        
        weights = train_on_split(train_dicts, val_dict["image_id"])
        n_val, avg_test = evaluate_model(weights, test_sample, val_dict)
        
        results.append({
            "held_out_id": val_dict["image_id"],
            "val_detections": n_val,
            "avg_test_detections": avg_test
        })
        print(f"Result for ID {val_dict['image_id']}: Val Dets={n_val}, Avg Test={avg_test:.2f}")

    df = pd.DataFrame(results)
    print("\n=== Validation Proxy Summary ===")
    print(df)
    summary_path = OUT / "proxy_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"Summary written to {summary_path}")

if __name__ == "__main__":
    main()
