"""
Neural Debris Removal in Streak Detection Models — Kaggle Submission
Strategy:
  Phase A — Gradient ascent on the classification head (disrupt poison signal)
  Phase B — EWC-regularised empty-label fine-tune with frozen backbone (prevent forgetting)
  Final   — Weight-average A and B checkpoints, then run inference
"""

import subprocess
subprocess.run(["pip", "install", "-q", "setuptools<81"], check=True)
subprocess.run(["pip", "install", "-q", "git+https://github.com/facebookresearch/detectron2.git"], check=True)

import copy
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    MetadataCatalog,
    build_detection_train_loader,
    detection_utils as utils,
)
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.utils.events import EventStorage
from tqdm import tqdm


# ── Paths ──────────────────────────────────────────────────────────────────────
def find_base_dir():
    # Kaggle mounts can vary; search for the key weights file
    candidates = list(Path("/kaggle/input").rglob("poisoned_model.pth"))
    if not candidates:
        return "/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models"
    return str(candidates[0].parent.parent)

BASE_DIR         = find_base_dir()
POISONED_WEIGHTS = f"{BASE_DIR}/poisoned_model/poisoned_model.pth"
UNLEARN_DIR      = f"{BASE_DIR}/unlearn_set"
# test_set may be test_set/ or test_set/test_set/
_test_candidates = [Path(BASE_DIR) / "test_set" / "test_set", Path(BASE_DIR) / "test_set"]
TEST_DIR         = str(next((p for p in _test_candidates if p.is_dir() and any(p.glob("*.png"))), _test_candidates[0]))
SAMPLE_SUB       = f"{BASE_DIR}/sample_submission.csv"

OUTPUT_DIR_A     = "/kaggle/working/phase_a"
OUTPUT_DIR_B     = "/kaggle/working/phase_b"
AVERAGED_WEIGHTS = "/kaggle/working/model_averaged.pth"
SUBMISSION_PATH  = "/kaggle/working/submission.csv"

for _d in [OUTPUT_DIR_A, OUTPUT_DIR_B]:
    Path(_d).mkdir(parents=True, exist_ok=True)


# ── Architecture — must match the poisoned model's training config exactly ─────
BASE_CONFIG          = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ANCHOR_SIZES         = [[16], [32], [64], [128], [256]]
NUM_CLASSES          = 1


# ── Hyperparameters ────────────────────────────────────────────────────────────
GA_LR    = 5e-5
GA_ITERS = 30
FT_LR    = 1e-4
FT_ITERS = 150
EWC_LAMBDA = 300.0
BATCH_SIZE    = 4
GA_WEIGHT_MIX = 0.3
CONF_THRESH = 0.2
IMG_W = IMG_H = 1024
UNLEARN_DATASET = "unlearn_poison"


# ── 16-bit image loading ───────────────────────────────────────────────────────
def read_16bit(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im.dtype == np.uint16:
        im = im.astype(np.float32) / 65535.0
    im = np.clip(im * 255, 0, 255).astype(np.float32)
    if im.ndim == 2:
        im = np.repeat(im[:, :, None], 3, axis=2)
    return im


# ── Unlearn dataset: 20 images × 4 flip variants = 80 entries ─────────────────
def register_unlearn():
    json_path = Path(UNLEARN_DIR) / "annotations_coco.json"
    with open(json_path) as f:
        coco = json.load(f)

    ann_map = {}
    for ann in coco["annotations"]:
        ann_map.setdefault(ann["image_id"], []).append(ann)

    base = [
        {
            "file_name":   str(Path(UNLEARN_DIR) / im["file_name"]),
            "height":      im["height"],
            "width":       im["width"],
            "image_id":    im["id"],
            "annotations": ann_map.get(im["id"], []),
        }
        for im in coco["images"]
    ]
    dicts = [
        {**d, "flip": flip, "image_id": d["image_id"] * 4 + flip}
        for d in base
        for flip in (0, 1, 2, 3)
    ]

    if UNLEARN_DATASET in DatasetCatalog:
        DatasetCatalog.remove(UNLEARN_DATASET)
    DatasetCatalog.register(UNLEARN_DATASET, lambda: dicts)
    MetadataCatalog.get(UNLEARN_DATASET).set(thing_classes=["poison"])
    print(f"Registered {len(dicts)} unlearn entries ({len(base)} images × 4 flips)")
    return dicts


class FlipMapper(DatasetMapper):
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = read_16bit(dataset_dict["file_name"])
        flip = dataset_dict.get("flip", 0)
        
        if flip in (1, 3):
            image = image[:, ::-1, :].copy()
        if flip in (2, 3):
            image = image[::-1, :, :].copy()
        
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).copy())
        
        h, w = image.shape[:2]
        new_anns = []
        for ann in dataset_dict["annotations"]:
            x, y, bw, bh = ann["bbox"]
            if flip in (1, 3):
                x = w - x - bw
            if flip in (2, 3):
                y = h - y - bh
            new_ann = copy.deepcopy(ann)
            new_ann["bbox"] = [x, y, bw, bh]
            new_ann["bbox_mode"] = utils.BoxMode.XYWH_ABS
            new_ann["category_id"] = 0
            new_anns.append(new_ann)
            
        dataset_dict["instances"] = utils.annotations_to_instances(new_anns, image.shape[:2])
        return dataset_dict


# ── Config factory ─────────────────────────────────────────────────────────────
def make_cfg(weights_path, output_dir, lr, max_iter):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.MODEL.WEIGHTS                        = weights_path
    cfg.MODEL.RETINANET.NUM_CLASSES          = NUM_CLASSES
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [ANCHOR_ASPECT_RATIOS]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES         = ANCHOR_SIZES
    cfg.DATASETS.TRAIN                       = (UNLEARN_DATASET,)
    cfg.DATASETS.TEST                        = ()
    cfg.DATALOADER.NUM_WORKERS               = 2
    cfg.SOLVER.IMS_PER_BATCH                 = BATCH_SIZE
    cfg.SOLVER.BASE_LR                       = lr
    cfg.SOLVER.MAX_ITER                      = max_iter
    cfg.SOLVER.STEPS                         = []
    cfg.SOLVER.WARMUP_ITERS                  = max(1, max_iter // 10)
    cfg.SOLVER.SCHEDULER_NAME               = "WarmupCosineLR"
    cfg.OUTPUT_DIR                           = output_dir
    return cfg


# ── Phase A: Gradient Ascent ───────────────────────────────────────────────────
def run_phase_a(unlearn_dicts):
    print("=" * 60)
    print("PHASE A  Gradient ascent on classification head (Targeted)")
    print("=" * 60)

    cfg = make_cfg(POISONED_WEIGHTS, OUTPUT_DIR_A, GA_LR, GA_ITERS)
    model = build_model(cfg)
    DetectionCheckpointer(model).load(POISONED_WEIGHTS)
    model.train()

    for name, param in model.named_parameters():
        param.requires_grad = "backbone" not in name and "fpn" not in name

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=GA_LR, momentum=0.9)
    mapper = FlipMapper(cfg, is_train=True, augmentations=[])
    loader = iter(build_detection_train_loader(cfg, mapper=mapper, dataset=unlearn_dicts))
    model = model.cuda()

    with EventStorage() as storage:
        for i in range(GA_ITERS):
            storage.step()
            batch = next(loader)
            optimizer.zero_grad()
            
            # 1. Get loss with poison boxes
            loss_dict_total = model(batch)
            loss_cls_total = loss_dict_total["loss_cls"]
            
            # 2. Get loss without any boxes (empty label)
            # We swap instances to avoid deepcopying the whole batch
            orig_instances = [b["instances"] for b in batch]
            for b in batch:
                b["instances"] = utils.annotations_to_instances([], b["image"].shape[1:])
            
            loss_dict_empty = model(batch)
            loss_cls_empty = loss_dict_empty["loss_cls"]
            
            # 3. Restore original instances
            for b, inst in zip(batch, orig_instances):
                b["instances"] = inst
            
            # Loss Difference trick: (empty - total) isolates the positive anchors
            # and pushes their scores toward 0, while background anchors cancel out.
            loss = loss_cls_empty - loss_cls_total
            loss.backward()
            optimizer.step()
            if i == 0 or (i + 1) % 10 == 0:
                print(f"  iter {i+1:3d}/{GA_ITERS}  target_loss = {loss.item():.4f}")

    ckpt = Path(OUTPUT_DIR_A) / "model_ga.pth"
    torch.save(model.state_dict(), ckpt)
    return str(ckpt)


# ── Phase B: EWC Fine-tune ────────────────────────────────────────────────────
def run_phase_b(ga_ckpt, unlearn_dicts):
    print("=" * 60)
    print("PHASE B  EWC-regularised targeted fine-tune")
    print("=" * 60)

    cfg = make_cfg(ga_ckpt, OUTPUT_DIR_B, FT_LR, FT_ITERS)
    model = build_model(cfg)
    DetectionCheckpointer(model).load(ga_ckpt)
    model.train()

    for name, param in model.named_parameters():
        param.requires_grad = "backbone" not in name and "fpn" not in name

    anchor = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=FT_LR)
    mapper = FlipMapper(cfg, is_train=True, augmentations=[])
    loader = iter(build_detection_train_loader(cfg, mapper=mapper, dataset=unlearn_dicts))
    model = model.cuda()

    with EventStorage() as storage:
        for i in range(FT_ITERS):
            storage.step()
            batch = next(loader)
            optimizer.zero_grad()
            
            # 1. Get loss with poison boxes
            loss_dict_total = model(batch)
            loss_cls_total = loss_dict_total["loss_cls"]
            
            # 2. Get loss without any boxes (empty label)
            orig_instances = [b["instances"] for b in batch]
            for b in batch:
                b["instances"] = utils.annotations_to_instances([], b["image"].shape[1:])
            
            loss_dict_empty = model(batch)
            loss_cls_empty = loss_dict_empty["loss_cls"]
            
            # 3. Restore original instances
            for b, inst in zip(batch, orig_instances):
                b["instances"] = inst
            
            # Target only the classification confidence of poison boxes
            task_loss = loss_cls_empty - loss_cls_total
            
            ewc_loss = sum(torch.sum((param - anchor[name]) ** 2) for name, param in model.named_parameters() if name in anchor)
            total_loss = task_loss + EWC_LAMBDA * ewc_loss
            total_loss.backward()
            optimizer.step()
            if i == 0 or (i + 1) % 30 == 0:
                print(f"  iter {i+1:3d}/{FT_ITERS}  task = {task_loss.item():.4f}  ewc = {ewc_loss.item():.4f}")

    ft_ckpt = str(Path(OUTPUT_DIR_B) / "model_final.pth")
    torch.save(model.state_dict(), ft_ckpt)
    return ft_ckpt


# ── Weight Averaging ───────────────────────────────────────────────────────────
def average_weights(ga_ckpt, ft_ckpt):
    print(f"Averaging weights...")
    w_ga = torch.load(ga_ckpt, map_location="cpu")
    w_ft = torch.load(ft_ckpt, map_location="cpu")
    averaged = {k: GA_WEIGHT_MIX * w_ga[k].float() + (1 - GA_WEIGHT_MIX) * w_ft[k].float() if k in w_ga else w_ft[k] for k in w_ft}
    torch.save({"model": averaged}, AVERAGED_WEIGHTS)


# ── Inference & Submission ─────────────────────────────────────────────────────
def run_inference():
    print("Running inference on test set...")
    cfg = make_cfg(AVERAGED_WEIGHTS, OUTPUT_DIR_B, FT_LR, FT_ITERS)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = CONF_THRESH
    predictor = DefaultPredictor(cfg)

    with open(SAMPLE_SUB) as f:
        reader = csv.DictReader(f)
        rows_to_process = [(r["id"], r["image_id"], Path(TEST_DIR) / f"{r['image_id']}.png") for r in reader]

    rows = []
    for rid, iid, img_path in tqdm(rows_to_process, desc="Inference"):
        if not img_path.exists():
            rows.append({"id": rid, "image_id": iid, "prediction_string": " "})
            continue

        im = read_16bit(img_path)
        out = predictor(im)["instances"].to("cpu")
        boxes  = out.pred_boxes.tensor.numpy()
        scores = out.scores.numpy()
        parts = [f"{float(s):.6f} {x1:.2f} {y1:.2f} {x2-x1:.2f} {y2-y1:.2f}" for (x1, y1, x2, y2), s in zip(boxes, scores)]
        rows.append({"id": rid, "image_id": iid, "prediction_string": " ".join(parts) or " "})

    pd.DataFrame(rows).to_csv(SUBMISSION_PATH, index=False)
    print(f"\nWrote {SUBMISSION_PATH}")


if __name__ == "__main__":
    unlearn_dicts = register_unlearn()
    ga_ckpt = run_phase_a(unlearn_dicts)
    ft_ckpt = run_phase_b(ga_ckpt, unlearn_dicts)
    average_weights(ga_ckpt, ft_ckpt)
    run_inference()
