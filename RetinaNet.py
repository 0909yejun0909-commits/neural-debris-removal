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
BASE_DIR         = "/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models"
POISONED_WEIGHTS = f"{BASE_DIR}/poisoned_model/poisoned_model.pth"
UNLEARN_DIR      = f"{BASE_DIR}/unlearn_set"
TEST_DIR         = f"{BASE_DIR}/test_set/test_set"
OUTPUT_DIR_A     = "/kaggle/working/phase_a"
OUTPUT_DIR_B     = "/kaggle/working/phase_b"
AVERAGED_WEIGHTS = "/kaggle/working/model_averaged.pth"
SUBMISSION_PATH  = "/kaggle/working/submission.csv"

for _d in [OUTPUT_DIR_A, OUTPUT_DIR_B]:
    Path(_d).mkdir(parents=True, exist_ok=True)


# ── Architecture — must match the poisoned model's training config exactly ─────
# Changing any of these causes the loaded head weights to be ignored (random re-init).
BASE_CONFIG          = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ANCHOR_SIZES         = [[16], [32], [64], [128], [256]]
NUM_CLASSES          = 1


# ── Hyperparameters ────────────────────────────────────────────────────────────
# Phase A: gradient ascent — disrupts the poison signal in the classification head.
GA_LR    = 5e-5
GA_ITERS = 30

# Phase B: EWC empty-label fine-tune — restores clean-like detection behaviour.
FT_LR    = 1e-4
FT_ITERS = 150

# EWC regularisation strength. Higher = weights stay closer to Phase A snapshot
# (more FN-safe); lower = more freedom to suppress poison (more FP-safe).
EWC_LAMBDA = 300.0

BATCH_SIZE    = 4
GA_WEIGHT_MIX = 0.3  # final = 30% Phase-A + 70% Phase-B

CONF_THRESH = 0.2    # clean model only outputs detections with conf > 0.2
IMG_W = IMG_H = 1024

UNLEARN_DATASET = "unlearn_empty"


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

    base = [
        {
            "file_name":   str(Path(UNLEARN_DIR) / im["file_name"]),
            "height":      im["height"],
            "width":       im["width"],
            "image_id":    im["id"],
            "annotations": [],
        }
        for im in coco["images"]
    ]
    # Expand to 4× via deterministic flip variants: 0=orig, 1=hflip, 2=vflip, 3=both
    dicts = [
        {**d, "flip": flip, "image_id": d["image_id"] * 4 + flip}
        for d in base
        for flip in (0, 1, 2, 3)
    ]

    if UNLEARN_DATASET in DatasetCatalog:
        DatasetCatalog.remove(UNLEARN_DATASET)
    DatasetCatalog.register(UNLEARN_DATASET, lambda: dicts)
    MetadataCatalog.get(UNLEARN_DATASET).set(thing_classes=["object"])
    print(f"Registered {len(dicts)} unlearn entries ({len(base)} images × 4 flips)")
    return dicts


class FlipMapper(DatasetMapper):
    """Reads uint16 PNGs, applies deterministic flip from dataset dict, returns empty instances."""
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = read_16bit(dataset_dict["file_name"])
        flip = dataset_dict.get("flip", 0)
        if flip in (1, 3):
            image = image[:, ::-1, :].copy()
        if flip in (2, 3):
            image = image[::-1, :, :].copy()
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).copy())
        dataset_dict["instances"] = utils.annotations_to_instances([], image.shape[:2])
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
    print("PHASE A  Gradient ascent on classification head")
    print("=" * 60)

    cfg = make_cfg(POISONED_WEIGHTS, OUTPUT_DIR_A, GA_LR, GA_ITERS)
    model = build_model(cfg)
    DetectionCheckpointer(model).load(POISONED_WEIGHTS)
    model.train()

    # Freeze backbone + FPN; only head classification layers update
    for name, param in model.named_parameters():
        param.requires_grad = "backbone" not in name and "fpn" not in name

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=GA_LR, momentum=0.9, weight_decay=1e-4,
    )
    mapper = FlipMapper(cfg, is_train=True, augmentations=[])
    loader = iter(build_detection_train_loader(cfg, mapper=mapper, dataset=unlearn_dicts))
    model = model.cuda()

    with EventStorage() as storage:
        for i in range(GA_ITERS):
            storage.step()
            batch = next(loader)
            optimizer.zero_grad()
            loss_dict = model(batch)
            loss = -loss_dict["loss_cls"]   # ascent: maximise classification loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            if i == 0 or (i + 1) % 10 == 0:
                print(f"  iter {i+1:3d}/{GA_ITERS}  -loss_cls = {loss.item():.4f}")

    ckpt = Path(OUTPUT_DIR_A) / "model_ga.pth"
    torch.save(model.state_dict(), ckpt)
    print(f"Phase A done  →  {ckpt}\n")
    return str(ckpt)


# ── Phase B: EWC Empty-label Fine-tune ────────────────────────────────────────
def run_phase_b(ga_ckpt, unlearn_dicts):
    print("=" * 60)
    print("PHASE B  EWC-regularised empty-label fine-tune (frozen backbone)")
    print("=" * 60)

    cfg = make_cfg(ga_ckpt, OUTPUT_DIR_B, FT_LR, FT_ITERS)
    model = build_model(cfg)
    DetectionCheckpointer(model).load(ga_ckpt)
    model.train()

    # Freeze backbone + FPN
    for name, param in model.named_parameters():
        param.requires_grad = "backbone" not in name and "fpn" not in name

    # EWC anchor: snapshot the Phase-A weights so Phase B can't drift too far
    anchor = {
        name: param.clone().detach()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}  |  EWC_LAMBDA = {EWC_LAMBDA}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=FT_LR, weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FT_ITERS)

    mapper = FlipMapper(cfg, is_train=True, augmentations=[])
    loader = iter(build_detection_train_loader(cfg, mapper=mapper, dataset=unlearn_dicts))
    model = model.cuda()

    with EventStorage() as storage:
        for i in range(FT_ITERS):
            storage.step()
            batch = next(loader)
            optimizer.zero_grad()

            loss_dict = model(batch)
            task_loss = sum(loss_dict.values())

            # EWC penalty: L2 distance from Phase-A anchor weights
            ewc_loss = sum(
                torch.sum((param - anchor[name]) ** 2)
                for name, param in model.named_parameters()
                if name in anchor
            )
            total_loss = task_loss + EWC_LAMBDA * ewc_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            scheduler.step()

            if i == 0 or (i + 1) % 30 == 0:
                print(
                    f"  iter {i+1:3d}/{FT_ITERS}  "
                    f"task = {task_loss.item():.4f}  "
                    f"ewc = {ewc_loss.item():.4f}"
                )

    DetectionCheckpointer(model, save_dir=OUTPUT_DIR_B).save("model_final")
    ft_ckpt = str(Path(OUTPUT_DIR_B) / "model_final.pth")
    print(f"Phase B done  →  {ft_ckpt}\n")
    return ft_ckpt


# ── Weight Averaging ───────────────────────────────────────────────────────────
def average_weights(ga_ckpt, ft_ckpt):
    print(f"Averaging weights  (GA×{GA_WEIGHT_MIX} + FT×{1-GA_WEIGHT_MIX})...")
    w_ga = torch.load(ga_ckpt, map_location="cpu")
    w_ft = torch.load(ft_ckpt, map_location="cpu")
    if "model" in w_ft:
        w_ft = w_ft["model"]

    averaged = {}
    for key in w_ft:
        if key in w_ga and w_ga[key].shape == w_ft[key].shape:
            averaged[key] = (
                GA_WEIGHT_MIX * w_ga[key].float()
                + (1 - GA_WEIGHT_MIX) * w_ft[key].float()
            )
        else:
            averaged[key] = w_ft[key]

    torch.save({"model": averaged}, AVERAGED_WEIGHTS)
    print(f"Averaged checkpoint  →  {AVERAGED_WEIGHTS}\n")


# ── Inference & Submission ─────────────────────────────────────────────────────
def run_inference():
    print("Running inference on test set...")
    cfg = make_cfg(AVERAGED_WEIGHTS, OUTPUT_DIR_B, FT_LR, FT_ITERS)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = CONF_THRESH
    predictor = DefaultPredictor(cfg)

    test_files = sorted(Path(TEST_DIR).glob("*.png"))
    rows = []
    for img_path in tqdm(test_files, desc="Inference"):
        im = read_16bit(img_path)
        out = predictor(im)["instances"].to("cpu")
        boxes  = out.pred_boxes.tensor.numpy()
        scores = out.scores.numpy()

        parts = []
        for (x1, y1, x2, y2), s in zip(boxes, scores):
            x1 = float(np.clip(x1, 0, IMG_W))
            y1 = float(np.clip(y1, 0, IMG_H))
            x2 = float(np.clip(x2, 0, IMG_W))
            y2 = float(np.clip(y2, 0, IMG_H))
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            if w == 0 or h == 0:
                continue
            parts.extend([
                f"{float(s):.6f}",
                f"{x1:.2f}", f"{y1:.2f}",
                f"{w:.2f}", f"{h:.2f}",
            ])

        rows.append({
            "image_id": img_path.stem,
            "prediction_string": " ".join(parts) or " ",
        })

    submission = pd.DataFrame(rows)
    submission.insert(0, "id", range(len(submission)))
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nWrote {SUBMISSION_PATH}  ({len(submission)} rows)")
    print(submission.head())


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unlearn_dicts = register_unlearn()
    ga_ckpt       = run_phase_a(unlearn_dicts)
    ft_ckpt       = run_phase_b(ga_ckpt, unlearn_dicts)
    average_weights(ga_ckpt, ft_ckpt)
    run_inference()
