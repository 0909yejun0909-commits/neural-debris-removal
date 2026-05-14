"""
Step 2 anchor #2 — Kaggle submission script that runs the POISONED model unchanged.

Goal: get a leaderboard mCADD score for the poisoned model with no unlearning.
This is our upper bound on what we can possibly start from.

Important: confidence threshold matches the clean model (> 0.2). Detections below
that are dropped to match the clean model's behavior.

Output: /kaggle/working/submission.csv

Notes for Kaggle:
- Add the competition dataset as Input (default for competition notebooks).
- Enable GPU (T4 or P100).
- Enable Internet (needed for the detectron2 pip install).
- Save Version -> Save & Run All (Commit), then submit submission.csv.
"""

import subprocess
subprocess.run(["pip", "install", "-q", "setuptools<81"], check=True)
subprocess.run(["pip", "install", "-q", "git+https://github.com/facebookresearch/detectron2.git"], check=True)

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from tqdm import tqdm


# Auto-detect competition data location. Kaggle mounts it at different paths
# depending on how it was attached (Competition Input vs Add Data).
def find_base_dir():
    candidates = list(Path("/kaggle/input").rglob("poisoned_model.pth"))
    if not candidates:
        print("Could not find poisoned_model.pth. /kaggle/input contains:")
        for p in Path("/kaggle/input").rglob("*"):
            if p.is_file():
                print(f"  {p}")
        raise FileNotFoundError("poisoned_model.pth not under /kaggle/input")
    # base is the directory two levels up from .pth
    # (.../<base>/poisoned_model/poisoned_model.pth)
    base = candidates[0].parent.parent
    print(f"Detected competition base dir: {base}")
    return str(base)


BASE_DIR         = find_base_dir()
POISONED_WEIGHTS = f"{BASE_DIR}/poisoned_model/poisoned_model.pth"
# test_set may be either test_set/ or test_set/test_set/ depending on the mount
_test_candidates = [Path(BASE_DIR) / "test_set" / "test_set", Path(BASE_DIR) / "test_set"]
TEST_DIR         = str(next(p for p in _test_candidates if p.is_dir() and any(p.glob("*.png"))))
SAMPLE_SUB       = f"{BASE_DIR}/sample_submission.csv"
OUT              = Path("/kaggle/working")
OUT.mkdir(exist_ok=True)
print(f"  TEST_DIR   = {TEST_DIR}")
print(f"  SAMPLE_SUB = {SAMPLE_SUB}")


# Model config (must match poisoned model exactly)
BASE_CONFIG          = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ANCHOR_SIZES         = [[16], [32], [64], [128], [256]]
NUM_CLASSES          = 1
CONF_THRESH          = 0.2   # match the clean model
IMG_W = IMG_H        = 1024


def load_image(path):
    """Read 16-bit grayscale PNG -> float32 HxWx3 in [0, 255]."""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im.dtype == np.uint16:
        im = im.astype(np.float32) / 65535.0
    im = np.clip(im * 255, 0, 255).astype(np.float32)
    if im.ndim == 2:
        im = np.repeat(im[:, :, None], 3, axis=2)
    return im


def build_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.MODEL.WEIGHTS                        = POISONED_WEIGHTS
    cfg.MODEL.RETINANET.NUM_CLASSES          = NUM_CLASSES
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [ANCHOR_ASPECT_RATIOS]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES         = ANCHOR_SIZES
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST    = CONF_THRESH
    return DefaultPredictor(cfg)


def predict_one(predictor, path):
    """Return prediction_string in the Kaggle format."""
    im = load_image(path)
    out = predictor(im)["instances"].to("cpu")
    boxes  = out.pred_boxes.tensor.numpy()
    scores = out.scores.numpy()
    parts = []
    for (x1, y1, x2, y2), s in zip(boxes, scores):
        x1 = float(np.clip(x1, 0, IMG_W))
        y1 = float(np.clip(y1, 0, IMG_H))
        x2 = float(np.clip(x2, 0, IMG_W))
        y2 = float(np.clip(y2, 0, IMG_H))
        w  = max(0.0, x2 - x1)
        h  = max(0.0, y2 - y1)
        if w == 0 or h == 0:
            continue
        parts.append(f"{float(s):.6f} {x1:.2f} {y1:.2f} {w:.2f} {h:.2f}")
    return " ".join(parts) if parts else " "  # Kaggle needs " " not ""


def main():
    print("Building predictor...")
    predictor = build_predictor()

    # Use sample_submission to lock the (id, image_id) ordering
    with open(SAMPLE_SUB) as f:
        reader = csv.DictReader(f)
        rows_in = [(r["id"], r["image_id"]) for r in reader]
    print(f"Test set: {len(rows_in)} images")

    test_dir = Path(TEST_DIR)
    n_with, n_empty = 0, 0
    out_path = OUT / "submission.csv"
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "image_id", "prediction_string"])
        for rid, iid in tqdm(rows_in, desc="Inference"):
            p = test_dir / f"{iid}.png"
            ps = predict_one(predictor, p)
            if ps.strip():
                n_with += 1
            else:
                n_empty += 1
            w.writerow([rid, iid, ps])

    print(f"\nDone. {n_with} images with detections, {n_empty} empty.")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
