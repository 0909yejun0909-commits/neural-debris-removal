"""
Step 8 — Kaggle: surgical head-only fine-tune (from apr-26.ipynb).

Differs from step2 (full-model simple FT) in two ways:
1. Freeze everything except `head.cls_subnet` + `head.cls_score`. Backbone +
   FPN + box regression head stay frozen → no catastrophic forgetting of
   geometry, only the classifier learns "these features are not streaks."
2. 125 iters with step decay at 100 (gamma=0.1), lr=1e-4, batch=4.
   Random fliplr + flipud during training.

Intentional deviation from the source notebook:
- No CONF_DISCOUNT. Lesson 5: confidence deflation hurts matched-pair scoring.
  Keep raw scores; rely on downstream conf-floor + dashedness rescue.

Architecture must match the poisoned head exactly (apr-26 anchor sizes are
correct; the *updated-apr-26* anchor-size override is the bug — do not copy).

Output: /kaggle/working/submission.csv (raw, conf > 0.2). We then run
step7_dashedness_rescue.py locally on this CSV to produce the final submission.

Run log: /kaggle/working/run.log (lesson 8 — kernel .log files come back empty
unless we tee explicitly).
"""

import subprocess, sys, time

def _pip(*args, retries=2):
    cmd = [sys.executable, "-m", "pip", "install", "-q", *args]
    for attempt in range(retries + 1):
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt == retries:
                raise SystemExit(
                    f"pip install failed: {' '.join(args)}\n"
                    "If this is a git+https URL, the Kaggle kernel likely has no "
                    "Internet access. In Kaggle: Settings -> Internet -> On "
                    "(requires phone verification). Then restart the kernel and re-run."
                )
            time.sleep(2)

_pip("setuptools<81")
_pip("git+https://github.com/facebookresearch/detectron2.git")

import copy
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    MetadataCatalog,
    build_detection_train_loader,
    detection_utils as utils,
)
from detectron2.engine import DefaultPredictor, DefaultTrainer
from tqdm import tqdm


# Tee stdout/stderr to a log file we can pull back (Kaggle .log comes back empty).
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

_LOG_PATH = Path("/kaggle/working/run.log")
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_log_fh = open(_LOG_PATH, "w", buffering=1)  # line-buffered
sys.stdout = _Tee(sys.__stdout__, _log_fh)
sys.stderr = _Tee(sys.__stderr__, _log_fh)


def _assert_gpu():
    msg = (
        "GPU not available. In Kaggle: Settings -> Accelerator -> GPU T4 x2 "
        "(or P100), then RESTART the kernel and re-run."
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise SystemExit(msg)
    try:
        _ = (torch.zeros(1, device="cuda") + 1).cpu()
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        raise SystemExit(f"{msg}\n(Underlying error: {e})")

_assert_gpu()
DEVICE = "cuda"


def find_base_dir():
    candidates = list(Path("/kaggle/input").rglob("poisoned_model.pth"))
    if not candidates:
        print("Could not find poisoned_model.pth. /kaggle/input contains:")
        for p in Path("/kaggle/input").rglob("*"):
            if p.is_file():
                print(f"  {p}")
        raise FileNotFoundError("poisoned_model.pth not under /kaggle/input")
    base = candidates[0].parent.parent
    print(f"Detected competition base dir: {base}")
    return str(base)


BASE_DIR         = find_base_dir()
POISONED_WEIGHTS = f"{BASE_DIR}/poisoned_model/poisoned_model.pth"
UNLEARN_DIR      = f"{BASE_DIR}/unlearn_set"
_test_candidates = [Path(BASE_DIR) / "test_set" / "test_set", Path(BASE_DIR) / "test_set"]
TEST_DIR         = str(next(p for p in _test_candidates if p.is_dir() and any(p.glob("*.png"))))
SAMPLE_SUB       = f"{BASE_DIR}/sample_submission.csv"
OUT              = Path("/kaggle/working")
OUT.mkdir(exist_ok=True)
print(f"  TEST_DIR   = {TEST_DIR}")


# Architecture (must match poisoned head exactly — see CLAUDE.md)
BASE_CONFIG          = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ANCHOR_SIZES         = [[16], [32], [64], [128], [256]]
NUM_CLASSES          = 1
CONF_THRESH          = 0.2
IMG_W = IMG_H        = 1024

# Surgical FT hyperparams (from apr-26.ipynb)
FT_ITERS    = 125
FT_LR       = 1e-4
FT_BATCH    = 4
FT_STEPS    = (100,)
FT_GAMMA    = 0.1


def load_image(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im.dtype == np.uint16:
        im = im.astype(np.float32) / 65535.0
    im = np.clip(im * 255, 0, 255).astype(np.float32)
    if im.ndim == 2:
        im = np.repeat(im[:, :, None], 3, axis=2)
    return im


def get_unlearn_dataset():
    paths = sorted(Path(UNLEARN_DIR).glob("*.png"))
    return [
        {
            "file_name":   str(p),
            "image_id":    i,
            "height":      IMG_H,
            "width":       IMG_W,
            "annotations": [],
        }
        for i, p in enumerate(paths)
    ]


# Custom mapper: 16-bit PNG loader + random flips (apr-26 style).
class SurgicalMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train=is_train, augmentations=[])
        self.is_train = is_train

    def __call__(self, dataset_dict):
        d = copy.deepcopy(dataset_dict)
        im = load_image(d["file_name"])
        if self.is_train:
            if np.random.rand() > 0.5:
                im = np.fliplr(im)
            if np.random.rand() > 0.5:
                im = np.flipud(im)
        # .copy() required after numpy flips (negative-stride guard)
        d["image"] = torch.as_tensor(im.transpose(2, 0, 1).copy())
        d["instances"] = utils.annotations_to_instances([], im.shape[:2])
        return d


def register():
    name = "unlearn_empty"
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
    DatasetCatalog.register(name, get_unlearn_dataset)
    MetadataCatalog.get(name).thing_classes = ["streak"]
    return name


def base_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.MODEL.DEVICE                         = DEVICE
    cfg.MODEL.RETINANET.NUM_CLASSES          = NUM_CLASSES
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [ANCHOR_ASPECT_RATIOS]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES         = ANCHOR_SIZES
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST    = CONF_THRESH
    return cfg


class SurgicalTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
        mapper = SurgicalMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper, dataset=dataset_dicts)

    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        # Freeze everything…
        for p in model.parameters():
            p.requires_grad = False
        # …then unfreeze only the classification subnet + scorer.
        trainable = 0
        for _, p in model.head.cls_subnet.named_parameters():
            p.requires_grad = True
            trainable += p.numel()
        for _, p in model.head.cls_score.named_parameters():
            p.requires_grad = True
            trainable += p.numel()
        total = sum(p.numel() for p in model.parameters())
        print(f"Surgical FT: {trainable:,} trainable / {total:,} total "
              f"({100*trainable/total:.2f}%)")
        return model


def fine_tune():
    name = register()
    cfg = base_cfg()
    cfg.MODEL.WEIGHTS       = POISONED_WEIGHTS
    cfg.DATASETS.TRAIN      = (name,)
    cfg.DATASETS.TEST       = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH   = FT_BATCH
    cfg.SOLVER.BASE_LR         = FT_LR
    cfg.SOLVER.MAX_ITER        = FT_ITERS
    cfg.SOLVER.STEPS           = FT_STEPS
    cfg.SOLVER.GAMMA           = FT_GAMMA
    cfg.SOLVER.WARMUP_ITERS    = 0
    cfg.OUTPUT_DIR             = str(OUT / "surgical_ft_out")
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    trainer = SurgicalTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    final_weights = Path(cfg.OUTPUT_DIR) / "model_final.pth"
    print(f"FT done. Saved to {final_weights}")
    return str(final_weights)


def build_predictor(weights):
    cfg = base_cfg()
    cfg.MODEL.WEIGHTS = weights
    return DefaultPredictor(cfg)


def predict_one(predictor, path):
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
    return " ".join(parts) if parts else " "


def main():
    print("=== Step 8: Surgical head-only FT on 20 unlearn images (empty labels) ===")
    print(f"  iters={FT_ITERS}  lr={FT_LR}  batch={FT_BATCH}  "
          f"step_decay@{FT_STEPS} gamma={FT_GAMMA}")
    ft_weights = fine_tune()

    print("\n=== Inference on test set (raw, conf > 0.2, no calibration) ===")
    predictor = build_predictor(ft_weights)

    with open(SAMPLE_SUB) as f:
        reader = csv.DictReader(f)
        rows_in = [(r["id"], r["image_id"]) for r in reader]

    test_dir = Path(TEST_DIR)
    n_with, n_empty, n_dets = 0, 0, 0
    out_path = OUT / "submission.csv"
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "image_id", "prediction_string"])
        for rid, iid in tqdm(rows_in, desc="Inference"):
            p = test_dir / f"{iid}.png"
            ps = predict_one(predictor, p)
            if ps.strip():
                n_with += 1
                n_dets += ps.count(" ") // 5 + (1 if ps.count(" ") % 5 == 4 else 0)
            else:
                n_empty += 1
            w.writerow([rid, iid, ps])

    print(f"\nDone. {n_with} images with detections, {n_empty} empty.")
    print(f"  total dets (approx): {n_dets}  ({n_dets/max(1,len(rows_in)):.3f}/img)")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
