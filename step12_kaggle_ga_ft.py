"""
Step 12 — Kaggle: Gradient Ascent + Empty-label FT + Weight Average.

Based on `shared notebooks/improving-the-baseline-fine-tuning.ipynb`. Goal: beat
the simple-FT-based 235.62 floor by introducing a fundamentally different
unlearning mechanism (active poison suppression via GA) rather than yet another
empty-label FT recipe.

Why this might break the floor:
- Simple-FT + rescue (235.62) is a "predict less everywhere" model. Its remaining
  errors are poison residue that survived 20 iters of empty-label pressure.
- GA *targets* the poison annotations specifically with `loss = -loss_cls`,
  pushing the head's response down at exactly those locations rather than
  uniformly suppressing all detections.
- The weight-average of GA + FT keeps GA's poison-specific disruption while
  letting FT restore the all-detection-class background prior.

Pipeline:
  Phase A — Gradient Ascent
    - Head-only trainable (backbone + FPN frozen)
    - Load POISON ANNOTATIONS from annotations_coco.json
    - 30 iters, lr=5e-5, SGD momentum=0.9, grad clip 1.0
    - Loss = -loss_cls (only the cls term, regression untouched)
  Phase B — Empty-label FT
    - Head-only trainable
    - Starts FROM Phase A checkpoint
    - 150 iters with periodic checkpoints (sweep), lr=1e-4
    - Loss = standard (empty labels → only background-pushing)
  Phase C — Weight average + inference
    - For each FT checkpoint: averaged = 0.3*GA + 0.7*FT_iter
    - Run inference on test set, dump submission_iter{X}.csv

Iter sweep: FT checkpoints at {20, 50, 100, 150}. Lesson 14 says iter count is
the dominant lever for empty-label FT; we want the same exploration we got
with surgical (which showed iter=25 was peak).

Output (all under /kaggle/working/):
  phase_a_ga/model_ga.pth
  phase_b_ft/model_iter{N}.pth   for N in {20, 50, 100, 150}
  submission_iter{N}.csv
  iter_summary.txt               (n_dets, conf distribution by iter)
  run.log                        (tee of stdout — lesson 8)
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
import json
import os
from pathlib import Path

import cv2
import numpy as np
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
from detectron2.structures import BoxMode
from detectron2.utils.events import EventStorage
from tqdm import tqdm


# ── Stdout tee (lesson 8: kernel .log files come back 0 bytes) ────────────────
OUT = Path("/kaggle/working")
OUT.mkdir(exist_ok=True)
_log_fp = open(OUT / "run.log", "w", buffering=1)
_orig_stdout = sys.stdout
_orig_stderr = sys.stderr

class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, b):
        for s in self.streams:
            s.write(b); s.flush()
    def flush(self):
        for s in self.streams: s.flush()

sys.stdout = _Tee(_orig_stdout, _log_fp)
sys.stderr = _Tee(_orig_stderr, _log_fp)


def _assert_gpu():
    msg = ("GPU not available. In Kaggle: Settings -> Accelerator -> GPU T4 x2 "
           "(or P100), then RESTART the kernel and re-run.")
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
UNLEARN_JSON     = f"{UNLEARN_DIR}/annotations_coco.json"
_test_candidates = [Path(BASE_DIR) / "test_set" / "test_set", Path(BASE_DIR) / "test_set"]
TEST_DIR         = str(next(p for p in _test_candidates if p.is_dir() and any(p.glob("*.png"))))
SAMPLE_SUB       = f"{BASE_DIR}/sample_submission.csv"

OUTPUT_A = OUT / "phase_a_ga"
OUTPUT_B = OUT / "phase_b_ft"
OUTPUT_A.mkdir(exist_ok=True)
OUTPUT_B.mkdir(exist_ok=True)
print(f"  TEST_DIR  = {TEST_DIR}")
print(f"  OUTPUT_A  = {OUTPUT_A}")
print(f"  OUTPUT_B  = {OUTPUT_B}")


# Architecture must-match (CLAUDE.md). DO NOT touch.
BASE_CONFIG          = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ANCHOR_SIZES         = [[16], [32], [64], [128], [256]]
NUM_CLASSES          = 1
CONF_THRESH          = 0.2
IMG_W = IMG_H        = 1024

# Hyperparameters (notebook defaults)
GA_LR      = 5e-5
GA_ITERS   = 30
FT_LR      = 1e-4
FT_ITERS   = 150
FT_BATCH   = 4
GA_BATCH   = 4
GA_MIX     = 0.3      # mix weight for GA in averaged checkpoint

CHECKPOINT_ITERS = [20, 50, 100, 150]


# ── Image loading ─────────────────────────────────────────────────────────────
def load_image(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im.dtype == np.uint16:
        im = im.astype(np.float32) / 65535.0
    im = np.clip(im * 255, 0, 255).astype(np.float32)
    if im.ndim == 2:
        im = np.repeat(im[:, :, None], 3, axis=2)
    return im


# ── Dataset registration ──────────────────────────────────────────────────────
def get_poison_dataset():
    """Unlearn set WITH poison bbox annotations (for GA — push *away* from these)."""
    with open(UNLEARN_JSON) as f:
        coco = json.load(f)
    by_img = {im["id"]: {"file_name": str(Path(UNLEARN_DIR) / im["file_name"]),
                         "height": im["height"], "width": im["width"],
                         "image_id": im["id"], "annotations": []}
              for im in coco["images"]}
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        by_img[ann["image_id"]]["annotations"].append({
            "bbox":      [x, y, w, h],
            "bbox_mode": BoxMode.XYWH_ABS,
            "category_id": 0,
        })
    return list(by_img.values())


def get_empty_dataset():
    """Unlearn set with EMPTY annotations (for FT — predict nothing)."""
    paths = sorted(Path(UNLEARN_DIR).glob("*.png"))
    return [{"file_name": str(p), "image_id": i,
             "height": IMG_H, "width": IMG_W, "annotations": []}
            for i, p in enumerate(paths)]


class UInt16PoisonMapper(DatasetMapper):
    """16-bit PNG → tensor + real instances built from `annotations`."""
    def __call__(self, dataset_dict):
        d = copy.deepcopy(dataset_dict)
        im = load_image(d["file_name"])
        d["image"] = torch.as_tensor(im.transpose(2, 0, 1).copy())
        anns = [{"bbox": a["bbox"], "bbox_mode": a["bbox_mode"], "category_id": a["category_id"]}
                for a in d.get("annotations", [])]
        d["instances"] = utils.annotations_to_instances(anns, im.shape[:2])
        return d


class UInt16EmptyMapper(DatasetMapper):
    """16-bit PNG → tensor + EMPTY instances."""
    def __call__(self, dataset_dict):
        d = copy.deepcopy(dataset_dict)
        im = load_image(d["file_name"])
        d["image"] = torch.as_tensor(im.transpose(2, 0, 1).copy())
        d["instances"] = utils.annotations_to_instances([], im.shape[:2])
        return d


def register():
    for name, builder in [("unlearn_poison", get_poison_dataset),
                          ("unlearn_empty",  get_empty_dataset)]:
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
        DatasetCatalog.register(name, builder)
        MetadataCatalog.get(name).thing_classes = ["streak"]


# ── Cfg builder ───────────────────────────────────────────────────────────────
def base_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.MODEL.DEVICE                         = DEVICE
    cfg.MODEL.RETINANET.NUM_CLASSES          = NUM_CLASSES
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [ANCHOR_ASPECT_RATIOS]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES         = ANCHOR_SIZES
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST    = CONF_THRESH
    return cfg


def freeze_backbone_and_fpn(model):
    frozen, trainable = 0, 0
    for name, p in model.named_parameters():
        if "backbone" in name or "fpn" in name:
            p.requires_grad = False
            frozen += 1
        else:
            trainable += 1
    print(f"  frozen params: {frozen}  trainable params: {trainable}")


# ── Phase A: Gradient Ascent ──────────────────────────────────────────────────
def phase_a_gradient_ascent():
    print("=" * 60)
    print("PHASE A: Gradient Ascent on poison annotations")
    print("=" * 60)
    cfg = base_cfg()
    cfg.MODEL.WEIGHTS         = POISONED_WEIGHTS
    cfg.DATASETS.TRAIN        = ("unlearn_poison",)
    cfg.DATASETS.TEST         = ()
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH  = GA_BATCH
    cfg.SOLVER.BASE_LR        = GA_LR
    cfg.SOLVER.MAX_ITER       = GA_ITERS
    cfg.OUTPUT_DIR            = str(OUTPUT_A)

    model = build_model(cfg)
    DetectionCheckpointer(model).load(POISONED_WEIGHTS)
    model.train()
    freeze_backbone_and_fpn(model)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=GA_LR, momentum=0.9, weight_decay=1e-4)

    dataset_dicts = DatasetCatalog.get("unlearn_poison")
    mapper = UInt16PoisonMapper(cfg, is_train=True, augmentations=[])
    loader = build_detection_train_loader(cfg, mapper=mapper, dataset=dataset_dicts)
    it = iter(loader)

    with EventStorage() as storage:
        for i in range(GA_ITERS):
            storage.step()
            batch = next(it)
            optimizer.zero_grad()
            loss_dict = model(batch)
            loss = -loss_dict["loss_cls"]   # NEGATE the cls loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  GA iter {i+1:3d}/{GA_ITERS}  -loss_cls = {loss.item():.4f}  "
                      f"loss_box_reg = {loss_dict.get('loss_box_reg', torch.tensor(0.)).item():.4f}")

    ga_path = OUTPUT_A / "model_ga.pth"
    torch.save({"model": model.state_dict()}, ga_path)
    print(f"Phase A done. Saved {ga_path}")
    return str(ga_path)


# ── Phase B: Empty-label FT with checkpoints ──────────────────────────────────
def phase_b_ft_with_checkpoints(ga_weights):
    print("\n" + "=" * 60)
    print(f"PHASE B: Empty-label FT (sweep checkpoints at {CHECKPOINT_ITERS})")
    print("=" * 60)
    cfg = base_cfg()
    cfg.MODEL.WEIGHTS         = ga_weights
    cfg.DATASETS.TRAIN        = ("unlearn_empty",)
    cfg.DATASETS.TEST         = ()
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH  = FT_BATCH
    cfg.SOLVER.BASE_LR        = FT_LR
    cfg.SOLVER.MAX_ITER       = FT_ITERS
    cfg.OUTPUT_DIR            = str(OUTPUT_B)

    model = build_model(cfg)
    DetectionCheckpointer(model).load(ga_weights)
    model.train()
    freeze_backbone_and_fpn(model)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=FT_LR, momentum=0.9, weight_decay=1e-4)

    dataset_dicts = DatasetCatalog.get("unlearn_empty")
    mapper = UInt16EmptyMapper(cfg, is_train=True, augmentations=[])
    loader = build_detection_train_loader(cfg, mapper=mapper, dataset=dataset_dicts)
    it = iter(loader)

    saved_paths = {}
    with EventStorage() as storage:
        for i in range(FT_ITERS):
            storage.step()
            batch = next(it)
            optimizer.zero_grad()
            loss_dict = model(batch)
            loss = sum(loss_dict.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            step = i + 1
            if step % 10 == 0 or step == 1:
                print(f"  FT iter {step:3d}/{FT_ITERS}  total={loss.item():.4f}  "
                      f"cls={loss_dict['loss_cls'].item():.4f}  "
                      f"reg={loss_dict.get('loss_box_reg', torch.tensor(0.)).item():.4f}")
            if step in CHECKPOINT_ITERS:
                p = OUTPUT_B / f"model_iter{step}.pth"
                torch.save({"model": model.state_dict()}, p)
                saved_paths[step] = str(p)
                print(f"    → checkpoint saved: {p}")
    return saved_paths


# ── Phase C: Weight-average + inference ───────────────────────────────────────
def average_weights(ga_path, ft_path, mix):
    """avg = mix*GA + (1-mix)*FT for matching keys, else FT."""
    w_ga = torch.load(ga_path, map_location="cpu")["model"]
    w_ft = torch.load(ft_path, map_location="cpu")["model"]
    out = {}
    for k, v_ft in w_ft.items():
        v_ga = w_ga.get(k)
        if v_ga is not None and v_ga.shape == v_ft.shape and v_ft.is_floating_point():
            out[k] = mix * v_ga.float() + (1 - mix) * v_ft.float()
        else:
            out[k] = v_ft
    return out


def build_predictor_from_state(state_dict, tmp_path):
    torch.save({"model": state_dict}, tmp_path)
    cfg = base_cfg()
    cfg.MODEL.WEIGHTS = str(tmp_path)
    return DefaultPredictor(cfg)


def predict_one(predictor, path):
    im = load_image(path)
    out = predictor(im)["instances"].to("cpu")
    boxes  = out.pred_boxes.tensor.numpy()
    scores = out.scores.numpy()
    parts = []
    for (x1, y1, x2, y2), s in zip(boxes, scores):
        x1 = float(np.clip(x1, 0, IMG_W)); y1 = float(np.clip(y1, 0, IMG_H))
        x2 = float(np.clip(x2, 0, IMG_W)); y2 = float(np.clip(y2, 0, IMG_H))
        w  = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
        if w == 0 or h == 0:
            continue
        parts.append(f"{float(s):.6f} {x1:.2f} {y1:.2f} {w:.2f} {h:.2f}")
    return (" ".join(parts) if parts else " "), float(np.max(scores) if len(scores) else 0.0), \
           [float(s) for s in scores]


def run_inference(state_dict, tag, sample_rows, summary):
    print(f"\n--- Inference for {tag} ---")
    tmp = OUT / f"_tmp_{tag}.pth"
    predictor = build_predictor_from_state(state_dict, tmp)
    test_dir = Path(TEST_DIR)
    all_scores = []
    n_with, n_empty = 0, 0
    out_path = OUT / f"submission_{tag}.csv"
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "image_id", "prediction_string"])
        for rid, iid in tqdm(sample_rows, desc=f"infer {tag}"):
            p = test_dir / f"{iid}.png"
            ps, _, scores = predict_one(predictor, p)
            if ps.strip():
                n_with += 1
                all_scores.extend(scores)
            else:
                n_empty += 1
            w.writerow([rid, iid, ps])
    tmp.unlink(missing_ok=True)

    if all_scores:
        arr = np.array(all_scores)
        line = (f"{tag:14s} | n_dets={len(arr):5d} n_img={n_with:4d} "
                f"dets/img={len(arr)/len(sample_rows):.3f} | "
                f"min={arr.min():.3f} med={np.median(arr):.3f} max={arr.max():.3f} | "
                f">=0.4={int((arr>=0.4).sum()):4d} >=0.5={int((arr>=0.5).sum()):4d} "
                f">=0.6={int((arr>=0.6).sum()):4d}")
    else:
        line = f"{tag:14s} | no detections"
    print(line)
    summary.append(line)
    return out_path


def main():
    register()

    ga_path  = phase_a_gradient_ascent()
    ft_paths = phase_b_ft_with_checkpoints(ga_path)

    with open(SAMPLE_SUB) as f:
        reader = csv.DictReader(f)
        sample_rows = [(r["id"], r["image_id"]) for r in reader]

    summary = []

    # Baseline: GA-only (mix=1.0) — sanity check, expected to be poor
    # Baseline: pure-FT (mix=0.0) at each iter — see if GA preprocessing alone helps
    # Main: averaged (mix=GA_MIX=0.3) at each iter
    for iter_n, ft_p in ft_paths.items():
        for mix, label in [(0.0, "pureFT"), (GA_MIX, "avg0.3")]:
            w = average_weights(ga_path, ft_p, mix)
            tag = f"iter{iter_n}_{label}"
            run_inference(w, tag, sample_rows, summary)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for line in summary:
        print(line)
    with open(OUT / "iter_summary.txt", "w") as f:
        f.write("\n".join(summary) + "\n")
    print(f"\nWrote /kaggle/working/iter_summary.txt")


if __name__ == "__main__":
    main()
