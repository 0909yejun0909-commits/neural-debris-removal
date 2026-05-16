"""
Step 8b — Kaggle: surgical FT iter-count sweep.

Background: step8 at 125 iters over-suppressed (max conf 0.55, 0 dets >= 0.6,
0.063 dets/img, scored 277.19 = tied with simple-FT raw). Conf flattening was
the failure mode, not the surgical-layer choice. Lesson 14 / lever #6 in
CLAUDE.md: empty-label FT iter count is the dominant conf-distribution lever.

Goal: find the iter count where surgical FT preserves a usable conf tail
(max conf >= 0.6) so the proven conf>=0.6 + dashedness rescue recipe applies.

Design:
- One training run to MAX_ITER=75.
- CHECKPOINT_PERIOD=25 saves model_0000024.pth, model_0000049.pth, model_final.pth.
- Inference pass on each → submission_iter{25,50,75}.csv.
- No step decay (we want apples-to-apples across iter counts; decay landed at
  iter 100 in step8, which is past our sweep range anyway).
- All other hyperparams match step8 (lr=1e-4, batch=4, random flips,
  surgical freeze: head.cls_subnet + head.cls_score only).

Outputs:
- /kaggle/working/submission_iter25.csv
- /kaggle/working/submission_iter50.csv
- /kaggle/working/submission_iter75.csv
- /kaggle/working/run.log  (tee'd stdout; see lesson 8)
- /kaggle/working/iter_summary.txt  (per-checkpoint: max/median conf, dets/img)
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


class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data); s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

_LOG_PATH = Path("/kaggle/working/run.log")
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_log_fh = open(_LOG_PATH, "w", buffering=1)
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


BASE_CONFIG          = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ANCHOR_SIZES         = [[16], [32], [64], [128], [256]]
NUM_CLASSES          = 1
CONF_THRESH          = 0.2
IMG_W = IMG_H        = 1024

FT_LR       = 1e-4
FT_BATCH    = 4
MAX_ITER    = 75
CKPT_PERIOD = 25      # → saves at iter 24, 49, 74 (0-indexed) + model_final.pth
SWEEP_ITERS = [25, 50, 75]


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
        {"file_name": str(p), "image_id": i,
         "height": IMG_H, "width": IMG_W, "annotations": []}
        for i, p in enumerate(paths)
    ]


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
        for p in model.parameters():
            p.requires_grad = False
        trainable = 0
        for _, p in model.head.cls_subnet.named_parameters():
            p.requires_grad = True; trainable += p.numel()
        for _, p in model.head.cls_score.named_parameters():
            p.requires_grad = True; trainable += p.numel()
        total = sum(p.numel() for p in model.parameters())
        print(f"Surgical FT: {trainable:,} trainable / {total:,} total "
              f"({100*trainable/total:.2f}%)")
        return model


def fine_tune():
    name = register()
    cfg = base_cfg()
    cfg.MODEL.WEIGHTS          = POISONED_WEIGHTS
    cfg.DATASETS.TRAIN         = (name,)
    cfg.DATASETS.TEST          = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH   = FT_BATCH
    cfg.SOLVER.BASE_LR         = FT_LR
    cfg.SOLVER.MAX_ITER        = MAX_ITER
    cfg.SOLVER.STEPS           = []        # no LR decay during sweep
    cfg.SOLVER.WARMUP_ITERS    = 0
    cfg.SOLVER.CHECKPOINT_PERIOD = CKPT_PERIOD
    cfg.OUTPUT_DIR             = str(OUT / "surgical_ft_sweep")
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    trainer = SurgicalTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return cfg.OUTPUT_DIR


def checkpoint_for_iter(out_dir, iter_n):
    """detectron2 saves at iter N-1 with zero-padded name model_{iter-1:07d}.pth.
    The MAX_ITER checkpoint is also saved as model_final.pth."""
    if iter_n == MAX_ITER:
        return str(Path(out_dir) / "model_final.pth")
    return str(Path(out_dir) / f"model_{iter_n - 1:07d}.pth")


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
        w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
        if w == 0 or h == 0:
            continue
        parts.append(f"{float(s):.6f} {x1:.2f} {y1:.2f} {w:.2f} {h:.2f}")
    return " ".join(parts) if parts else " "


def run_inference(weights, out_csv):
    """Returns (n_dets, n_nonempty, conf_array) for the summary."""
    predictor = build_predictor(weights)
    with open(SAMPLE_SUB) as f:
        reader = csv.DictReader(f)
        rows_in = [(r["id"], r["image_id"]) for r in reader]

    test_dir = Path(TEST_DIR)
    n_nonempty, all_confs = 0, []
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "image_id", "prediction_string"])
        for rid, iid in tqdm(rows_in, desc=Path(out_csv).name):
            ps = predict_one(predictor, test_dir / f"{iid}.png")
            if ps.strip():
                n_nonempty += 1
                # parse confs from the predication string (every 5th token)
                toks = ps.split()
                for i in range(0, len(toks), 5):
                    all_confs.append(float(toks[i]))
            w.writerow([rid, iid, ps])

    return len(all_confs), n_nonempty, np.array(all_confs)


def main():
    print(f"=== Step 8b: surgical FT iter-sweep ({SWEEP_ITERS}) ===")
    print(f"  lr={FT_LR}  batch={FT_BATCH}  max_iter={MAX_ITER}  "
          f"ckpt_period={CKPT_PERIOD}  no step decay")
    out_dir = fine_tune()
    print(f"Training complete. Checkpoints in {out_dir}")
    print("Available checkpoints:")
    for p in sorted(Path(out_dir).glob("model_*.pth")):
        print(f"  {p.name}")

    summary_lines = []
    summary_lines.append(f"{'iter':>5s} | {'n_dets':>7s} | {'n_img':>5s} | "
                         f"{'dets/img':>9s} | {'min':>6s} | {'med':>6s} | "
                         f"{'max':>6s} | {'>=0.4':>5s} | {'>=0.5':>5s} | {'>=0.6':>5s}")
    summary_lines.append("-" * 90)

    for iter_n in SWEEP_ITERS:
        weights = checkpoint_for_iter(out_dir, iter_n)
        if not Path(weights).exists():
            print(f"!! Missing checkpoint for iter={iter_n}: {weights}")
            continue
        print(f"\n=== Inference @ iter={iter_n} (weights={Path(weights).name}) ===")
        out_csv = OUT / f"submission_iter{iter_n}.csv"
        n_dets, n_nonempty, confs = run_inference(weights, out_csv)
        if len(confs) == 0:
            line = f"{iter_n:>5d} | {n_dets:>7d} | {n_nonempty:>5d} | {0.0:>9.3f} | (no dets)"
        else:
            line = (f"{iter_n:>5d} | {n_dets:>7d} | {n_nonempty:>5d} | "
                    f"{n_dets/2000:>9.3f} | {confs.min():>6.3f} | "
                    f"{np.median(confs):>6.3f} | {confs.max():>6.3f} | "
                    f"{(confs>=0.4).sum():>5d} | {(confs>=0.5).sum():>5d} | "
                    f"{(confs>=0.6).sum():>5d}")
        summary_lines.append(line)
        print(line)

    summary_path = OUT / "iter_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"\n=== Summary written to {summary_path} ===")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
