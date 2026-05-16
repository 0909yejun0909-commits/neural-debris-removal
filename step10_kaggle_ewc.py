"""
Step 10 — Kaggle: L2-anchored unlearning ("EWC-lite") with iter sweep.

Ported from `shared notebooks/neural-debris-removal-updated-apr-26.ipynb`.

Key differences from the source notebook:
1. **Fix anchor-size bug** — notebook has [[32],[64],[128],[256],[512]] which
   mismatches the poisoned head's weights (random init → garbage). Restore
   [[16],[32],[64],[128],[256]] (CLAUDE.md must-match values).
2. Iter sweep (checkpoint at 25/50/75/125) so we can read the conf
   distribution at each stage in a single kernel run. step8b showed iter
   count is the dominant lever; reuse that play.
3. Tee stdout to run.log (lesson 8).
4. No CONF_DISCOUNT at inference (lesson 5).
5. Inference at conf > 0.2; rescue applied locally afterward.

About the method: the notebook calls this "EWC" but it's actually uniform
L2-anchor regularization — there's no Fisher information matrix. The loss is
    L = empty_label_focal_loss(model) + EWC_LAMBDA * sum((w - w_orig)^2)
over the trainable params (only head.cls_score). This still prevents the
catastrophic-forgetting failure mode that broke step8 @ 125 iters: instead of
"predict background everywhere," the L2 anchor pulls weights back toward the
original. The hyperparameter knob is EWC_LAMBDA — author's rule:
  too many FPs → lower λ (more weight drift → more aggressive unlearning)
  too many FNs → raise λ (less drift → preserve original detection capacity)
Our regime is FN-dominated (235.62 best at 0.315 dets/img), so if this fails
on the FN side we'd try λ=200 or 500. λ=100 is the author's default starting
point.

Outputs:
- /kaggle/working/submission_iter{25,50,75,125}.csv
- /kaggle/working/iter_summary.txt
- /kaggle/working/run.log
"""

import subprocess, sys, time

def _pip(*args, retries=2):
    cmd = [sys.executable, "-m", "pip", "install", "-q", *args]
    for attempt in range(retries + 1):
        try:
            subprocess.run(cmd, check=True); return
        except subprocess.CalledProcessError:
            if attempt == retries:
                raise SystemExit(
                    f"pip install failed: {' '.join(args)}\n"
                    "If this is a git+https URL, the Kaggle kernel likely has no "
                    "Internet access. Settings -> Internet -> On, restart kernel."
                )
            time.sleep(2)

_pip("setuptools<81")
_pip("git+https://github.com/facebookresearch/detectron2.git")

import copy
import csv
import json
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
from detectron2.utils.events import EventStorage
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
    msg = ("GPU not available. Settings -> Accelerator -> GPU T4 x2 "
           "(or P100), restart kernel.")
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


# Architecture (MUST match poisoned head — bug-fixed anchors)
BASE_CONFIG          = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ANCHOR_SIZES         = [[16], [32], [64], [128], [256]]   # NOT [[32],..,[512]] (notebook bug)
NUM_CLASSES          = 1
CONF_THRESH          = 0.2
IMG_W = IMG_H        = 1024

# EWC-lite hyperparams (from updated-apr-26.ipynb)
MAX_ITER     = 125
EWC_LAMBDA   = 100.0     # author's default; 500 was their fallback
LR           = 1e-4
BATCH        = 4
GRAD_CLIP    = 1.0
WEIGHT_DECAY = 1e-5
SWEEP_ITERS  = [25, 50, 75, 125]
CKPT_DIR     = OUT / "ewc_ckpts"
CKPT_DIR.mkdir(exist_ok=True)


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


# Aggressive augmentation from updated-apr-26: flips + 90/180/270 rotations.
class EWCMapper(DatasetMapper):
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
            k = np.random.randint(0, 4)
            if k > 0:
                im = np.rot90(im, k=k, axes=(0, 1))
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
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS  = False
    cfg.DATALOADER.NUM_WORKERS               = 2
    cfg.SOLVER.IMS_PER_BATCH                 = BATCH
    cfg.DATASETS.TRAIN                       = (register(),)
    cfg.DATASETS.TEST                        = ()
    return cfg


def train_with_ewc(cfg):
    """Custom training loop with L2-anchor penalty. Saves checkpoints at each
    iter in SWEEP_ITERS. Returns dict {iter_n: checkpoint_path}."""
    model = build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(POISONED_WEIGHTS)
    model.train()

    # Surgical freeze: only head.cls_score is trainable.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.cls_score.parameters():
        p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count  = sum(p.numel() for p in trainable_params)
    total_count      = sum(p.numel() for p in model.parameters())
    print(f"EWC trainable: {trainable_count:,} / {total_count:,} "
          f"({100*trainable_count/total_count:.3f}%)  — head.cls_score only")

    # EWC anchor: snapshot of original trainable weights.
    orig_weights = {
        name: p.clone().detach()
        for name, p in model.named_parameters() if p.requires_grad
    }

    optimizer = torch.optim.AdamW(
        trainable_params, lr=LR, weight_decay=WEIGHT_DECAY
    )
    mapper = EWCMapper(cfg, is_train=True)
    data_loader = build_detection_train_loader(cfg, mapper=mapper)
    data_iter = iter(data_loader)

    sweep_set = set(SWEEP_ITERS)
    saved = {}

    print(f"Starting EWC unlearning: MAX_ITER={MAX_ITER}  LAMBDA={EWC_LAMBDA}  "
          f"LR={LR}  BATCH={BATCH}  clip={GRAD_CLIP}")

    with EventStorage() as storage:
        for it in range(1, MAX_ITER + 1):
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                data = next(data_iter)

            loss_dict     = model(data)
            standard_loss = sum(loss_dict.values())

            ewc_loss = 0.0
            for name, p in model.named_parameters():
                if p.requires_grad:
                    ewc_loss = ewc_loss + torch.sum((p - orig_weights[name]) ** 2)

            total_loss = standard_loss + EWC_LAMBDA * ewc_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=GRAD_CLIP)
            optimizer.step()

            storage.put_scalar("total_loss", float(total_loss.item()))

            if it in sweep_set:
                ckpt = CKPT_DIR / f"model_iter{it}.pth"
                # Save just the model weights (detectron2 checkpointer adds the suffix)
                DetectionCheckpointer(model, save_dir=str(CKPT_DIR)).save(f"model_iter{it}")
                # DetectionCheckpointer.save writes "model_iter{it}.pth" inside save_dir
                saved[it] = str(CKPT_DIR / f"model_iter{it}.pth")
                print(f"  iter {it:3d}  std_loss={float(standard_loss.item()):.5f}  "
                      f"ewc_loss={float(ewc_loss.item()) if torch.is_tensor(ewc_loss) else ewc_loss:.5f}  "
                      f"total={float(total_loss.item()):.5f}  → saved {ckpt.name}")

    return saved


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
                toks = ps.split()
                for i in range(0, len(toks), 5):
                    all_confs.append(float(toks[i]))
            w.writerow([rid, iid, ps])
    return len(all_confs), n_nonempty, np.array(all_confs)


def main():
    print(f"=== Step 10: EWC-lite (L2-anchor) iter-sweep {SWEEP_ITERS} ===")
    print(f"  EWC_LAMBDA={EWC_LAMBDA}  trainable=head.cls_score only "
          f"(true single-layer surgical, vs step8's cls_subnet+cls_score)")

    cfg = base_cfg()
    saved = train_with_ewc(cfg)
    print(f"\nTraining complete. Saved checkpoints: {list(saved.keys())}")

    summary = [
        f"{'iter':>5s} | {'n_dets':>7s} | {'n_img':>5s} | {'dets/img':>9s} | "
        f"{'min':>6s} | {'med':>6s} | {'max':>6s} | {'>=0.4':>5s} | "
        f"{'>=0.5':>5s} | {'>=0.6':>5s}",
        "-" * 95,
    ]
    for it in SWEEP_ITERS:
        weights = saved.get(it)
        if weights is None or not Path(weights).exists():
            print(f"!! Missing checkpoint for iter={it}")
            continue
        print(f"\n=== Inference @ iter={it} ===")
        out_csv = OUT / f"submission_iter{it}.csv"
        n_dets, n_nonempty, confs = run_inference(weights, out_csv)
        if len(confs) == 0:
            line = f"{it:>5d} | {n_dets:>7d} | {n_nonempty:>5d} | {0.0:>9.3f} | (no dets)"
        else:
            line = (f"{it:>5d} | {n_dets:>7d} | {n_nonempty:>5d} | "
                    f"{n_dets/2000:>9.3f} | {confs.min():>6.3f} | "
                    f"{np.median(confs):>6.3f} | {confs.max():>6.3f} | "
                    f"{(confs>=0.4).sum():>5d} | {(confs>=0.5).sum():>5d} | "
                    f"{(confs>=0.6).sum():>5d}")
        summary.append(line)
        print(line)

    (OUT / "iter_summary.txt").write_text("\n".join(summary) + "\n")
    print("\n=== iter_summary.txt ===")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
