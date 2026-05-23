"""
Step 25 (Phase 3): Surgical FT augmented with EMBEDDING HINGE LOSS.

Core idea: every patch-based POST-processing method plateaus at ~13% poison
concentration in its top tail (lesson: cls_subnet T=0.96 -> 226.31 floor;
DINOv2-small/base, NCC, width_var all max ~10-15% too). Use the proven
cls_subnet[-1] similarity signal AT TRAINING TIME — push the head to NOT
produce high-cos-sim-to-poison embeddings on test detections.

Loss:
    L = L_focal(unlearn, empty)   # standard surgical FT regularizer
      + LAMBDA * mean over test-batch dets of  ReLU(max_cos_sim_to_templates - THRESH)

The hinge means: predictions with cos_sim ≤ 0.96 are untouched; predictions
with cos_sim > 0.96 get gradient pushing their embedding away from poison
templates.  If the head's high-conf dets at sim≥0.96 are dominantly poison
(lesson: dropping them post-hoc gains +5.3 pts), then training-time
suppression of those locations should propagate further than post-hoc filtering.

Critical design constraints:
- FT_ITERS = 25 (lesson 14: empty-label FT collapses with too many iters)
- Hook captures cls_subnet[-1]; templates frozen at iter 0
- Test mini-batch: 4 images per iter for the embedding loss term
- All other architecture parameters match step8 (anchor sizes/aspects must match poisoned head)

Deploy on Kaggle as new kernel. Requires GPU.
"""

import subprocess, sys, time

def _pip(*args, retries=2):
    cmd = [sys.executable, "-m", "pip", "install", "-q", *args]
    for attempt in range(retries + 1):
        try:
            subprocess.run(cmd, check=True); return
        except subprocess.CalledProcessError:
            if attempt == retries: raise SystemExit(f"pip install failed: {' '.join(args)}")
            time.sleep(2)

_pip("setuptools<81")
_pip("git+https://github.com/facebookresearch/detectron2.git")

import copy, csv, json, random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    DatasetCatalog, DatasetMapper, MetadataCatalog,
    build_detection_train_loader, detection_utils as utils,
)
from detectron2.engine import DefaultPredictor


# Tee logs
class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data); s.flush()
    def flush(self):
        for s in self.streams: s.flush()

_LOG_PATH = Path("/kaggle/working/run.log")
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_log_fh = open(_LOG_PATH, "w", buffering=1)
sys.stdout = _Tee(sys.__stdout__, _log_fh)
sys.stderr = _Tee(sys.__stderr__, _log_fh)


def _assert_gpu():
    if not torch.cuda.is_available():
        raise SystemExit("GPU required. Settings -> Accelerator -> GPU")
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")

_assert_gpu()
DEVICE = "cuda"


def find_base_dir():
    cands = list(Path("/kaggle/input").rglob("poisoned_model.pth"))
    if not cands: raise FileNotFoundError("poisoned_model.pth not found")
    return str(cands[0].parent.parent)


BASE_DIR         = find_base_dir()
POISONED_WEIGHTS = f"{BASE_DIR}/poisoned_model/poisoned_model.pth"
UNLEARN_DIR      = f"{BASE_DIR}/unlearn_set"
UNLEARN_JSON     = f"{BASE_DIR}/unlearn_set/annotations_coco.json"
_test_cands = [Path(BASE_DIR)/"test_set"/"test_set", Path(BASE_DIR)/"test_set"]
TEST_DIR    = str(next(p for p in _test_cands if p.is_dir() and any(p.glob("*.png"))))
SAMPLE_SUB  = f"{BASE_DIR}/sample_submission.csv"
OUT         = Path("/kaggle/working")
OUT.mkdir(exist_ok=True)
print(f"BASE_DIR={BASE_DIR}  TEST_DIR={TEST_DIR}")


# Architecture must match poisoned head exactly
BASE_CONFIG          = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ANCHOR_SIZES         = [[16], [32], [64], [128], [256]]
NUM_CLASSES          = 1
CONF_THRESH          = 0.2
IMG_W = IMG_H        = 1024

# Phase 3 hyperparams — narrow sweep
FT_ITERS         = 25            # lesson 14: avoid empty-label collapse
FT_LR            = 1e-4
LAMBDA_EMB       = 1.0           # weight of embedding hinge loss
EMB_THRESH       = 0.96          # same threshold as proven inference filter
TEST_BATCH_SIZE  = 4             # test imgs per iter for emb loss
TEST_CACHE_N     = 200           # cache subset of test imgs for speed
TOP_K_DETS       = 10            # top-K predictions per image for emb loss


def load_image(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None: return None
    if im.dtype == np.uint16: im = im.astype(np.float32) / 65535.0
    im = np.clip(im * 255, 0, 255).astype(np.float32)
    if im.ndim == 2: im = np.repeat(im[:, :, None], 3, axis=2)
    return im


def get_unlearn_dataset():
    paths = sorted(Path(UNLEARN_DIR).glob("*.png"))
    return [{"file_name": str(p), "image_id": i, "height": IMG_H, "width": IMG_W,
             "annotations": []} for i, p in enumerate(paths)]


class SurgicalMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train=is_train, augmentations=[])
        self.is_train = is_train

    def __call__(self, d):
        d = copy.deepcopy(d)
        im = load_image(d["file_name"])
        if self.is_train and np.random.rand() > 0.5: im = np.fliplr(im)
        if self.is_train and np.random.rand() > 0.5: im = np.flipud(im)
        d["image"] = torch.as_tensor(im.transpose(2,0,1).copy())
        d["instances"] = utils.annotations_to_instances([], im.shape[:2])
        return d


def register():
    name = "unlearn_empty"
    if name in DatasetCatalog.list(): DatasetCatalog.remove(name)
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


def get_level_for_bbox(bbox):
    x1, y1, x2, y2 = bbox
    s = np.sqrt(max(1, (x2-x1)*(y2-y1)))
    return int(np.clip(int(np.floor(4 + np.log2(s/224 + 1e-6))), 3, 7))


class EmbHook:
    """Captures cls_subnet[-1] output during forward."""
    def __init__(self, model):
        self.fmaps = []
        model.head.cls_subnet[-1].register_forward_hook(self._hook)
    def _hook(self, m, i, o): self.fmaps.append(o)
    def reset(self): self.fmaps = []


def extract_emb_at_bbox(fmap_dict, bbox_xyxy, fpn_levels=(3,4,5,6,7)):
    """Extract pooled cls_subnet[-1] embedding at bbox via ROI align. fmap_dict[level]->feature map."""
    x1, y1, x2, y2 = bbox_xyxy
    lvl = get_level_for_bbox(bbox_xyxy)
    lvl = max(min(lvl, 7), 3)
    fmap = fmap_dict.get(lvl, fmap_dict[min(fmap_dict.keys())])
    rois = torch.tensor([[0, x1, y1, x2, y2]], dtype=torch.float32, device=fmap.device)
    rois[:, 1:] = rois[:, 1:] / (2 ** lvl)
    feat = roi_align(fmap, rois, output_size=(1,1), spatial_scale=1.0, aligned=True)
    return feat.view(-1)  # (256,)


def build_poison_templates(model, hook):
    """Run model on each unlearn image, extract cls_subnet[-1] emb at each poison ann bbox.
    Returns (N, 256) tensor on DEVICE, L2-normalized, NO GRADIENT."""
    print("Building 20 poison templates (frozen)...")
    with open(UNLEARN_JSON) as f: coco = json.load(f)
    id2fname = {im["id"]: im["file_name"] for im in coco["images"]}
    img2anns = {}
    for ann in coco["annotations"]: img2anns.setdefault(ann["image_id"], []).append(ann)
    templates = []
    model.eval()
    with torch.no_grad():
        for img_id, anns in img2anns.items():
            img = load_image(Path(UNLEARN_DIR) / id2fname[img_id])
            tens = torch.from_numpy(img[:,:,::-1].copy().transpose(2,0,1)).to(DEVICE)
            hook.reset()
            model([{"image": tens}])
            # cls_subnet runs at 5 FPN levels: order matches FPN P3..P7
            fmap_dict = {3+i: f for i, f in enumerate(hook.fmaps)}
            for ann in anns:
                x, y, w, h = ann["bbox"]
                emb = extract_emb_at_bbox(fmap_dict, [x, y, x+w, y+h])
                n = emb.norm()
                if n > 1e-6: emb = emb / n
                templates.append(emb.detach())
    templates = torch.stack(templates, dim=0).to(DEVICE)
    # Self-sim diagnostic
    sim = templates @ templates.T
    off = sim[~torch.eye(len(templates), dtype=torch.bool, device=DEVICE)]
    print(f"  templates: {templates.shape}  off-diag med={off.median().item():.4f} max={off.max().item():.4f} std={off.std().item():.4f}")
    return templates.detach()


def cache_test_images(n=TEST_CACHE_N):
    """Load N test image tensors (no augmentation, no grad)."""
    paths = sorted(Path(TEST_DIR).glob("*.png"))[:n]
    cached = []
    for p in paths:
        img = load_image(p)
        tens = torch.from_numpy(img[:,:,::-1].copy().transpose(2,0,1)).to(DEVICE)
        cached.append((p.stem, tens))
    print(f"Cached {len(cached)} test images")
    return cached


def emb_hinge_loss(model, hook, test_imgs_batch, templates, thresh=EMB_THRESH, top_k=TOP_K_DETS):
    """For each img in batch: forward, get top-K dets, compute embeddings, hinge loss on sim > thresh."""
    losses = []
    for img_id, tens in test_imgs_batch:
        hook.reset()
        # Forward in eval mode for proper inference output
        was_training = model.training
        model.eval()
        with torch.set_grad_enabled(True):
            outputs = model([{"image": tens}])
        if was_training: model.train()
        fmap_dict = {3+i: f for i, f in enumerate(hook.fmaps)}
        # Get predicted detections
        if "instances" not in outputs[0]: continue
        inst = outputs[0]["instances"]
        if len(inst) == 0: continue
        # Take top-K by score
        topk_idx = inst.scores.topk(min(top_k, len(inst))).indices
        boxes = inst.pred_boxes.tensor[topk_idx]  # (k, 4) xyxy
        # Extract embeddings
        for bbox in boxes:
            bbox_list = bbox.detach().cpu().tolist()
            emb = extract_emb_at_bbox(fmap_dict, bbox_list)
            n = emb.norm()
            if n < 1e-6: continue
            emb = emb / n
            max_sim = (templates @ emb).max()  # scalar
            losses.append(F.relu(max_sim - thresh))
    if not losses:
        return torch.tensor(0.0, device=DEVICE, requires_grad=True)
    return torch.stack(losses).mean()


def main():
    print("=== Step 25: Embedding hinge loss FT ===")
    print(f"  iters={FT_ITERS}  lr={FT_LR}  λ={LAMBDA_EMB}  thresh={EMB_THRESH}")

    name = register()
    cfg = base_cfg()
    cfg.MODEL.WEIGHTS       = POISONED_WEIGHTS
    cfg.DATASETS.TRAIN      = (name,)
    cfg.DATASETS.TEST       = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH   = 4
    cfg.SOLVER.BASE_LR         = FT_LR
    cfg.SOLVER.MAX_ITER        = FT_ITERS
    cfg.SOLVER.STEPS           = ()
    cfg.SOLVER.GAMMA           = 1.0
    cfg.SOLVER.WARMUP_ITERS    = 0

    model = build_model(cfg)
    DetectionCheckpointer(model).load(POISONED_WEIGHTS)
    # Freeze all, unfreeze cls_subnet + cls_score
    for p in model.parameters(): p.requires_grad = False
    trainable_params = []
    for n, p in model.head.cls_subnet.named_parameters():
        p.requires_grad = True; trainable_params.append(p)
    for n, p in model.head.cls_score.named_parameters():
        p.requires_grad = True; trainable_params.append(p)
    print(f"Trainable: {sum(p.numel() for p in trainable_params):,}")

    # Hook for embedding extraction
    hook = EmbHook(model)

    # Build templates BEFORE training (from poisoned model)
    templates = build_poison_templates(model, hook)

    # Cache test image subset for embedding loss
    test_cache = cache_test_images()

    # Setup optimizer
    optimizer = torch.optim.SGD(trainable_params, lr=FT_LR, momentum=0.9)

    # Training loop
    train_loader = build_detection_train_loader(cfg, mapper=SurgicalMapper(cfg, is_train=True),
                                                 dataset=DatasetCatalog.get(name))

    model.train()
    iter_loader = iter(train_loader)
    for it in range(FT_ITERS):
        # --- Step A: empty-label focal loss on unlearn batch ---
        batch = next(iter_loader)
        hook.reset()
        loss_dict = model(batch)
        loss_focal = sum(loss_dict.values())
        # --- Step B: embedding hinge loss on test mini-batch ---
        test_batch = random.sample(test_cache, TEST_BATCH_SIZE)
        loss_emb = emb_hinge_loss(model, hook, test_batch, templates)
        total = loss_focal + LAMBDA_EMB * loss_emb
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        print(f"  iter {it+1}/{FT_ITERS}  focal={loss_focal.item():.4f}  emb_hinge={loss_emb.item():.4f}  total={total.item():.4f}")

    # Save model
    ckpt_path = OUT / "step25_emb_loss_model.pth"
    torch.save({"model": model.state_dict()}, ckpt_path)
    print(f"Saved -> {ckpt_path}")

    # Inference on test set
    print("\n=== Inference on test set ===")
    pred_cfg = base_cfg()
    pred_cfg.MODEL.WEIGHTS = str(ckpt_path)
    predictor = DefaultPredictor(pred_cfg)

    with open(SAMPLE_SUB) as f:
        rows_in = [(r["id"], r["image_id"]) for r in csv.DictReader(f)]

    test_dir = Path(TEST_DIR)
    n_with, n_dets = 0, 0
    out_path = OUT / "submission.csv"
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "image_id", "prediction_string"])
        for ridx, (rid, iid) in enumerate(rows_in):
            im = load_image(test_dir / f"{iid}.png")
            inst = predictor(im)["instances"].to("cpu")
            boxes = inst.pred_boxes.tensor.numpy()
            scores = inst.scores.numpy()
            parts = []
            for (x1, y1, x2, y2), s in zip(boxes, scores):
                x1 = float(np.clip(x1, 0, IMG_W)); y1 = float(np.clip(y1, 0, IMG_H))
                x2 = float(np.clip(x2, 0, IMG_W)); y2 = float(np.clip(y2, 0, IMG_H))
                ww = max(0.0, x2-x1); hh = max(0.0, y2-y1)
                if ww == 0 or hh == 0: continue
                parts.append(f"{float(s):.6f} {x1:.2f} {y1:.2f} {ww:.2f} {hh:.2f}")
            ps = " ".join(parts) if parts else " "
            if parts: n_with += 1; n_dets += len(parts)
            w.writerow([rid, iid, ps])
            if (ridx+1) % 200 == 0: print(f"  pred {ridx+1}/{len(rows_in)}")

    print(f"\nDone. {n_with} non-empty, {n_dets} total dets ({n_dets/len(rows_in):.3f}/img)")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
