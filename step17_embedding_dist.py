
import subprocess
import os

# Install Detectron2 (needed for building the model architecture)
# Using a specific version that matches the training environment as closely as possible
subprocess.run(["pip", "install", "-q", "setuptools<81"], check=True)
subprocess.run(["pip", "install", "-q", "git+https://github.com/facebookresearch/detectron2.git"], check=True)

import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2 import model_zoo
from torchvision.ops import roi_align

# ── Paths ──────────────────────────────────────────────────────────────────────
def find_base_dir():
    candidates = list(Path("/kaggle/input").rglob("poisoned_model.pth"))
    if not candidates:
        return "/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models"
    return str(candidates[0].parent.parent)

BASE_DIR         = find_base_dir()
POISONED_WEIGHTS = f"{BASE_DIR}/poisoned_model/poisoned_model.pth"
UNLEARN_DIR      = f"{BASE_DIR}/unlearn_set"
UNLEARN_JSON     = f"{UNLEARN_DIR}/annotations_coco.json"

_test_candidates = [Path(BASE_DIR) / "test_set" / "test_set", Path(BASE_DIR) / "test_set"]
TEST_DIR         = str(next((p for p in _test_candidates if p.is_dir() and any(p.glob("*.png"))), _test_candidates[0]))

# Using the current best baseline CSV
BEST_CSV         = "/kaggle/input/debris-best-csvs/filter_length_uncond_stack_le40_or_45_51.csv"

OUT_DIR          = Path("/kaggle/working/step17_embedding")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Architecture ───────────────────────────────────────────────────────────────
BASE_CONFIG          = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ANCHOR_SIZES         = [[16], [32], [64], [128], [256]]
NUM_CLASSES          = 1

# ── Feature Extraction Setup ──────────────────────────────────────────────────
class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = None
        # Hook on the last layer of cls_subnet
        # In Detectron2 RetinaNet, head.cls_subnet is a Sequential of 4 Conv+ReLU
        self.model.head.cls_subnet[-1].register_forward_hook(self.hook_fn)
        
    def hook_fn(self, module, input, output):
        self.features = output

def get_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.MODEL.RETINANET.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.RETINANET.ANCHOR_SIZES = ANCHOR_SIZES
    cfg.MODEL.RETINANET.ASPECT_RATIOS = ANCHOR_ASPECT_RATIOS
    cfg.MODEL.WEIGHTS = POISONED_WEIGHTS
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = build_model(cfg)
    DetectionCheckpointer(model).load(POISONED_WEIGHTS)
    model.eval()
    return model, cfg

def read_16bit(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None: return None
    if im.dtype == np.uint16:
        im = im.astype(np.float32) / 65535.0
    im = np.clip(im * 255, 0, 255).astype(np.float32)
    if im.ndim == 2:
        im = np.repeat(im[:, :, None], 3, axis=2)
    return im

def extract_embeddings(model, extractor, img_path, bboxes):
    \"\"\"
    bboxes: List of [x1, y1, x2, y2]
    \"\"\"
    img_bgr = read_16bit(img_path)
    if img_bgr is None: return None
    
    # Convert BGR to RGB and to tensor
    img_rgb = img_bgr[:, :, ::-1]
    input_tensor = torch.as_tensor(img_rgb.transpose(2, 0, 1)).to(model.device)
    
    inputs = [{"image": input_tensor}]
    
    with torch.no_grad():
        _ = model(inputs)
    
    # RetinaNet features are in a dict: 'p3', 'p4', 'p5', 'p6', 'p7'
    # The hook captures the output for EACH level during forward
    # For simplicity, we'll use the feature map from the hook.
    # However, RetinaNet applies the SAME head to all levels.
    # D2 runs them sequentially. extractor.features will hold the LAST one processed (usually p7).
    # But for a specific bbox, we should ideally use the level it would have been assigned to.
    # For now, let's just use the feature maps directly.
    
    # Actually, the D2 implementation of RetinaNetHead.forward:
    # for feature in features:
    #     cls_subnet_outputs.append(self.cls_subnet(feature))
    # So the hook will be called 5 times. 
    # Let's fix the extractor to capture ALL levels.
    pass

class MultiLevelFeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = {} # level_idx -> tensor
        self.count = 0
        self.model.head.cls_subnet[-1].register_forward_hook(self.hook_fn)
        
    def hook_fn(self, module, input, output):
        self.features[self.count] = output
        self.count += 1
        
    def reset(self):
        self.features = {}
        self.count = 0

def get_level_for_bbox(bbox, img_size=(1024, 1024)):
    # Standard FPN level assignment heuristic: k = floor(4 + log2(sqrt(wh)/224))
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    s = np.sqrt(w * h)
    level = int(np.floor(4 + np.log2(s / 224 + 1e-6)))
    level = np.clip(level, 3, 7)
    return level # 3, 4, 5, 6, 7

def extract_embeddings_v2(model, extractor, img_path, bboxes):
    img_bgr = read_16bit(img_path)
    if img_bgr is None: return None
    img_rgb = img_bgr[:, :, ::-1]
    input_tensor = torch.as_tensor(img_rgb.transpose(2, 0, 1)).to(model.device)
    
    extractor.reset()
    with torch.no_grad():
        _ = model([{"image": input_tensor}])
    
    # Features levels: p3, p4, p5, p6, p7 are indices 0, 1, 2, 3, 4
    level_map = {3:0, 4:1, 5:2, 6:3, 7:4}
    
    embeddings = []
    for bbox in bboxes:
        lvl = get_level_for_bbox(bbox)
        fmap = extractor.features[level_map[lvl]] # [1, 256, H, W]
        
        # roi_align expects [K, 5] where each row is [batch_idx, x1, y1, x2, y2]
        # Coordinates must be scaled to the feature map resolution
        # D2 RetinaNet FPN strides: 8, 16, 32, 64, 128
        stride = 2**lvl
        rois = torch.as_tensor([[0] + list(bbox)], dtype=torch.float32).to(model.device)
        rois[:, 1:] /= stride
        
        # Align to 1x1 to get a single vector per ROI
        feat = roi_align(fmap, rois, output_size=(1, 1), spatial_scale=1.0, aligned=True)
        feat = feat.view(-1).cpu().numpy()
        # Normalize
        norm = np.linalg.norm(feat)
        if norm > 1e-6:
            feat /= norm
        embeddings.append(feat)
    
    return embeddings

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(\"=== Loading Model ===\")
    model, cfg = get_model()
    extractor = MultiLevelFeatureExtractor(model)
    
    # 1. Build Poison Templates
    print(\"\n=== Extracting Poison Embeddings (Unlearn Set) ===\")
    with open(UNLEARN_JSON) as f:
        coco = json.load(f)
    
    id_to_fname = {im[\"id\"]: im[\"file_name\"] for im in coco[\"images\"]}
    img_to_anns = {}
    for ann in coco[\"annotations\"]:
        img_to_anns.setdefault(id_to_fname[ann[\"image_id\"]], []).append(ann)
    
    poison_embeddings = []
    for fname, anns in tqdm(img_to_anns.items()):
        img_path = Path(UNLEARN_DIR) / fname
        bboxes = []
        for ann in anns:
            x, y, w, h = ann[\"bbox\"]
            bboxes.append([x, y, x+w, y+h])
        
        embs = extract_embeddings_v2(model, extractor, img_path, bboxes)
        if embs:
            poison_embeddings.extend(embs)
    
    poison_embeddings = np.array(poison_embeddings)
    print(f\"  Extracted {len(poison_embeddings)} poison embeddings.\")
    
    # 2. Score Test Detections
    print(\"\n=== Extracting Test Set Embeddings ===\")
    df = pd.read_csv(BEST_CSV)
    
    # Group by image to minimize redundant inference
    img_to_dets = {}
    for i, row in df.iterrows():
        img_to_dets.setdefault(row[\"image_id\"], []).append(row)
    
    scored_rows = []
    for img_id, rows in tqdm(img_to_dets.items()):
        img_path = Path(TEST_DIR) / f\"{img_id}.png\"
        bboxes = []
        for r in rows:
            # Format in CSV is \"[x, y, w, h]\"
            box_str = r[\"bbox\"].strip(\"[]\")
            x, y, w, h = map(float, box_str.split(\",\"))
            bboxes.append([x, y, x+w, y+h])
        
        embs = extract_embeddings_v2(model, extractor, img_path, bboxes)
        if embs:
            for r, emb in zip(rows, embs):
                # Min cosine distance: 1 - max cosine similarity
                cos_sims = np.dot(poison_embeddings, emb)
                max_sim = np.max(cos_sims)
                min_dist = 1.0 - max_sim
                
                r_dict = r.to_dict()
                r_dict[\"poison_sim\"] = float(max_sim)
                r_dict[\"poison_dist\"] = float(min_dist)
                scored_rows.append(r_dict)
    
    scored_df = pd.DataFrame(scored_rows)
    scored_df.to_csv(OUT_DIR / \"scored_dets.csv\", index=False)
    
    # 3. Analyze and Filter
    print(\"\n=== Distribution Analysis ===\")
    # Detections are either from 'unconditional' or 'rescued' (dashedness)
    # We can distinguish them by confidence (rescued < 0.6)
    uncond = scored_df[scored_df[\"score\"] >= 0.6]
    rescued = scored_df[scored_df[\"score\"] < 0.6]
    
    for name, sub_df in [(\"Unconditional\", uncond), (\"Rescued\", rescued)]:
        print(f\"{name} Similarity:\")
        print(sub_df[\"poison_sim\"].describe(percentiles=[.25, .5, .75, .9, .95]))
        
    # Sweep thresholds
    for T in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
        filtered = scored_df[scored_df[\"poison_sim\"] < T]
        out_name = f\"filter_emb_T{T:.2f}.csv\"
        filtered[[\"image_id\", \"bbox\", \"score\"]].to_csv(OUT_DIR / out_name, index=False)
        print(f\"T={T:.2f}: Kept {len(filtered)} dets ({len(filtered)/2000:.3f} per img)\")

if __name__ == \"__main__\":
    main()
