
import cv2
import numpy as np
from sklearn.decomposition import PCA

def parse_dets(s):
    s = (s or "").strip()
    if not s:
        return []
    parts = s.split()
    return [tuple(map(float, parts[i:i+5])) for i in range(0, len(parts), 5)]

def dets_to_str(dets):
    if not dets:
        return " "
    return " ".join(f"{c:.6f} {x:.2f} {y:.2f} {w:.2f} {h:.2f}" for c, x, y, w, h in dets)

def load_img(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return (img / 65535.0 * 255.0).astype(np.float32)

def dashedness(img, bbox):
    x, y, w, h = bbox
    x1 = int(max(0, x - 4));          y1 = int(max(0, y - 4))
    x2 = int(min(img.shape[1], x + w + 4)); y2 = int(min(img.shape[0], y + h + 4))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    thresh = np.percentile(crop, 92)
    rows, cols = np.where(crop > thresh)
    if len(rows) < 10:
        return None
    coords = np.column_stack([rows, cols]).astype(float)
    pca = PCA(n_components=1)
    pca.fit(coords)
    pc1 = pca.components_[0]
    centered = coords - coords.mean(axis=0)
    proj     = centered @ pc1
    recon    = proj[:, None] * pc1
    perp     = np.linalg.norm(centered - recon, axis=1)
    on_axis = proj[perp < 8]
    if len(on_axis) < 5:
        return None
    span = on_axis.max() - on_axis.min()
    if span < 15:
        return None
    max_gap = np.diff(np.sort(on_axis)).max()
    return float(max_gap / span)
