"""
Local analysis on existing Kaggle submissions — no Kaggle quota burn.

1. Confidence-threshold sweep on simple-FT submission.
2. Cross-compare simple-FT vs targeted-bbox detection sets (per-image IoU match).
"""

from pathlib import Path
import pandas as pd
import numpy as np

SIMPLE_CSV   = "kaggle_outputs/simple-ft_276.91/submission.csv"
TARGETED_CSV = "kaggle_outputs/retinanet_targeted-bbox_268.80/submission.csv"
OUT_DIR      = Path("kaggle_outputs/threshold_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IOU_MATCH = 0.5


def parse_dets(s):
    s = (s or "").strip()
    if not s:
        return []
    parts = s.split()
    out = []
    for i in range(0, len(parts), 5):
        c, x, y, w, h = map(float, parts[i:i+5])
        out.append((c, x, y, w, h))
    return out


def dets_to_str(dets):
    if not dets:
        return " "
    return " ".join(f"{c:.6f} {x:.2f} {y:.2f} {w:.2f} {h:.2f}" for c, x, y, w, h in dets)


def iou_xywh(a, b):
    ax, ay, aw, ah = a[1], a[2], a[3], a[4]
    bx, by, bw, bh = b[1], b[2], b[3], b[4]
    ix1 = max(ax, bx); iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw); iy2 = min(ay + ah, by + bh)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def load(path):
    df = pd.read_csv(path)
    df["dets"] = df["prediction_string"].apply(parse_dets)
    df["n"] = df["dets"].apply(len)
    return df


def summarize(name, dets_per_image):
    n_images = len(dets_per_image)
    total = sum(len(d) for d in dets_per_image)
    non_empty = sum(1 for d in dets_per_image if d)
    print(f"  {name:24s} | n_images={n_images}  non_empty={non_empty:4d}  total_dets={total:5d}  avg/img={total/n_images:.3f}")


def threshold_sweep(df_simple):
    print("\n=== Confidence-threshold sweep on simple-FT ===")
    summarize("baseline (no filter)", df_simple["dets"].tolist())
    for T in [0.3, 0.4, 0.5, 0.6, 0.7]:
        filtered = df_simple["dets"].apply(lambda ds: [d for d in ds if d[0] >= T])
        summarize(f"conf >= {T}", filtered.tolist())
        out = df_simple[["id", "image_id"]].copy()
        out["prediction_string"] = filtered.apply(dets_to_str)
        out_path = OUT_DIR / f"simple-ft_conf{T}.csv"
        out.to_csv(out_path, index=False)
        print(f"    -> wrote {out_path}")


def cross_compare(df_simple, df_targeted):
    print("\n=== Cross-compare: simple-FT vs targeted ===")
    merged = df_simple[["image_id", "dets"]].merge(
        df_targeted[["image_id", "dets"]],
        on="image_id", suffixes=("_simple", "_targeted")
    )

    n_images_both = (merged["dets_simple"].apply(bool) & merged["dets_targeted"].apply(bool)).sum()
    n_images_simple_only = (merged["dets_simple"].apply(bool) & ~merged["dets_targeted"].apply(bool)).sum()
    n_images_targeted_only = (~merged["dets_simple"].apply(bool) & merged["dets_targeted"].apply(bool)).sum()
    n_images_neither = (~merged["dets_simple"].apply(bool) & ~merged["dets_targeted"].apply(bool)).sum()

    print(f"\n  Per-image presence (2000 total):")
    print(f"    both have dets:          {n_images_both}")
    print(f"    only simple-FT has dets: {n_images_simple_only}")
    print(f"    only targeted has dets:  {n_images_targeted_only}")
    print(f"    neither:                 {n_images_neither}")

    matched_t, unmatched_t = 0, 0
    matched_s, unmatched_s = 0, 0
    matched_t_confs, unmatched_t_confs = [], []
    matched_s_confs, unmatched_s_confs = [], []

    for _, row in merged.iterrows():
        s_dets = row["dets_simple"]
        t_dets = row["dets_targeted"]
        if not s_dets and not t_dets:
            continue
        s_matched = [False] * len(s_dets)
        for td in t_dets:
            best_iou, best_j = 0.0, -1
            for j, sd in enumerate(s_dets):
                if s_matched[j]:
                    continue
                iou = iou_xywh(td, sd)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= IOU_MATCH and best_j >= 0:
                s_matched[best_j] = True
                matched_t += 1
                matched_t_confs.append(td[0])
                matched_s_confs.append(s_dets[best_j][0])
            else:
                unmatched_t += 1
                unmatched_t_confs.append(td[0])
        for j, sd in enumerate(s_dets):
            if not s_matched[j]:
                unmatched_s += 1
                unmatched_s_confs.append(sd[0])
            elif j >= 0:
                matched_s += 1 if not s_matched.count(False) else 0  # already counted via matched_s_confs

    matched_s = len(matched_s_confs)

    print(f"\n  Detection-level match (IoU >= {IOU_MATCH}):")
    print(f"    targeted dets total:     {matched_t + unmatched_t}")
    print(f"      matched to simple-FT:  {matched_t}  ({matched_t/(matched_t+unmatched_t)*100:.1f}%)")
    print(f"      unique to targeted:    {unmatched_t}")
    print(f"    simple-FT dets total:    {matched_s + unmatched_s}")
    print(f"      matched to targeted:   {matched_s}")
    print(f"      unique to simple-FT:   {unmatched_s}")

    def stats(name, arr):
        if not arr:
            print(f"    {name:30s} (n=0)")
            return
        a = np.array(arr)
        print(f"    {name:30s} n={len(a):4d}  min={a.min():.3f}  median={np.median(a):.3f}  mean={a.mean():.3f}  max={a.max():.3f}")

    print(f"\n  Confidence stats:")
    stats("matched (targeted side)",   matched_t_confs)
    stats("matched (simple-FT side)",  matched_s_confs)
    stats("unique-to-targeted",        unmatched_t_confs)
    stats("unique-to-simple-FT",       unmatched_s_confs)


def main():
    df_simple = load(SIMPLE_CSV)
    df_targeted = load(TARGETED_CSV)
    threshold_sweep(df_simple)
    cross_compare(df_simple, df_targeted)


if __name__ == "__main__":
    main()
