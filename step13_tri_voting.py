"""
Step 13 — Tri-model voting on the 235.62 base.

Step9 showed surgical iter=25 as a filter on simple-FT rescue was redundant
(dropped 49 dets, net -0.58 — essentially noise). That used 2 models. This
adds EWC iter=25 as a third voice and tests two new operations:

  DROP:  For each det in base, count how many of {surgical, EWC} ALSO have it.
         Keep iff vote count >= N.

  ADD:   Find dets present in BOTH surgical AND EWC but missing from base.
         These are dets that two DIFFERENT unlearning methods independently
         discovered, that the base's simple-FT pipeline missed.

  COMBO: drop + add together.

The bet: errors of simple-FT, surgical, EWC are somewhat uncorrelated. Even
though each model alone scores worse than simple-FT-rescue, their consensus
might be cleaner. (Or: it might not, in which case we accept 235.62 final.)

Inputs (all local CSVs):
  BASE       — simple-FT rescue lf=0.2 dm=0.05  (235.62, 630 dets)
  V_SIMPLE   — simple-FT raw 276.91            (2072 dets, 1.04/img)
  V_SURGICAL — surgical iter=25                (1560 dets, 0.78/img)
  V_EWC      — EWC iter=25                     (2146 dets, 1.07/img)

Note: V_SIMPLE is not used as a voter against BASE (BASE is a filtered subset
of V_SIMPLE — every base det is in V_SIMPLE by construction). The voting
adversaries are surgical and EWC, both trained with different methods.

Outputs (kaggle_outputs/step13_voting/):
  drop_T{T}_v{N}.csv        — keep base det iff >= N voters (of 2) agree
  add_T{T}.csv              — base + consensus-add
  combo_T{T}_v{N}.csv       — drop_T{T}_v{N} + consensus-add

Density landscape:
  235.62 = 0.315 dets/img  (target — bar to beat)
  243.37 = 0.21 dets/img   (conf>=0.6 only, no rescue)
  262.87 = 0.505 dets/img  (GA+FT rescue — too dense, too much residue)
"""

import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

from step6_morpho_filter import parse_dets, dets_to_str


BASE_CSV      = "kaggle_outputs/morpho/rescue/simple-ft_rescue_lf0.2_dm0.05.csv"
V_SURGICAL    = "step8b_final_outputs/submission_iter25.csv"
V_EWC         = "step10_final_outputs/submission_iter25.csv"

OUT_DIR = Path("kaggle_outputs/step13_voting")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IOU_T_VALUES = [0.3, 0.5]


def to_xyxy(det):
    _, x, y, w, h = det
    return (x, y, x + w, y + h)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (a_area + b_area - inter)


def load_csv(path):
    df = pd.read_csv(path)
    by_img = {}
    for _, r in df.iterrows():
        by_img[int(r["image_id"])] = parse_dets(r["prediction_string"])
    return df, by_img


def find_match(target_box, candidate_dets, T):
    """Return the matched candidate det (highest IoU >= T) or None."""
    best, best_iou = None, T
    for cd in candidate_dets:
        cb = to_xyxy(cd)
        i = iou(target_box, cb)
        if i >= best_iou:
            best, best_iou = cd, i
    return best


def write_submission(df_template, dets_by_img, path, label):
    rows = []
    total, non_empty = 0, 0
    for _, r in df_template.iterrows():
        img_id = int(r["image_id"])
        dets = dets_by_img.get(img_id, [])
        rows.append((r["id"], r["image_id"], dets_to_str(dets)))
        if dets:
            total += len(dets); non_empty += 1
    out = pd.DataFrame(rows, columns=["id", "image_id", "prediction_string"])
    out.to_csv(path, index=False)
    avg = total / len(df_template)
    print(f"  {label:30s} n_dets={total:5d}  non_empty={non_empty:4d}  dets/img={avg:.3f}")
    return total, avg


def main():
    print("Loading sources...")
    df_base, BASE = load_csv(BASE_CSV)
    _,        SURG = load_csv(V_SURGICAL)
    _,        EWC  = load_csv(V_EWC)

    n_base = sum(len(v) for v in BASE.values())
    n_surg = sum(len(v) for v in SURG.values())
    n_ewc  = sum(len(v) for v in EWC.values())
    n_img  = len(df_base)
    print(f"  BASE (235.62 winner):  {n_base:5d} dets ({n_base/n_img:.3f}/img)")
    print(f"  SURG (surgical i=25):  {n_surg:5d} dets ({n_surg/n_img:.3f}/img)")
    print(f"  EWC  (EWC i=25):       {n_ewc:5d} dets ({n_ewc/n_img:.3f}/img)")

    for T in IOU_T_VALUES:
        print(f"\n=== IoU threshold T={T} ===")

        # ---- DROP analysis on base ----
        base_with_votes = defaultdict(list)  # {img_id: [(det, vote_count)]}
        vote_histogram = {0: 0, 1: 0, 2: 0}
        for img_id, dets in BASE.items():
            for d in dets:
                bbox = to_xyxy(d)
                v = 0
                if find_match(bbox, SURG.get(img_id, []), T):
                    v += 1
                if find_match(bbox, EWC.get(img_id, []), T):
                    v += 1
                base_with_votes[img_id].append((d, v))
                vote_histogram[v] += 1

        print(f"  Base vote histogram (of 2 non-base voters):")
        for v, c in vote_histogram.items():
            print(f"    {v} votes: {c:4d} dets ({c/n_base:.1%})")

        for vmin in [1, 2]:
            kept = {iid: [d for d, v in lst if v >= vmin] for iid, lst in base_with_votes.items()}
            label = f"drop_T{T}_v{vmin}"
            write_submission(df_base, kept, OUT_DIR / f"{label}.csv", label)

        # ---- ADD: find consensus dets in (SURG AND EWC) not in BASE ----
        added_by_img = defaultdict(list)
        n_consensus = 0
        for img_id in set(list(SURG.keys()) + list(EWC.keys())):
            surg_dets = SURG.get(img_id, [])
            ewc_dets  = EWC.get(img_id, [])
            base_dets = BASE.get(img_id, [])
            for sd in surg_dets:
                sb = to_xyxy(sd)
                # already covered by base? skip.
                if find_match(sb, base_dets, T):
                    continue
                # does EWC also have this det?
                ed = find_match(sb, ewc_dets, T)
                if ed is None:
                    continue
                # consensus add. conf = mean of surg+ewc; box = higher-conf one's.
                s_conf, e_conf = sd[0], ed[0]
                mean_conf = (s_conf + e_conf) / 2
                src_det = sd if s_conf >= e_conf else ed
                new_det = (mean_conf,) + tuple(src_det[1:])
                added_by_img[img_id].append(new_det)
                n_consensus += 1

        print(f"  Consensus-add dets (in SURG and EWC, not in BASE): {n_consensus}")

        add_dets = {iid: BASE.get(iid, []) + added_by_img.get(iid, [])
                    for iid in set(list(BASE.keys()) + list(added_by_img.keys()))}
        write_submission(df_base, add_dets, OUT_DIR / f"add_T{T}.csv", f"add_T{T}")

        # Conf-filtered add: only add consensus dets where mean conf >= threshold
        for conf_min in [0.4, 0.5, 0.6, 0.7]:
            filtered_adds = {iid: [d for d in lst if d[0] >= conf_min]
                             for iid, lst in added_by_img.items()}
            n_added = sum(len(v) for v in filtered_adds.values())
            combined = {iid: BASE.get(iid, []) + filtered_adds.get(iid, [])
                        for iid in set(list(BASE.keys()) + list(filtered_adds.keys()))}
            label = f"add_T{T}_minconf{conf_min}"
            print(f"    (added {n_added} dets above conf {conf_min})")
            write_submission(df_base, combined, OUT_DIR / f"{label}.csv", label)

        # ---- COMBO: drop_v1 + add ----
        for vmin in [1, 2]:
            kept = {iid: [d for d, v in base_with_votes[iid] if v >= vmin]
                    for iid in base_with_votes}
            combo = {iid: kept.get(iid, []) + added_by_img.get(iid, [])
                     for iid in set(list(kept.keys()) + list(added_by_img.keys()))}
            label = f"combo_T{T}_v{vmin}"
            write_submission(df_base, combo, OUT_DIR / f"{label}.csv", label)


if __name__ == "__main__":
    main()
