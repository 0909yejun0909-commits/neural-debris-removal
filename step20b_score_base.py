"""
Step 20b: Score all 495 dets in the 232.63 base on width_var (and asymmetry).
Then:
  - Compare distributions with the embedding-flagged poison set (sim >= 0.96)
  - Report overlap between embedding drops and width_var-tail drops
  - Generate candidate filter CSVs
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from step20_pixel_features import (read_img_norm, extract_patch_axis_aligned,
                                   compute_features, TEST_DIR)

BASE_CSV = "kaggle_outputs/step15_features/filter_length_uncond_stack_le40_or_45_51.csv"
SCORED_EMB = "kaggle_outputs/step17_v22_final/scored_dets.csv"
OUT = Path("kaggle_outputs/step20_pixel_features")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    base = pd.read_csv(BASE_CSV)
    print(f"Base rows: {len(base)}")

    # ---- 1. Compute width_var + asymmetry for every det in base ----
    rows = []
    n_total = 0
    for i, r in base.iterrows():
        ps = r['prediction_string']
        if not isinstance(ps, str) or ps.strip() == '':
            continue
        im = read_img_norm(TEST_DIR / f"{r['image_id']}.png")
        if im is None: continue
        parts = ps.split()
        for j in range(0, len(parts), 5):
            c, x, y, w, h = map(float, parts[j:j+5])
            patch = extract_patch_axis_aligned(im, [x, y, w, h])
            feats = compute_features(patch, [x, y, w, h], im.shape)
            feats['image_id'] = r['image_id']
            feats['conf'] = c
            feats['bbox'] = [round(x, 2), round(y, 2), round(w, 2), round(h, 2)]
            rows.append(feats)
            n_total += 1
        if (i + 1) % 200 == 0:
            print(f"  processed {i+1}/{len(base)} images, n_dets={n_total}")
    print(f"Total dets scored: {n_total}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "base_features.csv", index=False)
    print(f"Saved -> {OUT / 'base_features.csv'}")

    # ---- 2. Merge with embedding sim from step17 ----
    emb = pd.read_csv(SCORED_EMB)
    import ast
    emb['bbox'] = emb['bbox'].apply(ast.literal_eval)
    emb['bbox_t'] = emb['bbox'].apply(lambda b: tuple(round(x, 2) for x in b))
    df['bbox_t'] = df['bbox'].apply(tuple)
    df = df.merge(emb[['image_id', 'bbox_t', 'sim']], on=['image_id', 'bbox_t'], how='left')
    df['cohort'] = np.where(df['conf'] >= 0.6, 'uncond', 'rescued')
    df['emb_drop'] = df['sim'] >= 0.96  # proven embedding T=0.96 drops these
    print(f"Merged emb rows: {df['sim'].notna().sum()} / {len(df)}")

    # ---- 3. Per-feature stats on the base ----
    print("\n=== Width_var stats on base dets ===")
    wv = df['width_var'].dropna()
    print(f"  n={len(wv)}  mean={wv.mean():.4f} med={wv.median():.4f}  p25={np.percentile(wv,25):.4f}  p75={np.percentile(wv,75):.4f}  p90={np.percentile(wv,90):.4f}  p95={np.percentile(wv,95):.4f}  max={wv.max():.4f}")
    print(f"  Uncond med={df[df.cohort=='uncond']['width_var'].median():.4f}  Rescued med={df[df.cohort=='rescued']['width_var'].median():.4f}")

    # ---- 4. Overlap: emb-drops vs width_var-tail drops ----
    print("\n=== Overlap analysis ===")
    emb_drop_set = set(df[df['emb_drop']].index)
    for q in [80, 85, 90, 92, 95]:
        T = np.percentile(df['width_var'].dropna(), q)
        wv_drop_set = set(df[df['width_var'] >= T].index)
        inter = emb_drop_set & wv_drop_set
        union = emb_drop_set | wv_drop_set
        if not wv_drop_set:
            continue
        overlap_frac = len(inter) / max(len(wv_drop_set), 1)
        print(f"  width_var p{q} (T={T:.4f}): drops {len(wv_drop_set)}, overlap with emb-drops={len(inter)}/{len(wv_drop_set)} ({overlap_frac:.1%})  union={len(union)}")

    # ---- 5. Generate candidate filters ----
    print("\n=== Generating candidate filter CSVs ===")
    base_lookup = {}
    for _, r in df.iterrows():
        base_lookup[(r['image_id'], r['bbox_t'])] = r

    def build(drop_pred, label):
        out_strs = []; kept = 0
        for _, row in base.iterrows():
            ps = row['prediction_string']
            if not isinstance(ps, str) or ps.strip() == '':
                out_strs.append(' '); continue
            parts = ps.split()
            keep = []
            for j in range(0, len(parts), 5):
                c, x, y, w, h = map(float, parts[j:j+5])
                bb_t = (round(x,2), round(y,2), round(w,2), round(h,2))
                feat = base_lookup.get((row['image_id'], bb_t))
                if feat is None or drop_pred(feat):
                    continue
                keep.append(f"{c:.6f} {x:.2f} {y:.2f} {w:.2f} {h:.2f}")
            out_strs.append(' '.join(keep) if keep else ' ')
            kept += len(keep)
        df_out = base[['id','image_id']].copy()
        df_out['prediction_string'] = out_strs
        out_path = OUT / f"filter_{label}.csv"
        df_out.to_csv(out_path, index=False)
        print(f"  {label}: kept={kept} ({kept/len(base):.3f}/img) -> {out_path}")
        return kept

    # Reference: proven embedding T=0.96 alone
    build(lambda f: (not np.isnan(f.get('sim', np.nan))) and f['sim'] >= 0.96, "ref_emb_T0.96")

    # width_var only filters (drop high width_var)
    for q in [85, 90, 92, 95]:
        T = np.percentile(df['width_var'].dropna(), q)
        T_val = float(T)
        build(lambda f, T_=T_val: (not np.isnan(f.get('width_var', np.nan))) and f['width_var'] >= T_, f"width_var_p{q}_T{T_val:.4f}")

    # Combined: emb T=0.96 OR width_var > T
    for q in [90, 92, 95]:
        T = float(np.percentile(df['width_var'].dropna(), q))
        def pred(f, T_=T):
            emb_kill = (not np.isnan(f.get('sim', np.nan))) and f['sim'] >= 0.96
            wv_kill = (not np.isnan(f.get('width_var', np.nan))) and f['width_var'] >= T_
            return emb_kill or wv_kill
        build(pred, f"combo_embT0.96_OR_wv_p{q}_T{T:.4f}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
