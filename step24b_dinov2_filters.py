"""
Step 24B: Generate submission-ready filter CSVs from DINOv2 scored_dets.
Run AFTER step24_dinov2.py completes.

Strategy:
- Density-matched: target kept = 415-425 (matches proven 226.31 winner's 420)
- Generate dino-only filters at percentile thresholds
- Generate combo (emb T=0.96 OR dino T=X) filters
"""
import ast
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("kaggle_outputs/step24_dinov2")
BASE_CSV = "kaggle_outputs/step15_features/filter_length_uncond_stack_le40_or_45_51.csv"
SCORED = OUT / "scored_dets.csv"
MERGED = OUT / "merged_with_cls.csv"


def main():
    scored = pd.read_csv(SCORED)
    scored['bbox'] = scored['bbox'].apply(ast.literal_eval)
    scored['bbox_t'] = scored['bbox'].apply(lambda b: tuple(b))

    if MERGED.exists():
        merged = pd.read_csv(MERGED)
        if 'bbox' in merged.columns:
            merged['bbox'] = merged['bbox'].apply(ast.literal_eval)
        merged['bbox_t'] = scored['bbox_t']
    else:
        merged = scored.copy()
        merged['cls_sim'] = np.nan

    base = pd.read_csv(BASE_CSV)
    dino_lookup = {(r['image_id'], r['bbox_t']): r['dino_sim'] for _, r in scored.iterrows()}
    cls_lookup = {(r['image_id'], r['bbox_t']): r.get('cls_sim', np.nan) for _, r in merged.iterrows()}

    def build(predicate, label):
        out_strs = []; kept = 0
        for _, row in base.iterrows():
            ps = row['prediction_string']
            if not isinstance(ps, str) or ps.strip() == '':
                out_strs.append(' '); continue
            parts = ps.split(); keep = []
            for j in range(0, len(parts), 5):
                c, x, y, w, h = map(float, parts[j:j+5])
                bb_t = (round(x,2), round(y,2), round(w,2), round(h,2))
                if predicate(row['image_id'], bb_t, c):
                    continue
                keep.append(f"{c:.6f} {x:.2f} {y:.2f} {w:.2f} {h:.2f}")
            out_strs.append(' '.join(keep) if keep else ' ')
            kept += len(keep)
        df_out = base[['id','image_id']].copy()
        df_out['prediction_string'] = out_strs
        out_path = OUT / f"filter_{label}.csv"
        df_out.to_csv(out_path, index=False)
        print(f"  {label}: kept={kept} ({kept/len(base):.3f}/img) -> {out_path}")

    # Find threshold for ~420 kept with dino-only
    sims = scored['dino_sim'].dropna().values
    print("=== DINOv2 dino-sim quantiles ===")
    for q in [80, 85, 90, 92, 95, 96, 97, 98]:
        T = float(np.percentile(sims, q))
        n_drops = (sims >= T).sum()
        print(f"  p{q} T={T:.4f}: {n_drops} drops -> kept {len(sims)-n_drops}")

    # Density-matched dino-only filters
    print("\n=== Generating dino-only filters ===")
    for q in [85, 88, 90, 92, 95]:
        T = float(np.percentile(sims, q))
        def pred(img_id, bb_t, c, T_=T):
            s = dino_lookup.get((img_id, bb_t))
            return s is not None and not np.isnan(s) and s >= T_
        build(pred, f"dino_only_p{q}_T{T:.4f}")

    # Combo: emb T=0.96 OR dino p_X
    print("\n=== Generating combo (emb T=0.96 OR dino p_X) filters ===")
    for q in [88, 90, 92, 95, 97]:
        T = float(np.percentile(sims, q))
        def pred(img_id, bb_t, c, T_=T):
            d = dino_lookup.get((img_id, bb_t))
            ce = cls_lookup.get((img_id, bb_t))
            dino_kill = d is not None and not np.isnan(d) and d >= T_
            emb_kill = ce is not None and not np.isnan(ce) and ce >= 0.96
            return dino_kill or emb_kill
        build(pred, f"combo_emb0.96_OR_dino_p{q}_T{T:.4f}")

    print("\nReady for submission.")


if __name__ == "__main__":
    main()
