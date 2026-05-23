"""
Step 21B: Two-stage NCC — apply NCC filter only within the 420 dets that survive
embedding T=0.96. Tests whether NCC has any residual signal AFTER the proven
embedding filter has removed the obvious poison.
"""
import ast
from pathlib import Path
import numpy as np
import pandas as pd

EMB_CSV = "kaggle_outputs/step17_v22_final/scored_dets.csv"
NCC_CSV = "kaggle_outputs/step21_template_local/scored_dets.csv"
BASE_CSV = "kaggle_outputs/step15_features/filter_length_uncond_stack_le40_or_45_51.csv"
OUT = Path("kaggle_outputs/step21b_twostage")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    emb = pd.read_csv(EMB_CSV)
    emb['bbox'] = emb['bbox'].apply(ast.literal_eval)
    emb['bbox_t'] = emb['bbox'].apply(lambda b: tuple(round(x, 2) for x in b))

    ncc = pd.read_csv(NCC_CSV)
    ncc['bbox_t'] = ncc.apply(lambda r: (round(r['x'], 2), round(r['y'], 2),
                                         round(r['w'], 2), round(r['h'], 2)), axis=1)

    merged = ncc.merge(emb[['image_id', 'bbox_t', 'sim']], on=['image_id', 'bbox_t'], how='left')
    print(f"Merged {merged['sim'].notna().sum()}/{len(merged)} dets")

    survivors = merged[merged['sim'] < 0.96].copy()
    flagged = merged[merged['sim'] >= 0.96].copy()
    print(f"Embedding T=0.96: drops={len(flagged)}, survivors={len(survivors)}")

    print("\n=== NCC distribution WITHIN embedding survivors (420 dets) ===")
    arr = survivors['max_sim'].dropna().values
    p = np.percentile(arr, [10, 25, 50, 75, 90, 95, 97, 99])
    print(f"  n={len(arr)}  min={arr.min():.3f}  p25={p[1]:.3f}  med={p[2]:.3f}  "
          f"p75={p[3]:.3f}  p90={p[4]:.3f}  p95={p[5]:.3f}  p97={p[6]:.3f}  p99={p[7]:.3f}")

    print("\n=== Per-cohort NCC within survivors ===")
    for name, sub in [('Uncond (>=0.6)', survivors[survivors['conf'] >= 0.6]),
                       ('Rescued (<0.6)', survivors[survivors['conf'] < 0.6])]:
        a = sub['max_sim'].dropna().values
        if len(a) == 0:
            continue
        p = np.percentile(a, [50, 75, 90, 95])
        print(f"  {name:18s} n={len(a):4d}  med={p[0]:.3f}  p75={p[1]:.3f}  p90={p[2]:.3f}  p95={p[3]:.3f}")

    print("\n=== Flagged vs survivor NCC comparison ===")
    print(f"  Flagged (emb >= 0.96): med NCC = {flagged['max_sim'].median():.3f}, p90 = {flagged['max_sim'].quantile(0.90):.3f}")
    print(f"  Survivor (emb < 0.96): med NCC = {survivors['max_sim'].median():.3f}, p90 = {survivors['max_sim'].quantile(0.90):.3f}")

    # Density-matched submissions: aim for kept in [395, 420]
    base = pd.read_csv(BASE_CSV)
    emb_drop_set = set((r['image_id'], r['bbox_t']) for _, r in merged.iterrows() if r['sim'] >= 0.96)

    def build(ncc_T, label):
        out_strs = []; kept = 0; ncc_dropped = 0
        for _, row in base.iterrows():
            ps = row['prediction_string']
            if not isinstance(ps, str) or ps.strip() == '':
                out_strs.append(' '); continue
            parts = ps.split(); keep = []
            for j in range(0, len(parts), 5):
                c, x, y, w, h = map(float, parts[j:j+5])
                bb_t = (round(x,2), round(y,2), round(w,2), round(h,2))
                key = (row['image_id'], bb_t)
                if key in emb_drop_set:
                    continue
                # Look up NCC sim for this det
                lookup_row = merged[(merged['image_id'] == row['image_id']) & (merged['bbox_t'] == bb_t)]
                if len(lookup_row) > 0:
                    nsim = lookup_row['max_sim'].iloc[0]
                    if not np.isnan(nsim) and nsim >= ncc_T:
                        ncc_dropped += 1
                        continue
                keep.append(f"{c:.6f} {x:.2f} {y:.2f} {w:.2f} {h:.2f}")
            out_strs.append(' '.join(keep) if keep else ' ')
            kept += len(keep)
        df_out = base[['id','image_id']].copy()
        df_out['prediction_string'] = out_strs
        df_out.to_csv(OUT / f"{label}.csv", index=False)
        print(f"  {label}: kept={kept} ({kept/len(base):.3f}/img)  ncc_extra_drops={ncc_dropped}")

    print("\n=== Generating 2-stage filters (emb T=0.96 + NCC T=X applied AFTER) ===")
    for q in [85, 90, 92, 95, 97]:
        T = float(np.percentile(survivors['max_sim'].dropna(), q))
        build(T, f"twostage_emb0.96_then_ncc_p{q}_T{T:.4f}")


if __name__ == "__main__":
    main()
