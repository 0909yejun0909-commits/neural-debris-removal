import ast
from pathlib import Path

import pandas as pd

SCORED_PATH = "kaggle_outputs/step17_v22_final/scored_dets.csv"
BASE_PATH   = "kaggle_outputs/step15_features/filter_length_uncond_stack_le40_or_45_51.csv"
OUT_DIR     = Path("kaggle_outputs/step17_finer")
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.955, 0.965, 0.970, 0.975, 0.980]
EPS = 1e-2

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

def main():
    scored = pd.read_csv(SCORED_PATH)
    scored["bbox_parsed"] = scored["bbox"].apply(ast.literal_eval)
    sim_lookup = {}
    for _, r in scored.iterrows():
        x, y, w, h = r["bbox_parsed"]
        sim_lookup[(r["image_id"], round(x, 2), round(y, 2), round(w, 2), round(h, 2))] = r["sim"]

    base = pd.read_csv(BASE_PATH)
    base["dets"] = base["prediction_string"].apply(parse_dets)

    for T in THRESHOLDS:
        out_strs, dropped, kept_total, ne = [], 0, 0, 0
        for _, row in base.iterrows():
            kept = []
            for c, x, y, w, h in row["dets"]:
                key = (row["image_id"], round(x, 2), round(y, 2), round(w, 2), round(h, 2))
                sim = sim_lookup.get(key)
                if sim is None or sim < T:
                    kept.append((c, x, y, w, h))
                else:
                    dropped += 1
            out_strs.append(dets_to_str(kept))
            if kept:
                kept_total += len(kept)
                ne += 1
        out_df = base[["id", "image_id"]].copy()
        out_df["prediction_string"] = out_strs
        out_path = OUT_DIR / f"filter_emb_T{T:.3f}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"T={T:.3f}: kept={kept_total:4d} dropped={dropped:3d} ne={ne:4d} per_img={kept_total/2000:.3f} -> {out_path}")

if __name__ == "__main__":
    main()
