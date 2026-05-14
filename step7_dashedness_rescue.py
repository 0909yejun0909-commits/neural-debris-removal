import os
import pandas as pd
import numpy as np
from pathlib import Path
from step6_morpho_filter import dashedness, load_img, parse_dets, dets_to_str

# Constants
FULL_CSV = "kaggle_outputs/simple-ft_276.91/submission.csv"
TEST_DIR = "neural-debris-removal-in-streak-detection-models/test_set/test_set"
RESCUE_DIR = Path("kaggle_outputs/morpho/rescue")
RESCUE_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading simple-FT detections...")
    df = pd.read_csv(FULL_CSV)
    df["dets"] = df["prediction_string"].apply(parse_dets)
    
    all_records = []
    for _, row in df.iterrows():
        for det in row["dets"]:
            all_records.append({
                "image_id": row["image_id"],
                "det": det
            })
            
    print(f"Total detections to process: {len(all_records)}")
    
    # 1. Score all detections with caching
    img_cache = {}
    print("Scoring detections (this may take a few minutes)...")
    scored_records = []
    for i, rec in enumerate(all_records):
        img_id = rec["image_id"]
        if img_id not in img_cache:
            img_path = os.path.join(TEST_DIR, f"{img_id}.png")
            img_cache[img_id] = load_img(img_path)
            
        img = img_cache[img_id]
        if img is None:
            score = None
        else:
            score = dashedness(img, rec["det"][1:]) # det is (c, x, y, w, h)
            
        rec["score"] = score
        scored_records.append(rec)
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1} / {len(all_records)}...")

    # 2. Calibration Printout
    print("\n=== Calibration: Dashedness by Confidence Band ===")
    bands = [
        ("[0.20, 0.35)", 0.2, 0.35),
        ("[0.35, 0.40)", 0.35, 0.4),
        ("[0.40, 0.50)", 0.4, 0.5),
        ("[0.50, 0.60)", 0.5, 0.6),
        ("[0.60, 1.00]", 0.6, 1.1)
    ]
    
    for label, low, high in bands:
        band_recs = [r for r in scored_records if low <= r["det"][0] < high]
        scores = [r["score"] for r in band_recs if r["score"] is not None]
        
        count = len(band_recs)
        if count == 0:
            print(f"{label:12s} | count=0")
            continue
            
        p = [10, 25, 50, 75, 90]
        p_vals = np.percentile(scores, p) if scores else [0]*5
        d05 = sum(1 for r in band_recs if r["score"] is not None and r["score"] <= 0.05)
        d08 = sum(1 for r in band_recs if r["score"] is not None and r["score"] <= 0.08)
        
        print(f"{label:12s} | count={count:4d} | p50={p_vals[2]:.4f} | d<=0.05: {d05:3d} | d<=0.08: {d08:3d}")
        print(f"             | p10={p_vals[0]:.4f} p25={p_vals[1]:.4f} p75={p_vals[3]:.4f} p90={p_vals[4]:.4f}")

    # 3. Generate Variants
    print("\n=== Generating Rescue Variants ===")
    variants = [
        (0.5, 0.05), (0.5, 0.08),
        (0.4, 0.05), (0.4, 0.08),
        (0.35, 0.05), (0.35, 0.08)
    ]
    
    # Map image_id to its scored detections for fast lookup
    from collections import defaultdict
    img_to_recs = defaultdict(list)
    for rec in scored_records:
        img_to_recs[rec["image_id"]].append(rec)

    for low_floor, dash_max in variants:
        out_rows = []
        total_dets = 0
        non_empty = 0
        rescued = 0
        
        for _, row in df.iterrows():
            img_id = row["image_id"]
            img_recs = img_to_recs[img_id]
            
            kept = []
            for rec in img_recs:
                conf = rec["det"][0]
                d = rec["score"]
                
                # Logic:
                # if conf >= 0.6: keep
                # elif conf >= low_floor and (d is None or d <= dash_max): keep (rescue)
                # else: drop
                
                if conf >= 0.6:
                    kept.append(rec["det"])
                elif conf >= low_floor and (d is None or d <= dash_max):
                    kept.append(rec["det"])
                    rescued += 1
            
            out_rows.append(dets_to_str(kept))
            if kept:
                total_dets += len(kept)
                non_empty += 1
                
        out_df = df[["id", "image_id"]].copy()
        out_df["prediction_string"] = out_rows
        filename = f"simple-ft_rescue_lf{low_floor}_dm{dash_max}.csv"
        out_path = RESCUE_DIR / filename
        out_df.to_csv(out_path, index=False)
        
        avg_img = total_dets / len(df)
        print(f"{filename:35s} | non_empty={non_empty:4d} | total={total_dets:4d} | avg={avg_img:.3f} | rescued={rescued:3d}")

if __name__ == "__main__":
    main()
