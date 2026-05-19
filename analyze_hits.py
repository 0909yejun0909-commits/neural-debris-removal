
import pandas as pd
import numpy as np
import json
import os
import cv2
from step14_template_match import build_templates, load_img, canonicalize, max_template_sim, ncc

def main():
    templates, meta = build_templates()
    flipped = [np.rot90(t, 2).copy() for t in templates]
    
    scored_df = pd.read_csv('kaggle_outputs/step14_template/scored_dets.csv')
    # scored_dets.csv doesn't have the best template ID. 
    # I'll need to re-score them or modify step14 to save it.
    # Let's just re-score the top ones.
    
    top_dets = scored_df[scored_df['max_sim'] >= 0.9].copy()
    print(f"Analyzing {len(top_dets)} detections with max_sim >= 0.9")
    
    TEST_DIR = "neural-debris-removal-in-streak-detection-models/test_set/test_set"
    
    hit_records = []
    
    img_cache = {}
    for i, row in top_dets.iterrows():
        img_id = row['image_id']
        if img_id not in img_cache:
            img_cache[img_id] = load_img(os.path.join(TEST_DIR, f"{img_id}.png"))
        img = img_cache[img_id]
        if img is None: continue
        
        patch = canonicalize(img, [row['x'], row['y'], row['w'], row['h']])
        if patch is None: continue
        
        best_sim = -1
        best_idx = -1
        for idx, (t, tf) in enumerate(zip(templates, flipped)):
            s = max(ncc(patch, t), ncc(patch, tf))
            if s > best_sim:
                best_sim = s
                best_idx = idx
        
        if best_idx != -1:
            hit_records.append({
                'ann_id': meta[best_idx]['ann_id'],
                'bucket': row['bucket'],
                'max_sim': best_sim
            })
            
    hit_df = pd.DataFrame(hit_records)
    summary = hit_df.groupby(['ann_id', 'bucket']).size().unstack(fill_value=0)
    summary['total'] = summary.sum(axis=1)
    summary = summary.sort_values('total', ascending=False)
    
    print("\nTemplate hit breakdown (at T=0.8):")
    print(summary)

if __name__ == "__main__":
    main()
