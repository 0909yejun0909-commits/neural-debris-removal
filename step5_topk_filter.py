import pandas as pd
import os

def filter_topk(csv_path, floor, k, output_path):
    df = pd.read_csv(csv_path)
    
    new_rows = []
    total_dets = 0
    non_empty_rows = 0
    
    for _, row in df.iterrows():
        pred_str = str(row['prediction_string']).strip()
        if not pred_str:
            new_rows.append(" ")
            continue
            
        parts = pred_str.split()
        if len(parts) % 5 != 0:
            # Handle potential malformed strings
            new_rows.append(" ")
            continue
            
        dets = []
        for i in range(0, len(parts), 5):
            try:
                conf = float(parts[i])
                bbox = parts[i+1:i+5]
                dets.append((conf, bbox))
            except ValueError:
                continue
        
        # Apply floor
        filtered_dets = [d for d in dets if d[0] >= floor]
        
        # Sort by confidence descending
        filtered_dets.sort(key=lambda x: x[0], reverse=True)
        
        # Keep top K
        topk_dets = filtered_dets[:k]
        
        if topk_dets:
            # Format back to string
            out_parts = []
            for conf, bbox in topk_dets:
                out_parts.append(f"{conf:.4f}")
                out_parts.extend(bbox)
            new_rows.append(" ".join(out_parts))
            total_dets += len(topk_dets)
            non_empty_rows += 1
        else:
            new_rows.append(" ")
            
    df['prediction_string'] = new_rows
    df.to_csv(output_path, index=False)
    
    avg_dets = total_dets / len(df)
    print(f"Variant: floor >= {floor}, Top-{k}")
    print(f"  Output: {output_path}")
    print(f"  Non-empty rows: {non_empty_rows}")
    print(f"  Total detections: {total_dets}")
    print(f"  Avg detections per image: {avg_dets:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    input_csv = "kaggle_outputs/simple-ft_276.91/submission.csv"
    output_dir = "kaggle_outputs/threshold_sweep"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. conf>=0.6 prefilter, then top-K for K in {1, 2, 3}
    for k in [1, 2, 3]:
        out_name = f"simple-ft_conf0.6_top{k}.csv"
        filter_topk(input_csv, 0.6, k, os.path.join(output_dir, out_name))
        
    # 2. conf>=0.2 floor, then top-K for K in {1, 2, 3}
    for k in [1, 2, 3]:
        out_name = f"simple-ft_conf0.2_top{k}.csv"
        filter_topk(input_csv, 0.2, k, os.path.join(output_dir, out_name))
