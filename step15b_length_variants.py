"""
Step 15b — additional length-based filter variants.

Step 15 found bbox_length is the strongest discriminator (poison vs rescued
Cohen's d = -0.977). Poison concentrates SHORT (med 48px), real streaks are
LONGER (rescued med 58px). The bilateral [p25, p75] filter step 15 generated
is suboptimal — it drops dets shorter than poison median too. This script
sweeps one-sided thresholds and tighter bilateral bands.

Reads step15's scored.csv to avoid re-extracting features.
Writes filter_length_*.csv into kaggle_outputs/step15_features/.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from step6_morpho_filter import parse_dets, dets_to_str

OUT_DIR    = Path("kaggle_outputs/step15_features")
SCORED_CSV = OUT_DIR / "scored.csv"
BEST_CSV   = "kaggle_outputs/morpho/rescue/simple-ft_rescue_lf0.2_dm0.05.csv"

# Poison percentiles from step15 calibration
POISON_P25  = 30.88
POISON_P50  = 48.19
POISON_P75  = 53.38

scored = pd.read_csv(SCORED_CSV)
template = pd.read_csv(BEST_CSV)
template["dets"] = template["prediction_string"].apply(parse_dets)

# Build {image_id: {bbox tuple: length}}
lookup = {}
for _, r in scored.iterrows():
    lookup.setdefault(r["image_id"], {})[(r["x"], r["y"], r["w"], r["h"])] = r["bbox_length"]


def apply(predicate, name):
    out, total, non_empty, dropped = [], 0, 0, 0
    for _, row in template.iterrows():
        kept = []
        per_img = lookup.get(row["image_id"], {})
        for det in row["dets"]:
            length = per_img.get(tuple(det[1:]))
            if length is not None and predicate(length):
                dropped += 1
            else:
                kept.append(det)
        out.append(dets_to_str(kept))
        if kept:
            total += len(kept); non_empty += 1
    out_df = template[["id", "image_id"]].copy()
    out_df["prediction_string"] = out
    out_df.to_csv(OUT_DIR / f"{name}.csv", index=False)
    print(f"  {name:45s}  non_empty={non_empty:4d}  total={total:4d}  "
          f"avg={total/len(template):.3f}  dropped={dropped}")


print("=== One-sided 'drop short' filters ===")
print("  (drop dets where length <= T; T near poison percentiles)")
for T in [POISON_P25, 35.0, 40.0, POISON_P50, 50.0, POISON_P75]:
    apply(lambda L, T=T: L <= T, f"filter_length_le_{T:.2f}")

print("\n=== Tight bilateral around poison median ===")
# Drop dets in [poison_med - W, poison_med + W]
for W in [3.0, 5.0, 8.0, 12.0]:
    lo, hi = POISON_P50 - W, POISON_P50 + W
    apply(lambda L, lo=lo, hi=hi: lo <= L <= hi,
          f"filter_length_in_{lo:.1f}_{hi:.1f}")

print("\n=== Hybrid: short AND inside poison range ===")
# Drop only short dets that are also in the poison sweet spot
apply(lambda L: 30.0 <= L <= 50.0, "filter_length_in_30_50")
apply(lambda L: 30.0 <= L <= 53.0, "filter_length_in_30_53")
apply(lambda L: 35.0 <= L <= 50.0, "filter_length_in_35_50")
