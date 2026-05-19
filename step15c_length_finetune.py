"""
Step 15c — finer length sweep + stacked filters around the 233.99 winner.

Known points:
  length <= 40        : drop 90  -> 233.99  (-1.63 from 235.62)  WINNER
  length in [45.2,51.2]: drop 94  -> 234.76  (-0.86)
  length <= 48.19     : drop 201 -> 243.11  (+7.49 — overshot)

Diagnosis: one-sided "drop short" is the right shape. Crossover where
poison/real density drops below 1 sits between 40 and 45px.

This sweep:
  A. Fine one-sided around T=40
  B. Stack ≤40 with bilateral [45.2, 51.2] (184 unique drops, no overlap)
  C. "Apply length filter to unconditional bucket only" — preserves the
     dashedness-rescued cohort untouched, which was already validated by
     a different signature.
"""

import pandas as pd
from pathlib import Path

from step6_morpho_filter import parse_dets, dets_to_str

OUT_DIR    = Path("kaggle_outputs/step15_features")
SCORED_CSV = OUT_DIR / "scored.csv"
BEST_CSV   = "kaggle_outputs/morpho/rescue/simple-ft_rescue_lf0.2_dm0.05.csv"

scored = pd.read_csv(SCORED_CSV)
template = pd.read_csv(BEST_CSV)
template["dets"] = template["prediction_string"].apply(parse_dets)

# Build {image_id: {bbox tuple: (length, group)}}
lookup = {}
for _, r in scored.iterrows():
    lookup.setdefault(r["image_id"], {})[(r["x"], r["y"], r["w"], r["h"])] = (
        r["bbox_length"], r["group"]
    )


def apply(predicate, name):
    """predicate(length, group) -> True = drop"""
    out, total, non_empty, dropped = [], 0, 0, 0
    for _, row in template.iterrows():
        kept = []
        per_img = lookup.get(row["image_id"], {})
        for det in row["dets"]:
            info = per_img.get(tuple(det[1:]))
            if info is not None and predicate(*info):
                dropped += 1
            else:
                kept.append(det)
        out.append(dets_to_str(kept))
        if kept:
            total += len(kept); non_empty += 1
    out_df = template[["id", "image_id"]].copy()
    out_df["prediction_string"] = out
    out_df.to_csv(OUT_DIR / f"{name}.csv", index=False)
    print(f"  {name:55s}  non_empty={non_empty:4d}  total={total:4d}  "
          f"avg={total/len(template):.3f}  dropped={dropped}")


print("=== A. Fine one-sided sweep around T=40 ===")
for T in [33.0, 35.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 44.0]:
    apply(lambda L, G, T=T: L <= T, f"filter_length_le_{T:.2f}")

print("\n=== B. Stacked <=40 OR in [45.2, 51.2] (probe additivity) ===")
apply(lambda L, G: L <= 40.0 or (45.2 <= L <= 51.2),
      "filter_length_stack_le40_or_45_51")
# Also tighter bilateral on the stack
apply(lambda L, G: L <= 40.0 or (46.0 <= L <= 50.0),
      "filter_length_stack_le40_or_46_50")
apply(lambda L, G: L <= 40.0 or (47.0 <= L <= 49.0),
      "filter_length_stack_le40_or_47_49")

print("\n=== C. Length filter UNCONDITIONAL bucket only (preserve rescued) ===")
# Same length cuts, but only drop dets from the conf>=0.6 bucket
for T in [37.0, 40.0, 42.0, 45.0]:
    apply(lambda L, G, T=T: G == "unconditional" and L <= T,
          f"filter_length_uncond_le_{T:.2f}")

# Stack on unconditional only
apply(lambda L, G: G == "unconditional" and (L <= 40.0 or (45.2 <= L <= 51.2)),
      "filter_length_uncond_stack_le40_or_45_51")
