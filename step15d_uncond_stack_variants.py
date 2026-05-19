"""
Step 15d — sweep around the 232.63 winner (uncond-only stack le40 OR [45.2, 51.2]).

Three axes to probe:
  A. Tighter bilateral second range (less density cost, less real-loss)
  B. Wider one-sided cut on uncond bucket (more poison removal, more density cost)
  C. Wider bilateral (more poison removal in the median band)

The 232.63 result beat naive additivity (predicted 233.13) by 0.5 pts —
the uncond restriction is structurally correct. All variants below keep that.
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

lookup = {}
for _, r in scored.iterrows():
    lookup.setdefault(r["image_id"], {})[(r["x"], r["y"], r["w"], r["h"])] = (
        r["bbox_length"], r["group"]
    )


def apply(predicate, name):
    out, total, non_empty, dropped = [], 0, 0, 0
    drop_buckets = {"unconditional": 0, "rescued": 0}
    for _, row in template.iterrows():
        kept = []
        per_img = lookup.get(row["image_id"], {})
        for det in row["dets"]:
            info = per_img.get(tuple(det[1:]))
            if info is not None and predicate(*info):
                dropped += 1
                drop_buckets[info[1]] += 1
            else:
                kept.append(det)
        out.append(dets_to_str(kept))
        if kept:
            total += len(kept); non_empty += 1
    out_df = template[["id", "image_id"]].copy()
    out_df["prediction_string"] = out
    out_df.to_csv(OUT_DIR / f"{name}.csv", index=False)
    print(f"  {name:60s}  total={total:4d}  drop={dropped:3d}  "
          f"(U={drop_buckets['unconditional']:3d}, R={drop_buckets['rescued']:3d})")


print("=== A. Tighter bilateral second range (uncond-only) ===")
print("    (232.63 baseline: le40 OR [45.2,51.2], drop=135)")
for lo, hi in [(46.0, 50.0), (47.0, 49.0), (46.5, 49.5), (45.5, 50.5)]:
    apply(lambda L, G, lo=lo, hi=hi: G == "unconditional" and (L <= 40 or lo <= L <= hi),
          f"filter_length_uncond_stack_le40_or_{lo:.1f}_{hi:.1f}")

print("\n=== B. Wider one-sided cut (uncond-only) ===")
print("    (single cut, no bilateral — test if widening helps without bilateral)")
for T in [41.0, 42.0, 43.0]:
    apply(lambda L, G, T=T: G == "unconditional" and L <= T,
          f"filter_length_uncond_le_{T:.1f}")

print("\n=== C. Wider bilateral second range (uncond-only) ===")
for lo, hi in [(44.0, 52.0), (43.0, 53.0), (44.0, 54.0), (45.0, 55.0)]:
    apply(lambda L, G, lo=lo, hi=hi: G == "unconditional" and (L <= 40 or lo <= L <= hi),
          f"filter_length_uncond_stack_le40_or_{lo:.1f}_{hi:.1f}")

print("\n=== D. Wider one-sided + bilateral combined ===")
for T_one, (lo, hi) in [(41.0, (45.2, 51.2)),
                          (42.0, (45.2, 51.2)),
                          (41.0, (46.0, 50.0)),
                          (42.0, (46.0, 50.0))]:
    apply(lambda L, G, T=T_one, lo=lo, hi=hi:
              G == "unconditional" and (L <= T or lo <= L <= hi),
          f"filter_length_uncond_stack_le{T_one:.0f}_or_{lo:.1f}_{hi:.1f}")
