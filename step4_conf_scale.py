"""Apply confidence scaling to an existing submission CSV."""

from pathlib import Path
import sys
import pandas as pd

SRC = "kaggle_outputs/threshold_sweep/simple-ft_conf0.6.csv"
OUT_DIR = Path("kaggle_outputs/threshold_sweep")


def scale_csv(src, scale):
    df = pd.read_csv(src)

    def scale_row(s):
        s = (s or "").strip()
        if not s:
            return " "
        parts = s.split()
        out = []
        for i in range(0, len(parts), 5):
            c, x, y, w, h = parts[i:i+5]
            c_new = float(c) * scale
            out.append(f"{c_new:.6f} {x} {y} {w} {h}")
        return " ".join(out)

    df["prediction_string"] = df["prediction_string"].apply(scale_row)
    out_path = OUT_DIR / f"simple-ft_conf0.6_scale{scale}.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    for scale in [0.5, 0.7]:
        scale_csv(SRC, scale)
