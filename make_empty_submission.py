"""
Step 2 anchor #1 — generate an empty submission locally (no GPU needed).

Output: empty_submission.csv with " " (single space) for every row.
This bounds the pure-FN penalty side of the mCADD metric.

Kaggle requires empty prediction_string to be " ", not "" — Kaggle treats
empty strings as null and rejects the row.
"""

import csv
from pathlib import Path

ROOT       = Path(__file__).parent
SAMPLE     = ROOT / "neural-debris-removal-in-streak-detection-models" / "sample_submission.csv"
OUT        = ROOT / "empty_submission.csv"


def main():
    with SAMPLE.open() as f:
        reader = csv.DictReader(f)
        rows = [(r["id"], r["image_id"]) for r in reader]

    print(f"Read {len(rows)} rows from sample_submission.csv")

    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "image_id", "prediction_string"])
        for rid, iid in rows:
            w.writerow([rid, iid, " "])

    print(f"Wrote {OUT}  ({len(rows)} empty rows)")


if __name__ == "__main__":
    main()
