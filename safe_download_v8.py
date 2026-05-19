import os
import sys
import io
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Kernel ID for a specific version
kernel_id = 'jasonkimmmmmmmm/step17-embedding-dist'
out_path = 'kaggle_outputs/step17_v8_retry'

if not os.path.exists(out_path):
    os.makedirs(out_path)

# Based on Version 8 logic: OUT_DIR = Path("/kaggle/working")
# The files should be at the root of the output.
# We can't specify version in kernel_output_file easily if the API doesn't support it,
# but we can try the URL-style ID or just hope latest version with files works if we find one.

# Actually, let's use the list_files logic first to SEE what's there
# We need to find the right API call for that.

print(f"Downloading from {kernel_id}...")
files_to_try = [
    'scored_dets.csv',
    'filter_emb_T0.50.csv',
    'filter_emb_T0.60.csv',
    'filter_emb_T0.70.csv',
    'filter_emb_T0.75.csv',
    'filter_emb_T0.80.csv',
    'filter_emb_T0.85.csv',
    'filter_emb_T0.90.csv'
]

for f in files_to_try:
    try:
        # Try downloading from the latest version that has these files
        api.kernel_output_file(kernel_id, f, out_path)
        print(f"  SUCCESS: {f}")
    except Exception as e:
        # print(f"  FAILED: {f}")
        pass

print("Done.")
