import os
import sys
import io
from kaggle.api.kaggle_api_extended import KaggleApi

# Force UTF-8 encoding for stdout/stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

api = KaggleApi()
api.authenticate()

kernel_id = 'jasonkimmmmmmmm/step17c-emb-233'
path = 'kaggle_outputs/step17c_233base'

if not os.path.exists(path):
    os.makedirs(path)

print(f"Downloading output files for {kernel_id}...")
try:
    api.kernels_output(kernel_id, path)
except Exception as e:
    print(f"Caught expected encoding error: {e}")

print("Checking if any files were downloaded...")
for f in os.listdir(path):
    print(f"  {f}")

print("Done.")
