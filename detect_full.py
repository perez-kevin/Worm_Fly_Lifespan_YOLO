import os
import re
import csv
from glob import glob
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# 1. Download weights
weights_path = hf_hub_download(
    repo_id="kprz/Worms_Fly_Lifespan_YOLO",
    filename="Worm_YOLO_Weights.pt"
    # filename="Fly_YOLO_Weights.pt"
)
model = YOLO(weights_path)

# --- Config ---
ExpName = '2'
OverlapDir = 'Images/Worms/RawJpg/output'          # where *_Overlapped.jpg were written
PredDir = os.path.join(OverlapDir, 'predictions')
os.makedirs(PredDir, exist_ok=True)
CsvPath = os.path.join(OverlapDir, f'{ExpName}_dead_worm_counts.csv')

# Optional: restrict to a specific "dead" class id. Set to None to count all boxes.
DEAD_CLASS_ID = None   # e.g. 0

# --- Collect overlapped images, sorted by the "next" day in the pair ---
# filenames look like: 2_1_to_4_Overlapped.jpg
pattern = os.path.join(OverlapDir, f'{ExpName}_*_to_*_Overlapped.jpg')
files = glob(pattern)

def pair_days(path):
    m = re.search(rf'{ExpName}_(\d+)_to_(\d+)_Overlapped\.jpg$', os.path.basename(path))
    return (int(m.group(1)), int(m.group(2))) if m else None

files = sorted([f for f in files if pair_days(f)], key=lambda f: pair_days(f)[1])
print('Overlapped images found:', [os.path.basename(f) for f in files])

# --- Predict and count ---
rows = []
for f in files:
    d_prev, d_next = pair_days(f)
    results = model(f)
    result = results[0]

    if DEAD_CLASS_ID is None:
        count = len(result.boxes)
    else:
        cls = result.boxes.cls.cpu().numpy().astype(int)
        count = int((cls == DEAD_CLASS_ID).sum())

    # save annotated image with boxes
    out_img = os.path.join(PredDir, f'{ExpName}_{d_prev}_to_{d_next}_pred.jpg')
    result.save(filename=out_img)

    # "timepoint" = the later day in the consecutive pair
    rows.append({'timepoint_day': d_next,
                 'pair': f'{d_prev}_to_{d_next}',
                 'dead_worms': count})
    print(f'{ExpName}_{d_prev}_to_{d_next}: {count} dead worms')

# --- Write CSV ---
with open(CsvPath, 'w', newline='') as fcsv:
    writer = csv.DictWriter(fcsv, fieldnames=['timepoint_day', 'pair', 'dead_worms'])
    writer.writeheader()
    writer.writerows(rows)

print('Saved CSV:', CsvPath)
print('Saved annotated images to:', PredDir)