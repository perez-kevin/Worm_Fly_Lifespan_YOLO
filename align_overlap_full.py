import cv2
import numpy as np
import os
import re
from glob import glob

# align_image: use src1 as the reference image to transform src2
def align_image(src1, src2, warp_mode=cv2.MOTION_TRANSLATION):
    img1_gray = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    num_iters = 1000
    termination_eps = 1e-8
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iters, termination_eps)

    (cc, warp_matrix) = cv2.findTransformECC(img1_gray, img2_gray, warp_matrix, warp_mode,
                                             criteria, inputMask=None, gaussFiltSize=1)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        img2_aligned = cv2.warpPerspective(src2, warp_matrix, (src1.shape[1], src1.shape[0]),
                                           flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        img2_aligned = cv2.warpAffine(src2, warp_matrix, (src1.shape[1], src1.shape[0]),
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img2_aligned


JpgDir = 'Images/Worms/RawJpg/'
ExpName = '2'                      # experiment / folder prefix
OutDir = os.path.join(JpgDir, 'output')
os.makedirs(OutDir, exist_ok=True)

# Find all images for this experiment and sort by day number
pattern = os.path.join(JpgDir, f'{ExpName}_*.jpg')
files = glob(pattern)

def day_num(path):
    m = re.match(rf'{ExpName}_(\d+)\.jpg$', os.path.basename(path))
    return int(m.group(1)) if m else None

files = sorted([f for f in files if day_num(f) is not None], key=day_num)
days = [day_num(f) for f in files]
print('Found days:', days)

if not files:
    raise SystemExit(f'No images matching {pattern}')

# Reference = first day
ref = cv2.imread(files[0], cv2.IMREAD_COLOR)

# 1) Align every day to day 1; store aligned images keyed by day
aligned = {days[0]: ref}            # day 1 aligned to itself = itself
for f, d in zip(files[1:], days[1:]):
    tgt = cv2.imread(f, cv2.IMREAD_COLOR)
    print(f'Aligning day {d} to day {days[0]} (findTransformECC may take a while)...')
    a = align_image(ref, tgt)
    aligned[d] = a
    cv2.imwrite(os.path.join(OutDir, f'{ExpName}_{d}_Aligned.jpg'), a)

# 2) Overlap from one day to the next (consecutive in the sorted list)
#    merged = [B,G of earlier day] + [R of later day]
for d_prev, d_next in zip(days[:-1], days[1:]):
    (B, G, R) = cv2.split(aligned[d_prev])
    (B1, G1, R1) = cv2.split(aligned[d_next])
    merged = cv2.merge([B, G, R1])
    cv2.imwrite(os.path.join(OutDir, f'{ExpName}_{d_prev}_to_{d_next}_Overlapped.jpg'), merged)
    print(f'Overlapped day {d_prev} -> {d_next}')