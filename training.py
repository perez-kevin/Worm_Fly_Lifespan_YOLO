## Training dataset not provided, available upon request: kevin.perez@ior.usi.ch

# Preferable to run code on Google Colab with GPU
# !pip install -U ultralytics

from ultralytics import YOLO

root = r"TRAINING_FOLDER"
config = root + r"config_v8.yaml"

# For worms
!yolo task=detect mode=train model=yolov8s.pt data='{config}' epochs=100 imgsz=640 cache=True batch=16 patience=35 hsv_h=0.1 hsv_s=0.2 hsv_v=0.2 degrees=0 translate=0.05 scale=0.05 shear=0 perspective=0 flipud=0.5 fliplr=0.5 mosaic=0.5 mixup=0 copy_paste=0.3

# For flies
!yolo task=detect mode=train model=yolov8m.pt data='{config}' epochs=100 imgsz=640 cache=True batch=16 patience=35 flipud=0.5 fliplr=0.5 mosaic=0.5 mixup=0 copy_paste=0.3

# Collect model weights
