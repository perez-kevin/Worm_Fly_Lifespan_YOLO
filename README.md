# Worm_Fly_Lifespan_YOLO

## Introduction
In this repository we provide the code and examples for training and inference of our detection models for dead worms and flies. 

Manuscript.
Perez et al., 2026 "A Cross-Species Drug-Discovery Platform to Accelerate the Identification of Lifespan-Extending Interventions".

Image acquisition. 
Worm images were acquired on a stereo zoom Nikon microscope (SMZ1270). Fly images on a Epson Perfection V600 flatbed scanners. 

## Training

Best to run code on Google Colab with GPU.
training.py: YOLO training code for worms and flies detection models
config_v8.yaml: config file for YOLO model

```
# Example training code for worms
!yolo task=detect mode=train model=yolov8s.pt data='{config}' epochs=100 imgsz=640 cache=True batch=16 patience=35 hsv_h=0.1 hsv_s=0.2 hsv_v=0.2 degrees=0 translate=0.05 scale=0.05 shear=0 perspective=0 flipud=0.5 fliplr=0.5 mosaic=0.5 mixup=0 copy_paste=0.3
```

Training dataset is not provided, available upon request.

## Alignment

align_overlap.py: code to align and overlap 2 consecutive worm images\

```
# align_image: use src1 as the reference image to transform src2
def align_image(src1, src2, warp_mode=cv2.MOTION_TRANSLATION):
```

align_overlap_full.py: code to align and overlap a full worm lifespan
Files should be labelled as ExpID_Timepoint: ex., ExpID_1, ExpID_4, ExpID_7...

## Inference

detect.py: YOLO detection code for worms and flies\
detect_full.py: YOLO detection code for full lifespan analysis

Model weights hosted on Hugging Face\
kprz/Worms_Fly_Lifespan_YOLO

## Correspondence

Kevin Perez\
Institute of Oncology Research (IOR), Bellinzona, Switzerland\
email: kevin.perez@ior.usi.ch

## Example results

![alt text](https://github.com/perez-kevin/Worm_Fly_Lifespan_YOLO/blob/main/Images/worm1.png))
![alt text](https://github.com/perez-kevin/Worm_Fly_Lifespan_YOLO/blob/main/Images/fly1.png)
