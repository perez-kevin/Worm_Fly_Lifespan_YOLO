# Worm_Fly_Lifespan_YOLO

## Introduction
In this repository we provide the code and examples for training and inference of our detection models for dead worms and flies. 

Manuscript.
Perez et al., 2026 "A Cross-Species Drug-Discovery Platform to Accelerate the Identification of Lifespan-Extending Interventions".

Image acquisition. 
Worm images were acquired on a stereo zoom Nikon microscope (SMZ1270). Fly images on a Epson Perfection V600 flatbed scanners. 

## File structure

align_overlap.py: code to align and overlap 2 consecutive worm images\
align_overlap_full.py: code to align and overlap a full worm lifespan

detect.py: YOLO detection code for worms and flies\
detect_full.py: YOLO detection code for full lifespan analysis

training.py: YOLO training code for worms and flies

## Additional information

Model weights hosted on Hugging Face\
kprz/Worms_Fly_Lifespan_YOLO

Training dataset not provided, available upon request kevin.perez@ior.usi.ch

## Example results

![alt text](https://github.com/perez-kevin/Worm_Fly_Lifespan_YOLO/blob/main/Images/worm1.png))
![alt text](https://github.com/perez-kevin/Worm_Fly_Lifespan_YOLO/blob/main/Images/fly1.png)
