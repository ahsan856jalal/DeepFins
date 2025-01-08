#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:36:31 2025

@author: ahsanjalal
"""

import numpy as np
import os
from os.path import join

# Helper function to calculate IoU
def calculate_iou(box1, box2):
    # Box format: [x_center, y_center, width, height]
    x1_min, y1_min = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    x1_max, y1_max = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    x2_min, y2_min = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    x2_max, y2_max = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Calculate union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0

# Directories
yolo_hist = 'yolo_predictions/predict/labels'
optical_kmeans_hist = 'motion_seg_classified'
save_dir = 'merged_SM_yolo'

# Read the test list
with open('test_list.txt', 'r') as a:
    video_used = a.readlines()

# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)

count = 0
for vids in video_used:
    count += 1
    print(f"Processing video {count}: {vids.strip()}")
    
    video_name = vids.rstrip().split('/')[-1]
    v_split = video_name.split('.')
    frame_from_vid = int(v_split[2])
    frame_on_gt = frame_from_vid
    img_name = f"{v_split[0]}.{v_split[1]}.{frame_on_gt}.txt"
    
    # Check if the file has already been processed
    output_file = join(save_dir, img_name)
    if os.path.exists(output_file):
        print(f"Skipping {img_name}, already processed.")
        continue

    yolo_flag = 0
    optical_flag = 0
    
    # Load YOLO predictions
    if os.path.exists(join(yolo_hist, img_name)):
        with open(join(yolo_hist, img_name), 'r') as a:
            yolo_txt = [list(map(float, line.split())) for line in a.readlines()]
        yolo_flag = 1
    else:
        yolo_txt = []

    # Load optical k-means predictions
    if os.path.exists(join(optical_kmeans_hist, img_name)):
        with open(join(optical_kmeans_hist, img_name), 'r') as b:
            optical_txt = [list(map(float, line.split())) for line in b.readlines()]
        optical_flag = 1
    else:
        optical_txt = []

    # Result storage
    merged_annotations = []

    # Case 1: Both YOLO and Optical annotations exist
    if yolo_flag and optical_flag:
        for yolo_ann in yolo_txt:
            matched = False
            for optical_ann in optical_txt:
                if calculate_iou(yolo_ann[1:], optical_ann[1:]) >= 0.5:
                    matched = True
                    merged_annotations.append(yolo_ann)
                    break
            if not matched:
                merged_annotations.append(yolo_ann)

        for optical_ann in optical_txt:
            matched = False
            for yolo_ann in yolo_txt:
                if calculate_iou(optical_ann[1:], yolo_ann[1:]) >= 0.5:
                    matched = True
                    if yolo_ann not in merged_annotations:
                        merged_annotations.append(yolo_ann)
                    break
            if not matched:
                merged_annotations.append(optical_ann)
    # Case 2: Only Optical annotations exist
    elif optical_flag and not yolo_flag:
        print(f"Only optical annotations found for {img_name}.")
        merged_annotations = optical_txt
    # Case 3: Only YOLO annotations exist
    elif yolo_flag and not optical_flag:
        print(f"Only YOLO annotations found for {img_name}.")
        merged_annotations = yolo_txt

    # Create output content
    with open(output_file, "w") as f:
        for ann in merged_annotations:
            f.write(f"{int(ann[0])} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")

print("Processing complete.")
