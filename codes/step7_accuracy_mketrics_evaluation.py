#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:48:50 2025

@author: ahsanjalal
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated to loop over gt_txt first.
"""

import os
import cv2
from os.path import join

# Directory paths
optical_dir = 'merged_SM_yolo'
gt_dir = 'original_test_frames'

# Read video list
with open('test_list.txt', 'r') as a:
    video_used = a.readlines()

# Parameters
width, height = 1920, 1080
TP, FP, FN = 0, 0, 0
total_gt, total_det = 0, 0
iou_sum, ct = 0, 0

# Process each video
for vid_counter, vids in enumerate(video_used, start=1):
    print(f"Processing video {vid_counter}")
    video_name = vids.strip().split('/')[-1]
    v_split = video_name.split('.')
    frame_from_vid = int(v_split[2])
    frame_on_gt = frame_from_vid
    img_name = f"{v_split[0]}.{v_split[1]}.{frame_on_gt}.txt"

    # Read YOLO detections
    yolo_txt = []
    if os.path.exists(join(optical_dir, img_name)):
        with open(join(optical_dir, img_name)) as file:
            yolo_txt = file.readlines()
            total_det += len(yolo_txt)

    # Read ground truth
    gt_txt = []
    if os.path.exists(join(gt_dir, img_name)):
        with open(join(gt_dir, img_name)) as file:
            gt_lines = file.readlines()
            gt_count = len(gt_lines)
            total_gt += gt_count
            ct += gt_count
    else:
        gt_lines = []

    # Process ground truth annotations
    for count_gt_line, line_gt in enumerate(gt_lines):
        line_gt = line_gt.strip()
        coords = line_gt.split(' ')
        label_gt = int(coords[0])
        w_gt = round(float(coords[3]) * width)
        h_gt = round(float(coords[4]) * height)
        x_gt = round(float(coords[1]) * width)
        y_gt = round(float(coords[2]) * height)

        xmin_gt = max(0, int(x_gt - w_gt / 2))
        ymin_gt = max(0, int(y_gt - h_gt / 2))
        xmax_gt = min(width, int(x_gt + w_gt / 2))
        ymax_gt = min(height, int(y_gt + h_gt / 2))

        match_flag = 0

        # Compare against YOLO detections
        for yolo_txt1 in yolo_txt:
            yolo_txt1 = yolo_txt1.strip()
            coords = yolo_txt1.split(' ')
            label_yolo = int(coords[0])
            w_yolo = round(float(coords[3]) * width)
            h_yolo = round(float(coords[4]) * height)
            x_yolo = round(float(coords[1]) * width)
            y_yolo = round(float(coords[2]) * height)

            xmin_yolo = max(0, int(x_yolo - w_yolo / 2))
            ymin_yolo = max(0, int(y_yolo - h_yolo / 2))
            xmax_yolo = min(width, int(x_yolo + w_yolo / 2))
            ymax_yolo = min(height, int(y_yolo + h_yolo / 2))

            # Calculate IOU
            xa = max(xmin_yolo, xmin_gt)
            ya = max(ymin_yolo, ymin_gt)
            xb = min(xmax_yolo, xmax_gt)
            yb = min(ymax_yolo, ymax_gt)

            if xb > xa and yb > ya:
                area_inter = (xb - xa + 1) * (yb - ya + 1)
                area_gt = (xmax_gt - xmin_gt + 1) * (ymax_gt - ymin_gt + 1)
                area_pred = (xmax_yolo - xmin_yolo + 1) * (ymax_yolo - ymin_yolo + 1)
                area_min = min(area_gt, area_pred)
                iou = float(area_inter) / area_min

                if iou >= 0.5:
                    TP += 1
                    match_flag = 1
                    iou_sum += iou
                    break

        # If no match found, count as false negative
        if match_flag == 0:
            FN += 1

    # Count false positives for unmatched YOLO detections
    if len(yolo_txt)>gt_count:
        FP += len(yolo_txt)-gt_count

# Calculate metrics
FN = abs(total_gt - TP)
PR = TP / (TP + FP) if (TP + FP) > 0 else 0
RE = TP / (TP + FN) if (TP + FN) > 0 else 0
F_SCORE = 2 * PR * RE / (PR + RE) if (PR + RE) > 0 else 0

# Print results
print(f"Precision: {PR:.4f}")
print(f"Recall: {RE:.4f}")
print(f"F-score: {F_SCORE:.4f}")
# print(f"Average IOU: {iou_sum / ct:.4f}" if ct > 0 else "Average IOU: N/A")
