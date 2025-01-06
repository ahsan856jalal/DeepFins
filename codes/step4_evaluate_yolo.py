#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 6 11:23:15 2025

@author: ahsanjalal
"""

from ultralytics import YOLO
import os
import glob
import numpy as np
from pathlib import Path

def load_yolo_labels(file_path):
    """Load YOLO format labels from a text file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    labels = [list(map(float, line.strip().split())) for line in lines]
    return np.array(labels)

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to corner coordinates
    box1 = [x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2]
    box2 = [x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2]

    # Intersection
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

# Parameters
model_path = "best.pt"
gt_dir = "original_test_frames"  # Directory containing test images
output_dir = "yolo_predictions"  # Directory to save predictions

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the YOLO model
model = YOLO(model_path)

# Check for already processed files
processed_files = set(Path(output_dir).glob("results/labels/*.txt"))

# Perform predictions and save results
for img_file in glob.glob(f"{gt_dir}/*.jpg"):  # Assuming test images are in JPG format
    file_name = Path(img_file).stem
    pred_file_path = os.path.join(output_dir, "results", "labels", f"{file_name}.txt")
    
    if pred_file_path in processed_files:
        print(f"Skipping {file_name}: Output file already exists.")
        continue

    # Run YOLO prediction
    results = model.predict(source=img_file, save=True, save_txt=True, project=output_dir, name="results")

# Prediction results directory
pred_dir = output_dir # check path if getting no results

# Metrics calculation
tp, fp, fn = 0, 0, 0

# Iterate through ground truth files
for gt_file in glob.glob(f"{gt_dir}/*.txt"):
    file_name = Path(gt_file).stem
    pred_file = os.path.join(pred_dir, f"{file_name}.txt")

    # Load ground truth and predictions
    gt_labels = load_yolo_labels(gt_file)
    pred_labels = load_yolo_labels(pred_file) if os.path.exists(pred_file) else np.array([])

    # Match ground truth and predictions
    for gt in gt_labels:
        matched = False
        for pred in pred_labels:
            if gt[0] == pred[0]:  # Same class
                iou = calculate_iou(gt[1:], pred[1:])  # Calculate IoU for bounding boxes
                if iou >= 0.5:  # IoU threshold
                    matched = True
                    break
        if matched:
            tp += 1
        else:
            fn += 1

    # False positives: predictions that don't match any ground truth
    for pred in pred_labels:
        matched = False
        for gt in gt_labels:
            if gt[0] == pred[0]:  # Same class
                iou = calculate_iou(gt[1:], pred[1:])  # Calculate IoU for bounding boxes
                if iou >= 0.5:
                    matched = True
                    break
        if not matched:
            fp += 1

# Calculate metrics
precision = tp / (tp + fp) if tp + fp > 0 else 0
recall = tp / (tp + fn) if tp + fn > 0 else 0
f_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
print('Standalone YOLO predictions')
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F-Score: {f_score:.4f}")
