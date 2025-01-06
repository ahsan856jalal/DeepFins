#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized KMeans Clustering for Motion Segmentation with Contour Threshold
"""

from skimage.io import imread
import numpy as np
import cv2
import os
from os.path import join
from sklearn.cluster import KMeans

# Parameters
n_colors = 24
threshold = 100000  # Contour threshold for cluster masking
adv_ms_save_dir = 'kmeans_optical_dense_24_color_100000'
data_dir = 'motion_segmentation'

# Create output directory
os.makedirs(adv_ms_save_dir, exist_ok=True)

# Read video file list
with open('test_list.txt', 'r') as file:
    video_used = [line.strip() for line in file]

# Process videos
for vid_counter, video_name in enumerate(video_used, 1):
    video_name1 = os.path.basename(video_name)
    v_split = video_name1.split('.')
    frame_on_gt = int(v_split[2])
    output_path = join(adv_ms_save_dir, f"{v_split[0]}.{v_split[1]}.{frame_on_gt}.png")
    
    if os.path.exists(output_path):
        print(f"Skipping {vid_counter}/{len(video_used)}: {video_name1}")
        continue

    print(f"Processing {vid_counter}/{len(video_used)}: {video_name1}")
    sample_img = imread(join(data_dir, video_name1))
    w, h, _ = sample_img.shape
    reshaped_img = sample_img.reshape(-1, 3)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_colors, random_state=0, n_init=5).fit(reshaped_img)
    labels = kmeans.labels_
    hist, _ = np.histogram(labels, bins=np.arange(n_colors + 1))

    # Identify clusters to mask (set to 0 if contour > threshold)
    identified_palette = kmeans.cluster_centers_.astype(int)
    mask = hist > threshold
    identified_palette[mask] = 0  # Zero out clusters exceeding the threshold

    # Recolor image based on the updated palette
    recolored_img = identified_palette[labels].reshape(w, h, 3)

    # Save histogram and recolored image
    np.savetxt(join(adv_ms_save_dir, f"{v_split[0]}.{v_split[1]}.{frame_on_gt}.txt"), hist, fmt='%d')
    cv2.imwrite(output_path, recolored_img)
