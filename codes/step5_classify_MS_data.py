import torch
from ultralytics import YOLO
import numpy as np
import cv2
import os
from os.path import join
from natsort import natsorted

# Directories
sot_save_dir = 'filtered_MS'
sot_classifier_dir = 'motion_seg_classified'
temp_dir = 'temp_dir'
rgb_dir = 'original_test_frames'

# Model paths
model_path = "best.pt"  # Replace with your YOLOv8 trained model

# Check for CUDA availability
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLOv8 model with the appropriate device
model = YOLO(model_path)
model.to(device)

# Ensure directories exist
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(sot_classifier_dir, exist_ok=True)

# Read test list
with open('test_list.txt', 'r') as a:
    image_files = a.readlines()
image_files = natsorted([img_name.strip() for img_name in image_files])

count = 0
for img_name in image_files:
    count += 1
    filename = os.path.basename(img_name)
    label_file = join(sot_classifier_dir, filename.replace('.png', '.txt'))
    
    # Check if the file is already processed
    if os.path.exists(label_file):
        print(f"Skipping image {count}: {img_name} (already processed)")
        continue

    print(f"Processing image {count}: {img_name}")

    blobs = []
    obj_arr = []
    
    rgb_img = cv2.imread(img_name)
    rgb_copy = rgb_img.copy()

    # Load the optical flow image if it exists
    if not os.path.exists(join(sot_save_dir, filename)):
        optical_img = np.zeros(rgb_img.shape[:2], dtype=np.uint8)
    else:
        optical_img = cv2.imread(join(sot_save_dir, filename), cv2.IMREAD_GRAYSCALE)

    # Find contours in the optical image
    contours, _ = cv2.findContours(optical_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 10000:
            blobs.append(cnt)

    # Process blobs and perform predictions
    for blb in blobs:
        (x, y, w, h) = cv2.boundingRect(blb)
        img_patch = rgb_img[y:y+h, x:x+w]
    
        # Perform YOLOv8 prediction on the patch
        results = model.predict(img_patch, conf=0.5, verbose=False, device=device)
        if len(results[0].boxes) > 0:
            # Calculate YOLO format coordinates
            x_center = (x + w / 2.0) / rgb_img.shape[1]
            y_center = (y + h / 2.0) / rgb_img.shape[0]
            norm_w = w / rgb_img.shape[1]
            norm_h = h / rgb_img.shape[0]
    
            obj_arr.append([0, x_center, y_center, norm_w, norm_h])

    # Save results
    xml_content = "\n".join([f"{obj[0]} {obj[1]:.6f} {obj[2]:.6f} {obj[3]:.6f} {obj[4]:.6f}" for obj in obj_arr])
    with open(label_file, "w") as f:
        f.write(xml_content)

print('GMM optical combination and classification is done.')
