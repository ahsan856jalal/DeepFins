import numpy as np
import os
from os.path import join

# Directories
yolo_hist = 'yolo_predictions'
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
    if not os.path.exists(join(yolo_hist, img_name)):
        yolo_txt = []
    else:
        with open(join(yolo_hist, img_name), 'r') as a:
            yolo_txt = a.readlines()
        yolo_flag = 1

    # Load optical k-means predictions
    if not os.path.exists(join(optical_kmeans_hist, img_name)):
        optical_txt = []
    else:
        with open(join(optical_kmeans_hist, img_name), 'r') as b:
            optical_txt = b.readlines()
        optical_flag = 1

    # Merge YOLO and optical k-means predictions
    if yolo_flag == 1 and optical_flag == 1:
        yolo_txt.extend(optical_txt)
    elif yolo_flag == 0 and optical_flag == 1:
        yolo_txt = optical_txt

    # Create output content
    xml_content = ""
    for obj in yolo_txt:
        obj = obj.rstrip().split(' ')
        xml_content += f"{int(obj[0])} {float(obj[1]):.6f} {float(obj[2]):.6f} {float(obj[3]):.6f} {float(obj[4]):.6f}\n"

    # Save the results
    with open(output_file, "w") as f:
        f.write(xml_content)

print("Processing complete.")
