#!/bin/bash

# Set the folder path
folder_path="DeepFins/codes"

# Move to the folder
cd "$folder_path" || { echo "Error: Could not change directory to $folder_path"; exit 1; }

# Run each script individually
python step2_motion_segmentation.py
python step3_filtering_MS.py
python step4_evaluate_yolo.py
python step5_classify_MS_data.py
python step6_merge_SM_yolo.py
python step7_accuracy_mketrics_evaluation.py
