#!/bin/bash

INPUT_DIR="input_tmp"
CAM_CONFIG_FILE="cameras.cfg"
EGO_POSE_FILE="lio_prior_sensor_coord.txt"
EGO_POSE_CSV="null_0_0_0_local2global_pose.csv"
CAM_POSE_FILE="null_0_0_0_local2global_cam_pose.csv"

# python3 gwm_init_cam_poses.py -c ${INPUT_DIR}/${CAM_CONFIG_FILE} -p ${INPUT_DIR}/${EGO_POSE_FILE} -e ${INPUT_DIR}/${EGO_POSE_CSV} -o ${INPUT_DIR}/${CAM_POSE_FILE}

# List of camera names
CAM_NAMES=("camera_1" "camera_4" "panoramic_1" "panoramic_2" "panoramic_3" "panoramic_4")

# 1. Check if the main input directory exists
if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: Input directory '$INPUT_DIR' does not exist."
  exit 1
fi

echo "Processing directories in '$INPUT_DIR'..."

# 2. Loop through the specified folder names
for CAMERA_NAME in "${CAM_NAMES[@]}"; do
  CURRENT_SUBDIR="${INPUT_DIR}/${CAMERA_NAME}"
  OUTPUT_FILE="${INPUT_DIR}/query_image_list_${CAMERA_NAME}.txt"

  # 2a. Check if the subdirectory exists
  if [ ! -d "$CURRENT_SUBDIR" ]; then
    echo "Warning: Subdirectory '$CURRENT_SUBDIR' not found. Skipping."
    continue # Skip to the next folder name
  fi

  echo "  Processing folder: '$CAMERA_NAME' -> '$OUTPUT_FILE'"

  # 2c. Find JPG files and write their names to the output file
  # -maxdepth 1: Prevents find from looking into sub-subdirectories
  # -type f: Ensures we only find regular files (not directories named .jpg)
  # -iname: Case-insensitive name matching (for .JPG, .jpg, etc.)
  # -printf "%f\n": Prints only the filename (basename) followed by a newline
  # Using -o for multiple extensions like .jpg and .jpeg
  find "$CURRENT_SUBDIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" \) -printf "%f\n" | sort > "$OUTPUT_FILE"

  # Count lines to give feedback (ensure file exists before wc, find with > creates it)
  if [ -f "$OUTPUT_FILE" ]; then
    NUM_FILES=$(wc -l < "$OUTPUT_FILE")
    echo "    Found $NUM_FILES image(s) in '$CAMERA_NAME' and wrote to '$OUTPUT_FILE'."
  else
    # This case should ideally not happen if find successfully ran and created the file,
    # unless no files were found and find somehow didn't create an empty file (unlikely with '>').
    # Or if find itself failed.
    echo "    No image files found or error creating '$OUTPUT_FILE' for '$CAMERA_NAME'."
  fi

done



CAM_NAMES=("panoramic_2")

export PYTHONPATH=../

echo "Starting processing for all cameras..."
overall_start_time=$(date +%s%N) # High precision start time (nanoseconds)

# Loop through the specified camera names
for CAMERA_NAME in "${CAM_NAMES[@]}"; do
  echo "----------------------------------------------------"
  echo "Processing CAMERA: ${CAMERA_NAME}"
  camera_start_time=$(date +%s%N)

  ./xvfb_wrapper.sh python3 main_SOPR.py -c "${CAMERA_NAME}" -s 1
  if [ $? -ne 0 ]; then
    echo "  ERROR: main_SOPR.py -s 1 failed for ${CAMERA_NAME}. Continuing with next step/camera."
    continue
  fi

  python3 main_SOPR.py -c "${CAMERA_NAME}" -s 2
  if [ $? -ne 0 ]; then
    echo "  ERROR: main_SOPR.py -s 2 failed for ${CAMERA_NAME}. Continuing with next step/camera."
    continue
  fi

  python3 main_SOPR.py -c "${CAMERA_NAME}" -s 3
  if [ $? -ne 0 ]; then
    echo "  ERROR: main_SOPR.py -s 3 failed for ${CAMERA_NAME}. Continuing with next step/camera."
    continue
  fi

  camera_end_time=$(date +%s%N)
  camera_duration_ns=$((camera_end_time - camera_start_time))
  camera_duration_s=$(awk -v ns="$camera_duration_ns" 'BEGIN {printf "%.3f", ns / 1000000000}')
  echo "Time taken for CAMERA ${CAMERA_NAME}: ${camera_duration_s} seconds."
  echo "----------------------------------------------------"
  
  vis_path="output/vis_proj"
  if [ ! -d "$vis_path" ]; then
     mkdir -p "$vis_path"
  fi
  cp -r output/hloc/visualizations/pnp/* "$vis_path"
  folder_path="output/renders"
  if [ -d "$folder_path" ]; then
    echo "Deleting tmp folder: $folder_path"
    rm -rf "$folder_path"
  fi
  folder_path="output/mid_data_cache"
  if [ -d "$folder_path" ]; then
    echo "Deleting tmp folder: $folder_path"
    rm -rf "$folder_path"
  fi
  folder_path="output/hloc"
  if [ -d "$folder_path" ]; then
    echo "Deleting tmp folder: $folder_path"
    rm -rf "$folder_path"
  fi

done

overall_end_time=$(date +%s%N)
overall_duration_ns=$((overall_end_time - overall_start_time))
overall_duration_s=$(awk -v ns="$overall_duration_ns" 'BEGIN {printf "%.3f", ns / 1000000000}')

echo ""
echo "===================================================="
echo "Total processing time for all cameras: ${overall_duration_s} seconds."
echo "===================================================="
