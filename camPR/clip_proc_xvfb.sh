#!/bin/bash

# --- Function for Usage/Help ---
usage() {
  echo "Usage: $0 [options]"
  echo ""
  echo "This script processes images from multiple cameras in parallel."
  echo ""
  echo "Options:"
  echo "  -b, --batch_size <N>   Set the number of parallel camera processing jobs. Default is 3."
  echo "  -h, --help             Display this help message and exit."
}

# --- CONFIGURATION (with defaults) ---
INPUT_DIR="input_tmp"
CAM_CONFIG_FILE="cameras.cfg"
EGO_POSE_CSV="null_0_0_0_local2global_pose.csv"
CAM_POSE_FILE="null_0_0_0_local2global_cam_pose.csv"
MAIN_SCRIPT="main_SOPR.py"
CAM_NAMES=("camera_1" "camera_4" "panoramic_1" "panoramic_2" "panoramic_3" "panoramic_4")

# Set the default batch size. This can be overridden by command-line arguments.
BATCH_SIZE=3

# --- PARSE COMMAND-LINE ARGUMENTS ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--batch_size)
      # Check if a value was provided for the batch size argument
      if ! [[ "$2" =~ ^[0-9]+$ ]]; then
        echo "Error: Argument for $1 is missing or not a number." >&2
        usage
        exit 1
      fi
      BATCH_SIZE="$2"
      shift 2 # Consume argument and value
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: Unknown option '$1'" >&2
      usage
      exit 1
      ;;
  esac
done


# --- INITIAL SETUP (Sequential) ---
echo "Running initial setup..."
python3 gwm_init_cam_poses.py -c ${INPUT_DIR}/${CAM_CONFIG_FILE} -p ${INPUT_DIR}/${EGO_POSE_CSV} -o ${INPUT_DIR}/${CAM_POSE_FILE}

# 1. Check if the main input directory exists
if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: Input directory '$INPUT_DIR' does not exist."
  exit 1
fi

echo "Generating query image lists for all cameras..."
# 2. Loop through the specified folder names to create image lists
for CAMERA_NAME in "${CAM_NAMES[@]}"; do
  CURRENT_SUBDIR="${INPUT_DIR}/${CAMERA_NAME}"
  OUTPUT_FILE="${INPUT_DIR}/query_image_list_${CAMERA_NAME}.txt"

  if [ ! -d "$CURRENT_SUBDIR" ]; then
    echo "Warning: Subdirectory '$CURRENT_SUBDIR' not found. Skipping."
    continue
  fi

  # Find JPG/JPEG files and write their names to the output file
  find "$CURRENT_SUBDIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" \) -printf "%f\n" | sort > "$OUTPUT_FILE"

  if [ -f "$OUTPUT_FILE" ]; then
    NUM_FILES=$(wc -l < "$OUTPUT_FILE")
    echo "  Found $NUM_FILES image(s) in '$CAMERA_NAME' and wrote to '$OUTPUT_FILE'."
  else
    echo "  No image files found or error creating '$OUTPUT_FILE' for '$CAMERA_NAME'."
  fi
done

# --- PARALLEL PROCESSING SECTION ---

# Define a function that contains the work for a single camera.
process_camera() {
  local CAMERA_NAME=$1
  local LOG_FILE="output/log_${CAMERA_NAME}.txt"
  
  echo "--- Starting processing for CAMERA: ${CAMERA_NAME} (Log: ${LOG_FILE}) ---"
  
  local camera_start_time=$(date +%s%N)

  {
    echo "Running Step 1 for ${CAMERA_NAME}..."
    ./xvfb_wrapper.sh python3 ${MAIN_SCRIPT} -c "${CAMERA_NAME}" -s 1
    if [ $? -ne 0 ]; then
      echo "  ERROR: ${MAIN_SCRIPT} -s 1 failed for ${CAMERA_NAME}. Aborting this camera."
      exit 1
    fi

    echo "Running Step 2 for ${CAMERA_NAME}..."
    python3 ${MAIN_SCRIPT} -c "${CAMERA_NAME}" -s 2
    if [ $? -ne 0 ]; then
      echo "  ERROR: ${MAIN_SCRIPT} -s 2 failed for ${CAMERA_NAME}. Aborting this camera."
      exit 1
    fi

    echo "Running Step 3 for ${CAMERA_NAME}..."
    python3 ${MAIN_SCRIPT} -c "${CAMERA_NAME}" -s 3
    if [ $? -ne 0 ]; then
      echo "  ERROR: ${MAIN_SCRIPT} -s 3 failed for ${CAMERA_NAME}. Aborting this camera."
      exit 1
    fi

    local camera_end_time=$(date +%s%N)
    local camera_duration_ns=$((camera_end_time - camera_start_time))
    local camera_duration_s=$(awk -v ns="$camera_duration_ns" 'BEGIN {printf "%.3f", ns / 1000000000}')
    echo "Time taken for CAMERA ${CAMERA_NAME}: ${camera_duration_s} seconds."
    
    echo "Running cleanup for ${CAMERA_NAME}..."
    local vis_path="output/vis_proj"
    if [ ! -d "$vis_path" ]; then
       mkdir -p "$vis_path"
    fi
    
    cp -r "output/${CAMERA_NAME}/hloc/visualizations/pnp/"* "$vis_path/"
    cp "output/${CAMERA_NAME}/"*.txt "output/"
    cp "output/${CAMERA_NAME}/"*.csv "output/"
    
    local folder_path="output/${CAMERA_NAME}"
    if [ -d "$folder_path" ]; then
      echo "Deleting tmp folder: $folder_path"
      rm -rf "$folder_path"
    fi
    echo "--- Finished processing for CAMERA: ${CAMERA_NAME} ---"

  } > "$LOG_FILE" 2>&1
}

# Export the function and necessary variables so they are available to subshells.
export -f process_camera
export MAIN_SCRIPT

# Set PYTHONPATH for all subshells
export PYTHONPATH=../

# Ensure output directories exist before starting
mkdir -p output/vis_proj

echo "Starting parallel processing for all cameras (Batch Size: ${BATCH_SIZE})..."
overall_start_time=$(date +%s%N)

# Use xargs to run the `process_camera` function in parallel
printf "%s\n" "${CAM_NAMES[@]}" | xargs -I {} -P ${BATCH_SIZE} bash -c 'process_camera "{}"'

overall_end_time=$(date +%s%N)
overall_duration_ns=$((overall_end_time - overall_start_time))
overall_duration_s=$(awk -v ns="$overall_duration_ns" 'BEGIN {printf "%.3f", ns / 1000000000}')

echo ""
echo "===================================================="
echo "Parallel processing finished."
echo "Total processing time for all cameras: ${overall_duration_s} seconds."
echo "Check individual logs in output/ for details (e.g., output/log_camera_1.txt)."
echo "===================================================="