#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

# --- CONFIGURATION ---
ALL_CAMERA_NAMES=("camera_1" "camera_4" "panoramic_1" "panoramic_2" "panoramic_3" "panoramic_4")
INPUT_BASE_DIR="input_tmp"
REFINED_DATA_DIR="output"
COLMAP_PROJECT_NAME="colmap_project_mvs"
PROJECT_DIR="${REFINED_DATA_DIR}/${COLMAP_PROJECT_NAME}"
PREPARE_SCRIPT="prepare_colmap_project.py"
GENERATE_PAIRS_SCRIPT="generate_fusion_cfg.py"

# --- MVS PARAMETERS ---
# You can easily adjust these values if needed.
# Minimum depth for stereo search, in meters.
DEPTH_MIN=2.0
# Maximum depth for stereo search, in meters.
DEPTH_MAX=100.0

# --- FUNCTION DEFINITIONS ---
log_info() { echo -e "\n========================================================================\n$(date '+%Y-%m-%d %H:%M:%S') - INFO: $1\n========================================================================"; }
log_error() { echo -e "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n$(date '+%Y-%m-%d %H:%M:%S') - ERROR: $1\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"; exit 1; }
check_dependency() { if ! command -v "$1" &> /dev/null; then log_error "$1 could not be found."; fi; }

# --- SCRIPT EXECUTION ---
log_info "Checking for required command-line tools..."
check_dependency "python3"
check_dependency "colmap"
echo "All dependencies found."

log_info "Setting up COLMAP project directory structure..."
if [ -d "$PROJECT_DIR" ]; then
  log_info "Project directory '${PROJECT_DIR}' already exists. Removing it."
  rm -rf "$PROJECT_DIR"
fi
mkdir -p "${PROJECT_DIR}/images"
mkdir -p "${PROJECT_DIR}/dense"
echo "Directory structure created at '${PROJECT_DIR}'."

log_info "Running Python script to generate COLMAP model files..."
if [ ! -f "$PREPARE_SCRIPT" ]; then log_error "'${PREPARE_SCRIPT}' not found."; fi
python3 "$PREPARE_SCRIPT" --input_dir "$INPUT_BASE_DIR" --refined_data_dir "$REFINED_DATA_DIR" --colmap_project_dir "$PROJECT_DIR" --cameras "${ALL_CAMERA_NAMES[@]}"
if [ ! -f "${PROJECT_DIR}/sparse/cameras.txt" ]; then log_error "Python script failed to create model files."; fi
log_info "Successfully generated COLMAP text model."

log_info "Copying original camera images into the project..."
for cam_name in "${ALL_CAMERA_NAMES[@]}"; do
  cam_image_dir="${INPUT_BASE_DIR}/${cam_name}"
  if [ -d "$cam_image_dir" ]; then
    echo "  Copying images from ${cam_image_dir}"
    mkdir -p "${PROJECT_DIR}/images/${cam_name}"
    cp -r "${cam_image_dir}/"* "${PROJECT_DIR}/images/${cam_name}/"
  else
    echo "  Warning: Image directory '${cam_image_dir}' not found. Skipping."
  fi
done
echo "Image copying complete."

log_info "Starting COLMAP MVS pipeline..."

log_info "[Step 5a] Importing text model to binary format..."
colmap model_converter --input_path "${PROJECT_DIR}/sparse" --output_path "${PROJECT_DIR}/sparse" --output_type BIN
echo "Binary model created successfully."

log_info "[Step 5b] Undistorting images for MVS..."
colmap image_undistorter --image_path "${PROJECT_DIR}/images" --input_path "${PROJECT_DIR}/sparse" --output_path "${PROJECT_DIR}/dense" --output_type COLMAP
echo "Image undistortion complete."

log_info "[Step 5c-pre] Generating custom image pairs for stereo matching..."
if [ ! -f "$GENERATE_PAIRS_SCRIPT" ]; then log_error "'${GENERATE_PAIRS_SCRIPT}' not found."; fi
python3 "$GENERATE_PAIRS_SCRIPT" --workspace_path "$PROJECT_DIR"
if [ ! -f "${PROJECT_DIR}/dense/fusion.cfg" ]; then log_error "Failed to generate fusion.cfg."; fi
log_info "Custom stereo pairs generated."

log_info "[Step 5c] Running patch match stereo (dense matching)... This will take a while."
log_info "Using depth range: Min=${DEPTH_MIN}m, Max=${DEPTH_MAX}m"
# --- MODIFICATION: Added depth_min and depth_max flags ---
colmap patch_match_stereo \
  --workspace_path "${PROJECT_DIR}/dense" \
  --workspace_format "COLMAP" \
  --PatchMatchStereo.geom_consistency true \
  --config_path "${PROJECT_DIR}/dense/fusion.cfg" \
  --PatchMatchStereo.depth_min ${DEPTH_MIN} \
  --PatchMatchStereo.depth_max ${DEPTH_MAX}
echo "Patch match stereo complete."

log_info "[Step 5d] Fusing depth maps into a final point cloud..."
colmap stereo_fusion \
  --workspace_path "${PROJECT_DIR}/dense" \
  --output_path "${PROJECT_DIR}/dense/fused.ply" \
  --input_type "photometric" \
  --StereoFusion.min_num_pixels 3 \
  --StereoFusion.max_reproj_error 4.0
echo "Stereo fusion complete."

log_info "MVS Pipeline Finished Successfully!"
echo "The final dense point cloud is at: ${PROJECT_DIR}/dense/fused.ply"