#!/usr/bin/env python3

import argparse
import logging
import traceback
from pathlib import Path
import numpy as np
import csv
import re
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import functools
import pandas as pd

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions (Some are new or modified) ---

def parse_value(value_str):
    """Helper to parse values from the .cfg file."""
    value_str = value_str.strip()
    if value_str.endswith(','): value_str = value_str[:-1]
    if value_str.startswith('"') and value_str.endswith('"'): return value_str[1:-1]
    try:
        if value_str.endswith('f'): value_str = value_str[:-1]
        return float(value_str)
    except ValueError: pass
    try: return int(value_str)
    except ValueError: pass
    return value_str

def parse_camera_configs(config_text):
    """Parses the text of a cameras.cfg file into a list of dictionaries."""
    configs = []
    current_config_data = None
    dict_stack = []
    for line in config_text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'): continue
        if line == 'config {':
            current_config_data = {}
            dict_stack = [current_config_data]
            continue
        if line == '}':
            if not dict_stack: continue
            dict_stack.pop()
            if not dict_stack and current_config_data is not None:
                configs.append(current_config_data)
                current_config_data = None
            continue
        if not dict_stack: continue
        match_block = re.match(r'^(\w+)\s*\{$', line)
        if match_block:
            key = match_block.group(1)
            new_dict = {}
            dict_stack[-1][key] = new_dict
            dict_stack.append(new_dict)
            continue
        match_kv = re.match(r'^(\w+)\s*:\s*(.+)$', line)
        if match_kv:
            key, value_str = match_kv.groups()
            dict_stack[-1][key] = parse_value(value_str.strip())
    return configs

def load_original_camera_configs(config_path: Path) -> dict | None:
    """Loads and parses the original cameras.cfg to get ground truth dimensions."""
    if not config_path.is_file():
        logging.error(f"Original camera config file not found: {config_path}")
        return None
    try:
        config_text = config_path.read_text()
        parsed_configs = parse_camera_configs(config_text)
        configs_map = {}
        for cfg in parsed_configs:
            cam_name = cfg.get("camera_dev")
            intrinsic = cfg.get("parameters", {}).get("intrinsic", {})
            width = intrinsic.get("img_width")
            height = intrinsic.get("img_height")
            if cam_name and width and height:
                configs_map[cam_name] = {'width': int(width), 'height': int(height)}
        logging.info(f"Loaded original dimensions for {len(configs_map)} cameras from {config_path.name}")
        return configs_map
    except Exception as e:
        logging.error(f"Failed to load or parse {config_path}: {e}")
        return None


# --- Other helpers remain the same ---
def load_matrix(filepath: Path) -> np.ndarray | None:
    if not filepath.is_file(): logging.error(f"Matrix file not found: {filepath}"); return None
    try: return np.loadtxt(str(filepath))
    except Exception as e: logging.error(f"Failed to load matrix from {filepath}: {e}"); return None

def load_vector(filepath: Path) -> tuple[np.ndarray | None, str | None]:
    if not filepath.is_file(): logging.error(f"Vector file not found: {filepath}"); return None, None
    try:
        vec = np.loadtxt(str(filepath))
        model_type = None
        with open(filepath, 'r') as f:
            for line in f:
                if "# model_type:" in line: model_type = line.split(":")[-1].strip(); break
        return vec, model_type
    except Exception as e: logging.error(f"Failed to load vector from {filepath}: {e}"); return None, None

def load_delta_t_map(filepath: Path) -> dict | None:
    if not filepath.is_file(): logging.error(f"Delta_t CSV file not found: {filepath}"); return None
    dt_map = {}
    try:
        with open(filepath, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if len(row) >= 2: dt_map[row[0]] = float(row[1])
        logging.info(f"Loaded {len(dt_map)} delta_t entries from {filepath.name}")
        return dt_map
    except Exception as e: logging.error(f"Failed to parse delta_t CSV {filepath}: {e}"); return None

def load_and_prepare_ego_poses(csv_path: Path):
    if not csv_path.is_file(): logging.error(f"Ego pose CSV not found: {csv_path}"); return None, None
    try:
        col_names = ['timestamp'] + [f'p{i}' for i in range(16)]
        df = pd.read_csv(str(csv_path), header=None, names=col_names, comment='#')
        df.dropna(inplace=True)
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').astype(np.int64)
        df.sort_values(by='timestamp', inplace=True)
        poses, timestamps_us = [], []
        for _, row in df.iterrows():
            poses.append(row[col_names[1:]].values.astype(np.float64).reshape(4, 4))
            timestamps_us.append(row['timestamp'])
        logging.info(f"Prepared {len(poses)} ego poses."); return np.array(timestamps_us), poses
    except Exception as e: logging.error(f"Error processing ego pose CSV '{csv_path}': {e}", exc_info=True); return None, None

def get_pose_for_timestamp(query_ts_us, timestamps_us, poses, tolerance_us=1000):
    from bisect import bisect_left
    from scipy.spatial.transform import Slerp
    if timestamps_us is None or not poses: return None
    idx = bisect_left(timestamps_us, query_ts_us)
    if idx < len(timestamps_us) and abs(timestamps_us[idx] - query_ts_us) <= tolerance_us: return poses[idx]
    if idx > 0 and abs(timestamps_us[idx - 1] - query_ts_us) <= tolerance_us: return poses[idx-1]
    if idx == 0 or idx == len(timestamps_us): return None
    t0_us, t1_us, pose0, pose1 = timestamps_us[idx - 1], timestamps_us[idx], poses[idx - 1], poses[idx]
    if t1_us <= t0_us: return None
    alpha = (query_ts_us - t0_us) / (t1_us - t0_us)
    R0, R1 = R.from_matrix(pose0[:3, :3]), R.from_matrix(pose1[:3, :3])
    T0, T1 = pose0[:3, 3], pose1[:3, 3]
    slerp = Slerp([t0_us, t1_us], R.concatenate([R0, R1])); R_interp = slerp([query_ts_us])[0]
    T_interp = T0 + alpha * (T1 - T0)
    pose_interp = np.eye(4); pose_interp[:3, :3], pose_interp[:3, 3] = R_interp.as_matrix(), T_interp
    return pose_interp

def load_lidar_points(filepath: Path) -> np.ndarray | None:
    if not filepath.is_file(): logging.error(f"LiDAR map not found: {filepath}"); return None
    try:
        points = np.asarray(o3d.io.read_point_cloud(str(filepath)).points)
        logging.info(f"Loaded {len(points)} points from {filepath.name}"); return points
    except Exception as e: logging.error(f"Failed to load LiDAR points from {filepath}: {e}"); return None

def write_cameras_txt(cameras_map: dict, filepath: Path):
    logging.info(f"Writing {len(cameras_map)} camera model(s) to {filepath}...");
    try:
        with open(filepath, 'w') as f:
            f.write("# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n# Number of cameras: {}\n".format(len(cameras_map)))
            for cam in sorted(cameras_map.values(), key=lambda x: x['id']):
                f.write(f"{cam['id']} {cam['model']} {cam['width']} {cam['height']} {' '.join(map(str, cam['params']))}\n")
        logging.info("Successfully wrote cameras.txt")
    except Exception as e: logging.error(f"Failed to write cameras.txt: {e}")

def write_images_txt(images_list: list, filepath: Path):
    logging.info(f"Writing {len(images_list)} image entries to {filepath}...")
    try:
        with open(filepath, 'w') as f:
            f.write("# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n# Number of images: {}, mean observations per image: 0\n".format(len(images_list)))
            for img in sorted(images_list, key=lambda x: x['id']):
                q, t = img['q'], img['t']
                f.write(f"{img['id']} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {img['cam_id']} {img['name']}\n\n")
        logging.info("Successfully wrote images.txt")
    except Exception as e: logging.error(f"Failed to write images.txt: {e}")

def write_points3D_txt(points: np.ndarray, filepath: Path):
    if points is None or points.shape[0] == 0:
        logging.warning("No 3D points provided. Writing an empty points3D.txt")
        try:
            with open(filepath, 'w') as f:
                f.write("# 3D point list with one line of data per point:\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n# Number of points: 0\n")
            return
        except Exception as e: logging.error(f"Failed to write empty points3D.txt: {e}"); return
    logging.info(f"Writing {len(points)} 3D points to {filepath}...")
    try:
        with open(filepath, 'w') as f:
            f.write("# 3D point list with one line of data per point:\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n# Number of points: {}\n".format(len(points)))
            for i, p in enumerate(points):
                f.write(f"{i+1} {p[0]} {p[1]} {p[2]} 128 128 128 0\n")
        logging.info("Successfully wrote points3D.txt")
    except Exception as e: logging.error(f"Failed to write points3D.txt: {e}")

# --- Main Logic ---

def main(args):
    colmap_project_dir = Path(args.colmap_project_dir)
    refined_data_dir = Path(args.refined_data_dir)
    input_base_dir = Path(args.input_dir)
    camera_names = args.cameras

    sparse_model_dir = colmap_project_dir / "sparse"
    sparse_model_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"COLMAP text model will be generated in: {sparse_model_dir}")

    all_camera_models, all_images_data = {}, []
    next_camera_id, next_image_id = 1, 1

    # --- MODIFICATION: Load original camera configs first ---
    original_configs = load_original_camera_configs(input_base_dir / "cameras.cfg")
    if not original_configs:
        logging.critical("Could not load original camera dimensions from cameras.cfg. Aborting.")
        return

    ego_timestamps_us, ego_poses_list = load_and_prepare_ego_poses(input_base_dir / "null_0_0_0_local2global_pose.csv")
    if ego_timestamps_us is None:
        logging.critical("Failed to load ego poses. Aborting."); return

    ego_interpolator = functools.partial(get_pose_for_timestamp, timestamps_us=ego_timestamps_us, poses=ego_poses_list, tolerance_us=1000)

    for cam_name in camera_names:
        logging.info(f"--- Processing data for camera: {cam_name} ---")

        extrinsics_file = refined_data_dir / f"refined_extrinsics_{cam_name}.txt"
        k_matrix_file = refined_data_dir / f"refined_intrinsics_K_sensor_{cam_name}.txt"
        d_coeffs_file = refined_data_dir / f"refined_intrinsics_D_sensor_{cam_name}.txt"
        delta_t_file = refined_data_dir / f"refined_delta_t_{cam_name}.csv"

        T_ego_cam = load_matrix(extrinsics_file)
        K_sensor = load_matrix(k_matrix_file)
        D_sensor, model_type_from_file = load_vector(d_coeffs_file)
        delta_t_map = load_delta_t_map(delta_t_file)

        if any(v is None for v in [T_ego_cam, K_sensor, D_sensor, delta_t_map]):
            logging.warning(f"Missing one or more refined files for '{cam_name}'. Skipping.")
            continue

        # --- MODIFICATION: Use width/height from original_configs ---
        if cam_name not in original_configs:
            logging.error(f"'{cam_name}' not found in original cameras.cfg. Cannot get dimensions. Skipping.")
            continue
        width = original_configs[cam_name]['width']
        height = original_configs[cam_name]['height']
        logging.info(f"Using original image size for {cam_name}: {width}x{height}")
        # --- END MODIFICATION ---

        if model_type_from_file == "KANNALA_BRANDT": colmap_model_name, intrinsic_params = "OPENCV_FISHEYE", [K_sensor[0,0], K_sensor[1,1], K_sensor[0,2], K_sensor[1,2]] + D_sensor.flatten().tolist()
        elif model_type_from_file == "PINHOLE": colmap_model_name, intrinsic_params = "OPENCV", [K_sensor[0,0], K_sensor[1,1], K_sensor[0,2], K_sensor[1,2]] + (D_sensor.flatten().tolist() + [0]*8)[:8]
        else: logging.error(f"Unknown model '{model_type_from_file}' for {cam_name}. Skipping."); continue
        logging.info(f"Mapped model '{model_type_from_file}' to COLMAP model '{colmap_model_name}'.")

        if cam_name not in all_camera_models:
            all_camera_models[cam_name] = {'id': next_camera_id, 'model': colmap_model_name, 'width': width, 'height': height, 'params': intrinsic_params}
            this_camera_id = next_camera_id; next_camera_id += 1
            logging.info(f"Added new camera model '{cam_name}' with ID {this_camera_id}.")
        else: this_camera_id = all_camera_models[cam_name]['id']

        image_list_file = input_base_dir / f"query_image_list_{cam_name}.txt"
        if not image_list_file.is_file():
            logging.warning(f"Image list not found for {cam_name}. Skipping images."); continue

        for basename in [line.strip() for line in image_list_file.read_text().splitlines()]:
            try:
                rec_ts_us = int(Path(basename).stem)
                true_ts_us = rec_ts_us + (delta_t_map.get(basename, 0.0) * 1_000_000)
                T_map_ego = ego_interpolator(true_ts_us)
                if T_map_ego is None: logging.warning(f"Could not get ego pose for {basename}. Skipping."); continue
                T_cam_map = np.linalg.inv(T_map_ego @ T_ego_cam)
                quat_xyzw = R.from_matrix(T_cam_map[:3, :3]).as_quat()
                all_images_data.append({'id': next_image_id, 'q': [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], 't': T_cam_map[:3, 3], 'cam_id': this_camera_id, 'name': f"{cam_name}/{basename}"})
                next_image_id += 1
            except Exception as e: logging.error(f"Failed to process image {basename} for {cam_name}: {e}"); logging.debug(traceback.format_exc())

    lidar_points = load_lidar_points(input_base_dir / "whole_map.pcd")
    write_cameras_txt(all_camera_models, sparse_model_dir / "cameras.txt")
    write_images_txt(all_images_data, sparse_model_dir / "images.txt")
    write_points3D_txt(lidar_points, sparse_model_dir / "points3D.txt")
    logging.info("\n" + "="*50 + "\nCOLMAP Project Generation Complete.\n" + "="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate COLMAP text model files from refined calibration outputs for MVS.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input_dir', type=str, default="input_tmp", help="Base directory containing camera image lists and LiDAR map.")
    parser.add_argument('-r', '--refined_data_dir', type=str, default="output", help="Directory containing the flat refined .txt and .csv files from the main pipeline.")
    parser.add_argument('-o', '--colmap_project_dir', type=str, default="output/colmap_project_mvs", help="Path to create the new COLMAP project directory.")
    parser.add_argument('-c', '--cameras', nargs='+', required=True, help="List of camera names to process (e.g., camera_1 panoramic_1).")
    main(parser.parse_args())