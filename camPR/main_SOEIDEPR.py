
from __future__ import annotations
import open3d as o3d
import logging
import numpy as np
import cv2
import time
import os
import h5py
try:
    import pycolmap
    PYCOLMAP_AVAILABLE = True
except ImportError:
    PYCOLMAP_AVAILABLE = False
    logging.warning("pycolmap not found. Pose refinement step will be skipped.")
from scipy.spatial import KDTree
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from bisect import bisect_left
from pathlib import Path
from hloc import (
    extract_features,
    match_features,
)
import traceback
from scipy.spatial.distance import cdist
import json
import matplotlib.pyplot as plt # For colormap
import matplotlib.cm as cm
from scipy.optimize import (
    least_squares,
    differential_evolution,
)
import functools
from dataclasses import dataclass
import argparse
import csv
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Number of distortion parameters
NUM_K_PARAMS = 4

# --- Preprocessing Function ---
# Includes: Tensor API loading, parallel normals, intensity handling, KDTree
def preprocess_lidar_map(
    lidar_file_path,
    min_height=-2.0, # <-- New parameter for height filtering
    normal_radius=0.1,
    normal_max_nn=50,
    voxel_down_sample_size=0.03,
    device_str="auto"
):
    """
    Loads LiDAR map using Tensor API, filters points below a minimum height,
    optionally downsamples, computes attributes (normals), handles intensity,
    assigns indices, and builds KDTree.

    Args:
        lidar_file_path (str): Path to the LiDAR point cloud file (e.g., .pcd, .ply).
        min_height (float, optional): Minimum Z coordinate. Points with Z values
            strictly less than this will be removed *before* downsampling.
            Set to None to disable height filtering. Defaults to -2.0.
        normal_radius (float): Radius for normal estimation neighborhood search.
        normal_max_nn (int): Max number of neighbors for normal estimation.
        voxel_down_sample_size (float or None): Voxel size for downsampling.
            If None or <= 0, downsampling is skipped. Defaults to 0.03.
        device_str (str): Device for Open3D Tensor operations ('auto', 'CPU:0', 'CUDA:0').
            Defaults to 'auto'.

    Returns:
        tuple: A tuple containing:
            - pcd_tensor (o3d.t.geometry.PointCloud): The processed point cloud
              on the selected device (potentially filtered and downsampled).
            - lidar_data_original (dict): A dictionary containing NumPy arrays
              and the KDTree of the final processed points (after filtering
              and downsampling):
                - 'points': (N, 3) float32 NumPy array of point coordinates.
                - 'indices': (N,) uint32 NumPy array of original 1-based indices
                           (relative to the state *after* filtering/downsampling).
                - 'normals': (N, 3) float32 NumPy array of estimated normals.
                - 'intensities': (N,) float32 NumPy array of normalized intensities (0-1).
                - 'kdtree': KDTree object built on the final 'points'.

    Raises:
        RuntimeError: If loading fails or normal estimation fails unexpectedly.
        ValueError: If the loaded point cloud is empty, or becomes empty after
                    filtering or downsampling.
    """
    print(f"Preprocessing {lidar_file_path}...")
    # --- Device Selection ---
    if device_str == "auto":
        if o3d.core.cuda.is_available():
            device = o3d.core.Device("CUDA:0")
        else:
            device = o3d.core.Device("CPU:0")
    else:
        try:
            device = o3d.core.Device(device_str)
        except Exception as e:
            raise ValueError(f"Invalid device_str '{device_str}': {e}")
    print(f"Using device: {device}")

    # --- Load ---
    try:
        pcd_tensor = o3d.t.io.read_point_cloud(lidar_file_path)
        pcd_tensor = pcd_tensor.to(device) # Move after loading
    except Exception as e:
        raise RuntimeError(f"Failed to load point cloud '{lidar_file_path}': {e}")

    if len(pcd_tensor.point.positions) == 0:
        raise ValueError("Loaded point cloud is empty.")
    print(f"Loaded {len(pcd_tensor.point.positions):,} points. Initial Attrs: {list(pcd_tensor.point)}")

    # --- Height Filtering (BEFORE Downsampling) ---
    if min_height is not None:
        print(f"Filtering points below Z = {min_height}...")
        num_pts_before_filter = len(pcd_tensor.point.positions)
        # Ensure min_height is a tensor on the correct device for comparison
        min_height_tensor = o3d.core.Tensor([min_height], dtype=o3d.core.Dtype.Float32, device=device)
        keep_mask = pcd_tensor.point.positions[:, 2] >= min_height_tensor
        pcd_tensor = pcd_tensor.select_by_mask(keep_mask)
        num_pts_after_filter = len(pcd_tensor.point.positions)
        if num_pts_after_filter == 0:
            raise ValueError(f"Point cloud empty after filtering points below Z = {min_height}.")
        print(f"Filtered {num_pts_before_filter - num_pts_after_filter:,} points. Remaining: {num_pts_after_filter:,}")
    else:
        print("Skipping height filtering.")


    # --- Downsample (Optional) ---
    if voxel_down_sample_size is not None and voxel_down_sample_size > 0:
        print(f"Downsampling with voxel size {voxel_down_sample_size}...")
        num_pts_before_downsample = len(pcd_tensor.point.positions)
        pcd_tensor = pcd_tensor.voxel_down_sample(voxel_down_sample_size)
        num_pts_after_downsample = len(pcd_tensor.point.positions)
        if num_pts_after_downsample == 0:
            raise ValueError("Point cloud empty after voxel downsampling.")
        print(f"Downsampled from {num_pts_before_downsample:,} to {num_pts_after_downsample:,} points.")
    else:
        print("Skipping voxel downsampling.")


    # --- Intensity ---
    print("Processing intensity...")
    intensities_tensor = None
    num_pts = len(pcd_tensor.point.positions)
    # Define comparison tensors once
    zero_tensor = o3d.core.Tensor([0.0], dtype=o3d.core.Dtype.Float32, device=device)
    one_tensor = o3d.core.Tensor([1.0], dtype=o3d.core.Dtype.Float32, device=device)
    max_val_255 = o3d.core.Tensor([255.0], dtype=o3d.core.Dtype.Float32, device=device)
    max_val_65535 = o3d.core.Tensor([65535.0], dtype=o3d.core.Dtype.Float32, device=device)

    if 'intensity' in pcd_tensor.point:
        print("Found 'intensity' attribute.")
        intensities_tensor = pcd_tensor.point.intensity.to(o3d.core.Dtype.Float32)
        max_intensity_val = intensities_tensor.max()
        if max_intensity_val > one_tensor:
            if max_intensity_val <= max_val_255 * 1.01:
                print("Normalizing intensity from assumed 0-255 range.")
                intensities_tensor = intensities_tensor / max_val_255
            elif max_intensity_val <= max_val_65535 * 1.01:
                print("Normalizing intensity from assumed 0-65535 range.")
                intensities_tensor = intensities_tensor / max_val_65535
            else:
                print(f"WARN: Max intensity {max_intensity_val.item()} > 1.0 but not in common ranges. Clamping to 0-1.")
        # FIX: Replace .maximum().minimum() with .clip()
        intensities_tensor = intensities_tensor.clip(zero_tensor, one_tensor)

    elif 'colors' in pcd_tensor.point:
        print("Using first channel of 'colors' attribute as intensity.")
        colors_tensor = pcd_tensor.point.colors
        if colors_tensor.dtype in (o3d.core.Dtype.UInt8, o3d.core.Dtype.UInt16):
            print("Normalizing colors from integer type.")
            max_dtype_val = o3d.core.Tensor([np.iinfo(colors_tensor.dtype.to_numpy_dtype()).max],
                                            dtype=o3d.core.Dtype.Float32, device=device)
            intensities_tensor = colors_tensor[:, 0].to(o3d.core.Dtype.Float32) / max_dtype_val
        elif colors_tensor.dtype in (o3d.core.Dtype.Float32, o3d.core.Dtype.Float64):
            intensities_tensor = colors_tensor[:, 0].to(o3d.core.Dtype.Float32)
        else:
            print(f"WARN: Unsupported color dtype {colors_tensor.dtype}. Using default intensity.")
            intensities_tensor = o3d.core.Tensor.full((num_pts,), 0.5, dtype=o3d.core.Dtype.Float32, device=device)
        # FIX: Replace .maximum().minimum() with .clip()
        intensities_tensor = intensities_tensor.clip(zero_tensor, one_tensor)

    else:
        print("No 'intensity' or 'colors' attribute found. Using default intensity value 0.5.")
        intensities_tensor = o3d.core.Tensor.full((num_pts,), 0.5, dtype=o3d.core.Dtype.Float32, device=device)

    # --- Normals ---
    print(f"Estimating normals (radius={normal_radius}, max_nn={normal_max_nn})...")
    normals_tensor = None # Initialize
    # Re-check num_pts as it's crucial here
    num_pts = len(pcd_tensor.point.positions)

    if num_pts == 0:
         print("WARN: Point cloud has 0 points. Skipping normal estimation.")
         normals_tensor = o3d.core.Tensor.zeros((0, 3), dtype=o3d.core.Dtype.Float32, device=device)
         # Ensure normals attribute exists even if empty
         if 'normals' not in pcd_tensor.point:
             pcd_tensor.point.add('normals', normals_tensor)
         else:
             pcd_tensor.point.normals = normals_tensor # Update potentially existing one
    else:
        try:
            # Ensure positions exist before estimating normals
            if 'positions' not in pcd_tensor.point or len(pcd_tensor.point.positions) == 0:
                 # This condition is redundant due to num_pts check above, but safe
                 raise RuntimeError("Cannot estimate normals on point cloud with no points.")

            pcd_tensor.estimate_normals(max_nn=normal_max_nn, radius=normal_radius)

            if 'normals' not in pcd_tensor.point or len(pcd_tensor.point.normals) != num_pts:
                # This case should ideally not happen if estimate_normals succeeded without error
                # but handle defensively
                raise RuntimeError("Normal estimation called but 'normals' attribute not created or has wrong size.")

            normals_tensor = pcd_tensor.point.normals

            # Check for NaN/Inf in normals using o3d.core.Tensor methods
            # FIX: Use tensor methods isnan() and isinf()
            nan_mask = normals_tensor.isnan().any(dim=1)
            inf_mask = normals_tensor.isinf().any(dim=1)
            bad_mask = nan_mask | inf_mask # Element-wise OR

            # FIX: Use o3d.core.any() to check if any bad normals exist
            if bad_mask.any():
                 print("WARN: NaNs or Infs detected in estimated normals. Replacing affected normals with [0, 0, 0].")
                 zero_normals = o3d.core.Tensor.zeros_like(normals_tensor)
                 # Use o3d.core.where for conditional selection
                 # Ensure mask is broadcastable: reshape (N,) to (N, 1)
                 normals_tensor = o3d.core.where(bad_mask.reshape(-1, 1), zero_normals, normals_tensor)
                 pcd_tensor.point.normals = normals_tensor # Update the point cloud attribute

        except Exception as e:
            print(f"WARN: Normal estimation failed: {e}. Using zero vectors [0, 0, 0] for normals.")
            # Re-check num_pts inside exception handler as a safeguard
            num_pts = len(pcd_tensor.point.positions)
            if num_pts > 0:
                 normals_tensor = o3d.core.Tensor.zeros((num_pts, 3), dtype=o3d.core.Dtype.Float32, device=device)
            else: # Should not happen if initial check passed, but safety first
                 normals_tensor = o3d.core.Tensor.zeros((0, 3), dtype=o3d.core.Dtype.Float32, device=device)

            # Ensure normals attribute exists even after failure
            if 'normals' not in pcd_tensor.point:
                 pcd_tensor.point.add('normals', normals_tensor)
            else:
                 pcd_tensor.point.normals = normals_tensor # Overwrite potentially problematic partial results

    # --- Prepare CPU Data & KDTree ---
    print("Converting final data to NumPy & Building KDTree...")
    # Ensure tensors exist before converting
    if 'positions' in pcd_tensor.point:
        points_lidar_np = pcd_tensor.point.positions.cpu().numpy()
    else:
        points_lidar_np = np.empty((0, 3), dtype=np.float32) # Handle edge case

    # Use the final normals_tensor which is guaranteed to exist
    normals_lidar_np = normals_tensor.cpu().numpy()
    # Use the final intensities_tensor which is guaranteed to exist
    intensities_lidar_np = intensities_tensor.cpu().numpy()


    num_points = len(points_lidar_np)
    # *** Generate indices RELATIVE TO THE FINAL points ***
    indices_lidar_np = np.arange(1, num_points + 1, dtype=np.uint32) # 1-based index for N final points

    kdtree_lidar = None
    if num_points > 0:
        try:
            kdtree_lidar = KDTree(points_lidar_np)
        except Exception as e:
            logging.warning(f"WARN: Failed to build KDTree: {e}. KDTree will be None.")
            kdtree_lidar = None
    else:
        logging.warning("WARN: Final point cloud has 0 points. Cannot build KDTree.")

    # *** RENAME the returned dictionary ***
    processed_lidar_data = {
        'points': points_lidar_np,           # (N_processed, 3)
        'indices': indices_lidar_np,         # (N_processed,) -> values [1, ..., N_processed]
        'normals': normals_lidar_np,         # (N_processed, 3)
        'intensities': intensities_lidar_np, # (N_processed,)
        'kdtree': kdtree_lidar
    }
    logging.info(f"Preprocessing done. Final point count: {num_points:,}")
    # pcd_tensor still contains the same processed points
    return pcd_tensor, processed_lidar_data # Return renamed dict

# --- Rendering Function (MODIFIED) ---
def render_geometric_viewpoint_open3d(
    pcd_tensor_processed: o3d.t.geometry.PointCloud,
    processed_lidar_data: dict, # Needs 'points', 'normals', 'intensities'
    T_map_cam_render: np.ndarray,
    K_camera: np.ndarray,
    width: int,
    height: int,
    shading_mode: str = 'normal',
    checkerboard_scale: float = 1.0,
    point_size: float = 3.0,  # <--- Default geometric point size back to 3.0
    depth_mask_point_size: float = 2.0, # Fixed size for depth/mask
    bg_color: tuple = (0.0, 0.0, 0.0, 1.0),
    intensity_highlight_threshold: float = None,
    highlight_color_rgb: tuple = (1.0, 0.0, 0.0)
):
    """
    Renders geometric image, Depth Map, and Render Mask using PROCESSED point cloud data.
    Generates a mask where pixels with projected points are white (255), others black (0).

    DIFFERENT POINT SIZES are used:
      - Geometric Image uses the 'point_size' argument (default 3.0).
      - Depth Map and Render Mask use a fixed point_size of 1.0 for better accuracy.

    Args:
        pcd_tensor_processed (o3d.t.geometry.PointCloud): Processed point cloud tensor.
        processed_lidar_data (dict): Dict with processed point data (NumPy arrays).
                                     MUST contain 'points', 'normals', 'intensities'.
        T_map_cam_render (np.ndarray): 4x4 pose matrix (Map -> Camera).
        K_camera (np.ndarray): 3x3 intrinsic matrix.
        width (int): Render width.
        height (int): Render height.
        shading_mode (str): 'normal' or 'checkerboard' for geometric render.
        checkerboard_scale (float): Scale for checkerboard pattern.
        point_size (float): Size of rendered points for the GEOMETRIC image. Defaults to 3.0.
        bg_color (tuple): Background color (RGBA float 0-1).
        intensity_highlight_threshold (float, optional): Intensity threshold for highlighting.
        highlight_color_rgb (tuple): RGB color (float 0-1) for highlighted points.

    Returns:
        dict or None: Dictionary with 'geometric_image', 'depth', 'render_mask', 'pose', or None on failure.
                      'depth' is a float32 NumPy array.
    """
    logging.info(f"Rendering W={width}, H={height} with '{shading_mode}' shading...")
    logging.info(f"  Geometric Point Size: {point_size}")
    logging.info(f"  Depth/Mask Point Size: {depth_mask_point_size}") # Explicitly log the fixed size
    start_time = time.time()

    # --- Input Checks & Data Validation ---
    if not isinstance(pcd_tensor_processed, o3d.t.geometry.PointCloud):
        logging.error("ERROR: Input pcd_tensor_processed is not valid.")
        return None
    if not isinstance(processed_lidar_data, dict):
        logging.error("ERROR: Input processed_lidar_data is not a dictionary.")
        return None

    original_device = pcd_tensor_processed.device
    cpu_device = o3d.core.Device("CPU:0")

    try:
        # --- Data Validation (Simplified) ---
        num_points = len(pcd_tensor_processed.point.positions)
        if num_points == 0: raise ValueError("Input pcd_tensor_processed has 0 points.")
        points_np = processed_lidar_data['points']
        normals_np = processed_lidar_data['normals']
        intensities_np = processed_lidar_data['intensities']
        num_points_np = points_np.shape[0]
        if num_points != num_points_np: raise ValueError("Point count mismatch.")
        logging.info(f"Validated tensor and NumPy data for {num_points:,} points.")
        # --- End Validation ---

    except Exception as e:
        logging.error(f"ERROR: Input Data validation failed: {e}", exc_info=True)
        return None

    # --- Renderer Setup ---
    try:
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

        # *** Create TWO material records ***
        # Material for Geometric Render
        mat_geom = o3d.visualization.rendering.MaterialRecord()
        mat_geom.shader = "defaultUnlit"
        mat_geom.point_size = float(point_size) # Use the function argument

        # Material for Depth and Mask Renders
        mat_depth_mask = o3d.visualization.rendering.MaterialRecord()
        mat_depth_mask.shader = "defaultUnlit"
        mat_depth_mask.point_size = float(depth_mask_point_size) # Use fixed size 1.0 for accuracy
        # *** End Create TWO material records ***

        K_camera_f64 = K_camera.astype(np.float64)
        T_map_cam_render_f64 = np.ascontiguousarray(T_map_cam_render, dtype=np.float64)
        T_view_matrix = np.linalg.inv(T_map_cam_render_f64) # Camera <- Map
        renderer.setup_camera(K_camera_f64, T_view_matrix, width, height)
        renderer.scene.set_background(list(bg_color))
    except Exception as e_setup:
        logging.error(f"ERROR: Renderer setup failed: {e_setup}", exc_info=True)
        return None

    # --- Prepare Point Cloud Data for Rendering ---
    pcd_geom_render = None
    pcd_mask_render = None # No separate index PCD needed
    try:
        # Geometric PCD (remains same, needed for depth/mask render too)
        pcd_geom_render = pcd_tensor_processed.clone()
        if shading_mode == 'normal':
            geom_colors_np = ((normals_np + 1.0) / 2.0).astype(np.float32)
        elif shading_mode == 'checkerboard':
            if checkerboard_scale <= 0: checkerboard_scale = 1.0
            grid_coords = np.floor(points_np / checkerboard_scale).astype(int)
            checker_pattern = (grid_coords[:, 0] + grid_coords[:, 1] + grid_coords[:, 2]) % 2
            geom_colors_scalar = np.where(checker_pattern == 0, 0.2, 0.8)
            geom_colors_np = np.tile(geom_colors_scalar[:, None], (1, 3)).astype(np.float32)
        else: raise ValueError(f"Unsupported shading mode: {shading_mode}")
        if intensity_highlight_threshold is not None:
            logging.info(f"Applying intensity highlighting (threshold={intensity_highlight_threshold:.3f})")
            highlight_mask = intensities_np > intensity_highlight_threshold # Shape (N,)
            if highlight_mask.any():
                highlight_color_np = np.array(highlight_color_rgb, dtype=np.float32) # Shape (3,)
                indices_to_highlight = np.where(highlight_mask)[0] # Get integer indices of True values
                if indices_to_highlight.size > 0: # Check if any indices were found
                    geom_colors_np[indices_to_highlight, :] = highlight_color_np # Assign color to specific rows
                logging.info(f"Highlighted {indices_to_highlight.size} points.")
        pcd_geom_render.point['colors'] = o3d.core.Tensor(geom_colors_np, dtype=o3d.core.Dtype.Float32, device=original_device)

        # Mask PCD (Positions + White Color) - Still useful
        pcd_mask_render = o3d.t.geometry.PointCloud(pcd_tensor_processed.point.positions)
        white_colors_np = np.ones((num_points, 3), dtype=np.float32)
        pcd_mask_render.point['colors'] = o3d.core.Tensor(white_colors_np, dtype=o3d.core.Dtype.Float32, device=original_device)

        # Move to CPU for Rendering
        pcd_geom_cpu = pcd_geom_render.to(cpu_device)
        pcd_mask_cpu = pcd_mask_render.to(cpu_device)

    except Exception as e_prep:
        logging.error(f"ERROR preparing point cloud data for rendering: {e_prep}", exc_info=True)
        return None

    # --- Render All Outputs ---
    img_geom_gray_u8 = None
    depth_map_f32 = None # Store depth map
    render_mask_u8 = None
    try:
        # Render Geom (using mat_geom with input point_size)
        logging.debug("Rendering geometric view...")
        renderer.scene.clear_geometry()
        renderer.scene.add_geometry("__geom_render", pcd_geom_cpu, mat_geom) # Use mat_geom
        img_geom_o3d = renderer.render_to_image()
        if img_geom_o3d is None: raise RuntimeError("render_to_image returned None for geom")
        img_geom_rgb_u8 = np.asarray(img_geom_o3d).astype(np.uint8)
        img_geom_gray_u8 = cv2.cvtColor(img_geom_rgb_u8, cv2.COLOR_RGB2GRAY)

        # *** Render Depth ***
        # Render using the geometry but with the material having point_size=1.0
        logging.debug("Rendering depth map...")
        renderer.scene.clear_geometry() # Clear previous geometry
        # Add the *geometric* point cloud BUT use the *depth/mask* material
        renderer.scene.add_geometry("__depth_render", pcd_geom_cpu, mat_depth_mask)
        # --- FIX: Use correct keyword argument ---
        img_depth_o3d = renderer.render_to_depth_image(z_in_view_space=True) # Get depth in camera coords
        # --- END FIX ---
        if img_depth_o3d is None: raise RuntimeError("render_to_depth_image returned None")
        depth_map_f32 = np.asarray(img_depth_o3d).astype(np.float32)
        # Optional logging for depth range
        finite_depth = depth_map_f32[np.isfinite(depth_map_f32)]
        min_d = np.min(finite_depth) if finite_depth.size > 0 else 0
        max_d = np.max(finite_depth) if finite_depth.size > 0 else 0
        logging.info(f"Rendered depth map. Min: {min_d:.2f}, Max (finite): {max_d:.2f}")


        # Render Mask (using mat_depth_mask with point_size=1.0)
        logging.debug("Rendering mask...")
        renderer.scene.clear_geometry() # Clear previous geometry
        renderer.scene.add_geometry("__mask_render", pcd_mask_cpu, mat_depth_mask) # Use mat_depth_mask
        img_mask_o3d = renderer.render_to_image()
        if img_mask_o3d is None: raise RuntimeError("render_to_image returned None for mask")
        img_mask_rgb_u8 = np.asarray(img_mask_o3d).astype(np.uint8)
        render_mask_u8 = cv2.cvtColor(img_mask_rgb_u8, cv2.COLOR_RGB2GRAY)
        _, render_mask_u8 = cv2.threshold(render_mask_u8, 1, 255, cv2.THRESH_BINARY)

    except Exception as e_render:
        logging.error(f"ERROR during rendering: {e_render}", exc_info=True)
        return None
    finally:
        try: renderer.scene.clear_geometry()
        except: pass

    logging.info(f"Rendering done: {time.time() - start_time:.3f}s")
    return {
        'geometric_image': img_geom_gray_u8,
        'depth': depth_map_f32, # Return depth map
        'render_mask': render_mask_u8,
        'pose': T_map_cam_render
    }

def match_by_distance(
    features_path: Path,
    query_image_list_file: Path,
    render_image_list_file: Path,
    matches_output_path: Path,
    distance_threshold_px: float = 10.0 # Max pixel distance for a match
):
    """
    Performs simple nearest-neighbor matching based on Euclidean distance
    in pixel coordinates between keypoints of corresponding query/render images.

    Assumes a one-to-one correspondence between the lists.
    Saves matches in HLOC format ('matches0' dataset).

    Args:
        features_path (Path): Path to the HDF5 file containing keypoints.
        query_image_list_file (Path): Path to the text file listing query image names.
        render_image_list_file (Path): Path to the text file listing render image names.
        matches_output_path (Path): Path to save the output matches HDF5 file.
        distance_threshold_px (float): Maximum Euclidean distance in pixels
                                      to consider two keypoints a match.

    Returns:
        bool: True if matching completed successfully, False otherwise.
    """
    logging.info(f"--- Running Simple Distance-Based Matching (Threshold: {distance_threshold_px} px) ---")
    start_time = time.time()
    matching_ok = False
    processed_pairs = 0
    total_matches_found = 0

    try:
        # --- Read Image Lists ---
        if not query_image_list_file.is_file() or not render_image_list_file.is_file():
            logging.error("One or both image list files not found.")
            return False
        if not features_path.is_file():
             logging.error(f"Features file not found: {features_path}")
             return False

        query_names = [line.strip() for line in query_image_list_file.read_text().splitlines() if line.strip()]
        render_names = [line.strip() for line in render_image_list_file.read_text().splitlines() if line.strip()]

        if not query_names or not render_names:
            logging.error("One or both image lists are empty.")
            return False

        # --- Crucial Check: Ensure lists have the same length ---
        if len(query_names) != len(render_names):
            logging.error(f"Query ({len(query_names)}) and Render ({len(render_names)}) lists must have the same length for distance matching.")
            return False

        num_pairs = len(query_names)
        logging.info(f"Found {num_pairs} corresponding query-render pairs.")

        # --- Open Files ---
        with h5py.File(features_path, 'r') as features_db, \
             h5py.File(matches_output_path, 'w') as matches_db: # Open output in write mode

            for i in range(num_pairs):
                q_name = query_names[i]
                r_name = render_names[i]
                pair_key = f"{q_name}/{r_name}" # HLOC standard key format
                logging.debug(f"Processing pair {i+1}/{num_pairs}: {pair_key}")

                # --- Load Keypoints ---
                try:
                    # Handle potential missing keys or different key naming conventions
                    q_feature_key = q_name if q_name in features_db else next((k for k in features_db if k.endswith(q_name)), None)
                    r_feature_key = r_name if r_name in features_db else next((k for k in features_db if k.endswith(r_name)), None)

                    if not q_feature_key or not r_feature_key:
                         logging.warning(f"Missing feature key for query '{q_name}' or render '{r_name}'. Skipping pair.")
                         continue

                    kps_q = features_db[q_feature_key]['keypoints'][()] # Shape (N, >=2)
                    kps_r = features_db[r_feature_key]['keypoints'][()] # Shape (M, >=2)
                except KeyError as e:
                    logging.warning(f"KeyError loading keypoints for pair {pair_key}: {e}. Skipping.")
                    continue
                except Exception as e:
                     logging.error(f"Error loading keypoints for pair {pair_key}: {e}", exc_info=True)
                     continue # Skip pair on error

                num_kps_q = kps_q.shape[0]
                num_kps_r = kps_r.shape[0]

                if num_kps_q == 0 or num_kps_r == 0:
                    logging.debug(f"Zero keypoints found for query ({num_kps_q}) or render ({num_kps_r}) in pair {pair_key}. Skipping.")
                    # Create empty group in output? Optional, maybe skip.
                    continue

                # Extract XY coordinates (ensure correct dtype for cdist)
                kps_q_xy = kps_q[:, :2].astype(np.float64)
                kps_r_xy = kps_r[:, :2].astype(np.float64)

                # --- Calculate Pairwise Distances ---
                # dist_matrix[i, j] = distance between kps_q_xy[i] and kps_r_xy[j]
                dist_matrix = cdist(kps_q_xy, kps_r_xy, metric='euclidean')

                # --- Find Nearest Neighbor and Apply Threshold ---
                # For each query keypoint (row i), find the index of the closest render keypoint (column j)
                min_render_indices = np.argmin(dist_matrix, axis=1) # Shape (N,)
                # Get the actual minimum distance values corresponding to those indices
                min_distances = dist_matrix[np.arange(num_kps_q), min_render_indices] # Shape (N,)

                # --- Create matches0 array ---
                matches0 = np.full(num_kps_q, -1, dtype=np.int32) # Initialize with -1 (no match)
                # Find which query keypoints have a match within the threshold
                match_mask = min_distances <= distance_threshold_px
                # Assign the index of the closest render keypoint where the distance is acceptable
                matches0[match_mask] = min_render_indices[match_mask]

                num_matches_pair = np.sum(match_mask)
                total_matches_found += num_matches_pair
                logging.debug(f"Found {num_matches_pair} matches for pair {pair_key} within {distance_threshold_px}px.")

                # --- Save to HDF5 ---
                pair_group = matches_db.create_group(pair_key)
                pair_group.create_dataset('matches0', data=matches0, dtype=np.int32)
                # Optionally save scores (e.g., inverse distance or just confidence 1) if needed
                # scores0 = np.zeros(num_kps_q, dtype=np.float32)
                # scores0[match_mask] = 1.0 / (1.0 + min_distances[match_mask]) # Example score
                # pair_group.create_dataset('matching_scores0', data=scores0)

                processed_pairs += 1

        # End of HDF5 context managers
        logging.info(f"Distance matching completed. Processed {processed_pairs}/{num_pairs} pairs.")
        logging.info(f"Total matches found across all pairs: {total_matches_found}")
        matching_ok = True # Consider successful if it ran through

    except FileNotFoundError as e:
        logging.error(f"File not found during distance matching: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during distance matching:")
        logging.error(traceback.format_exc())

    logging.info(f"Distance matching finished in {time.time() - start_time:.2f} seconds. Success: {matching_ok}")
    return matching_ok

def link_matches_via_depth(
    query_image_name: str,
    features_path: Path,
    matches_path: Path,
    rendered_views_info: list, # Expects 'depth_map_path' and 'pose'
    processed_lidar_data: dict, # Expects 'points' and 'kdtree'
    camera_intrinsics: np.ndarray,
    nn_distance_threshold: float, # Max distance for NN match
    max_depth_value: float = 100.0 # Ignore points further than this distance
):
    """
    Links hloc matches (query <-> render) to PROCESSED LiDAR points using
    depth map back-projection and KDTree nearest neighbor search.

    Args:
        query_image_name (str): Name of the query image.
        features_path (Path): Path to the feature HDF5 file.
        matches_path (Path): Path to the matches HDF5 file.
        rendered_views_info (list): Info about rendered views (depth maps, poses).
        processed_lidar_data (dict): Dict containing processed lidar data ('points', 'kdtree').
        camera_intrinsics (np.ndarray): 3x3 camera intrinsics matrix.
        nn_distance_threshold (float): Max distance allowed between back-projected point
                                      and its nearest neighbor in the processed cloud.
        max_depth_value (float): Maximum valid depth reading from the depth map.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of (query_kp_coords, processed_lidar_coords).
    """
    logging.info(f"\nLinking matches for {query_image_name} using Depth Maps + KDTree (NN thresh={nn_distance_threshold:.3f}m)...")

    empty_2d = np.empty((0, 2), dtype=np.float32)
    empty_3d = np.empty((0, 3), dtype=np.float32)

    # --- Input Checks ---
    if not isinstance(camera_intrinsics, np.ndarray) or camera_intrinsics.shape != (3, 3):
        logging.error("Invalid camera_intrinsics."); return empty_2d, empty_3d
    if not isinstance(processed_lidar_data, dict) or 'points' not in processed_lidar_data or 'kdtree' not in processed_lidar_data:
        logging.error("processed_lidar_data missing 'points' or 'kdtree'."); return empty_2d, empty_3d
    kdtree = processed_lidar_data['kdtree']
    processed_points_np = processed_lidar_data['points']
    if kdtree is None or not isinstance(kdtree, KDTree):
         logging.error("KDTree is invalid or None in processed_lidar_data."); return empty_2d, empty_3d
    if len(processed_points_np) == 0:
        logging.warning("Processed LiDAR data has 0 points."); return empty_2d, empty_3d
    num_processed_points = len(processed_points_np)
    logging.info(f"Using KDTree with {num_processed_points:,} points.")

    # Extract camera intrinsic parameters
    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]
    K_inv = np.linalg.inv(camera_intrinsics) # Pre-calculate inverse

    query_kp_coords_list = []
    processed_lidar_coords_list = []

    try:
        with h5py.File(features_path, 'r') as features_db, \
             h5py.File(matches_path, 'r') as matches_db:

            # Find Query Features
            found_feature_key = None
            if query_image_name in features_db: found_feature_key = query_image_name
            else:
                for k in features_db.keys():
                    if k.endswith(query_image_name): found_feature_key = k; break
            if not found_feature_key: logging.warning(f"Query features '{query_image_name}' not found."); return empty_2d, empty_3d
            try: query_kps = features_db[found_feature_key]['keypoints'][()]
            except KeyError: logging.error(f"Key 'keypoints' not found for query."); return empty_2d, empty_3d
            num_query_kps = len(query_kps)
            if num_query_kps == 0: logging.info(f"Query {query_image_name} has 0 keypoints."); return empty_2d, empty_3d

            link_count = 0
            processed_query_indices = set()
            depth_fail_count = 0
            nn_fail_count = 0

            # Iterate through Render Views
            for render_info in rendered_views_info:
                if not isinstance(render_info, dict): continue
                render_path_obj = render_info.get('geometric_image_path') # Still useful for key name
                depth_map_path = render_info.get('depth_map_path')
                T_map_cam_render = render_info.get('pose')

                if not isinstance(render_path_obj, Path) or not isinstance(depth_map_path, Path) or not isinstance(T_map_cam_render, np.ndarray):
                     logging.warning(f"Render info dict missing path/pose. Skipping."); continue

                render_key_name = render_path_obj.name

                # Check Matches
                pair_key_simple = f'{query_image_name}/{render_key_name}'
                pair_key_hloc_style = f'{found_feature_key}/{render_key_name}'
                matches_pair_group = None
                if pair_key_simple in matches_db: matches_pair_group = matches_db[pair_key_simple]
                elif pair_key_hloc_style in matches_db: matches_pair_group = matches_db[pair_key_hloc_style]
                else: continue # No match pair found
                if 'matches0' not in matches_pair_group: continue
                try: matches0 = matches_pair_group['matches0'][()]
                except Exception: continue
                if matches0.ndim != 1 or len(matches0) != num_query_kps: continue
                if np.all(matches0 == -1): continue

                # Load Render Features
                render_feature_key = None
                if render_key_name in features_db: render_feature_key = render_key_name
                else:
                    for k in features_db.keys():
                         if k.endswith(render_key_name): render_feature_key = k; break
                if not render_feature_key: continue
                try: render_kps = features_db[render_feature_key]['keypoints'][()]
                except KeyError: continue
                num_render_kps = len(render_kps)
                if num_render_kps == 0: continue

                # Load Depth Map
                if not depth_map_path.is_file(): logging.warning(f"Depth map not found: {depth_map_path}."); continue
                try: depth_map = np.load(depth_map_path)
                except Exception as e: logging.warning(f"Failed to load depth map {depth_map_path}: {e}"); continue
                height, width = depth_map.shape

                # Iterate through Query Keypoints
                for query_idx in range(num_query_kps):
                    if query_idx in processed_query_indices: continue

                    render_idx = matches0[query_idx]
                    if render_idx == -1: continue
                    if not (0 <= render_idx < num_render_kps): continue

                    try: render_kp_coords = render_kps[render_idx][:2] # (x, y) float
                    except IndexError: continue

                    u_f, v_f = render_kp_coords[0], render_kp_coords[1]
                    u, v = int(round(u_f)), int(round(v_f)) # Use rounded coords for lookup

                    if not (0 <= u < width and 0 <= v < height): continue # Outside image bounds

                    # --- 1. Get Depth ---
                    try: depth_value = depth_map[v, u]
                    except IndexError: continue

                    # Validate depth value
                    if depth_value <= 0 or depth_value > max_depth_value or not np.isfinite(depth_value):
                        logging.debug(f"Invalid depth {depth_value:.2f} at ({u},{v}) in {render_key_name}. Skipping.")
                        depth_fail_count += 1
                        continue

                    # --- 2. Back-project to Camera Frame ---
                    # Use the precise float coordinates (u_f, v_f) for potentially better accuracy
                    # P_cam = depth_value * K_inv @ np.array([u_f, v_f, 1.0]) # Matrix multiplication way
                    # Or direct formula:
                    x_cam = (u_f - cx) * depth_value / fx
                    y_cam = (v_f - cy) * depth_value / fy
                    z_cam = depth_value
                    P_cam = np.array([x_cam, y_cam, z_cam])


                    # --- 3. Transform to Map Frame ---
                    P_map_h = T_map_cam_render @ np.array([x_cam, y_cam, z_cam, 1.0])
                    P_map = P_map_h[:3] / P_map_h[3] # Dehomogenize


                    # --- 4. KDTree Nearest Neighbor Search ---
                    try:
                        distance, nn_idx = kdtree.query(P_map, k=1) # Find 1 nearest neighbor
                    except Exception as e_nn:
                         logging.warning(f"KDTree query failed for point {P_map}: {e_nn}")
                         continue

                    # --- 5. Validate NN Distance ---
                    if distance > nn_distance_threshold:
                        # logging.debug(f"NN search failed for Q_idx={query_idx}, R_idx={render_idx} in {render_key_name}. "
                        #               f"Dist: {distance:.3f}m > {nn_distance_threshold:.3f}m. Backproj: {P_map}")
                        nn_fail_count += 1
                        continue

                    # --- 6. If Valid, Add the Link ---
                    try:
                        query_kp_coords = query_kps[query_idx][:2]
                        processed_lidar_coords = processed_points_np[nn_idx] # Get coords of NN
                    except IndexError:
                         logging.warning(f"IndexError accessing query_kps[{query_idx}] or processed_points_np[{nn_idx}].")
                         continue

                    logging.debug(f"Link Found (Depth): Q_idx={query_idx} -> R_idx={render_idx} @ ({u},{v}) -> Depth={depth_value:.2f}m -> NN_Idx={nn_idx}, Dist={distance:.3f}m")
                    query_kp_coords_list.append(query_kp_coords)
                    processed_lidar_coords_list.append(processed_lidar_coords)
                    link_count += 1
                    processed_query_indices.add(query_idx)

            # End loop through rendered_views_info

    # --- Handle Exceptions ---
    except Exception as e:
        logging.error(f"An unexpected error occurred during depth linking for {query_image_name}: {e}")
        logging.error(traceback.format_exc())
        return empty_2d, empty_3d

    # --- Final Report ---
    logging.info(f"Linking complete for {query_image_name}.")
    logging.info(f"  Found {link_count} unique 2D(Query)<->3D(LiDAR) matches via depth map + NN.")
    if depth_fail_count > 0: logging.info(f"  Skipped {depth_fail_count} potential links due to invalid depth.")
    if nn_fail_count > 0: logging.info(f"  Skipped {nn_fail_count} potential links due to NN distance > {nn_distance_threshold:.3f}m.")

    final_query_kps = np.array(query_kp_coords_list, dtype=np.float32)
    final_processed_coords = np.array(processed_lidar_coords_list, dtype=np.float32)

    min_pnp_points_required = 15
    if link_count < min_pnp_points_required:
         logging.warning(f"Final link count {link_count} is less than minimum required for PnP ({min_pnp_points_required}).")

    return final_query_kps, final_processed_coords

def visualize_pnp_matches(
    query_image_path: Path,
    points2D: np.ndarray, # Shape (N, 2) - Query keypoints
    points3D: np.ndarray, # Shape (N, 3) - Corresponding original LiDAR points
    camera_intrinsics: np.ndarray, # Shape (3, 3)
    output_path: Path,
    # --- NEW ARGUMENT ---
    # Pose transforming Map/World points TO Camera coordinates (Camera <- Map/World)
    # Can be a 4x4 matrix or None (if None, uses dummy pose at origin)
    estimated_pose_cam_from_map: np.ndarray = None,
    # --------------------
    dist_coeffs: np.ndarray = None,
    max_points_to_draw: int = 100,
    inlier_indices: np.ndarray = None # 1D array of inlier indices
):
    """
    Visualizes 2D-3D correspondences by projecting 3D points onto the query
    image using the provided estimated camera pose (or a dummy pose if None).
    Highlights PnP inliers if provided.

    Args:
        query_image_path (Path): Path to the query image.
        points2D (np.ndarray): (N, 2) array of detected 2D keypoints.
        points3D (np.ndarray): (N, 3) array of corresponding 3D world points.
        camera_intrinsics (np.ndarray): 3x3 camera intrinsics matrix.
        output_path (Path): Path to save the visualization image.
        estimated_pose_cam_from_map (np.ndarray | None): 4x4 matrix defining the
            Camera-from-Map transformation. If None, uses a dummy pose at origin.
        dist_coeffs (np.ndarray, optional): Distortion coefficients for projection.
        max_points_to_draw (int): Max points to draw (random sample).
        inlier_indices (np.ndarray | None): 1D array of indices marking PnP inliers.
    """
    logging.info(f"Visualizing PnP matches for {query_image_path.name} -> {output_path.name}")
    if estimated_pose_cam_from_map is None:
        logging.warning("No estimated pose provided to visualize_pnp_matches, projecting from world origin (likely incorrect).")

    if not query_image_path.is_file():
        logging.error(f"Query image not found: {query_image_path}")
        return

    if points2D.shape[0] != points3D.shape[0] or points2D.shape[1] != 2 or points3D.shape[1] != 3:
         logging.error(f"Cannot visualize: Input point shape mismatch - 2D {points2D.shape}, 3D {points3D.shape}")
         return
    if points2D.shape[0] == 0:
        logging.warning(f"No points provided for visualization for {query_image_path.name}.")
        # Create an empty image or copy the original? Let's copy.
        img = cv2.imread(str(query_image_path))
        if img is not None:
            cv2.putText(img, "No Points to Visualize", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            try:
                 output_path.parent.mkdir(parents=True, exist_ok=True)
                 cv2.imwrite(str(output_path), img)
            except Exception as e:
                 logging.error(f"Error saving empty visualization image {output_path}: {e}")
        return


    img = cv2.imread(str(query_image_path))
    if img is None:
        logging.error(f"Failed to load query image: {query_image_path}")
        return

    # --- Determine Pose for Projection ---
    if estimated_pose_cam_from_map is not None and estimated_pose_cam_from_map.shape == (4, 4):
        R_cam_map = estimated_pose_cam_from_map[:3, :3]
        t_cam_map = estimated_pose_cam_from_map[:3, 3]
        try:
            rvec, _ = cv2.Rodrigues(R_cam_map) # Convert rotation matrix to rotation vector
            tvec = t_cam_map.reshape(3, 1)     # Ensure tvec is (3, 1)
        except Exception as e_conv:
            logging.error(f"Error converting estimated pose to rvec/tvec: {e_conv}. Using dummy pose.", exc_info=True)
            rvec = np.zeros((3, 1), dtype=np.float64)
            tvec = np.zeros((3, 1), dtype=np.float64)
    else:
        # Use dummy pose at origin if no valid pose provided
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)

    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # --- Project 3D points using the determined pose ---
    try:
        projected_points, _ = cv2.projectPoints(
            points3D,
            rvec, # Camera pose relative to world (Map)
            tvec, # Camera pose relative to world (Map)
            camera_intrinsics,
            dist_coeffs
        )
        projected_points = projected_points.squeeze() # Shape (N, 2)
        if projected_points.ndim == 1: # Handle case with only one point
             projected_points = projected_points.reshape(1, 2)

    except Exception as e:
        logging.error(f"Error projecting 3D points for visualization: {e}", exc_info=True)
        # Optionally save the image with an error message
        cv2.putText(img, "Projection Error", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        try:
             output_path.parent.mkdir(parents=True, exist_ok=True)
             cv2.imwrite(str(output_path), img)
        except Exception as e_save:
             logging.error(f"Error saving error visualization image {output_path}: {e_save}")
        return

    # --- Select points and Draw Visualization (same as before) ---
    num_points = points2D.shape[0]
    indices_to_draw = np.arange(num_points)
    if num_points > max_points_to_draw:
        indices_to_draw = np.random.choice(num_points, max_points_to_draw, replace=False)
        logging.info(f"Drawing a random subset of {max_points_to_draw}/{num_points} matches.")

    color_outlier = (0, 0, 255)  # Red
    color_inlier = (0, 255, 0)   # Green
    color_line = (255, 128, 0)   # Blue/Cyan
    radius_kp = 3
    radius_proj = 5
    thickness = 1

    inlier_set = set(inlier_indices.flatten()) if inlier_indices is not None else None

    # Draw points
    for i in indices_to_draw:
        pt2d = tuple(points2D[i].round().astype(int))
        pt_proj = tuple(projected_points[i].round().astype(int))

        is_inlier = inlier_set is not None and i in inlier_set
        point_color = color_inlier if is_inlier else color_outlier

        # Check if projected point is within image bounds before drawing heavily
        img_h, img_w = img.shape[:2]
        if 0 <= pt_proj[0] < img_w and 0 <= pt_proj[1] < img_h :
            cv2.line(img, pt2d, pt_proj, color_line, thickness, cv2.LINE_AA)
            cv2.drawMarker(img, pt_proj, point_color, markerType=cv2.MARKER_TILTED_CROSS, markerSize=radius_proj*2, thickness=thickness, line_type=cv2.LINE_AA)
        else:
            # Optionally draw line to edge or just skip drawing projected point/line
            pass # Skip drawing projected point if outside bounds for clarity

        # Always draw the detected 2D keypoint
        cv2.circle(img, pt2d, radius_kp, point_color, -1, cv2.LINE_AA)

    # Add text about inliers if available
    if inlier_indices is not None:
         inlier_count = len(inlier_set) if inlier_set else 0
         cv2.putText(img, f"Inliers: {inlier_count}/{num_points}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
         cv2.putText(img, f"Points: {num_points}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


    # --- Save Image ---
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(output_path), img)
        if success:
            logging.info(f"Saved PnP match visualization to {output_path}")
        else:
            logging.error(f"Failed to save visualization image to {output_path}")
    except Exception as e:
        logging.error(f"Error saving visualization image {output_path}: {e}", exc_info=True)

# ==============================================================================
# Helper Function: Visualize Map Projection (CORRECTED)
# ==============================================================================
def visualize_map_projection(
    original_image_path: str, # Path to the ORIGINAL (distorted) image
    processed_lidar_data: dict,
    camera_intrinsics_K: np.ndarray, # Current/Refined K matrix (3x3)
    camera_distortion_D: np.ndarray, # Current/Refined D coeffs (flat array)
    model_type: str,                 # "KANNALA_BRANDT" or "PINHOLE"
    pose_cam_from_map: np.ndarray,   # T_cam_map (4x4)
    output_path: str,
    point_size: int = 1,
    max_vis_points: int = 500000,
    color_map_name: str = 'jet', # Renamed to avoid conflict with cm module
    filter_distance: float = 50.0,
    color_by: str = 'intensity',
    intensity_stretch_percentiles: tuple[float, float] = (10.0, 80.0)
):
    logging.debug(f"Visualizing map projection on ORIGINAL image {Path(original_image_path).name} -> {Path(output_path).name}")
    logging.debug(f"Using K:\n{camera_intrinsics_K}\nD:{camera_distortion_D}\nModel: {model_type}")
    logging.debug(f"Coloring by: {color_by}, Filter distance: {filter_distance}, Stretch: {intensity_stretch_percentiles}")

    try:
        image_original_distorted = cv2.imread(original_image_path)
        if image_original_distorted is None:
            logging.error(f"Could not load original image for visualization: {original_image_path}")
            return

        points3D_map_original = processed_lidar_data.get('points')
        intensities_map_original = None
        # ... (intensity loading logic remains the same) ...
        if color_by == 'intensity':
            intensities_map_original = processed_lidar_data.get('intensities')
            if intensities_map_original is None:
                logging.warning("Requested coloring by intensity, but 'intensities' not found. Falling back to depth.")
                color_by = 'depth' # Fallback
            elif points3D_map_original is not None and intensities_map_original.shape[0] != points3D_map_original.shape[0]:
                logging.warning("Mismatch points/intensities. Falling back to depth.")
                color_by = 'depth'; intensities_map_original = None

        if points3D_map_original is None or points3D_map_original.shape[0] == 0:
            logging.warning("No points in processed_lidar_data for visualization.")
            # Save the original image if no points
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, image_original_distorted)
            return

        if pose_cam_from_map is None or pose_cam_from_map.shape != (4, 4):
            # ... (handle invalid pose, save original image) ...
            logging.error(f"Invalid pose_cam_from_map provided for {Path(original_image_path).name}")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, image_original_distorted)
            return

        # --- Filtering and Subsampling (remains largely the same) ---
        points3D_map_current = points3D_map_original
        intensities_map_current = intensities_map_original
        points_cam_h = None # Will be computed

        # Transform to camera frame first for distance filtering and depth values
        points3D_map_h = np.hstack((points3D_map_current, np.ones((points3D_map_current.shape[0], 1))))
        points_cam_h = (pose_cam_from_map @ points3D_map_h.T).T # These are in camera coordinates

        if filter_distance is not None:
            distances_cam = np.linalg.norm(points_cam_h[:, :3], axis=1)
            keep_mask_dist = distances_cam < filter_distance
            points3D_map_current = points3D_map_current[keep_mask_dist]
            points_cam_h = points_cam_h[keep_mask_dist] # Filter transformed points too
            if intensities_map_current is not None:
                intensities_map_current = intensities_map_current[keep_mask_dist]
            if points3D_map_current.shape[0] == 0:
                 Path(output_path).parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(output_path, image_original_distorted); return

        points3D_to_project = points3D_map_current
        intensities_to_project = intensities_map_current
        # points_cam_h is already filtered if distance filter was applied

        num_points = points3D_to_project.shape[0]
        if max_vis_points is not None and num_points > max_vis_points:
            sample_indices = np.random.choice(num_points, max_vis_points, replace=False)
            points3D_to_project = points3D_to_project[sample_indices]
            points_cam_h = points_cam_h[sample_indices] # Subsample transformed points
            if intensities_to_project is not None:
                intensities_to_project = intensities_to_project[sample_indices]
        
        if points3D_to_project.shape[0] == 0:
             Path(output_path).parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(output_path, image_original_distorted); return

        # --- Projection using provided K, D, model_type ---
        R_cam_map_mat = pose_cam_from_map[:3, :3]
        t_cam_map_vec = pose_cam_from_map[:3, 3]
        try:
            rvec, _ = cv2.Rodrigues(R_cam_map_mat)
        except cv2.error as e_rot:
            logging.error(f"cv2.Rodrigues conversion failed: {e_rot}."); return

        image_points_distorted = None # This will store the final 2D points in the distorted image
        D_coeffs_for_proj = camera_distortion_D.flatten() if camera_distortion_D is not None else None

        if model_type == "KANNALA_BRANDT":
            P3D_reshaped = points3D_to_project.reshape(-1, 1, 3).astype(np.float32)
            d_kb = D_coeffs_for_proj[:4] if D_coeffs_for_proj is not None and len(D_coeffs_for_proj) >=4 else np.zeros(4, dtype=np.float64)
            image_points_distorted, _ = cv2.fisheye.projectPoints(
                P3D_reshaped, rvec, t_cam_map_vec, camera_intrinsics_K, d_kb
            )
        elif model_type == "PINHOLE":
            P3D_standard = points3D_to_project.astype(np.float32)
            # Ensure D is suitable for cv2.projectPoints (e.g., 5 or 8 elements)
            if D_coeffs_for_proj is None:
                d_pinhole = np.zeros(5, dtype=np.float64) # Assume 5 (k1,k2,p1,p2,k3) if none
            elif len(D_coeffs_for_proj) == 4: # k1,k2,p1,p2 -> add k3=0
                d_pinhole = np.array([D_coeffs_for_proj[0],D_coeffs_for_proj[1],D_coeffs_for_proj[2],D_coeffs_for_proj[3],0.0], dtype=np.float64)
            elif len(D_coeffs_for_proj) >= 5:
                d_pinhole = D_coeffs_for_proj
            else: # Fallback for unexpected D length
                d_pinhole = np.zeros(5, dtype=np.float64)

            image_points_distorted, _ = cv2.projectPoints(
                P3D_standard, rvec, t_cam_map_vec, camera_intrinsics_K, d_pinhole
            )
        else:
            logging.error(f"Unsupported model type '{model_type}' for projection in visualization.")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(output_path, image_original_distorted); return

        if image_points_distorted is None:
            logging.warning("Projection returned None.")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(output_path, image_original_distorted); return
        
        image_points_distorted = image_points_distorted.reshape(-1, 2)

        # --- Coloring and Drawing (largely same, but uses image_original_distorted) ---
        h_img, w_img = image_original_distorted.shape[:2]
        # Depths are from points_cam_h (which were filtered/subsampled along with points3D_to_project)
        depths_cam_for_color = points_cam_h[:, 2]

        # Shape check (image_points_distorted vs depths_cam_for_color vs intensities_to_project)
        mismatched_shapes = False
        if image_points_distorted.shape[0] != depths_cam_for_color.shape[0]: mismatched_shapes = True
        if intensities_to_project is not None and image_points_distorted.shape[0] != intensities_to_project.shape[0]: mismatched_shapes = True
        if mismatched_shapes:
             logging.error(f"Shape mismatch: image_points {image_points_distorted.shape}, depths {depths_cam_for_color.shape}, intensities {getattr(intensities_to_project, 'shape', 'N/A')}.")
             Path(output_path).parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(output_path, image_original_distorted); return


        # Valid mask: points in front of camera AND within image bounds
        # Depths check already implicitly handled by points_cam_h structure
        valid_mask_proj = (depths_cam_for_color > 0.1) & \
                          (image_points_distorted[:, 0] >= 0) & (image_points_distorted[:, 0] < w_img) & \
                          (image_points_distorted[:, 1] >= 0) & (image_points_distorted[:, 1] < h_img)

        image_points_valid = image_points_distorted[valid_mask_proj].astype(int)
        if image_points_valid.shape[0] == 0:
            logging.warning("No valid projected points after all checks.")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(output_path, image_original_distorted); return

        # ... (Coloring logic for norm_values_for_cmap using depths_cam_for_color[valid_mask_proj]
        #      or intensities_to_project[valid_mask_proj] remains the same) ...
        norm_values_for_cmap = None
        if color_by == 'intensity' and intensities_to_project is not None:
            intensities_valid = intensities_to_project[valid_mask_proj]
            if intensities_valid.shape[0] > 0:
                # ... (percentile stretching as before) ...
                if intensity_stretch_percentiles is not None and len(intensity_stretch_percentiles) == 2:
                    min_p, max_p = intensity_stretch_percentiles
                    val_min = np.percentile(intensities_valid.flatten(), min_p)
                    val_max = np.percentile(intensities_valid.flatten(), max_p)
                    if val_max - val_min < 1e-6: norm_values_for_cmap = np.full_like(intensities_valid.flatten(), 0.5, dtype=np.float32)
                    else: norm_values_for_cmap = (intensities_valid.flatten() - val_min) / (val_max - val_min)
                else: norm_values_for_cmap = intensities_valid.flatten().astype(np.float32)
                norm_values_for_cmap = np.clip(norm_values_for_cmap, 0, 1)
            else: color_by = 'depth' # Fallback
        
        if color_by == 'depth': # (either originally or as fallback)
            depths_valid_for_color = depths_cam_for_color[valid_mask_proj]
            if depths_valid_for_color.shape[0] > 0:
                min_depth, max_depth = np.min(depths_valid_for_color), np.max(depths_valid_for_color)
                if max_depth - min_depth < 1e-6: norm_values_for_cmap = np.full_like(depths_valid_for_color, 0.5, dtype=np.float32)
                else: norm_values_for_cmap = (depths_valid_for_color - min_depth) / (max_depth - min_depth)
                norm_values_for_cmap = np.clip(norm_values_for_cmap, 0, 1)

        colors_bgr = None
        if norm_values_for_cmap is not None and norm_values_for_cmap.shape[0] > 0:
            # ... (colormap application as before, using color_map_name) ...
            cmap_obj = plt.get_cmap(color_map_name)
            colors_rgba = cmap_obj(norm_values_for_cmap.squeeze()) 
            # ... (handling RGBA/RGB/Grayscale from colormap as before) ...
            if isinstance(colors_rgba, np.ndarray) and colors_rgba.ndim == 3 and colors_rgba.shape[1] == 1: colors_rgba = colors_rgba.squeeze(axis=1)
            num_channels = colors_rgba.shape[1]
            if num_channels == 4: colors_bgr_float = colors_rgba[:, [2, 1, 0]]
            elif num_channels == 3: colors_bgr_float = colors_rgba[:, [2, 1, 0]]
            elif num_channels == 1: gray_channel = colors_rgba[:, 0]; colors_bgr_float = np.stack((gray_channel, gray_channel, gray_channel), axis=-1)
            else: raise ValueError(f"Colormap unexpected channels: {num_channels}")
            colors_bgr = (colors_bgr_float * 255).astype(np.uint8)

        elif image_points_valid.shape[0] > 0 :
            colors_bgr = np.full((image_points_valid.shape[0], 3), (200, 200, 200), dtype=np.uint8) # Default gray

        # Draw points on image_original_distorted
        output_image_viz = image_original_distorted.copy() # Draw on a copy
        radius = max(1, int(point_size))
        if colors_bgr is not None and image_points_valid.shape[0] == colors_bgr.shape[0]:
            for i in range(image_points_valid.shape[0]):
                center = tuple(image_points_valid[i])
                color = tuple(map(int, colors_bgr[i]))
                cv2.circle(output_image_viz, center, radius, color, thickness=-1)
        elif image_points_valid.shape[0] > 0:
             logging.warning("Projected points exist, but colors not determined. Not drawing.")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save_success = cv2.imwrite(output_path, output_image_viz) # Save the image with drawings
        if not save_success:
             logging.error(f"Failed to save map projection visualization to {output_path}")

    except ImportError:
         logging.error("Matplotlib required for colormaps. pip install matplotlib")
    except cv2.error as e:
        logging.error(f"OpenCV error during visualization: {e} for {original_image_path}")
    except Exception as e:
        logging.error(f"Error during map projection visualization for {original_image_path}: {e}", exc_info=True)

# --- Pose Estimation Function ---
def estimate_final_pose_opencv(points2D: np.ndarray, points3D: np.ndarray, camera_intrinsics: np.ndarray, IMG_WIDTH: int, IMG_HEIGHT: int, dist_coeffs: np.ndarray = None):
    """
    Estimates pose using OpenCV's PnP+RANSAC (cv2.solvePnPRansac).
    Optionally uses pycolmap for refinement if available and successful.
    """
    # ... (Input validation checks remain the same) ...
    points2D = np.ascontiguousarray(points2D, dtype=np.float64)
    points3D = np.ascontiguousarray(points3D, dtype=np.float64)
    min_pnp_points = 15
    if points2D.shape[0] < min_pnp_points: logging.warning(f"Not enough points ({points2D.shape[0]}) for PnP."); return None, None
    if points2D.shape[0] != points3D.shape[0] or points2D.shape[1] != 2 or points3D.shape[1] != 3: logging.error(f"Shape mismatch: 2D {points2D.shape}, 3D {points3D.shape}"); return None, None
    if camera_intrinsics.shape != (3, 3): logging.error(f"Invalid intrinsics shape: {camera_intrinsics.shape}."); return None, None
    if dist_coeffs is None: dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    else: dist_coeffs = np.ascontiguousarray(dist_coeffs, dtype=np.float64)

    try:
        logging.info(f"Running cv2.solvePnPRansac with {points2D.shape[0]} points...")
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=points3D, imagePoints=points2D, cameraMatrix=camera_intrinsics,
            distCoeffs=dist_coeffs, iterationsCount=500, reprojectionError=5.0,
            confidence=0.999999, flags=cv2.SOLVEPNP_SQPNP
        )
        if not success: logging.warning("OpenCV PnP RANSAC failed."); return None, None

        if inliers is None:
             logging.warning("OpenCV PnP RANSAC returned None for inliers.")
             num_inliers = points2D.shape[0]; inlier_indices = np.arange(num_inliers)
        else:
             num_inliers = len(inliers); inlier_indices = inliers.flatten()
        if num_inliers < min_pnp_points: logging.warning(f"Too few PnP inliers ({num_inliers} < {min_pnp_points})."); return None, None
        logging.info(f"OpenCV PnP RANSAC successful. Inliers: {num_inliers} / {points2D.shape[0]}")

        R_cam_map_mat, _ = cv2.Rodrigues(rvec)
        t_cam_map_vec = tvec.flatten()
        T_cam_map_pnp_mat = np.eye(4); T_cam_map_pnp_mat[:3, :3] = R_cam_map_mat; T_cam_map_pnp_mat[:3, 3] = t_cam_map_vec

        T_cam_map_final_mat = T_cam_map_pnp_mat
        if PYCOLMAP_AVAILABLE:
            try:
                logging.info("Attempting pose refinement using pycolmap...")

                R_pnp_colmap = pycolmap.Rotation3d(T_cam_map_pnp_mat[:3, :3])
                t_pnp_colmap = T_cam_map_pnp_mat[:3, 3]
                cam_from_world_pnp = pycolmap.Rigid3d(rotation=R_pnp_colmap, translation=t_pnp_colmap)

                fx, fy, cx, cy = camera_intrinsics[0, 0], camera_intrinsics[1, 1], camera_intrinsics[0, 2], camera_intrinsics[1, 2]
                cam_model_pyc = pycolmap.Camera(model='PINHOLE', width=IMG_WIDTH, height=IMG_HEIGHT, params=[fx, fy, cx, cy])

                refine_options = pycolmap.AbsolutePoseRefinementOptions()
                refine_options.refine_extra_params = True
                refine_options.refine_focal_length = True
                refine_options.max_num_iterations = 500
                refine_options.print_summary = True

                points2D_inliers = points2D[inlier_indices]
                points3D_inliers = points3D[inlier_indices]
                inlier_mask_colmap = np.ones((num_inliers, 1), dtype=bool)

                # Call using positional arguments (as fixed before)
                refine_ret = pycolmap.refine_absolute_pose(
                    cam_from_world_pnp,
                    points2D_inliers,
                    points3D_inliers,
                    inlier_mask_colmap,
                    cam_model_pyc,
                    refine_options
                 )

                if refine_ret is not None and 'cam_from_world' in refine_ret:
                     # Assume success if 'cam_from_world' is present
                     T_cam_map_final_refined = refine_ret['cam_from_world'] # This is a pycolmap.Rigid3d object
                     if isinstance(T_cam_map_final_refined, pycolmap.Rigid3d): # Extra safety check
                         T_cam_map_final_mat_3by4 = T_cam_map_final_refined.matrix() # Update final matrix
                         T_cam_map_final_mat = np.eye(4); T_cam_map_final_mat[:3, :4] = T_cam_map_final_mat_3by4
                         logging.info(f"Pycolmap pose refinement successful.") # Simplified message
                     else:
                         logging.warning("Pycolmap refinement returned 'cam_from_world' but it wasn't a Rigid3d object. Using OpenCV PnP result.")
                else:
                     # Handle failure case (None returned or key missing)
                     logging.warning("Pycolmap pose refinement failed or returned unexpected dictionary. Using OpenCV PnP result.")
                     # T_cam_map_final_mat remains the OpenCV PnP result

            except AttributeError as e_refine_attr:
                 logging.error(f"AttributeError during pycolmap pose refinement: {e_refine_attr}. Using OpenCV PnP result.")
                 logging.error(traceback.format_exc())
            except TypeError as e_type_pyc:
                 logging.error(f"TypeError during pycolmap.refine_absolute_pose call: {e_type_pyc}. Using OpenCV PnP result.")
                 logging.error(traceback.format_exc())
            except Exception as e_refine:
                logging.error(f"ERROR during pycolmap pose refinement: {e_refine}")
                logging.error(traceback.format_exc())
                logging.warning("Using OpenCV PnP result due to refinement error.")

        # --- Convert final pose T_cam_map to output format T_map_cam ---
        try:
            T_map_cam_final_mat = np.linalg.inv(T_cam_map_final_mat)
        except np.linalg.LinAlgError:
             logging.error("Failed to invert the final pose matrix (T_cam_map). Returning None.")
             return None, None

        return T_map_cam_final_mat, inlier_indices

    except Exception as e_pnp:
        logging.error(f"ERROR during OpenCV PnP main block: {e_pnp}")
        logging.error(traceback.format_exc())
        return None, None

# ==============================================================================
# Helper Function: Visualize Features
# ==============================================================================
def visualize_features(h5_feature_path, image_list_path, image_base_dir, vis_output_dir, num_to_vis=1, prefix="vis"):
    """Visualizes keypoints from an HDF5 file onto images."""
    logging.info(f"--- Starting Visualization ({prefix}) ---")
    os.makedirs(vis_output_dir, exist_ok=True)

    if not os.path.exists(h5_feature_path):
        logging.error(f"Feature file not found at {h5_feature_path}. Cannot visualize.")
        return

    try:
        with open(image_list_path, 'r') as f:
            image_names = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logging.error(f"Image list file not found at {image_list_path}. Cannot visualize.")
        return

    if not image_names:
        logging.warning("No image names found in list file.")
        return

    visualized_count = 0
    try:
        with h5py.File(h5_feature_path, 'r') as f:
            h5_keys = list(f.keys())
            logging.info(f"Found {len(h5_keys)} keys in {os.path.basename(h5_feature_path)}")

            for i, image_name in enumerate(image_names):
                if visualized_count >= num_to_vis:
                    break # Stop after visualizing the desired number

                image_path = os.path.join(image_base_dir, image_name)
                logging.info(f"Attempting visualization for: {image_path}")

                # --- Find corresponding key in HDF5 ---
                found_key = None
                if image_name in f:
                    found_key = image_name
                else:
                    # Check if keys might contain relative paths from image_dir used during extraction
                    # (hloc behaviour can vary)
                    potential_key = os.path.join(os.path.basename(image_base_dir), image_name)
                    if potential_key in f:
                        found_key = potential_key
                    else: # Last resort: check endswith
                        for k in h5_keys:
                           if k.endswith(image_name):
                               found_key = k
                               logging.warning(f"Used fuzzy match: H5 key '{k}' for image name '{image_name}'")
                               break

                if not found_key:
                    logging.warning(f"Key corresponding to '{image_name}' not found in HDF5 file. Skipping visualization.")
                    continue

                # --- Load Image ---
                image = cv2.imread(image_path)
                if image is None:
                    logging.error(f"Could not load image {image_path}. Skipping.")
                    continue

                # --- Read and Draw Keypoints ---
                try:
                    keypoints = f[found_key]['keypoints'][()]
                    logging.info(f"Found {len(keypoints)} keypoints for H5 key '{found_key}'.")

                    radius = 3
                    color = (0, 0, 255)  # Red (BGR)
                    thickness = 1

                    for kp in keypoints:
                        x, y = int(round(kp[0])), int(round(kp[1]))
                        # Add boundary check just in case
                        h, w = image.shape[:2]
                        if 0 <= x < w and 0 <= y < h:
                           cv2.circle(image, (x, y), radius, color, thickness)

                    # --- Save ---
                    safe_image_name = os.path.splitext(os.path.basename(image_name))[0].replace('/','_')
                    vis_filename = f"{prefix}_{safe_image_name}.png"
                    vis_save_path = os.path.join(vis_output_dir, vis_filename)
                    cv2.imwrite(vis_save_path, image)
                    logging.info(f"Visualization saved to: {vis_save_path}")
                    visualized_count += 1

                except KeyError:
                     logging.error(f"Dataset 'keypoints' not found for key '{found_key}' in HDF5. Skipping.")
                except Exception as e:
                    logging.error(f"Error processing keypoints for '{found_key}': {e}")

    except Exception as e:
        logging.error(f"An error occurred during visualization: {e}", exc_info=True)

# ==============================================================================
# Helper Function: Apply Masks Post-Extraction
# ==============================================================================
def apply_masks_to_features(feature_file_path: Path, image_list_path: Path, image_base_dir: Path, mask_suffix="_mask.png", neighborhood_size=4, expected_descriptor_dim=256):
    """
    Filters features in an HDF5 file based on corresponding mask images,
    using a stricter neighborhood check. Assumes descriptors are stored
    as (descriptor_dim, num_keypoints) and saves them back in the same format.

    Args:
        feature_file_path (Path): Path to the .h5 feature file.
        image_list_path (Path): Path to the text file listing image names.
        image_base_dir (Path): Directory containing images and masks.
        mask_suffix (str): Suffix to find mask files.
        neighborhood_size (int): The size (width/height) of the square neighborhood
                                 to check around each keypoint in the mask.
        expected_descriptor_dim (int): The expected dimension (e.g., 256).

    Returns:
        bool: True if the process completed without critical errors, False otherwise.
    """
    logging.info(f"--- Applying Masks Post-Extraction to {feature_file_path.name} (Neighborhood Check: {neighborhood_size}x{neighborhood_size}) ---")
    if neighborhood_size <= 0:
        logging.error("Neighborhood size must be positive.")
        return False
    if not feature_file_path.exists(): # Use Path.exists()
        logging.error(f"Feature file not found at {feature_file_path}. Cannot apply masks.")
        return False

    try:
        # Use Path.read_text()
        image_names = [line.strip() for line in image_list_path.read_text().splitlines() if line.strip()]
    except FileNotFoundError:
        logging.error(f"Image list file not found at {image_list_path}. Cannot apply masks.")
        return False
    except Exception as e:
        logging.error(f"Error reading image list {image_list_path}: {e}", exc_info=True)
        return False


    logging.info(f"Applying masks for {len(image_names)} images listed in {image_list_path.name}")
    processed_count = 0; skipped_count = 0; error_count = 0
    masking_successful = True

    # --- Define neighborhood offsets (same logic) ---
    if neighborhood_size % 2 == 0: offset_start = - (neighborhood_size // 2 - 1); offset_end = neighborhood_size // 2 + 1
    else: offset_start = - (neighborhood_size // 2); offset_end = neighborhood_size // 2 + 1
    rel_offsets = np.arange(offset_start, offset_end)
    offset_grid = np.array(np.meshgrid(rel_offsets, rel_offsets)).T.reshape(-1, 2)
    logging.debug(f"Using {neighborhood_size}x{neighborhood_size} neighborhood with relative offsets:\n{offset_grid}")

    try:
        with h5py.File(feature_file_path, 'r+') as hfile:
            image_keys_in_h5 = list(hfile.keys())
            logging.info(f"Opened HDF5 file. Found {len(image_keys_in_h5)} existing keys.")

            for image_name in image_names:
                # --- Find corresponding key (Pathlib aware - simplified) ---
                # Hloc often stores keys relative to an 'export' dir, or just base names.
                # Try direct name first, then check if any key *ends with* the name.
                found_key = None
                if image_name in hfile:
                    found_key = image_name
                else:
                    for k in image_keys_in_h5:
                         # Check if the HDF5 key ends with the image name from the list
                         # This handles cases where hloc might prepend path components
                         if k.endswith(image_name):
                             found_key = k
                             logging.warning(f"Used suffix match: H5 key '{k}' for image name '{image_name}'")
                             break
                if not found_key:
                    logging.warning(f"Key for '{image_name}' not found in HDF5 file '{feature_file_path.name}'. Skipping mask.")
                    skipped_count += 1; continue

                # --- Load Mask (using Pathlib) ---
                mask_path = image_base_dir / f"{Path(image_name).stem}{mask_suffix}"
                if not mask_path.exists():
                    logging.warning(f"Mask file '{mask_path}' not found for '{image_name}'. Skipping mask."); skipped_count += 1; continue
                # Use str() for cv2.imread
                mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_image is None:
                    logging.warning(f"Could not read mask file '{mask_path}'. Skipping mask for '{image_name}'."); skipped_count += 1; continue

                # --- Process Features for this image ---
                try:
                    feature_group = hfile[found_key]
                    keypoints = feature_group['keypoints'][()]
                    descriptors_stored = feature_group['descriptors'][()] # Shape (expected_dim, N)
                    scores = feature_group.get('scores');
                    if scores is not None: scores = scores[()]

                    kp_count = keypoints.shape[0]
                    desc_shape_stored = descriptors_stored.shape

                    # --- Transpose Descriptors for Processing ---
                    # We need (N, expected_dim) for filtering logic
                    descriptors_proc = None
                    if kp_count == 0:
                        if descriptors_stored.size == 0:
                             logging.debug(f"Key '{found_key}' has 0 keypoints. Skipping masking logic.")
                             processed_count += 1; continue
                        else:
                             logging.error(f"MISMATCH for key '{found_key}': Kpts(0) but stored Descs shape {desc_shape_stored}. Skipping.")
                             error_count += 1; masking_successful = False; continue

                    if len(desc_shape_stored) == 2 and desc_shape_stored[0] == expected_descriptor_dim and desc_shape_stored[1] == kp_count:
                        # Correct stored format (expected_dim, N). Transpose for processing.
                        descriptors_proc = descriptors_stored.T
                        logging.debug(f"Transposing stored descriptors {desc_shape_stored} to {descriptors_proc.shape} for processing key '{found_key}'.")
                    elif len(desc_shape_stored) == 2 and desc_shape_stored[0] == kp_count and desc_shape_stored[1] == expected_descriptor_dim:
                         # Found (N, expected_dim) format stored - incorrect based on new goal.
                         # Use as is for processing, but warn.
                         logging.warning(f"Descriptors for key '{found_key}' were stored as {desc_shape_stored} (N, dim). Using directly for processing, but check storage logic.")
                         descriptors_proc = descriptors_stored # Use directly
                    else:
                        # Unexpected shape found in storage
                        logging.error(f"UNEXPECTED stored descriptor shape for key '{found_key}': {desc_shape_stored}, Kpts {kp_count}. Skipping masking.")
                        error_count += 1; masking_successful = False; continue

                    # --- Sanity Check (using processing format) ---
                    desc_shape_proc = descriptors_proc.shape
                    if kp_count != desc_shape_proc[0]:
                        logging.error(f"CRITICAL MISMATCH for key '{found_key}' after transpose for processing: Kpts({kp_count}) != Proc Descs({desc_shape_proc[0]}). Skipping.")
                        error_count += 1; masking_successful = False; continue

                    initial_kp_count = kp_count # Redundant check above, but keep for clarity

                    # --- Stricter Filtering Logic (same as before) ---
                    kp_coords = keypoints[:, :2].round().astype(int)
                    mask_h, mask_w = mask_image.shape[:2]
                    neighborhood_coords = kp_coords[:, np.newaxis, :] + offset_grid[np.newaxis, :, :]
                    neighborhood_coords[..., 0] = np.clip(neighborhood_coords[..., 0], 0, mask_w - 1)
                    neighborhood_coords[..., 1] = np.clip(neighborhood_coords[..., 1], 0, mask_h - 1)
                    mask_values = mask_image[neighborhood_coords[..., 1], neighborhood_coords[..., 0]]
                    keep_indices = np.all(mask_values > 0, axis=1)

                    # --- Apply filter to processing format descriptors (N, dim) ---
                    filtered_keypoints = keypoints[keep_indices]
                    filtered_descriptors_proc = descriptors_proc[keep_indices] # Filter rows
                    filtered_scores = scores[keep_indices] if scores is not None else None
                    final_kp_count = len(filtered_keypoints)

                    # --- Transpose Filtered Descriptors Back for Saving ---
                    # Target storage format is (dim, N_filtered)
                    filtered_descriptors_save = filtered_descriptors_proc.T

                    # --- Update HDF5 File In-Place ---
                    logging.debug(f"Updating HDF5 for key '{found_key}'")
                    del feature_group['keypoints']; del feature_group['descriptors']
                    if 'scores' in feature_group: del feature_group['scores']

                    feature_group.create_dataset('keypoints', data=filtered_keypoints, compression='gzip')
                    # Save the transposed (dim, N_filtered) descriptors
                    feature_group.create_dataset('descriptors', data=filtered_descriptors_save, compression='gzip')
                    if filtered_scores is not None: feature_group.create_dataset('scores', data=filtered_scores, compression='gzip')

                    logging.info(f"Masked '{found_key}' (strict {neighborhood_size}x{neighborhood_size}): {initial_kp_count} -> {final_kp_count} kpts. Saved descs shape: {filtered_descriptors_save.shape}")
                    processed_count += 1

                except Exception as e:
                    logging.error(f"Error processing/filtering features for '{found_key}': {e}", exc_info=True)
                    error_count += 1; masking_successful = False

    except IOError as e:
        logging.error(f"I/O Error accessing HDF5 file {feature_file_path}: {e}")
        masking_successful = False
    except Exception as e:
        logging.error(f"An unexpected error occurred during mask application: {e}", exc_info=True)
        masking_successful = False

    logging.info(f"Masking summary: Processed={processed_count}, Skipped={skipped_count}, Errors={error_count}")
    if error_count > 0:
        logging.warning("Masking process encountered errors.")
        masking_successful = False

    return masking_successful

def check_and_fix_features(feature_file_path: Path, expected_descriptor_dim: int = 256):
    """
    Checks features in an HDF5 file for descriptor shape and mismatches.
    Ensures descriptors are stored in the format (descriptor_dim, num_keypoints),
    i.e., (expected_dim, N). Fixes shapes by transposing if needed.

    Args:
        feature_file_path (Path): Path to the .h5 feature file to check and potentially modify.
        expected_descriptor_dim (int): The expected dimension of a single descriptor vector
                                       (e.g., 256 for SuperPoint).

    Returns:
        bool: True if the process completed without critical errors (mismatches),
              False otherwise. Note: Fixing shapes is logged but does not cause False return.
    """
    logging.info(f"--- Checking and Fixing Features in {feature_file_path.name} (Ensuring format: {expected_descriptor_dim}, N) ---")
    if not feature_file_path.exists():
        logging.error(f"Feature file not found at {feature_file_path}. Cannot perform checks.")
        return False

    processed_count = 0
    fixed_count = 0
    mismatch_errors = 0
    other_errors = 0
    check_successful = True # Assume success unless a mismatch occurs

    try:
        with h5py.File(feature_file_path, 'r+') as hfile:
            image_keys = list(hfile.keys())
            logging.info(f"Opened HDF5 file. Checking {len(image_keys)} keys.")

            for key in image_keys:
                try:
                    feature_group = hfile[key]
                    if 'keypoints' not in feature_group or 'descriptors' not in feature_group:
                        logging.warning(f"Key '{key}' is missing 'keypoints' or 'descriptors'. Skipping checks.")
                        continue

                    keypoints = feature_group['keypoints'][()]
                    descriptors_orig = feature_group['descriptors'][()] # Load original

                    kp_count = keypoints.shape[0]
                    desc_shape = descriptors_orig.shape
                    needs_update = False
                    descriptors_to_save = descriptors_orig # Start with original

                    # Skip checks if no keypoints exist
                    if kp_count == 0:
                        if descriptors_orig.size == 0:
                            logging.debug(f"Key '{key}' has 0 keypoints and 0 descriptors. OK.")
                            processed_count += 1
                            continue
                        else:
                             logging.error(f"MISMATCH for key '{key}': Kpts(0) but Descs shape {desc_shape}. Skipping.")
                             mismatch_errors += 1; check_successful = False; continue

                    # --- 1. Check and Fix Shape ---
                    # Goal: Store as (expected_dim, N)
                    if len(desc_shape) == 2:
                        if desc_shape[0] == kp_count and desc_shape[1] == expected_descriptor_dim:
                            # Found (N, expected_dim), needs transpose for storage
                            logging.warning(f"Descriptors for key '{key}' have shape {desc_shape} but target is ({expected_descriptor_dim}, N). Fixing by transposing.")
                            descriptors_to_save = descriptors_orig.T # Transpose for saving
                            desc_shape = descriptors_to_save.shape # Update shape
                            needs_update = True
                            fixed_count += 1
                        elif desc_shape[0] == expected_descriptor_dim and desc_shape[1] == kp_count:
                            # Found (expected_dim, N), correct format, no change needed
                            logging.debug(f"Descriptors for key '{key}' already have target shape {desc_shape}. OK.")
                            descriptors_to_save = descriptors_orig # Already correct
                        else:
                            # Found unexpected shape
                            logging.error(f"UNEXPECTED shape for key '{key}': Descs {desc_shape}, Kpts {kp_count}. Cannot fix. Skipping.")
                            mismatch_errors += 1; check_successful = False; continue
                    else:
                        # Not 2D array
                        logging.error(f"UNEXPECTED descriptor dimensions for key '{key}': {len(desc_shape)}. Skipping.")
                        mismatch_errors += 1; check_successful = False; continue

                    # --- 2. Sanity Check: Keypoints and (Corrected) Descriptors Match ---
                    # After correction, shape should be (expected_dim, N)
                    if kp_count != desc_shape[1]: # Compare N with second dim
                        logging.error(f"CRITICAL MISMATCH for key '{key}' after shape check: Kpts({kp_count}) != Descs second dim ({desc_shape[1]}). Final intended shape was {desc_shape}. Skipping update.")
                        mismatch_errors += 1
                        check_successful = False
                        continue # Don't save if counts are wrong

                    # --- 3. Update HDF5 File If Fix Applied ---
                    if needs_update:
                        logging.info(f"Updating descriptors for key '{key}' to shape {desc_shape} in HDF5 file.")
                        del feature_group['descriptors']
                        feature_group.create_dataset('descriptors', data=descriptors_to_save, compression='gzip')

                    processed_count += 1

                except Exception as e:
                    logging.error(f"Error during check/fix for key '{key}': {e}", exc_info=False)
                    logging.error(f"Traceback for key '{key}':\n{traceback.format_exc()}")
                    other_errors += 1
                    check_successful = False

    except IOError as e:
        logging.error(f"I/O Error accessing HDF5 file {feature_file_path}: {e}")
        check_successful = False
    except Exception as e:
        logging.error(f"An unexpected error occurred during feature checking/fixing: {e}", exc_info=True)
        check_successful = False

    logging.info(f"Feature Check/Fix Summary: Processed={processed_count}, Fixed Shape={fixed_count}, Mismatch Errors={mismatch_errors}, Other Errors={other_errors}")
    if not check_successful:
        logging.error("Feature checking process encountered critical errors.")

    return check_successful

def undistort_images_fisheye(
    image_list_path,
    original_image_dir,
    output_image_dir,
    K, D, new_K=None, new_size=None
):
    """
    Undistorts images listed in a file using the fisheye model.

    Args:
        image_list_path (str): Path to the text file listing image filenames.
        original_image_dir (str): Directory containing the original distorted images.
        output_image_dir (str): Directory where undistorted images will be saved.
        K (np.ndarray): Original 3x3 camera intrinsic matrix.
        D (np.ndarray): Fisheye distortion coefficients (k1, k2, k3, k4). Should be shape (4,1) or (1,4).
        new_K (np.ndarray, optional): Camera matrix for the undistorted image. If None, K is used.
        new_size (tuple, optional): Size (width, height) of the undistorted image. If None, original size is used.

    Returns:
        bool: True if all images were processed successfully, False otherwise.
    """
    logging.info(f"--- Starting Fisheye Undistortion ---")
    logging.info(f"Original images: {original_image_dir}")
    logging.info(f"Outputting to: {output_image_dir}")

    if not os.path.exists(original_image_dir):
        logging.error(f"Original image directory not found: {original_image_dir}")
        return False

    os.makedirs(output_image_dir, exist_ok=True)

    try:
        with open(image_list_path, 'r') as f:
            image_names = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logging.error(f"Image list file not found: {image_list_path}")
        return False

    if not image_names:
        logging.warning("Image list is empty. No images to undistort.")
        return True # Technically successful, nothing to do

    overall_success = True
    processed_count = 0
    error_count = 0

    # Use K as new_K if not provided
    if new_K is None:
        new_K = K

    for image_name in image_names:
        original_path = os.path.join(original_image_dir, image_name)
        output_path = os.path.join(output_image_dir, image_name)

        try:
            if not os.path.exists(original_path):
                logging.error(f"  Image not found: {original_path}")
                error_count += 1
                overall_success = False
                continue

            img_distorted = cv2.imread(original_path)
            if img_distorted is None:
                logging.error(f"  Failed to load image: {original_path}")
                error_count += 1
                overall_success = False
                continue

            # Determine output size if not specified
            h, w = img_distorted.shape[:2]
            current_new_size = new_size if new_size is not None else (w, h)

            # Perform undistortion using fisheye model
            img_undistorted = cv2.fisheye.undistortImage(
                img_distorted, K, D, Knew=new_K, new_size=current_new_size
            )

            # Save the undistorted image
            if not cv2.imwrite(output_path, img_undistorted):
                 logging.error(f"  Failed to save undistorted image to: {output_path}")
                 error_count += 1
                 overall_success = False
                 continue

            processed_count += 1
            logging.debug(f"  Successfully undistorted and saved: {output_path}")

        except Exception as e:
            logging.error(f"  Error processing {image_name}: {e}", exc_info=True)
            error_count += 1
            overall_success = False

    logging.info(f"--- Fisheye Undistortion Finished ---")
    logging.info(f"Processed: {processed_count}, Errors: {error_count}")
    return overall_success

def save_processed_data(
    output_dir: Path,
    processed_lidar_data: dict,
    rendered_views_info: list,
    lidar_data_filename: str = "processed_lidar_data.npz",
    render_info_filename: str = "rendered_views_info.json"
):
    """
    Saves processed LiDAR data arrays and rendered views info to files.

    Args:
        output_dir (Path): Directory to save the files in.
        processed_lidar_data (dict): Dictionary containing processed NumPy arrays
                                    ('points', 'indices', 'normals', 'intensities').
                                    The 'kdtree' key will be ignored.
        rendered_views_info (list): List of dictionaries containing render info.
                                    Path objects and NumPy arrays will be converted.
        lidar_data_filename (str): Filename for the NumPy archive (.npz).
        render_info_filename (str): Filename for the JSON file (.json).

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    logging.info(f"Saving processed data to directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_ok = True

    # --- Save processed_lidar_data (NumPy arrays) ---
    lidar_data_path = output_dir / lidar_data_filename
    try:
        arrays_to_save = {}
        required_keys = ['points', 'indices', 'normals', 'intensities']
        for key in required_keys:
            if key in processed_lidar_data and isinstance(processed_lidar_data[key], np.ndarray):
                arrays_to_save[key] = processed_lidar_data[key]
            else:
                raise ValueError(f"Missing or invalid NumPy array for key '{key}' in processed_lidar_data")

        np.savez_compressed(lidar_data_path, **arrays_to_save)
        logging.info(f"Saved LiDAR data arrays to: {lidar_data_path}")

    except Exception as e:
        logging.error(f"Error saving processed LiDAR data to {lidar_data_path}: {e}", exc_info=True)
        save_ok = False

    # --- Save rendered_views_info (JSON) ---
    render_info_path = output_dir / render_info_filename
    try:
        serializable_render_info = []
        for info_dict in rendered_views_info:
            serializable_dict = {}
            for key, value in info_dict.items():
                if isinstance(value, Path):
                    serializable_dict[key] = str(value) # Convert Path to string
                elif isinstance(value, np.ndarray):
                    serializable_dict[key] = value.tolist() # Convert NumPy array to list
                else:
                    serializable_dict[key] = value # Keep other types (str, int, etc.)
            serializable_render_info.append(serializable_dict)

        with render_info_path.open('w') as f:
            json.dump(serializable_render_info, f, indent=4)
        logging.info(f"Saved rendered views info to: {render_info_path}")

    except Exception as e:
        logging.error(f"Error saving rendered views info to {render_info_path}: {e}", exc_info=True)
        save_ok = False

    return save_ok

def load_processed_data(
    output_dir: Path,
    lidar_data_filename: str = "processed_lidar_data.npz",
    render_info_filename: str = "rendered_views_info.json",
    rebuild_kdtree: bool = True # Option to rebuild KDTree on load
):
    """
    Loads processed LiDAR data arrays and rendered views info from files.
    Optionally rebuilds the KDTree.

    Args:
        output_dir (Path): Directory where the files were saved.
        lidar_data_filename (str): Filename of the NumPy archive (.npz).
        render_info_filename (str): Filename of the JSON file (.json).
        rebuild_kdtree (bool): If True, rebuilds the KDTree from loaded points.

    Returns:
        tuple[dict | None, list | None]: A tuple containing:
            - processed_lidar_data (dict): Loaded data with NumPy arrays and
              optionally a rebuilt 'kdtree'. None on failure.
            - rendered_views_info (list): Loaded render info with Path objects
              and NumPy arrays restored. None on failure.
    """
    logging.info(f"Loading processed data from directory: {output_dir}")
    processed_lidar_data = None
    rendered_views_info = None
    load_ok = True

    # --- Load processed_lidar_data (NumPy arrays) ---
    lidar_data_path = output_dir / lidar_data_filename
    try:
        if not lidar_data_path.is_file():
            raise FileNotFoundError(f"LiDAR data file not found: {lidar_data_path}")

        loaded_npz = np.load(lidar_data_path)
        processed_lidar_data = {}
        required_keys = ['points', 'indices', 'normals', 'intensities'] # Assuming these were saved
        for key in required_keys:
            if key not in loaded_npz:
                raise KeyError(f"Key '{key}' not found in loaded npz file: {lidar_data_path}")
            processed_lidar_data[key] = loaded_npz[key]
        loaded_npz.close()

        processed_lidar_data['kdtree'] = None
        if rebuild_kdtree:
            points_np = processed_lidar_data.get('points')
            if points_np is not None and len(points_np) > 0:
                logging.info("Rebuilding KDTree from loaded points...")
                try:
                    processed_lidar_data['kdtree'] = KDTree(points_np)
                    logging.info("KDTree rebuilt successfully.")
                except Exception as e_kdtree:
                     logging.warning(f"Failed to rebuild KDTree: {e_kdtree}. 'kdtree' will be None.")
            else:
                 logging.warning("Cannot rebuild KDTree: No points loaded or points array is empty.")

        logging.info(f"Loaded LiDAR data arrays from: {lidar_data_path}")

    except Exception as e:
        logging.error(f"Error loading processed LiDAR data from {lidar_data_path}: {e}", exc_info=True)
        processed_lidar_data = None
        load_ok = False

    # --- Load rendered_views_info (JSON) ---
    render_info_path = output_dir / render_info_filename
    try:
        if not render_info_path.is_file():
            # Only raise error if lidar data was successfully loaded, otherwise it might just be the first run
            if load_ok:
                 raise FileNotFoundError(f"Render info file not found: {render_info_path}")
            else:
                 logging.warning(f"Render info file not found, continuing without it: {render_info_path}")
                 rendered_views_info = [] # Return empty list if file not found and lidar failed too

        else: # File exists, try to load
            with render_info_path.open('r') as f:
                serializable_render_info = json.load(f)

            rendered_views_info = []
            # --- FIX: Add 'depth_map_path' here ---
            path_keys = ['geometric_image_path', 'index_map_path', 'mask_path', 'depth_map_path']
            # --- END FIX ---
            pose_key = 'pose'

            for serializable_dict in serializable_render_info:
                restored_dict = {}
                for key, value in serializable_dict.items():
                    if key in path_keys and isinstance(value, str):
                        restored_dict[key] = Path(value) # Convert string back to Path
                    elif key == pose_key and isinstance(value, list):
                        restored_dict[key] = np.array(value) # Convert list back to NumPy array
                    # --- Add Check for missing keys gracefully ---
                    elif key not in path_keys and key != pose_key:
                         restored_dict[key] = value # Keep other types
                # --- Ensure required keys for depth linking are present after potential conversion ---
                if 'geometric_image_path' in restored_dict and \
                   'depth_map_path' in restored_dict and \
                   'pose' in restored_dict:
                    rendered_views_info.append(restored_dict)
                else:
                    logging.warning(f"Skipping loaded render info dict due to missing keys after conversion: {serializable_dict.keys()}")


            logging.info(f"Loaded {len(rendered_views_info)} valid rendered views info entries from: {render_info_path}")

    except json.JSONDecodeError as e_json:
         logging.error(f"Error decoding JSON from {render_info_path}: {e_json}", exc_info=True)
         rendered_views_info = None # Set to None on JSON error
         load_ok = False
    except Exception as e:
        logging.error(f"Error loading or processing rendered views info from {render_info_path}: {e}", exc_info=True)
        rendered_views_info = None # Set to None on other errors
        load_ok = False

    # --- Final Check ---
    if not load_ok and processed_lidar_data is None: # If both failed
         logging.error("Loading BOTH processed LiDAR data AND render info failed.")
         return None, None
    elif not load_ok: # If only render info failed but lidar data might be ok
         logging.error("Loading rendered views info failed.")
         # Return lidar data but None for render info
         return processed_lidar_data, None


    return processed_lidar_data, rendered_views_info

def load_and_prepare_poses(csv_path, target_camera):
    """Loads poses from CSV, filters by camera, sorts by time, and converts to matrices."""
    logging.info(f"Loading poses for camera '{target_camera}' from {csv_path}")
    try:
        # Define column names for clarity
        col_names = ['timestamp', 'camera_name'] + [f'p{i}' for i in range(16)]
        df = pd.read_csv(csv_path, header=None, names=col_names)
        logging.info(f"Read {len(df)} total rows from CSV.")

        # Filter for the target camera
        df_cam = df[df['camera_name'] == target_camera].copy()
        logging.info(f"Found {len(df_cam)} rows for camera '{target_camera}'.")

        if df_cam.empty:
            logging.warning(f"No poses found for camera '{target_camera}' in the CSV.")
            return None, None # Modified to return two Nones as per load_and_prepare_ego_poses

        # Convert timestamp and pose columns to numeric first
        df_cam['timestamp'] = pd.to_numeric(df_cam['timestamp'], errors='coerce')
        pose_cols = [f'p{i}' for i in range(16)]
        for col in pose_cols:
            df_cam[col] = pd.to_numeric(df_cam[col], errors='coerce')

        df_cam.dropna(subset=['timestamp'] + pose_cols, inplace=True)
        if df_cam.empty:
             logging.error(f"No valid numeric pose data found for camera '{target_camera}' after cleaning.")
             return None, None # Modified

        try:
            # Timestamps are microseconds
            df_cam['timestamp'] = df_cam['timestamp'].astype(np.int64)
        except ValueError as e:
            logging.error(f"Could not convert cleaned timestamps to int64 for camera '{target_camera}': {e}. Check data format.")
            return None, None # Modified

        df_cam.sort_values(by='timestamp', inplace=True)
        df_cam.reset_index(drop=True, inplace=True)

        poses = []
        timestamps = []
        for index, row in df_cam.iterrows():
            try:
                pose_flat = row[pose_cols].values.astype(np.float64)
                pose_matrix = pose_flat.reshape(4, 4)
                poses.append(pose_matrix)
                timestamps.append(row['timestamp']) # row['timestamp'] is already int64 (microseconds)
            except ValueError:
                logging.warning(f"Could not reshape pose at timestamp {row['timestamp']}. Skipping row {index}.")
                continue
            except Exception as e:
                 logging.warning(f"Error processing pose at timestamp {row['timestamp']}: {e}. Skipping row {index}.")
                 continue

        if not timestamps:
            logging.error(f"No valid poses could be constructed for camera '{target_camera}'.")
            return None, None # Modified

        logging.info(f"Prepared {len(timestamps)} valid, sorted poses for '{target_camera}'.")
        return np.array(timestamps, dtype=np.int64), poses # Timestamps are microseconds

    except FileNotFoundError:
        logging.error(f"Pose CSV file not found at: {csv_path}")
        return None, None # Modified
    except Exception as e:
        logging.error(f"Error reading or processing pose CSV: {e}", exc_info=True)
        return None, None # Modified

def load_and_prepare_ego_poses(csv_path):
    """
    Loads EGO VEHICLE poses (T_map_ego) from a CSV file.
    Assumes CSV format: timestamp (microseconds), p0, p1, ..., p15 (17 columns total).
    Sorts by time and converts to a list of 4x4 matrices.

    Args:
        csv_path (str or Path): Path to the ego pose CSV file.

    Returns:
        tuple[np.ndarray | None, list | None]: A tuple containing:
            - sorted_timestamps_us (np.ndarray): NumPy array of sorted microsecond timestamps.
            - sorted_poses (list): List of corresponding 4x4 NumPy pose matrices (T_map_ego).
            Returns (None, None) on failure.
    """
    logging.info(f"Loading ego vehicle poses from {csv_path}")
    try:
        col_names = ['timestamp'] + [f'p{i}' for i in range(16)]
        df = pd.read_csv(csv_path, header=None, names=col_names)
        logging.info(f"Read {len(df)} total rows from ego pose CSV.")

        if df.empty:
            logging.error(f"Ego pose CSV file '{csv_path}' is empty.")
            return None, None

        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        pose_cols = [f'p{i}' for i in range(16)]
        for col in pose_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')

        df.dropna(subset=['timestamp'] + pose_cols, inplace=True)
        logging.info(f"Found {len(df)} rows with valid numeric data after cleaning.")
        if df.empty:
             logging.error(f"No valid numeric pose data found in '{csv_path}' after cleaning.")
             return None, None

        try:
            # Timestamps are microseconds
            df['timestamp'] = df['timestamp'].astype(np.int64)
        except ValueError as e:
            logging.error(f"Could not convert cleaned timestamps to int64 in ego poses: {e}. Check data format.")
            return None, None

        df.sort_values(by='timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)

        poses = []
        timestamps_us = []
        for index, row in df.iterrows():
            try:
                pose_flat = row[pose_cols].values.astype(np.float64)
                pose_matrix = pose_flat.reshape(4, 4)
                R_mat = pose_matrix[:3, :3]
                if not np.isclose(np.linalg.det(R_mat), 1.0, atol=1e-4):
                     logging.warning(f"Invalid rotation matrix determinant ({np.linalg.det(R_mat):.4f}) found at timestamp {row['timestamp']}. Skipping row {index}.")
                     continue
                if not np.allclose(R_mat.T @ R_mat, np.eye(3), atol=1e-4):
                     logging.warning(f"Rotation matrix not orthogonal for timestamp {row['timestamp']}. Skipping row {index}.")
                     continue
                poses.append(pose_matrix)
                timestamps_us.append(row['timestamp']) # row['timestamp'] is already int64 (microseconds)
            except ValueError:
                logging.warning(f"Could not reshape pose at timestamp {row['timestamp']}. Skipping row {index}.")
                continue
            except Exception as e:
                 logging.warning(f"Error processing pose row {index} (ts: {row['timestamp']}): {e}. Skipping.")
                 continue

        if not timestamps_us:
            logging.error(f"No valid ego poses could be constructed from '{csv_path}'.")
            return None, None

        sorted_timestamps_us_np = np.array(timestamps_us, dtype=np.int64)

        if len(sorted_timestamps_us_np) > 1 and not np.all(np.diff(sorted_timestamps_us_np) > 0):
            logging.warning(f"Timestamps in {csv_path} are not strictly increasing after processing. Re-sorting.")
            sort_indices = np.argsort(sorted_timestamps_us_np)
            sorted_timestamps_us_np = sorted_timestamps_us_np[sort_indices]
            poses = [poses[i] for i in sort_indices]

        logging.info(f"Prepared {len(poses)} valid, sorted ego poses.")
        return sorted_timestamps_us_np, poses # Timestamps are microseconds

    except FileNotFoundError:
        logging.error(f"Ego pose CSV file not found at: {csv_path}")
        return None, None
    except Exception as e:
        logging.error(f"Error reading or processing ego pose CSV '{csv_path}': {e}", exc_info=True)
        return None, None

def get_pose_for_timestamp(query_ts_us, timestamps_us, poses, tolerance_us=1000):
    """
    Finds exact pose or interpolates between two poses using SLERP and LERP.
    Timestamps are expected in microseconds. Tolerance is in microseconds.
    """
    if timestamps_us is None or not poses or len(timestamps_us) == 0: # Added len check
        logging.warning(f"No pose data available to query for timestamp {query_ts_us}.")
        return None

    n_poses = len(timestamps_us)

    idx = bisect_left(timestamps_us, query_ts_us)

    if idx < n_poses and abs(timestamps_us[idx] - query_ts_us) <= tolerance_us:
        logging.debug(f"Exact match found for timestamp {query_ts_us} at index {idx}.")
        return poses[idx]
    if idx > 0 and abs(timestamps_us[idx - 1] - query_ts_us) <= tolerance_us:
         logging.debug(f"Exact match found for timestamp {query_ts_us} at index {idx-1}.")
         return poses[idx-1]

    if idx == 0:
        logging.warning(f"Query timestamp {query_ts_us} is before the first pose timestamp {timestamps_us[0]}. Cannot interpolate.")
        return None

    if idx == n_poses:
        logging.warning(f"Query timestamp {query_ts_us} is after the last pose timestamp {timestamps_us[-1]}. Cannot interpolate.")
        return None

    t0_us = timestamps_us[idx - 1]
    t1_us = timestamps_us[idx]
    pose0 = poses[idx - 1]
    pose1 = poses[idx]

    if t1_us <= t0_us:
        logging.error(f"Invalid timestamp order for interpolation: t0_us={t0_us}, t1_us={t1_us}. Check sorting.")
        return None

    alpha = (query_ts_us - t0_us) / (t1_us - t0_us)
    logging.debug(f"Interpolating for {query_ts_us} between {t0_us} and {t1_us} with alpha={alpha:.4f}")

    try:
        R0 = R.from_matrix(pose0[:3, :3])
        R1 = R.from_matrix(pose1[:3, :3])
        T0 = pose0[:3, 3]
        T1 = pose1[:3, 3]
    except ValueError as e:
         logging.error(f"Failed to create Rotation object from matrices at {t0_us} or {t1_us}. Invalid rotation? Error: {e}")
         return None

    try:
        # Slerp uses the actual timestamp values for interpolation points
        slerp_interpolator = Slerp([t0_us, t1_us], R.concatenate([R0, R1]))
        R_interp = slerp_interpolator([query_ts_us])[0]
    except Exception as e:
        logging.error(f"SLERP failed between timestamps {t0_us} and {t1_us}: {e}")
        return None

    T_interp = T0 + alpha * (T1 - T0)

    pose_interp = np.eye(4)
    pose_interp[:3, :3] = R_interp.as_matrix()
    pose_interp[:3, 3] = T_interp
    return pose_interp

def get_init_poses(init_pose_path, camera_name, query_img_list_path_str):
    pose_data = load_and_prepare_poses(init_pose_path, camera_name)
    if pose_data is None or pose_data[0] is None: # Check if load_and_prepare_poses returned (None,None)
        logging.error(f"Failed to load pose data for {camera_name}. Exiting.")
        exit(1) # Ensure exit if critical data is missing
    sorted_timestamps_us, sorted_poses = pose_data # Timestamps are in microseconds

    query_poses = []
    not_found_count = 0

    try:
        with open(query_img_list_path_str, 'r') as f: # Use renamed arg
            query_image_names = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logging.error(f"Query image list file not found: {query_img_list_path_str}")
        exit(1) # Ensure exit

    logging.info(f"\n--- Looking up poses for {len(query_image_names)} query images ---")
    start_lookup_time = time.time()

    for image_name in query_image_names:
        try:
            base_name = os.path.splitext(image_name)[0]
            query_ts_us = int(base_name) # Timestamp from filename is microseconds

            interpolated_pose = get_pose_for_timestamp(
                query_ts_us, # Pass microsecond timestamp
                sorted_timestamps_us, # Pass sorted microsecond timestamps
                sorted_poses
                # Default tolerance_us=1000 (1ms) from get_pose_for_timestamp
            )

            if interpolated_pose is not None:
                query_poses.append(interpolated_pose)
                logging.debug(f"Found/Interpolated pose for {image_name} (ts_us: {query_ts_us})")
            else:
                logging.warning(f"Could not determine pose for {image_name} (ts_us: {query_ts_us}).")
                not_found_count += 1
        except ValueError:
            logging.warning(f"Could not parse timestamp from query image name: {image_name}. Skipping.")
            not_found_count += 1
        except Exception as e:
            logging.error(f"Error processing query image {image_name}: {e}", exc_info=True)
            not_found_count += 1

    lookup_duration = time.time() - start_lookup_time
    logging.info(f"--- Pose lookup finished in {lookup_duration:.2f} seconds ---")
    logging.info(f"Successfully found/interpolated poses for {len(query_poses)} images.")
    if not_found_count > 0:
        logging.warning(f"Could not determine poses for {not_found_count} images.")
    
    if not query_poses and query_image_names: # If list was not empty but no poses found
        logging.error(f"No initial poses could be determined for any query image in {camera_name}. This is critical.")
        # Depending on pipeline, might need to exit here if RENDER_POSES is essential
        # exit(1) 
    
    return query_poses

def skew_symmetric(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def se3_to_SE3(xi):
    omega = xi[:3]
    v = xi[3:]
    theta = np.linalg.norm(omega)
    if np.isclose(theta, 0.0):
        R_mat = np.eye(3)
        t = v
    else:
        # axis = omega / theta # Not directly used in from_rotvec
        omega_skew = skew_symmetric(omega) # Skew of omega (axis * theta)
        # R_mat = R.from_rotvec(omega).as_matrix() # Correct way
        # Using Rodrigues' formula for R (from_rotvec essentially does this)
        # R = I + sin(theta)/theta * omega_skew + (1-cos(theta))/theta^2 * omega_skew @ omega_skew
        # Simpler: use scipy
        R_mat = R.from_rotvec(omega).as_matrix()

        # For V matrix (mapping se(3) translation to SE(3) translation)
        # V = I + (1-cos(theta))/theta^2 * omega_skew + (theta-sin(theta))/theta^3 * omega_skew @ omega_skew
        # omega_skew is skew of (axis * theta)
        V = np.eye(3) + \
            ((1 - np.cos(theta)) / (theta**2)) * omega_skew + \
            ((theta - np.sin(theta)) / (theta**3)) * (omega_skew @ omega_skew)
        t = V @ v
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T

def SE3_to_se3(T_SE3):
    R_mat = T_SE3[:3, :3]
    t_vec = T_SE3[:3, 3]
    r_obj = R.from_matrix(R_mat)
    omega = r_obj.as_rotvec() 
    theta = np.linalg.norm(omega)

    if np.isclose(theta, 0.0):
        V_inv = np.eye(3)
    else:
        omega_skew = skew_symmetric(omega) # Skew of omega (axis*theta)
        # V_inv = I - 1/2 * omega_skew + (1/theta^2 - cot(theta/2)/(2*theta)) * omega_skew^2
        # cot(x) = 1/tan(x)
        # term_coeff = (1.0 / theta**2) - (1.0 / (2.0 * theta * np.tan(theta / 2.0)))
        # This formula for V_inv using unnormalized omega_skew is common.
        # Let's use the one based on normalized axis from a reference (e.g. Barfoot R_so3.cpp logmap)
        # V_inv = I - 0.5 * omega_skew + (1 - theta*cot(theta/2)/2) / theta^2 * (omega_skew @ omega_skew)
        # where cot(theta/2) = cos(theta/2)/sin(theta/2)
        
        # Simplified from common robotics texts:
        # V_inv = I - (1/2) * omega_skew + ( (1/theta**2) - ( (1+cos(theta)) / (2*theta*sin(theta)) ) ) * omega_skew**2
        # Handle sin(theta) close to zero if theta is near pi for the above formula.
        # The provided code's version seems based on normalized axis, let's stick to its structure but ensure omega is used correctly.
        # The original code used `omega_skew_axis` which implies `axis` was `omega/theta`.
        # omega_skew = skew_symmetric(omega)
        
        # Using the formula from the original code, which seems to be:
        # V_inv = I - (1/2) * skew(omega) + (1/theta^2 - cot(theta/2)/(2*theta)) * skew(omega)^2
        # Let's use the axis-angle based one which is often more stable or clearer.
        # omega = axis * theta
        # omega_skew_axis = skew_symmetric(omega / theta)
        # V_inv = np.eye(3) - (theta / 2.0) * omega_skew_axis + \
        #        (1.0 - (theta / 2.0) * (np.cos(theta / 2.0) / np.sin(theta / 2.0))) * (omega_skew_axis @ omega_skew_axis)

        # The original code's V_inv was:
        half_theta = theta / 2.0
        if np.isclose(np.sin(half_theta), 0.0): # Should be caught by theta approx 0
             V_inv = np.eye(3) - 0.5 * skew_symmetric(omega)
        else:
            term_cot_coeff = (theta / 2.0) * (np.cos(half_theta) / np.sin(half_theta))
            # omega_skew_axis was skew_symmetric(axis), where axis = omega/theta.
            # So (omega_skew_axis @ omega_skew_axis) is (skew(omega/theta) @ skew(omega/theta))
            # We can write skew(omega) = theta * skew(omega/theta)
            # So skew(omega/theta) = (1/theta) * skew(omega)
            # (skew(omega/theta))^2 = (1/theta^2) * skew(omega)^2
            # The formula becomes: I - (1/2)skew(omega) + (1 - term_cot_coeff)/theta^2 * skew(omega)^2
            # This looks like the (1/theta^2 - cot(theta/2)/(2*theta)) type term.

            # Let's use a standard V_inv formula:
            # V_inv = I - 0.5*skew(omega) + (1/theta^2 - (1+cos(theta))/(2*theta*sin(theta))) * skew(omega)^2 for sin(theta)!=0
            # If sin(theta) is zero (theta = k*pi), different forms are used.
            # If theta = pi, V_inv = I - 0.5*skew(omega) + (1/pi^2)*skew(omega)^2
            
            # Reverting to previous V_inv which seemed okay, just ensuring omega (not axis) is used where appropriate.
            # The previous version used omega_skew_axis = skew_symmetric(axis)
            # and terms like (theta/2)*omega_skew_axis and (omega_skew_axis @ omega_skew_axis)
            # This is correct application of axis-angle version if axis = omega / theta.
            axis = omega / theta
            omega_skew_axis = skew_symmetric(axis)
            V_inv = np.eye(3) - (theta / 2.0) * omega_skew_axis + \
                    (1.0 - term_cot_coeff) * (omega_skew_axis @ omega_skew_axis)
            
    v_se3 = V_inv @ t_vec
    xi = np.concatenate((omega, v_se3))
    return xi

def visualize_2d_points_on_image(
    original_image_path: str,
    points_2d_to_draw: np.ndarray,
    output_path: str,
    point_color: tuple = (0, 0, 255), # BGR: Red
    point_size: int = 3,
    label: str = "Points"
):
    """
    Loads an image and draws the provided 2D points on it.
    """
    logging.debug(f"Visualizing {points_2d_to_draw.shape[0]} {label} on {Path(original_image_path).name} -> {Path(output_path).name}")

    try:
        image_original = cv2.imread(original_image_path)
        if image_original is None:
            logging.error(f"Could not load original image for visualization: {original_image_path}")
            return

        output_image_viz = image_original.copy()
        
        if points_2d_to_draw is None or points_2d_to_draw.shape[0] == 0:
            logging.warning(f"No {label} provided to draw for {original_image_path}.")
        else:
            points_2d_int = points_2d_to_draw.reshape(-1, 2).astype(int)
            
            h_img, w_img = output_image_viz.shape[:2]
            
            drawn_count = 0
            for i in range(points_2d_int.shape[0]):
                pt_x, pt_y = points_2d_int[i, 0], points_2d_int[i, 1]
                if 0 <= pt_x < w_img and 0 <= pt_y < h_img:
                    cv2.circle(output_image_viz, (pt_x, pt_y), point_size, point_color, thickness=-1)
                    drawn_count +=1
            logging.debug(f"Drew {drawn_count}/{points_2d_int.shape[0]} {label} within image bounds.")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save_success = cv2.imwrite(output_path, output_image_viz)
        if not save_success:
            logging.error(f"Failed to save {label} visualization to {output_path}")
        else:
            logging.info(f"Saved {label} visualization to {output_path}")

    except Exception as e:
        logging.error(f"Error during {label} visualization for {original_image_path}: {e}", exc_info=True)

# --- Helper: Generate Target Distorted Points ---
def generate_target_distorted_points(
    p2d_undistorted_pixels: np.ndarray, # Ideal points in undistorted pixel frame (of K_ideal_fixed)
    K_sensor_current: np.ndarray,       # Current K_sensor being evaluated/optimized
    D_sensor_current: np.ndarray,       # Current D_sensor being evaluated/optimized
    model_type: str,
    K_ideal_fixed_for_normalization: np.ndarray, # The K corresponding to p2d_undistorted_pixels input
    img_width: int,
    img_height: int
):
    """
    Distorts ideal undistorted 2D points (normalized using K_ideal_fixed_for_normalization)
    to get target points on a sensor image defined by K_sensor_current and D_sensor_current.
    """
    if p2d_undistorted_pixels.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)

    p2d_target_dist_pixels_out = np.zeros_like(p2d_undistorted_pixels, dtype=np.float64)

    # 1. Normalize p2d_undistorted_pixels using K_ideal_fixed_for_normalization
    fx_ideal_norm = K_ideal_fixed_for_normalization[0, 0]
    fy_ideal_norm = K_ideal_fixed_for_normalization[1, 1]
    cx_ideal_norm = K_ideal_fixed_for_normalization[0, 2]
    cy_ideal_norm = K_ideal_fixed_for_normalization[1, 2]

    p2d_normalized_coords = np.zeros_like(p2d_undistorted_pixels, dtype=np.float64)
    p2d_normalized_coords[:, 0] = (p2d_undistorted_pixels[:, 0] - cx_ideal_norm) / fx_ideal_norm
    p2d_normalized_coords[:, 1] = (p2d_undistorted_pixels[:, 1] - cy_ideal_norm) / fy_ideal_norm
    
    p2d_normalized_reshaped = p2d_normalized_coords.reshape(-1, 1, 2)

    if model_type == "KANNALA_BRANDT":
        D_sensor_kb_prepared = np.zeros(4, dtype=np.float64)
        if D_sensor_current is not None:
            d_flat = D_sensor_current.flatten()
            D_sensor_kb_prepared[:min(4, len(d_flat))] = d_flat[:min(4, len(d_flat))].astype(np.float64)

        # cv2.fisheye.distortPoints takes NORMALIZED points and outputs DISTORTED PIXEL points
        # using the K (K_sensor_current) and D (D_sensor_kb_prepared) passed to it.
        distorted_pixel_pts_cv = cv2.fisheye.distortPoints( # No underscore for the return here
            p2d_normalized_reshaped,                             # Input: Normalized coordinates
            np.ascontiguousarray(K_sensor_current, dtype=np.float64), # K of the target distorted camera
            np.ascontiguousarray(D_sensor_kb_prepared, dtype=np.float64)  # D of the target distorted camera
        )
        p2d_target_dist_pixels_out = distorted_pixel_pts_cv.reshape(-1, 2)

    elif model_type == "PINHOLE":
        # For Pinhole, we project pseudo-3D points (normalized coords with Z=1)
        # P3D_ideal_cam are effectively (X/Z, Y/Z, 1) where X/Z, Y/Z are p2d_normalized_coords
        P3D_in_normalized_frame = np.hstack((p2d_normalized_coords, np.ones((p2d_normalized_coords.shape[0], 1)))).astype(np.float32)
        
        rvec_ident = np.zeros(3, dtype=np.float32)
        tvec_ident = np.zeros(3, dtype=np.float32)
        
        D_sensor_pinhole_prepared = None
        if D_sensor_current is not None:
            D_sensor_pinhole_prepared = D_sensor_current.flatten().astype(np.float64)
            # Optional: Pad/truncate D_sensor_pinhole_prepared if needed for cv2.projectPoints
            # e.g., if expecting 5 params for standard pinhole
            expected_pinhole_d_len = 5
            if len(D_sensor_pinhole_prepared) < expected_pinhole_d_len:
                _temp_D = np.zeros(expected_pinhole_d_len, dtype=np.float64)
                _temp_D[:len(D_sensor_pinhole_prepared)] = D_sensor_pinhole_prepared
                D_sensor_pinhole_prepared = _temp_D
            elif len(D_sensor_pinhole_prepared) > expected_pinhole_d_len:
                 D_sensor_pinhole_prepared = D_sensor_pinhole_prepared[:expected_pinhole_d_len]


        distorted_pixel_pts_cv, _ = cv2.projectPoints(
            P3D_in_normalized_frame, rvec_ident, tvec_ident,
            np.ascontiguousarray(K_sensor_current, dtype=np.float64), # K of the target distorted camera
            D_sensor_pinhole_prepared # D of the target distorted camera
        )
        p2d_target_dist_pixels_out = distorted_pixel_pts_cv.reshape(-1, 2)
    else:
        logging.error(f"Unsupported model_type '{model_type}' in generate_target_distorted_points.")
        # Fallback: if model is unknown, what should target be? Perhaps undistorted points if K_sensor is similar to K_ideal
        # Or, more safely, log error and return empty or raise exception. For now, copy normalized.
        # This part needs careful consideration if other models are genuinely used.
        # If K_sensor_current is used to "un-normalize" p2d_normalized_coords:
        fx_s_curr = K_sensor_current[0,0]; fy_s_curr = K_sensor_current[1,1]
        cx_s_curr = K_sensor_current[0,2]; cy_s_curr = K_sensor_current[1,2]
        p2d_target_dist_pixels_out[:,0] = p2d_normalized_coords[:,0] * fx_s_curr + cx_s_curr
        p2d_target_dist_pixels_out[:,1] = p2d_normalized_coords[:,1] * fy_s_curr + cy_s_curr


    # --- DEBUG VISUALIZATION CALL within generate_target_distorted_points ---
    # This needs to be adapted if you want to debug generate_target_distorted_points itself
    # For now, the debug call is in OptimizationData.__init__ and the main refine_all_parameters_hybrid
    # It's better to keep the debug viz at higher levels that call this, passing necessary image names etc.

    return p2d_target_dist_pixels_out

class OptimizationData:
    def __init__(self,
                 inlier_matches_map_us: dict,
                 ego_interpolator_us,
                 K_initial_ideal_for_opt_data_targets: np.ndarray, # K_sensor for target generation inside THIS OptData
                 D_initial_sensor_for_opt_data_targets: np.ndarray, # D_sensor for target generation inside THIS OptData
                 model_type: str,
                 img_width: int,
                 img_height: int,
                 t_rec_map_us: dict,
                 num_images_passed_to_constructor: int,
                 # This K_ideal is the one defining the plane of the input p2d_undistorted_pixels.
                 # It's fixed globally for the normalization step within generate_target_distorted_points.
                 K_GLOBAL_IDEAL_FOR_UNDIST_NORMALIZATION: np.ndarray,
                 # Debug parameters
                 debug_image_idx_to_name_map: dict = None,
                 debug_query_image_dir: Path = None,
                 debug_output_vis_dir: Path = None,
                 debug_vis_image_indices: list = None):

        self.inlier_matches_orig_undist = inlier_matches_map_us
        self.ego_interpolator = ego_interpolator_us
        
        # K and D used by THIS OptData instance to generate its internal target points.
        # For final LS, these would be K_sensor_best_de, D_sensor_best_de.
        self.K_sensor_for_internal_targets = K_initial_ideal_for_opt_data_targets.astype(np.float64)
        self.D_sensor_for_internal_targets = D_initial_sensor_for_opt_data_targets.copy() if D_initial_sensor_for_opt_data_targets is not None else None
        
        self.model_type = model_type
        self.img_width = img_width
        self.img_height = img_height
        self.t_rec_map = t_rec_map_us

        # Store the length of D parameters this OptData instance was initialized with.
        # This is crucial for compute_residuals to correctly slice the optimization vector X.
        if self.D_sensor_for_internal_targets is not None:
            self.num_initial_d_params = len(self.D_sensor_for_internal_targets.flatten())
        else:
            self.num_initial_d_params = 0
            
        # Store the global K_ideal used for normalizing the input p2d_undistorted_pixels
        self.K_GLOBAL_IDEAL_FOR_UNDIST_NORMALIZATION = K_GLOBAL_IDEAL_FOR_UNDIST_NORMALIZATION.astype(np.float64)


        # Store debug parameters
        self.debug_image_idx_to_name_map = debug_image_idx_to_name_map
        self.debug_query_image_dir = debug_query_image_dir
        self.debug_output_vis_dir = debug_output_vis_dir
        self.debug_vis_image_indices = debug_vis_image_indices if debug_vis_image_indices is not None else []

        self.image_indices = []
        self.all_p2d_distorted_target = [] 
        self.all_P3d = []
        self.image_idx_to_dt_param_idx_map = {}

        current_dt_param_idx = 0
        sorted_input_image_indices_for_optdata = sorted(inlier_matches_map_us.keys())

        if self.debug_output_vis_dir:
            self.debug_output_vis_dir.mkdir(parents=True, exist_ok=True)

        for original_img_idx in sorted_input_image_indices_for_optdata:
            p2d_undistorted_pixels, P3d_world = self.inlier_matches_orig_undist[original_img_idx]

            if p2d_undistorted_pixels.shape[0] == 0:
                continue

            internal_p2d_target_dist_pixels = generate_target_distorted_points(
                p2d_undistorted_pixels=p2d_undistorted_pixels,
                K_sensor_current=self.K_sensor_for_internal_targets, 
                D_sensor_current=self.D_sensor_for_internal_targets, 
                model_type=self.model_type,
                K_ideal_fixed_for_normalization=self.K_GLOBAL_IDEAL_FOR_UNDIST_NORMALIZATION, # Use the global K_ideal here
                img_width=self.img_width,
                img_height=self.img_height
            )
            
            # ... (debug visualization call remains the same, using self.K_sensor_for_internal_targets for label) ...
            if self.debug_output_vis_dir and \
               self.debug_image_idx_to_name_map and \
               self.debug_query_image_dir and \
               original_img_idx in self.debug_vis_image_indices:
                image_filename = self.debug_image_idx_to_name_map.get(original_img_idx)
                if image_filename:
                    original_image_file_path_str = str(self.debug_query_image_dir / image_filename)
                    if Path(original_image_file_path_str).exists():
                        vis_path_str = str(self.debug_output_vis_dir / f"OptData_internal_target_img_{original_img_idx}_{Path(image_filename).stem}.png")
                        logging.info(f"DEBUG VIS (OptData Init): Visualizing internal target for img_idx {original_img_idx} to {vis_path_str}")
                        visualize_2d_points_on_image(
                            original_image_path=original_image_file_path_str,
                            points_2d_to_draw=internal_p2d_target_dist_pixels.copy(),
                            output_path=vis_path_str,
                            label=f"OptData Internal Target (idx {original_img_idx}) using K_sensor:\n{np.round(self.K_sensor_for_internal_targets[:2,:],2)}", # Label with the K used
                            point_color=(0, 255, 255)
                        )
            
            self.image_indices.append(original_img_idx)
            self.all_p2d_distorted_target.append(internal_p2d_target_dist_pixels)
            self.all_P3d.append(P3d_world)
            self.image_idx_to_dt_param_idx_map[original_img_idx] = current_dt_param_idx
            current_dt_param_idx += 1
        
        if self.all_p2d_distorted_target:
            self.all_p2d_distorted_target = np.concatenate(self.all_p2d_distorted_target, axis=0)
            self.all_P3d = np.concatenate(self.all_P3d, axis=0)
            self.total_residuals = self.all_p2d_distorted_target.shape[0] * 2
        else:
            self.all_p2d_distorted_target = np.empty((0,2), dtype=np.float64)
            self.all_P3d = np.empty((0,3), dtype=np.float64)
            self.total_residuals = 0
        
        self.num_opt_images = current_dt_param_idx
        
        if len(self.image_indices) != self.num_opt_images:
            logging.error(f"OptData init: Mismatch len(image_indices)={len(self.image_indices)} vs num_opt_images={self.num_opt_images}.")
        if len(self.image_idx_to_dt_param_idx_map) != self.num_opt_images:
             logging.error(f"OptData init: Mismatch len(dt_param_map)={len(self.image_idx_to_dt_param_idx_map)} vs num_opt_images={self.num_opt_images}.")

def compute_residuals(X_params_vector, opt_data: OptimizationData):
    residuals = np.zeros(opt_data.total_residuals, dtype=np.float64)
    if opt_data.total_residuals == 0:
        return residuals if residuals.shape[0] > 0 else np.array([], dtype=np.float64)

    try:
        # --- Unpack Parameters from X_params_vector (current trial values) ---
        xi_ego_cam_iter = X_params_vector[:6]

        fx_iter = X_params_vector[6]
        fy_iter = X_params_vector[7]
        cx_iter = X_params_vector[8]
        cy_iter = X_params_vector[9]
        K_iter = np.array([[fx_iter, 0, cx_iter], [0, fy_iter, cy_iter], [0, 0, 1]], dtype=np.float64)

        # Use opt_data.num_initial_d_params to know how many D params were configured
        # and thus how many are present in X_params_vector for D.
        num_d_params_in_X = opt_data.num_initial_d_params # Corrected: Use stored length

        idx_start_D = 6 + NUM_K_PARAMS 
        idx_end_D = idx_start_D + num_d_params_in_X # Corrected: Use num_d_params_in_X
        D_iter_flat = X_params_vector[idx_start_D:idx_end_D]

        idx_start_dt = idx_end_D
        all_dt_seconds_iter = X_params_vector[idx_start_dt:]
        # --- End Unpack Parameters ---

        T_ego_cam_iter = se3_to_SE3(xi_ego_cam_iter) 
        current_residual_offset = 0
        point_counter_for_slicing = 0

        for i in range(opt_data.num_opt_images): # Iterate through images this OptData instance knows about
            original_img_idx = opt_data.image_indices[i] # Get the global image index
            
            # Retrieve the pre-generated target points and 3D points for this image
            # The number of matches is implicitly handled by the slicing below.
            num_matches_for_this_image_in_optdata = opt_data.inlier_matches_orig_undist[original_img_idx][0].shape[0]
            
            if num_matches_for_this_image_in_optdata == 0: # Should not happen if OptData built correctly
                continue

            # These are the internal targets generated by OptData using K_sensor_from_DE, D_sensor_from_DE
            p2d_distorted_target_slice = opt_data.all_p2d_distorted_target[point_counter_for_slicing : point_counter_for_slicing + num_matches_for_this_image_in_optdata]
            P3d_inliers_slice = opt_data.all_P3d[point_counter_for_slicing : point_counter_for_slicing + num_matches_for_this_image_in_optdata]
            point_counter_for_slicing += num_matches_for_this_image_in_optdata

            # Get the dt parameter index for this image within the dt part of X_params_vector
            dt_param_idx_for_image = opt_data.image_idx_to_dt_param_idx_map[original_img_idx]
            dt_i_seconds = all_dt_seconds_iter[dt_param_idx_for_image]

            t_rec_i_us = opt_data.t_rec_map[original_img_idx]
            dt_i_us = dt_i_seconds * 1_000_000.0
            t_true_i_us = float(t_rec_i_us) + dt_i_us

            T_map_ego_i = opt_data.ego_interpolator(t_true_i_us)
            if T_map_ego_i is None:
                residuals[current_residual_offset : current_residual_offset + num_matches_for_this_image_in_optdata * 2] = 1e6
                current_residual_offset += num_matches_for_this_image_in_optdata * 2
                continue

            T_map_cam_i = T_map_ego_i @ T_ego_cam_iter
            try:
                T_cam_map_i = np.linalg.inv(T_map_cam_i)
                R_cam_map_i_for_rodrigues = np.ascontiguousarray(T_cam_map_i[:3, :3])
                rvec_iter, _ = cv2.Rodrigues(R_cam_map_i_for_rodrigues)
                rvec_iter = np.ascontiguousarray(rvec_iter)
                tvec_slice = T_cam_map_i[:3, 3]
                tvec_iter = np.ascontiguousarray(tvec_slice.reshape(3, 1))
            except np.linalg.LinAlgError:
                residuals[current_residual_offset : current_residual_offset + num_matches_for_this_image_in_optdata * 2] = 1e6
                current_residual_offset += num_matches_for_this_image_in_optdata * 2
                continue
            
            p_proj_distorted_iter_pixels = np.full((num_matches_for_this_image_in_optdata, 2), 1e6, dtype=np.float64)
            
            # Prepare 3D points for projection functions based on model type
            P3d_for_cv_proj = None
            if opt_data.model_type == "KANNALA_BRANDT":
                P3d_for_cv_proj = P3d_inliers_slice.reshape(-1, 1, 3).astype(np.float32) # Needs to be float32 for cv2.fisheye
            elif opt_data.model_type == "PINHOLE":
                P3d_for_cv_proj = P3d_inliers_slice.astype(np.float32) # Needs to be float32 for cv2.projectPoints
            else: # Should not happen if model_type is validated earlier
                 P3d_for_cv_proj = P3d_inliers_slice.astype(np.float32)


            P3d_h = np.hstack((P3d_inliers_slice, np.ones((num_matches_for_this_image_in_optdata, 1))))
            P_cam_h = (T_cam_map_i @ P3d_h.T).T
            front_of_camera_mask = P_cam_h[:, 2] > 1e-3 # Check Z > 0 in camera frame
            
            if np.any(front_of_camera_mask):
                P3d_to_project_actual = np.ascontiguousarray(P3d_for_cv_proj[front_of_camera_mask])

                projected_points_cv = None
                if P3d_to_project_actual.shape[0] > 0: # Ensure there are points to project
                    if opt_data.model_type == "KANNALA_BRANDT":
                        D_fisheye_iter = np.zeros(4, dtype=np.float64) 
                        if num_d_params_in_X >= 4: # Use num_d_params_in_X from opt_data
                            D_fisheye_iter = D_iter_flat[:4].astype(np.float64)
                        elif num_d_params_in_X > 0 :
                            D_fisheye_iter[:num_d_params_in_X] = D_iter_flat.astype(np.float64)
                        D_fisheye_iter = np.ascontiguousarray(D_fisheye_iter)
                        projected_points_cv, _ = cv2.fisheye.projectPoints(
                            P3d_to_project_actual, rvec_iter, tvec_iter, K_iter, D_fisheye_iter
                        )
                    elif opt_data.model_type == "PINHOLE":
                        D_pinhole_iter = np.array([], dtype=np.float64)
                        if num_d_params_in_X > 0: # Use num_d_params_in_X from opt_data
                            D_pinhole_iter = D_iter_flat.astype(np.float64)
                        # Pad/truncate D_pinhole_iter for OpenCV if needed
                        expected_pinhole_d_len = 5 
                        if len(D_pinhole_iter) < expected_pinhole_d_len and num_d_params_in_X > 0:
                            _temp_D = np.zeros(expected_pinhole_d_len, dtype=np.float64)
                            _temp_D[:len(D_pinhole_iter)] = D_pinhole_iter
                            D_pinhole_iter = _temp_D
                        elif len(D_pinhole_iter) > expected_pinhole_d_len:
                            D_pinhole_iter = D_pinhole_iter[:expected_pinhole_d_len]
                        elif num_d_params_in_X == 0: # No D params being optimized
                             D_pinhole_iter = None # Pass None to projectPoints

                        D_pinhole_iter_arg = np.ascontiguousarray(D_pinhole_iter) if D_pinhole_iter is not None else None
                        projected_points_cv, _ = cv2.projectPoints(
                            P3d_to_project_actual, rvec_iter, tvec_iter, K_iter, D_pinhole_iter_arg
                        )
                
                if projected_points_cv is not None:
                    projected_points_flat = projected_points_cv.reshape(-1, 2)
                    p_proj_distorted_iter_pixels[front_of_camera_mask] = projected_points_flat

            r_i_slice = p2d_distorted_target_slice - p_proj_distorted_iter_pixels
            residuals[current_residual_offset : current_residual_offset + num_matches_for_this_image_in_optdata * 2] = r_i_slice.flatten()
            current_residual_offset += num_matches_for_this_image_in_optdata * 2

    except Exception as e:
        logging.error(f"Error in compute_residuals: {e}\n{traceback.format_exc()}")
        return np.full(opt_data.total_residuals if opt_data.total_residuals > 0 else 1, 1e6, dtype=np.float64)

    if not np.all(np.isfinite(residuals)):
        residuals[~np.isfinite(residuals)] = 1e6
    return residuals

# --- Revised timestamp-only residual and refinement functions ---
def compute_residuals_timestamp_only_revised_for_de(dt_scalar_seconds, # Input: dt for the current image
                                     # Fixed parameters for this evaluation:
                                     fixed_T_ego_cam,
                                     fixed_K_matrix_sensor, # K of the sensor (potentially being optimized)
                                     fixed_D_coeffs_sensor, # D of the sensor (potentially being optimized)
                                     fixed_model_type,
                                     fixed_img_width, 
                                     fixed_img_height,
                                     # Per-image data:
                                     # p2d_undistorted_ideal_kps, # Undistorted keypoints in K_ideal frame
                                     p2d_target_distorted_pixels, # TARGET points for current K_sensor, D_sensor
                                     P3d_world_points,
                                     t_rec_us,
                                     ego_interpolator_us
                                     # K_initial_ideal # K corresponding to p2d_undistorted_ideal_kps
                                     ):
    num_matches = P3d_world_points.shape[0]
    residuals_flat = np.full(num_matches * 2, 1e6, dtype=np.float64) 

    if num_matches == 0:
        return np.array([], dtype=np.float64)

    try:
        dt_value_seconds = dt_scalar_seconds[0] if isinstance(dt_scalar_seconds, np.ndarray) and dt_scalar_seconds.size == 1 else dt_scalar_seconds
        dt_value_us = dt_value_seconds * 1_000_000.0
        t_true_i_us = float(t_rec_us) + dt_value_us

        T_map_ego_i = ego_interpolator_us(t_true_i_us)
        if T_map_ego_i is None: return residuals_flat 

        T_map_cam_i = T_map_ego_i @ fixed_T_ego_cam
        try:
            T_cam_map_i = np.linalg.inv(T_map_cam_i)
            R_cam_map_i_for_rodrigues = np.ascontiguousarray(T_cam_map_i[:3, :3])
            rvec_iter, _ = cv2.Rodrigues(R_cam_map_i_for_rodrigues)
            rvec_iter = np.ascontiguousarray(rvec_iter)
            tvec_slice = T_cam_map_i[:3, 3]
            tvec_iter = np.ascontiguousarray(tvec_slice.reshape(3, 1))
        except np.linalg.LinAlgError: return residuals_flat

        # --- Projection ---
        p_proj_distorted_iter_pixels_current_sensor = np.full((num_matches, 2), 1e6, dtype=np.float64)
        
        P3d_h_world = np.hstack((P3d_world_points, np.ones((num_matches, 1))))
        P_cam_h = (T_cam_map_i @ P3d_h_world.T).T
        front_of_camera_mask = P_cam_h[:, 2] > 1e-3

        if np.any(front_of_camera_mask):
            P3d_world_to_project_actual_non_cont = P3d_world_points[front_of_camera_mask]
            
            P3d_to_project_cv_input = None
            if fixed_model_type == "KANNALA_BRANDT":
                P3d_world_to_project_actual_non_cont_reshaped = P3d_world_to_project_actual_non_cont.reshape(-1,1,3)
                P3d_to_project_cv_input = np.ascontiguousarray(P3d_world_to_project_actual_non_cont_reshaped)
            else: # PINHOLE
                P3d_to_project_cv_input = np.ascontiguousarray(P3d_world_to_project_actual_non_cont)

            projected_points_cv_current_sensor = None
            if fixed_model_type == "KANNALA_BRANDT":
                D_fisheye_iter = np.zeros(4, dtype=np.float64)
                if fixed_D_coeffs_sensor is not None:
                    d_flat = fixed_D_coeffs_sensor.flatten()
                    D_fisheye_iter[:min(4, len(d_flat))] = d_flat[:min(4, len(d_flat))]
                D_fisheye_iter = np.ascontiguousarray(D_fisheye_iter)
                projected_points_cv_current_sensor, _ = cv2.fisheye.projectPoints(
                    P3d_to_project_cv_input, rvec_iter, tvec_iter, 
                    np.ascontiguousarray(fixed_K_matrix_sensor, dtype=np.float64), 
                    D_fisheye_iter
                )
            elif fixed_model_type == "PINHOLE":
                D_pinhole_iter = np.array([], dtype=np.float64)
                if fixed_D_coeffs_sensor is not None:
                    D_pinhole_iter = fixed_D_coeffs_sensor.astype(np.float64).flatten()
                D_pinhole_iter = np.ascontiguousarray(D_pinhole_iter)
                projected_points_cv_current_sensor, _ = cv2.projectPoints(
                    P3d_to_project_cv_input, rvec_iter, tvec_iter, 
                    np.ascontiguousarray(fixed_K_matrix_sensor, dtype=np.float64), 
                    D_pinhole_iter
                )
            
            if projected_points_cv_current_sensor is not None:
                projected_points_flat = projected_points_cv_current_sensor.reshape(-1, 2)
                p_proj_distorted_iter_pixels_current_sensor[front_of_camera_mask] = projected_points_flat
        
        # The target is p2d_target_distorted_pixels, which was generated using K_sensor, D_sensor
        r_i = p2d_target_distorted_pixels - p_proj_distorted_iter_pixels_current_sensor
        residuals_flat = r_i.flatten()

    except Exception as e:
        return np.full(num_matches * 2, 1e6, dtype=np.float64)

    if not np.all(np.isfinite(residuals_flat)):
        residuals_flat[~np.isfinite(residuals_flat)] = 1e6
    return residuals_flat

def refine_timestamp_only_revised_for_de(
    # Fixed parameters for this image's dt optimization (from DE trial)
    current_T_ego_cam, current_K_matrix_sensor, current_D_coeffs_sensor, current_model_type,
    current_img_width, current_img_height,
    # Per-image data
    p2d_target_distorted_pixels, # Target points for the current K_sensor, D_sensor
    P3d_world_points,
    t_rec_us, ego_interpolator_us,
    # K_initial_ideal_for_undist_input, # K for the undistorted points
    # Optimization settings
    dt_bounds_seconds=(-0.02, 0.02), # Tighter bounds for speed
    loss_function='linear',          # Linear loss for speed in inner loop
    verbose=0):

    if P3d_world_points.shape[0] == 0:
        return None, -1, "No inliers", np.inf, 0.0 

    x0_dt_seconds = [0.0]
    # Ensure bounds are correctly formatted as ([low], [high])
    bounds_dt_seconds_least_sq = ([dt_bounds_seconds[0]], [dt_bounds_seconds[1]])


    residual_func_partial = functools.partial(
        compute_residuals_timestamp_only_revised_for_de, 
        fixed_T_ego_cam=current_T_ego_cam,
        fixed_K_matrix_sensor=current_K_matrix_sensor,
        fixed_D_coeffs_sensor=current_D_coeffs_sensor,
        fixed_model_type=current_model_type,
        fixed_img_width=current_img_width,
        fixed_img_height=current_img_height,
        p2d_target_distorted_pixels=p2d_target_distorted_pixels,
        P3d_world_points=P3d_world_points,
        t_rec_us=t_rec_us,
        ego_interpolator_us=ego_interpolator_us
        # K_initial_ideal=K_initial_ideal_for_undist_input
    )

    dt_val = None; status_val = -1; message_val = "Optimization not run"
    optimality_val = np.inf; final_cost_val = np.inf

    initial_residuals = residual_func_partial(x0_dt_seconds)
    initial_cost = np.inf
    if initial_residuals is not None and initial_residuals.size > 0:
        initial_cost = 0.5 * np.sum(initial_residuals**2)
    
    if not np.isfinite(initial_cost) or initial_residuals.size == 0 :
        return 0.0, 0, "Bad initial or no residuals", 0.0, initial_cost

    try:
        result = least_squares(
            residual_func_partial, x0_dt_seconds, jac='2-point', bounds=bounds_dt_seconds_least_sq,
            method='trf', ftol=1e-6, xtol=1e-6, gtol=1e-6, # More relaxed for speed
            loss=loss_function, verbose=0, 
            max_nfev=50 # Further reduced for speed
        )
        status_val = result.status; message_val = result.message; optimality_val = result.optimality
        
        # Check if cost improved. If not, or if failed, revert to dt=0.0
        if result.success and result.cost < initial_cost:
            dt_val = result.x[0]
            final_cost_val = result.cost
        else:
            dt_val = 0.0 # Revert if no improvement or failure
            final_cost_val = initial_cost
            if verbose > 1 and result.cost >= initial_cost : logging.debug(f"dt_opt for t_rec={t_rec_us}: cost did not improve ({result.cost:.2e} vs {initial_cost:.2e}), dt=0.0")
            elif verbose > 1 and not result.success : logging.debug(f"dt_opt for t_rec={t_rec_us}: failed, dt=0.0")


    except Exception as e:
        # logging.warning(f"dt_opt CRASHED for t_rec={t_rec_us}: {e}", exc_info=False)
        dt_val = 0.0 # Revert on crash
        final_cost_val = initial_cost
        status_val = -100; message_val = str(e)

    return dt_val, status_val, message_val, optimality_val, final_cost_val

# --- PnP Derived Initial Guess ---
def get_pnp_derived_initial_T_ego_cam(
    successful_pnp_results: list[PnPResult], 
    query_timestamps_rec_us_indexed: dict, 
    ego_interpolator_us, 
    initial_T_ego_cam_guess_fallback: np.ndarray
):
    # (Implementation from thought block - kept same)
    logging.info("Deriving initial T_ego_cam guess from PnP results...")
    T_ego_cam_estimates_se3 = []
    for res in successful_pnp_results:
        if res.initial_T_map_cam is None:
            logging.debug(f"PnP result for {res.query_name} missing T_map_cam. Skipping for T_ego_cam averaging.")
            continue
        
        t_rec_us = query_timestamps_rec_us_indexed.get(res.query_idx)
        if t_rec_us is None:
            logging.warning(f"Missing recorded timestamp for PnP-successful image {res.query_name}. Skipping for T_ego_cam averaging.")
            continue

        T_map_ego_pnp = ego_interpolator_us(float(t_rec_us)) # dt=0 for this initial guess
        if T_map_ego_pnp is None:
            logging.warning(f"Could not interpolate T_map_ego for {res.query_name} (ts={t_rec_us}). Skipping for T_ego_cam averaging.")
            continue
        
        try:
            T_ego_map_pnp = np.linalg.inv(T_map_ego_pnp)
            T_ego_cam_estimate_i = T_ego_map_pnp @ res.initial_T_map_cam
            xi_i = SE3_to_se3(T_ego_cam_estimate_i)
            T_ego_cam_estimates_se3.append(xi_i)
        except Exception as e:
            logging.warning(f"Error processing T_ego_cam for {res.query_name}: {e}. Skipping.")
            continue
            
    if not T_ego_cam_estimates_se3:
        logging.warning("No valid T_ego_cam estimates from PnP. Using fallback initial guess.")
        return np.copy(initial_T_ego_cam_guess_fallback) # Return a copy

    xi_ego_cam_avg = np.mean(np.array(T_ego_cam_estimates_se3), axis=0)
    T_ego_cam_pnp_derived = se3_to_SE3(xi_ego_cam_avg)
    logging.info(f"Derived PnP-based initial T_ego_cam (from {len(T_ego_cam_estimates_se3)} estimates):\n{np.round(T_ego_cam_pnp_derived, 4)}")
    
    if not np.isclose(np.linalg.det(T_ego_cam_pnp_derived[:3,:3]), 1.0):
        logging.warning("PnP-derived T_ego_cam has non-unit determinant. Using fallback.")
        return np.copy(initial_T_ego_cam_guess_fallback) # Return a copy
        
    return T_ego_cam_pnp_derived

def calculate_robust_cost_for_de(residuals, loss_type='cauchy', scale_c=1.0):
    """
    Calculates a robust cost from raw residuals, similar to how scipy.least_squares would.
    The cost is 0.5 * sum(rho_i), where rho_i is the loss for residual f_i.

    Args:
        residuals (np.ndarray): Raw residual values (f_i).
        loss_type (str): Type of robust loss ('linear', 'cauchy', 'huber', etc.).
        scale_c (float): Scale parameter for the robust loss function (like f_scale in least_squares).

    Returns:
        float: The calculated robust cost.
    """
    if residuals is None or residuals.size == 0:
        # If there are no residuals (e.g., no points for an image),
        # the contribution to the total cost should ideally be zero or reflect this.
        # For an objective function being minimized, a large penalty might be if no points can be processed.
        # However, if it's just one image out of many, sum of zero is fine.
        # Let's assume if residuals.size == 0 it means no error from this component.
        return 0.0 

    if loss_type == 'linear':
        return 0.5 * np.sum(residuals**2)
    elif loss_type == 'cauchy':
        # rho(f_i) = C^2 * log(1 + (f_i/C)^2)
        # Cost = 0.5 * sum(rho(f_i))
        scaled_residuals_sq = (residuals / scale_c)**2
        cost = 0.5 * np.sum(scale_c**2 * np.log1p(scaled_residuals_sq)) # log1p(x) = log(1+x)
        return cost
    elif loss_type == 'huber':
        # rho(f_i) = f_i^2 if |f_i| <= C
        #          = 2*C*|f_i| - C^2 if |f_i| > C
        # Cost = 0.5 * sum(rho(f_i))
        abs_residuals = np.abs(residuals)
        is_small_res = abs_residuals <= scale_c
        rho_values = np.zeros_like(residuals)
        rho_values[is_small_res] = residuals[is_small_res]**2
        rho_values[~is_small_res] = 2 * scale_c * abs_residuals[~is_small_res] - scale_c**2
        return 0.5 * np.sum(rho_values)
    # Add other scipy.optimize.least_squares compatible losses if needed:
    # 'soft_l1': rho(f_i) = 2 * C * (sqrt(1 + (f_i/C)^2) - 1)
    # 'arctan': rho(f_i) = C^2 * arctan((f_i/C)^2) -> This is unusual, scipy's is likely different.
    # Check scipy's exact definitions if expanding further.
    else:
        logging.warning(f"Unknown robust loss type '{loss_type}' for DE. Using linear.")
        return 0.5 * np.sum(residuals**2)

# --- Hybrid Refinement Function ---
de_eval_count_global = 0 # Keep this global or make it part of objective_data_dict if preferred

def de_objective_func_global(X_de_trial, objective_data_dict): # X_de_trial: [xi_ego_cam, k_params_sensor, d_params_sensor]
    global de_eval_count_global
    de_eval_count_global += 1

    # Unpack X_de_trial (Global parameters from DE)
    xi_ego_cam_trial = X_de_trial[:6]
    k_params_sensor_trial = X_de_trial[6:6+NUM_K_PARAMS] # NUM_K_PARAMS must be defined
    num_d_params_from_data = objective_data_dict["num_d_params"]
    d_params_sensor_trial = X_de_trial[6+NUM_K_PARAMS : 6+NUM_K_PARAMS+num_d_params_from_data]

    # Get DE robust loss settings from objective_data_dict
    de_robust_loss_type = objective_data_dict.get("de_robust_loss_type", "linear")
    de_robust_loss_scale = objective_data_dict.get("de_robust_loss_scale", 1.0)

    try:
        T_ego_cam_trial = se3_to_SE3(xi_ego_cam_trial) # se3_to_SE3 must be defined
        K_sensor_trial = np.array([[k_params_sensor_trial[0], 0, k_params_sensor_trial[2]],
                                   [0, k_params_sensor_trial[1], k_params_sensor_trial[3]],
                                   [0, 0, 1]])
    except Exception:
        # logging.debug(f"DE Eval {de_eval_count_global}: Invalid se3 or K params. Cost=inf") # Reduce log noise
        return np.inf # Invalid transform or K parameters, penalize heavily

    total_robust_cost_for_de_trial = 0.0
    total_points_contributing = 0
    inner_dt_optim_failed_count = 0

    # Iterate through each image that has PnP inliers
    for img_idx in objective_data_dict["query_indices_for_opt"]:
        # P3d_world_img are the corresponding 3D world points
        P3d_world_img = objective_data_dict["inlier_matches_map_undistorted_ideal"][img_idx][1]

        # --- MODIFICATION: Retrieve PRE-CALCULATED target distorted points ---
        # These targets were generated using the INITIAL K_sensor and D_sensor
        # and represent the "ground truth" distorted points on the sensor for the given ideal keypoints.
        p2d_target_distorted_on_sensor_fixed_img = objective_data_dict["precalculated_target_distorted_points"][img_idx]
        # --- END MODIFICATION ---

        if P3d_world_img.shape[0] == 0 or p2d_target_distorted_on_sensor_fixed_img.shape[0] == 0:
            # This case implies the pre-calculation of targets might have failed for this image,
            # or there were no 3D points.
            # If p2d_target_distorted_on_sensor_fixed_img is empty, it implies no valid targets to match.
            # Add a penalty or skip. For now, if either is empty, this image contributes nothing or a penalty.
            # If P3d_world_img.shape[0] > 0 but target is empty, it's an issue.
            if P3d_world_img.shape[0] > 0 and p2d_target_distorted_on_sensor_fixed_img.shape[0] == 0:
                 penalty_per_point_val = 1e12
                 if de_robust_loss_type == 'cauchy': penalty_per_point_val = 0.5 * de_robust_loss_scale**2 * np.log1p( (1e6/de_robust_loss_scale)**2 )
                 total_robust_cost_for_de_trial += penalty_per_point_val * P3d_world_img.shape[0] * 2
                 total_points_contributing += P3d_world_img.shape[0]
            continue


        t_rec_us_img = objective_data_dict["query_timestamps_rec_us"][img_idx]

        # --- Inner Loop: Optimize dt for the current T_ego_cam_trial, K_sensor_trial, D_sensor_trial ---
        # The compute_residuals_timestamp_only_revised_for_de function will project P3d_world_img
        # using the *trial* T_ego_cam, K_sensor, D_sensor and the *optimized dt* for this image,
        # and compare against the p2d_target_distorted_on_sensor_fixed_img.
        
        # dt_optimized_s, status_dt_opt, _, _, _ = refine_timestamp_only_revised_for_de(
        #     current_T_ego_cam=T_ego_cam_trial,
        #     current_K_matrix_sensor=K_sensor_trial,
        #     current_D_coeffs_sensor=d_params_sensor_trial,
        #     current_model_type=objective_data_dict["model_type"],
        #     current_img_width=objective_data_dict["img_width"],
        #     current_img_height=objective_data_dict["img_height"],
        #     p2d_target_distorted_pixels=p2d_target_distorted_on_sensor_fixed_img, # Fixed target points
        #     P3d_world_points=P3d_world_img,
        #     t_rec_us=t_rec_us_img,
        #     ego_interpolator_us=objective_data_dict["ego_interpolator_us"],
        #     dt_bounds_seconds=objective_data_dict["dt_bounds_seconds_de_inner_opt"],
        #     loss_function='linear', # Inner LS for dt always linear for speed
        #     verbose=0 
        # )

        # current_dt_for_residuals = 0.0 # Default if dt opt fails
        # if dt_optimized_s is not None and status_dt_opt >= 0 :
        #     current_dt_for_residuals = dt_optimized_s
        # else:
        #     inner_dt_optim_failed_count += 1

        current_dt_for_residuals = 0.0
            
        # Re-evaluate residuals with the (potentially optimized) dt to get raw residuals
        # for the outer DE robust cost calculation.
        raw_residuals_for_image = compute_residuals_timestamp_only_revised_for_de(
            dt_scalar_seconds=current_dt_for_residuals, 
            fixed_T_ego_cam=T_ego_cam_trial,
            fixed_K_matrix_sensor=K_sensor_trial,
            fixed_D_coeffs_sensor=d_params_sensor_trial,
            fixed_model_type=objective_data_dict["model_type"],
            fixed_img_width=objective_data_dict["img_width"],
            fixed_img_height=objective_data_dict["img_height"],
            p2d_target_distorted_pixels=p2d_target_distorted_on_sensor_fixed_img,
            P3d_world_points=P3d_world_img,
            t_rec_us=t_rec_us_img,
            ego_interpolator_us=objective_data_dict["ego_interpolator_us"]
        )

        # Apply robust loss to the raw residuals for the outer DE objective
        if raw_residuals_for_image.size > 0 and np.all(np.isfinite(raw_residuals_for_image)):
            cost_this_image_robust = calculate_robust_cost_for_de(
                raw_residuals_for_image,
                loss_type=de_robust_loss_type,
                scale_c=de_robust_loss_scale
            )
            total_robust_cost_for_de_trial += cost_this_image_robust
            total_points_contributing += P3d_world_img.shape[0] 
        else:
            penalty_per_point_val = 1e12 
            if de_robust_loss_type == 'cauchy': penalty_per_point_val = 0.5 * de_robust_loss_scale**2 * np.log1p( (1e6/de_robust_loss_scale)**2 )
            elif de_robust_loss_type == 'huber': penalty_per_point_val = 0.5 * (2 * de_robust_loss_scale * 1e6 - de_robust_loss_scale**2)
            total_robust_cost_for_de_trial += penalty_per_point_val * P3d_world_img.shape[0] * 2 
            total_points_contributing += P3d_world_img.shape[0]

    if total_points_contributing == 0: 
        return np.inf

    avg_robust_cost = total_robust_cost_for_de_trial / (total_points_contributing * 2.0) if total_points_contributing > 0 else np.inf

    log_freq = objective_data_dict.get('de_popsize_factor_for_log', 15) * 20 
    if de_eval_count_global % log_freq == 0:
        xi_str = ", ".join([f"{x:.3f}" for x in X_de_trial[:3]]) 
        k_str = ", ".join([f"{x:.2f}" for x in X_de_trial[6:8]]) 
        dt_fails_str = f", dt_fails={inner_dt_optim_failed_count}" if inner_dt_optim_failed_count > 0 else ""
        logging.debug(f"DE Eval {de_eval_count_global}: AvgRobCost={avg_robust_cost:.4e} (Loss: {de_robust_loss_type}, Scale: {de_robust_loss_scale}) for xi_rot=[{xi_str}], K_f=[{k_str}]{dt_fails_str}")

    return avg_robust_cost

def de_objective_func_stage1_extrinsics_only(xi_ego_cam_trial, objective_data_dict_stage1):
    """
    Objective function for DE Stage 1: Optimize T_ego_cam (extrinsics) only.
    Intrinsics (K_sensor, D_sensor) are taken as fixed (initial/trusted values from config).
    The target points are pre-calculated based on these initial fixed intrinsics.
    Time offsets (dt) are assumed to be 0.
    """
    # Fixed K and D for projection in this stage (initial sensor config values)
    K_sensor_for_projection_s1 = objective_data_dict_stage1["K_sensor_for_projection_in_stage1"]
    D_sensor_for_projection_s1 = objective_data_dict_stage1["D_sensor_for_projection_in_stage1"]
    model_type = objective_data_dict_stage1["model_type"]
    img_width = objective_data_dict_stage1["img_width"]
    img_height = objective_data_dict_stage1["img_height"]

    # Global fixed target points (pre-calculated using initial sensor config)
    global_fixed_targets_map = objective_data_dict_stage1["precalculated_target_distorted_points_map_GLOBAL"]

    # Robust loss settings
    de_robust_loss_type = objective_data_dict_stage1.get("de_robust_loss_type", "linear")
    de_robust_loss_scale = objective_data_dict_stage1.get("de_robust_loss_scale", 1.0)

    try:
        T_ego_cam_trial = se3_to_SE3(xi_ego_cam_trial) # Extrinsics being optimized
    except Exception:
        return np.inf

    total_robust_cost_stage1 = 0.0
    total_points_contributing_stage1 = 0

    for img_idx in objective_data_dict_stage1["query_indices_for_opt"]:
        # P3d_world_img are the 3D points for this image
        _, P3d_world_img = objective_data_dict_stage1["inlier_matches_map_undistorted_ideal"][img_idx]
        
        # Fetch the globally fixed target for this image
        p2d_fixed_target_distorted_img = global_fixed_targets_map.get(img_idx)

        if P3d_world_img.shape[0] == 0 or p2d_fixed_target_distorted_img is None or p2d_fixed_target_distorted_img.shape[0] == 0:
            # If P3d points exist but target is missing/empty, penalize.
            # (This case should ideally be prevented by ensuring global_fixed_targets_map is complete for query_indices_for_opt)
            if P3d_world_img.shape[0] > 0:
                penalty_per_point = 1e12 
                if de_robust_loss_type == 'cauchy': penalty_per_point = 0.5 * de_robust_loss_scale**2 * np.log1p( (1e6/de_robust_loss_scale)**2 )
                total_robust_cost_stage1 += penalty_per_point * P3d_world_img.shape[0] * 2 
                total_points_contributing_stage1 += P3d_world_img.shape[0]
            continue
        
        t_rec_us_img = objective_data_dict_stage1["query_timestamps_rec_us"][img_idx]
            
        # Compute raw reprojection residuals using:
        # - TRIAL T_ego_cam_trial
        # - FIXED K_sensor_for_projection_s1, D_sensor_for_projection_s1 (from initial config)
        # - dt = 0
        # - Against the p2d_fixed_target_distorted_img
        raw_residuals_for_image = compute_residuals_timestamp_only_revised_for_de(
            dt_scalar_seconds=0.0, 
            fixed_T_ego_cam=T_ego_cam_trial,                 # Trial extrinsics
            fixed_K_matrix_sensor=K_sensor_for_projection_s1, # Fixed K for projection
            fixed_D_coeffs_sensor=D_sensor_for_projection_s1, # Fixed D for projection
            fixed_model_type=model_type,
            fixed_img_width=img_width,
            fixed_img_height=img_height,
            p2d_target_distorted_pixels=p2d_fixed_target_distorted_img, # Global fixed target
            P3d_world_points=P3d_world_img,
            t_rec_us=t_rec_us_img,
            ego_interpolator_us=objective_data_dict_stage1["ego_interpolator_us"]
        )

        # Apply robust loss
        if raw_residuals_for_image.size > 0 and np.all(np.isfinite(raw_residuals_for_image)):
            cost_this_image_robust = calculate_robust_cost_for_de(
                raw_residuals_for_image,
                loss_type=de_robust_loss_type,
                scale_c=de_robust_loss_scale
            )
            total_robust_cost_stage1 += cost_this_image_robust
            total_points_contributing_stage1 += P3d_world_img.shape[0]
        else: # Penalize if residuals are bad
            penalty_per_point = 1e12
            if de_robust_loss_type == 'cauchy': penalty_per_point = 0.5 * de_robust_loss_scale**2 * np.log1p((1e6/de_robust_loss_scale)**2)
            total_robust_cost_stage1 += penalty_per_point * P3d_world_img.shape[0] * 2
            total_points_contributing_stage1 += P3d_world_img.shape[0]


    if total_points_contributing_stage1 == 0:
        return np.inf

    avg_robust_cost = total_robust_cost_stage1 / (total_points_contributing_stage1 * 2.0) if total_points_contributing_stage1 > 0 else np.inf
    return avg_robust_cost

def de_objective_func_stage2_intrinsics_only(intrinsics_params_trial, objective_data_dict_stage2):
    """
    Objective function for DE Stage 2: Optimize K_sensor, D_sensor (intrinsics) only.
    T_ego_cam (extrinsics) is taken as fixed (from Stage 1 result).
    The target points are pre-calculated based on the initial sensor config.
    Time offsets (dt) are assumed to be 0.
    """
    # intrinsics_params_trial = [k_params_sensor_trial (4), d_params_sensor_trial (N_d)]
    k_params_sensor_trial = intrinsics_params_trial[:NUM_K_PARAMS]
    num_d_params_from_data = objective_data_dict_stage2["num_d_params"]
    d_params_sensor_trial = intrinsics_params_trial[NUM_K_PARAMS : NUM_K_PARAMS + num_d_params_from_data]

    # Fixed T_ego_cam for this stage
    T_ego_cam_fixed = objective_data_dict_stage2["T_ego_cam_fixed"] # From Stage 1
    model_type = objective_data_dict_stage2["model_type"]
    img_width = objective_data_dict_stage2["img_width"]
    img_height = objective_data_dict_stage2["img_height"]

    # Global fixed target points (pre-calculated using initial sensor config)
    global_fixed_targets_map = objective_data_dict_stage2["precalculated_target_distorted_points_map_GLOBAL"]

    # Robust loss settings
    de_robust_loss_type = objective_data_dict_stage2.get("de_robust_loss_type", "linear")
    de_robust_loss_scale = objective_data_dict_stage2.get("de_robust_loss_scale", 1.0)

    try:
        K_sensor_trial = np.array([[k_params_sensor_trial[0], 0, k_params_sensor_trial[2]],
                                   [0, k_params_sensor_trial[1], k_params_sensor_trial[3]],
                                   [0, 0, 1]])
    except Exception:
        return np.inf # Invalid K parameters

    total_robust_cost_stage2 = 0.0
    total_points_contributing_stage2 = 0

    for img_idx in objective_data_dict_stage2["query_indices_for_opt"]:
        # P3d_world_img are the 3D points for this image
        _, P3d_world_img = objective_data_dict_stage2["inlier_matches_map_undistorted_ideal"][img_idx]
        
        # Fetch the globally fixed target for this image
        p2d_fixed_target_distorted_img = global_fixed_targets_map.get(img_idx)
        
        if P3d_world_img.shape[0] == 0 or p2d_fixed_target_distorted_img is None or p2d_fixed_target_distorted_img.shape[0] == 0:
            if P3d_world_img.shape[0] > 0: # Penalize if 3D points exist but target missing
                penalty_per_point = 1e12
                if de_robust_loss_type == 'cauchy': penalty_per_point = 0.5 * de_robust_loss_scale**2 * np.log1p((1e6/de_robust_loss_scale)**2)
                total_robust_cost_stage2 += penalty_per_point * P3d_world_img.shape[0] * 2
                total_points_contributing_stage2 += P3d_world_img.shape[0]
            continue
            
        t_rec_us_img = objective_data_dict_stage2["query_timestamps_rec_us"][img_idx]
            
        # Compute raw reprojection residuals using:
        # - FIXED T_ego_cam_fixed (from Stage 1)
        # - TRIAL K_sensor_trial, d_params_sensor_trial
        # - dt = 0
        # - Against the p2d_fixed_target_distorted_img
        raw_residuals_for_image = compute_residuals_timestamp_only_revised_for_de(
            dt_scalar_seconds=0.0, 
            fixed_T_ego_cam=T_ego_cam_fixed,                 # Fixed T from Stage 1
            fixed_K_matrix_sensor=K_sensor_trial,           # Trial K for projection
            fixed_D_coeffs_sensor=d_params_sensor_trial,    # Trial D for projection
            fixed_model_type=model_type,
            fixed_img_width=img_width,
            fixed_img_height=img_height,
            p2d_target_distorted_pixels=p2d_fixed_target_distorted_img, # Global fixed target
            P3d_world_points=P3d_world_img,
            t_rec_us=t_rec_us_img,
            ego_interpolator_us=objective_data_dict_stage2["ego_interpolator_us"]
        )

        # Apply robust loss
        if raw_residuals_for_image.size > 0 and np.all(np.isfinite(raw_residuals_for_image)):
            cost_this_image_robust = calculate_robust_cost_for_de(
                raw_residuals_for_image,
                loss_type=de_robust_loss_type,
                scale_c=de_robust_loss_scale
            )
            total_robust_cost_stage2 += cost_this_image_robust
            total_points_contributing_stage2 += P3d_world_img.shape[0]
        else: # Penalize
            penalty_per_point = 1e12
            if de_robust_loss_type == 'cauchy': penalty_per_point = 0.5 * de_robust_loss_scale**2 * np.log1p((1e6/de_robust_loss_scale)**2)
            total_robust_cost_stage2 += penalty_per_point * P3d_world_img.shape[0] * 2
            total_points_contributing_stage2 += P3d_world_img.shape[0]


    if total_points_contributing_stage2 == 0:
        return np.inf

    avg_robust_cost = total_robust_cost_stage2 / (total_points_contributing_stage2 * 2.0) if total_points_contributing_stage2 > 0 else np.inf
    return avg_robust_cost

def refine_all_parameters_staged(
    # --- Renamed for clarity ---
    initial_T_ego_cam_from_config: np.ndarray,
    pnp_derived_T_ego_cam_guess: np.ndarray, 
    inlier_matches_map_undistorted_ideal: dict,
    ego_timestamps_us: np.ndarray,
    ego_poses: list,
    query_indices_for_opt: list,
    query_timestamps_rec_us: dict,
    K_ideal_plane_definition: np.ndarray, # K defining the ideal undistorted plane of input 2D points
    K_initial_sensor_cfg: np.ndarray,         # Initial K of the physical sensor (from config)
    initial_D_sensor_coeffs_cfg: np.ndarray,  # Initial D of the physical sensor (from config)
    # --- End Renamed ---
    model_type: str,
    img_width: int, img_height: int,
    stage1_de_xi_bounds_abs_offset: np.ndarray = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]),
    stage1_de_popsize_factor: int = 15,
    stage1_de_maxiter: int = 100,
    stage2_de_k_bounds_rel_offset: tuple[float, float] = (0.9, 1.1),
    stage2_de_cxcy_bounds_abs_offset_px: float = 30.0,
    stage2_de_d_bounds_abs_offset: np.ndarray = None,
    stage2_de_popsize_factor: int = 15,
    stage2_de_maxiter: int = 100,
    de_robust_loss_type: str = 'cauchy',
    de_robust_loss_scale: float = 5.0,
    de_workers: int = -1,
    dt_bounds_seconds_final_ls: tuple[float, float] = (-0.05, 0.05),
    loss_function_final_ls: str = 'cauchy',
    final_ls_verbose: int = 1,
    final_ls_xi_bounds_abs_offset: np.ndarray = np.array([0.03, 0.03, 0.03, 0.08, 0.08, 0.08]),
    final_ls_k_bounds_rel_offset: tuple[float, float] = (0.98, 1.02),
    final_ls_cxcy_bounds_abs_offset_px: float = 10.0,
    final_ls_d_bounds_abs_offset_scale: float = 0.05,
    final_ls_d_bounds_abs_offset_const: float = 0.008,
    interpolation_tolerance_us: int = 1,
    intrinsic_bounds_config: dict = None, 
    debug_opt_data_vis_params: dict = None
):
    logging.info("--- Starting STAGED Extrinsics, Intrinsics, and Timestamp Refinement (Fixed Global Target Version) ---")
    global de_eval_count_global
    de_eval_count_global = 0

    num_query_images_for_opt = len(query_indices_for_opt)
    if num_query_images_for_opt == 0:
        logging.error("No query images for staged refinement."); return None, None, None, None, None

    num_d_params = 0
    flat_initial_D_sensor_cfg = np.array([])
    if initial_D_sensor_coeffs_cfg is not None:
        flat_initial_D_sensor_cfg = initial_D_sensor_coeffs_cfg.flatten()
        num_d_params = len(flat_initial_D_sensor_cfg)

    # This is K_sensor from config, used as initial guess for Stage 2 DE & final LS, and for global target gen
    initial_k_params_sensor_cfg = np.array([
        K_initial_sensor_cfg[0, 0], K_initial_sensor_cfg[1, 1],
        K_initial_sensor_cfg[0, 2], K_initial_sensor_cfg[1, 2]
    ])
    
    ego_interpolator_func = functools.partial(get_pose_for_timestamp,
                                            timestamps_us=ego_timestamps_us,
                                            poses=ego_poses,
                                            tolerance_us=interpolation_tolerance_us)

    # --- Pre-calculate GLOBAL fixed target points based on initial sensor config ---
    precalculated_target_distorted_points_map_GLOBAL = {}
    logging.info("Pre-calculating GLOBAL fixed target distorted points for all DE/LS stages...")
    # Debug visualization for these global targets
    global_target_vis_dir = None
    if debug_opt_data_vis_params and debug_opt_data_vis_params.get('output_vis_dir_base'):
        global_target_vis_dir = Path(debug_opt_data_vis_params['output_vis_dir_base']) / "GLOBAL_fixed_targets_debug"
        global_target_vis_dir.mkdir(parents=True, exist_ok=True)

    for img_idx in query_indices_for_opt:
        p2d_undist_ideal_kps_img, _ = inlier_matches_map_undistorted_ideal[img_idx]
        if p2d_undist_ideal_kps_img.shape[0] == 0:
            precalculated_target_distorted_points_map_GLOBAL[img_idx] = np.empty((0,2), dtype=np.float64)
            continue
        
        targets = generate_target_distorted_points(
            p2d_undistorted_pixels=p2d_undist_ideal_kps_img,
            K_sensor_current=K_initial_sensor_cfg,          # Use initial configured sensor K
            D_sensor_current=initial_D_sensor_coeffs_cfg,   # Use initial configured sensor D
            model_type=model_type,
            K_ideal_fixed_for_normalization=K_ideal_plane_definition, # K defining the ideal plane
            img_width=img_width,
            img_height=img_height
        )
        precalculated_target_distorted_points_map_GLOBAL[img_idx] = targets
        
        if global_target_vis_dir and \
           debug_opt_data_vis_params.get('image_idx_to_name_map') and \
           debug_opt_data_vis_params.get('query_image_dir') and \
           img_idx in debug_opt_data_vis_params.get('vis_image_indices_global_targets', []): # New key for these
            
            image_filename = debug_opt_data_vis_params['image_idx_to_name_map'].get(img_idx)
            if image_filename:
                original_image_file_path_str = str(Path(debug_opt_data_vis_params['query_image_dir']) / image_filename)
                if Path(original_image_file_path_str).exists():
                    vis_path_str = str(global_target_vis_dir / f"GLOBAL_target_img_{img_idx}_{Path(image_filename).stem}.png")
                    logging.info(f"DEBUG VIS (Global Target): Visualizing for img_idx {img_idx} to {vis_path_str}")
                    visualize_2d_points_on_image(
                        original_image_path=original_image_file_path_str, points_2d_to_draw=targets.copy(),
                        output_path=vis_path_str, label=f"GLOBAL Fixed Target (idx {img_idx})", point_color=(0, 128, 255)
                    )
    # --- End Pre-calculation ---

    # --- STAGE 1: Optimize T_ego_cam (Extrinsics) using DE ---
    logging.info("\n--- STAGE 1: Optimizing T_ego_cam (Extrinsics) with DE ---")
    T_ego_cam_stage1_center = initial_T_ego_cam_from_config
    # (Optional: use pnp_derived_T_ego_cam_guess to center DE search if valid)
    # ...
    try:
        xi_ego_cam_stage1_initial_center = SE3_to_se3(T_ego_cam_stage1_center)
    except:
        logging.error("Failed to convert T_ego_cam_stage1_center to se3. Using zeros."); xi_ego_cam_stage1_initial_center = np.zeros(6)

    bounds_stage1_xi_low = xi_ego_cam_stage1_initial_center - stage1_de_xi_bounds_abs_offset
    bounds_stage1_xi_high = xi_ego_cam_stage1_initial_center + stage1_de_xi_bounds_abs_offset
    de_bounds_stage1 = list(zip(bounds_stage1_xi_low, bounds_stage1_xi_high))

    objective_data_dict_stage1 = {
        "inlier_matches_map_undistorted_ideal": inlier_matches_map_undistorted_ideal,
        "query_indices_for_opt": query_indices_for_opt,
        "query_timestamps_rec_us": query_timestamps_rec_us,
        "ego_interpolator_us": ego_interpolator_func,
        "model_type": model_type, "img_width": img_width, "img_height": img_height,
        "precalculated_target_distorted_points_map_GLOBAL": precalculated_target_distorted_points_map_GLOBAL,
        "K_sensor_for_projection_in_stage1": K_initial_sensor_cfg.copy(), # K for projection matches target gen K
        "D_sensor_for_projection_in_stage1": initial_D_sensor_coeffs_cfg.copy() if initial_D_sensor_coeffs_cfg is not None else np.array([]),
        "de_robust_loss_type": de_robust_loss_type, "de_robust_loss_scale": de_robust_loss_scale,
        "de_popsize_factor_for_log": stage1_de_popsize_factor
    }
    # ... (DE call for Stage 1 remains same, using de_objective_func_stage1_extrinsics_only)
    de_result_stage1 = differential_evolution(
        de_objective_func_stage1_extrinsics_only, bounds=de_bounds_stage1,
        args=(objective_data_dict_stage1,), strategy='best1bin', maxiter=stage1_de_maxiter,
        popsize=stage1_de_popsize_factor, tol=0.001, polish=False, disp=True,
        workers=de_workers, updating='deferred'
    )
    if not de_result_stage1.success:
        logging.warning(f"Stage 1 DE (Extrinsics) did not converge: {de_result_stage1.message}. Using its best found parameters.")
    xi_ego_cam_from_stage1 = de_result_stage1.x
    T_ego_cam_from_stage1 = se3_to_SE3(xi_ego_cam_from_stage1)
    logging.info(f"Stage 1 DE (Extrinsics) finished. Cost: {de_result_stage1.fun:.6e}")
    logging.info(f"T_ego_cam from Stage 1:\n{np.round(T_ego_cam_from_stage1, 5)}")


    # --- STAGE 2: Optimize K_sensor, D_sensor (Intrinsics) using DE, with T_ego_cam fixed ---
    logging.info("\n--- STAGE 2: Optimizing K_sensor, D_sensor (Intrinsics) with DE (fixed T_ego_cam) ---")
    # Bounds for K (relative to initial K_sensor_cfg)
    k_s2_low = initial_k_params_sensor_cfg * stage2_de_k_bounds_rel_offset[0]
    k_s2_high = initial_k_params_sensor_cfg * stage2_de_k_bounds_rel_offset[1]
    # ... (cx, cy, D bounds setup remains same, using flat_initial_D_sensor_cfg)
    k_s2_low[2] = initial_k_params_sensor_cfg[2] - stage2_de_cxcy_bounds_abs_offset_px
    k_s2_high[2] = initial_k_params_sensor_cfg[2] + stage2_de_cxcy_bounds_abs_offset_px
    k_s2_low[3] = initial_k_params_sensor_cfg[3] - stage2_de_cxcy_bounds_abs_offset_px
    k_s2_high[3] = initial_k_params_sensor_cfg[3] + stage2_de_cxcy_bounds_abs_offset_px
    k_s2_low[2] = max(0, k_s2_low[2]); k_s2_high[2] = min(img_width -1, k_s2_high[2])
    k_s2_low[3] = max(0, k_s2_low[3]); k_s2_high[3] = min(img_height -1, k_s2_high[3])

    d_s2_low, d_s2_high = np.array([]), np.array([])
    if num_d_params > 0:
        d_offset = stage2_de_d_bounds_abs_offset
        if d_offset is None: 
            if model_type == "KANNALA_BRANDT": d_offset = np.array([0.05, 0.03, 0.01, 0.01])[:num_d_params]
            elif model_type == "PINHOLE": d_offset = np.array([0.1, 0.05, 0.005, 0.005, 0.05])[:num_d_params]
            else: d_offset = np.full(num_d_params, 0.02)
        d_offset_corrected = np.resize(d_offset, flat_initial_D_sensor_cfg.shape)
        d_s2_low = flat_initial_D_sensor_cfg - d_offset_corrected
        d_s2_high = flat_initial_D_sensor_cfg + d_offset_corrected

    de_bounds_stage2_list = list(zip(np.concatenate((k_s2_low, d_s2_low)),
                                     np.concatenate((k_s2_high, d_s2_high))))

    objective_data_dict_stage2 = {
        "inlier_matches_map_undistorted_ideal": inlier_matches_map_undistorted_ideal,
        "query_indices_for_opt": query_indices_for_opt,
        "query_timestamps_rec_us": query_timestamps_rec_us,
        "ego_interpolator_us": ego_interpolator_func,
        "model_type": model_type, "img_width": img_width, "img_height": img_height,
        "num_d_params": num_d_params,
        "T_ego_cam_fixed": T_ego_cam_from_stage1.copy(),
        "precalculated_target_distorted_points_map_GLOBAL": precalculated_target_distorted_points_map_GLOBAL,
        "de_robust_loss_type": de_robust_loss_type, "de_robust_loss_scale": de_robust_loss_scale,
        "de_popsize_factor_for_log": stage2_de_popsize_factor
    }
    # ... (DE call for Stage 2 remains same, using de_objective_func_stage2_intrinsics_only)
    de_result_stage2 = differential_evolution(
        de_objective_func_stage2_intrinsics_only, bounds=de_bounds_stage2_list,
        args=(objective_data_dict_stage2,), strategy='best1bin', maxiter=stage2_de_maxiter,
        popsize=stage2_de_popsize_factor, tol=0.001, polish=False, disp=True,
        workers=de_workers, updating='deferred'
    )
    if not de_result_stage2.success:
        logging.warning(f"Stage 2 DE (Intrinsics) did not converge: {de_result_stage2.message}. Using its best found parameters.")
    intrinsics_params_from_stage2 = de_result_stage2.x
    k_params_from_stage2 = intrinsics_params_from_stage2[:NUM_K_PARAMS]
    d_params_from_stage2 = intrinsics_params_from_stage2[NUM_K_PARAMS : NUM_K_PARAMS + num_d_params]
    K_sensor_from_stage2 = np.array([[k_params_from_stage2[0], 0, k_params_from_stage2[2]],
                                     [0, k_params_from_stage2[1], k_params_from_stage2[3]], [0,0,1]])
    D_sensor_from_stage2 = d_params_from_stage2 
    logging.info(f"Stage 2 DE (Intrinsics) finished. Cost: {de_result_stage2.fun:.6e}")
    logging.info(f"K_sensor from Stage 2:\n{np.round(K_sensor_from_stage2, 4)}")
    logging.info(f"D_sensor from Stage 2: {np.round(D_sensor_from_stage2, 6)}")


    # --- STAGE 3: Final Joint Refinement (LS of T_ego_cam, K, D, and all dt_i) ---
    logging.info("\n--- STAGE 3: Final Joint Refinement (T_ego_cam, K, D, dt_i) with Least Squares ---")
    # Initialize dt_i based on T_ego_cam_from_stage1 and K/D_from_stage2
    # For dt calculation, the "target" (p2d_target_dist_final_ls_init) should be based on K_sensor_from_stage2, D_sensor_from_stage2.
    # However, for the final LS itself, OptimizationData will use the GLOBAL fixed target.
    logging.info("Calculating optimal dt_i values based on Stage 1 & 2 results for final LS initialization...")
    dt_init_for_final_ls_map = {}
    for img_idx in query_indices_for_opt:
        p2d_undist_ideal_kps_img, P3d_world_img = inlier_matches_map_undistorted_ideal[img_idx]
        if p2d_undist_ideal_kps_img.shape[0] == 0:
            dt_init_for_final_ls_map[img_idx] = 0.0; continue
        t_rec_us_img = query_timestamps_rec_us[img_idx]
        
        # Generate target for dt optimization using Stage 2 K/D results
        p2d_target_for_dt_opt = generate_target_distorted_points(
            p2d_undist_ideal_kps_img, 
            K_sensor_current=K_sensor_from_stage2,       # K from Stage 2
            D_sensor_current=D_sensor_from_stage2,       # D from Stage 2
            model_type=model_type, 
            K_ideal_fixed_for_normalization=K_ideal_plane_definition, 
            img_width=img_width, img_height=img_height
        )
        if p2d_target_for_dt_opt.shape[0] == 0:
            dt_init_for_final_ls_map[img_idx] = 0.0; continue

        dt_opt_s, _, _, _, _ = refine_timestamp_only_revised_for_de(
            T_ego_cam_from_stage1, K_sensor_from_stage2, D_sensor_from_stage2, model_type,
            img_width, img_height,
            p2d_target_for_dt_opt, # Target based on K/D from Stage 2
            P3d_world_img, t_rec_us_img,
            ego_interpolator_func,
            dt_bounds_seconds=dt_bounds_seconds_final_ls, loss_function='linear' 
        )
        dt_init_for_final_ls_map[img_idx] = dt_opt_s if dt_opt_s is not None else 0.0
    
    # Create OptimizationData for the final LS step.
    # Its internal targets will be the GLOBAL fixed targets.
    opt_data_final_ls = OptimizationData(
        inlier_matches_map_us=inlier_matches_map_undistorted_ideal,
        ego_interpolator_us=ego_interpolator_func,
        K_initial_ideal_for_opt_data_targets=K_initial_sensor_cfg, # For OptData's internal target gen
        D_initial_sensor_for_opt_data_targets=initial_D_sensor_coeffs_cfg, # For OptData's internal target gen
        model_type=model_type,
        img_width=img_width, img_height=img_height,
        t_rec_map_us=query_timestamps_rec_us,
        num_images_passed_to_constructor=len(query_indices_for_opt),
        K_GLOBAL_IDEAL_FOR_UNDIST_NORMALIZATION=K_ideal_plane_definition, # K_ideal plane
        debug_image_idx_to_name_map=debug_opt_data_vis_params.get('image_idx_to_name_map') if debug_opt_data_vis_params else None,
        debug_query_image_dir=debug_opt_data_vis_params.get('query_image_dir') if debug_opt_data_vis_params else None,
        debug_output_vis_dir= (Path(debug_opt_data_vis_params['output_vis_dir_base']) / "LS_OptData_targets_debug_staged") if debug_opt_data_vis_params and debug_opt_data_vis_params.get('output_vis_dir_base') else None,
        debug_vis_image_indices=debug_opt_data_vis_params.get('vis_image_indices_final_ls_optdata', []) if debug_opt_data_vis_params else []
    )
    # ... (LS initial guess x0, bounds, and call remain mostly the same,
    #      initialized with T_ego_cam_from_stage1, K_sensor_from_stage2, D_sensor_from_stage2)
    if opt_data_final_ls.num_opt_images == 0: # Should be caught earlier by query_indices_for_opt check
        logging.error("OptData for Stage 3 LS has zero images. Returning Stage 1 & 2 results.")
        return T_ego_cam_from_stage1, K_sensor_from_stage2, D_sensor_from_stage2, dt_init_for_final_ls_map, None

    xi_ego_cam_ls_init = xi_ego_cam_from_stage1 
    k_params_ls_init = k_params_from_stage2   
    d_params_ls_init = D_sensor_from_stage2   
    
    dt_ls_init_seconds_vec = np.zeros(opt_data_final_ls.num_opt_images)
    for i, img_idx_in_opt_data in enumerate(opt_data_final_ls.image_indices):
        dt_ls_init_seconds_vec[i] = dt_init_for_final_ls_map.get(img_idx_in_opt_data, 0.0)
    
    x0_final_ls = np.concatenate((xi_ego_cam_ls_init, k_params_ls_init, d_params_ls_init, dt_ls_init_seconds_vec))

    # Bounds for final LS (tighter, around Stage 1/2 results)
    bounds_ls_low = list(xi_ego_cam_ls_init - final_ls_xi_bounds_abs_offset)
    bounds_ls_high = list(xi_ego_cam_ls_init + final_ls_xi_bounds_abs_offset)
    # K bounds
    k_ls_b_low = k_params_ls_init * final_ls_k_bounds_rel_offset[0]
    k_ls_b_high = k_params_ls_init * final_ls_k_bounds_rel_offset[1]
    k_ls_b_low[2] = k_params_ls_init[2] - final_ls_cxcy_bounds_abs_offset_px
    k_ls_b_high[2] = k_params_ls_init[2] + final_ls_cxcy_bounds_abs_offset_px
    k_ls_b_low[3] = k_params_ls_init[3] - final_ls_cxcy_bounds_abs_offset_px
    k_ls_b_high[3] = k_params_ls_init[3] + final_ls_cxcy_bounds_abs_offset_px
    k_ls_b_low[2] = max(0,k_ls_b_low[2]); k_ls_b_high[2] = min(img_width-1,k_ls_b_high[2])
    k_ls_b_low[3] = max(0,k_ls_b_low[3]); k_ls_b_high[3] = min(img_height-1,k_ls_b_high[3])
    bounds_ls_low.extend(k_ls_b_low); bounds_ls_high.extend(k_ls_b_high)
    # D bounds
    if num_d_params > 0:
        d_abs = np.abs(d_params_ls_init)
        d_ls_b_low = list(d_params_ls_init - final_ls_d_bounds_abs_offset_scale * d_abs - final_ls_d_bounds_abs_offset_const)
        d_ls_b_high = list(d_params_ls_init + final_ls_d_bounds_abs_offset_scale * d_abs + final_ls_d_bounds_abs_offset_const)
        for i_d in range(len(d_ls_b_low)):
            if d_ls_b_low[i_d] >= d_ls_b_high[i_d]:
                d_mid = (d_ls_b_low[i_d] + d_ls_b_high[i_d])/2.0
                d_ls_b_low[i_d] = d_mid - 1e-4; d_ls_b_high[i_d] = d_mid + 1e-4
        bounds_ls_low.extend(d_ls_b_low); bounds_ls_high.extend(d_ls_b_high)
    # dt bounds
    bounds_ls_low.extend([dt_bounds_seconds_final_ls[0]] * opt_data_final_ls.num_opt_images)
    bounds_ls_high.extend([dt_bounds_seconds_final_ls[1]] * opt_data_final_ls.num_opt_images)
    bounds_final_ls_staged = (bounds_ls_low, bounds_ls_high)

    f_scale_final_ls = de_robust_loss_scale 
    try:
        initial_final_ls_residuals = compute_residuals(x0_final_ls, opt_data_final_ls)
        initial_final_ls_cost_robust = calculate_robust_cost_for_de(
            initial_final_ls_residuals, loss_function_final_ls, f_scale_final_ls
        )
        logging.info(f"Stage 3 LS: Initial cost (robust '{loss_function_final_ls}' eval, scale={f_scale_final_ls}): {initial_final_ls_cost_robust:.4e}")
    except Exception as e_ls_init_cost:
        logging.error(f"Failed to compute initial cost for Stage 3 LS: {e_ls_init_cost}. Returning Stage 1&2 results.", exc_info=True)
        return T_ego_cam_from_stage1, K_sensor_from_stage2, D_sensor_from_stage2, dt_init_for_final_ls_map, None
    
    final_ls_result_obj = None
    try:
        final_ls_result_obj = least_squares(
            compute_residuals, x0_final_ls, jac='2-point', bounds=bounds_final_ls_staged, method='trf',
            ftol=1e-9, xtol=1e-9, gtol=1e-9, loss=loss_function_final_ls, f_scale=f_scale_final_ls,
            verbose=final_ls_verbose, max_nfev=2500 * len(x0_final_ls), args=(opt_data_final_ls,)
        )
        logging.info(f"Stage 3 LS optimization finished. Status: {final_ls_result_obj.status} ({final_ls_result_obj.message})")
        logging.info(f"Stage 3 LS cost: {final_ls_result_obj.cost:.4e}. Optimality: {final_ls_result_obj.optimality:.4e}")
    except Exception as e_ls_final:
        logging.error(f"Stage 3 LS optimization crashed: {e_ls_final}. Returning Stage 1&2 results.", exc_info=True)
        return T_ego_cam_from_stage1, K_sensor_from_stage2, D_sensor_from_stage2, dt_init_for_final_ls_map, None

    # --- Unpack Final Results from Stage 3 LS ---
    T_ego_cam_final_staged = T_ego_cam_from_stage1
    K_sensor_final_staged = K_sensor_from_stage2
    D_sensor_final_staged = D_sensor_from_stage2
    final_refined_delta_t_map_seconds_staged = dt_init_for_final_ls_map

    if final_ls_result_obj and (final_ls_result_obj.success or final_ls_result_obj.cost < initial_final_ls_cost_robust * 0.999): # Allow tiny increase if success flag is good
        X_ls_final = final_ls_result_obj.x
        T_ego_cam_final_staged = se3_to_SE3(X_ls_final[:6])
        k_final_params = X_ls_final[6:6+NUM_K_PARAMS]
        K_sensor_final_staged = np.array([[k_final_params[0], 0, k_final_params[2]],
                                          [0, k_final_params[1], k_final_params[3]], [0,0,1]])
        D_sensor_final_staged = X_ls_final[6+NUM_K_PARAMS : 6+NUM_K_PARAMS+num_d_params]
        dt_final_seconds_flat = X_ls_final[6+NUM_K_PARAMS+num_d_params:]
        for i, img_idx_in_opt_data in enumerate(opt_data_final_ls.image_indices):
            if img_idx_in_opt_data in final_refined_delta_t_map_seconds_staged:
                 final_refined_delta_t_map_seconds_staged[img_idx_in_opt_data] = dt_final_seconds_flat[i]
        logging.info("Successfully refined all parameters using 3-Stage optimization (DE-Extr -> DE-Intr -> LS-Joint).")
    else:
        logging.warning(f"Stage 3 LS did not improve upon Stage 1&2 results (Cost DE/S2: {initial_final_ls_cost_robust:.4e} vs LS: {final_ls_result_obj.cost:.4e if final_ls_result_obj else 'N/A'}) or failed. Returning Stage 1&2 parameters with LS-initialized dt.")

    logging.info(f"Final Refined T_ego_cam (Staged):\n{np.round(T_ego_cam_final_staged, 6)}")
    logging.info(f"Final Refined K_sensor (Staged):\n{np.round(K_sensor_final_staged, 4)}")
    if num_d_params > 0: logging.info(f"Final Refined D_sensor (Staged):\n{np.round(D_sensor_final_staged, 7)}")

    return T_ego_cam_final_staged, K_sensor_final_staged, D_sensor_final_staged, final_refined_delta_t_map_seconds_staged, final_ls_result_obj

def refine_parameters_de_extr_then_ls_joint(
    # --- Inputs for Initial State & Data ---
    initial_T_ego_cam_from_config: np.ndarray,
    pnp_derived_T_ego_cam_guess: np.ndarray,
    inlier_matches_map_undistorted_ideal: dict,
    ego_timestamps_us: np.ndarray,
    ego_poses: list,
    query_indices_for_opt: list,
    query_timestamps_rec_us: dict,
    K_initial_sensor: np.ndarray,   # Initial K_sensor (e.g., from config)
    initial_D_sensor_coeffs: np.ndarray, # Initial D_sensor (e.g., from config)
    model_type: str,
    img_width: int, img_height: int,
    K_ideal_for_undistorted_input_normalization: np.ndarray,

    # --- Stage 1 DE (Extrinsics) Parameters ---
    # Wider bounds for DE exploration of extrinsics
    stage1_de_xi_bounds_abs_offset: np.ndarray = np.array([0.2, 0.2, 0.2, 0.4, 0.4, 0.4]), # Made wider
    stage1_de_popsize_factor: int = 15,
    stage1_de_maxiter: int = 100,
    
    # --- Shared DE Parameters ---
    de_robust_loss_type: str = 'cauchy',
    de_robust_loss_scale: float = 5.0,
    de_workers: int = -1,
    
    # --- Stage 2 (Final Joint LS) Parameters ---
    final_ls_dt_bounds_seconds: tuple[float, float] = (-0.05, 0.05),
    final_ls_loss_function: str = 'cauchy',
    final_ls_verbose: int = 1,
    # `final_ls_xi_bounds_abs_offset` will be same as `stage1_de_xi_bounds_abs_offset`
    # `final_ls_intrinsic_bounds_from_config` will be the cfg_intrinsic_bounds from main
    final_ls_intrinsic_bounds_from_config: dict = None, # To pass cfg_intrinsic_bounds

    # --- Common params ---
    interpolation_tolerance_us: int = 1,
    debug_opt_data_vis_params: dict = None
):
    logging.info("--- Starting 2-STAGE Refinement (V2): DE (Extr) -> LS (Joint, Intrinsics bounded by config) ---")
    global de_eval_count_global
    de_eval_count_global = 0

    # --- Initial Parameter Setup (common) ---
    # (This part remains the same: num_d_params, ego_interpolator_func)
    if not query_indices_for_opt:
        logging.error("No query images for staged refinement."); return None, None, None, None, None

    num_d_params = 0
    flat_initial_D_sensor_coeffs = np.array([])
    if initial_D_sensor_coeffs is not None:
        flat_initial_D_sensor_coeffs = initial_D_sensor_coeffs.flatten()
        num_d_params = len(flat_initial_D_sensor_coeffs)
    
    ego_interpolator_func = functools.partial(get_pose_for_timestamp,
                                            timestamps_us=ego_timestamps_us,
                                            poses=ego_poses,
                                            tolerance_us=interpolation_tolerance_us)

    # --- STAGE 1: Optimize T_ego_cam (Extrinsics) using DE ---
    # (This stage remains the same as in the previous response)
    # It uses `stage1_de_xi_bounds_abs_offset`
    logging.info("\n--- STAGE 1: Optimizing T_ego_cam (Extrinsics) with DE ---")
    logging.info(f"Stage 1 DE will use FIXED K_sensor:\n{K_initial_sensor}")
    logging.info(f"Stage 1 DE will use FIXED D_sensor: {initial_D_sensor_coeffs}")

    T_ego_cam_stage1_center = initial_T_ego_cam_from_config
    # if pnp_derived_T_ego_cam_guess is not None:
    #     try:
    #         if np.isclose(np.linalg.det(pnp_derived_T_ego_cam_guess[:3,:3]), 1.0):
    #             T_ego_cam_stage1_center = pnp_derived_T_ego_cam_guess
    #             logging.info("Using PnP-derived T_ego_cam as center for Stage 1 DE search.")
    #     except: pass
    
    try:
        xi_ego_cam_stage1_initial_center = SE3_to_se3(T_ego_cam_stage1_center)
    except:
        logging.error("Failed to convert T_ego_cam_stage1_center to se3 for DE. Using zeros.")
        xi_ego_cam_stage1_initial_center = np.zeros(6)

    bounds_stage1_xi_low = xi_ego_cam_stage1_initial_center - stage1_de_xi_bounds_abs_offset
    bounds_stage1_xi_high = xi_ego_cam_stage1_initial_center + stage1_de_xi_bounds_abs_offset
    de_bounds_stage1 = list(zip(bounds_stage1_xi_low, bounds_stage1_xi_high))

    objective_data_dict_stage1 = {
        "inlier_matches_map_undistorted_ideal": inlier_matches_map_undistorted_ideal,
        "query_indices_for_opt": query_indices_for_opt,
        "query_timestamps_rec_us": query_timestamps_rec_us,
        "ego_interpolator_us": ego_interpolator_func,
        "K_initial_ideal_for_undistorted_input": K_ideal_for_undistorted_input_normalization,
        "model_type": model_type, "img_width": img_width, "img_height": img_height,
        "K_sensor_fixed": K_initial_sensor.copy(),
        "D_sensor_fixed": initial_D_sensor_coeffs.copy() if initial_D_sensor_coeffs is not None else np.array([]),
        "de_robust_loss_type": de_robust_loss_type,
        "de_robust_loss_scale": de_robust_loss_scale,
        "de_popsize_factor_for_log": stage1_de_popsize_factor
    }

    de_result_stage1 = differential_evolution(
        de_objective_func_stage1_extrinsics_only,
        bounds=de_bounds_stage1,
        args=(objective_data_dict_stage1,),
        strategy='best1bin', maxiter=stage1_de_maxiter, popsize=stage1_de_popsize_factor,
        tol=0.001, polish=False, disp=True, workers=de_workers, updating='deferred'
    )
    
    if not de_result_stage1.success and de_result_stage1.fun > 1e5:
        logging.error(f"Stage 1 DE (Extrinsics) failed badly (cost: {de_result_stage1.fun:.2e}). Aborting.")
        dt_map_fallback = {idx: 0.0 for idx in query_indices_for_opt}
        return initial_T_ego_cam_from_config, K_initial_sensor, initial_D_sensor_coeffs, dt_map_fallback, de_result_stage1

    xi_ego_cam_from_stage1 = de_result_stage1.x
    T_ego_cam_from_stage1 = se3_to_SE3(xi_ego_cam_from_stage1)
    logging.info(f"Stage 1 DE (Extrinsics) finished. Cost: {de_result_stage1.fun:.6e}")
    logging.info(f"T_ego_cam from Stage 1 DE:\n{np.round(T_ego_cam_from_stage1, 5)}")


    # --- STAGE 2: Final Joint Refinement (LS of T_ego_cam, K_sensor, D_sensor, and all dt_i) ---
    logging.info("\n--- STAGE 2: Final Joint Refinement (T_ego_cam, K, D, dt_i) with Least Squares ---")

    # K_sensor and D_sensor for LS are initialized from the *original input values*.
    K_sensor_ls_init = K_initial_sensor.copy()
    D_sensor_ls_init_flat = flat_initial_D_sensor_coeffs.copy()

    # dt_i initialization (same as before)
    logging.info("Calculating initial dt_i values for Stage 2 LS (using Stage 1 T_ego_cam and initial K/D)...")
    dt_init_for_final_ls_map = {}
    # (dt calculation loop is the same, using T_ego_cam_from_stage1, K_sensor_ls_init, D_sensor_ls_init_flat)
    for img_idx in query_indices_for_opt:
        p2d_undist_ideal_kps_img, P3d_world_img = inlier_matches_map_undistorted_ideal[img_idx]
        if p2d_undist_ideal_kps_img.shape[0] == 0:
            dt_init_for_final_ls_map[img_idx] = 0.0; continue
        t_rec_us_img = query_timestamps_rec_us[img_idx]
        p2d_target_dist_final_ls_init = generate_target_distorted_points(
            p2d_undist_ideal_kps_img, K_sensor_ls_init, D_sensor_ls_init_flat, model_type,
            K_ideal_for_undistorted_input_normalization, img_width, img_height
        )
        if p2d_target_dist_final_ls_init.shape[0] == 0:
            dt_init_for_final_ls_map[img_idx] = 0.0; continue
        dt_opt_s, _, _, _, _ = refine_timestamp_only_revised_for_de(
            T_ego_cam_from_stage1, K_sensor_ls_init, D_sensor_ls_init_flat, model_type, img_width, img_height,
            p2d_target_dist_final_ls_init, P3d_world_img, t_rec_us_img, ego_interpolator_func,
            dt_bounds_seconds=final_ls_dt_bounds_seconds, loss_function='linear'
        )
        dt_init_for_final_ls_map[img_idx] = dt_opt_s if dt_opt_s is not None else 0.0

    # OptData for final LS (initial K/D for OptData are the K_sensor_ls_init / D_sensor_ls_init_flat)
    # (OptData creation is the same)
    try:
        opt_data_final_ls = OptimizationData(
            inlier_matches_map_us=inlier_matches_map_undistorted_ideal,
            ego_interpolator_us=ego_interpolator_func,
            K_initial_ideal=K_sensor_ls_init,
            D_initial_sensor=D_sensor_ls_init_flat,
            model_type=model_type,
            img_width=img_width, img_height=img_height,
            t_rec_map_us=query_timestamps_rec_us,
            num_images_passed_to_constructor=len(query_indices_for_opt),
            debug_image_idx_to_name_map=debug_opt_data_vis_params.get('image_idx_to_name_map') if debug_opt_data_vis_params else None,
            debug_query_image_dir=debug_opt_data_vis_params.get('query_image_dir') if debug_opt_data_vis_params else None,
            debug_output_vis_dir=debug_opt_data_vis_params.get('output_vis_dir') if debug_opt_data_vis_params else None,
            debug_vis_image_indices=debug_opt_data_vis_params.get('vis_image_indices') if debug_opt_data_vis_params else []
        )
    except Exception as e_optdata:
        logging.error(f"Failed to create OptimizationData for Stage 2 LS: {e_optdata}", exc_info=True)
        return T_ego_cam_from_stage1, K_initial_sensor, initial_D_sensor_coeffs, dt_init_for_final_ls_map, None

    if opt_data_final_ls.num_opt_images == 0:
        logging.error("OptData for Stage 2 LS has zero images. Returning Stage 1 results & initial K/D.")
        return T_ego_cam_from_stage1, K_initial_sensor, initial_D_sensor_coeffs, dt_init_for_final_ls_map, None

    # Initial guess x0 for final LS
    xi_ego_cam_ls_init = xi_ego_cam_from_stage1 # From Stage 1
    k_params_ls_init = np.array([
        K_sensor_ls_init[0,0], K_sensor_ls_init[1,1],
        K_sensor_ls_init[0,2], K_sensor_ls_init[1,2]
    ])
    d_params_ls_init = D_sensor_ls_init_flat
    
    dt_ls_init_seconds_vec = np.zeros(opt_data_final_ls.num_opt_images)
    for i, img_idx_in_opt_data in enumerate(opt_data_final_ls.image_indices):
        dt_ls_init_seconds_vec[i] = dt_init_for_final_ls_map.get(img_idx_in_opt_data, 0.0)
    x0_final_ls = np.concatenate((xi_ego_cam_ls_init, k_params_ls_init, d_params_ls_init, dt_ls_init_seconds_vec))

    # Bounds for final LS
    bounds_ls_low = []
    bounds_ls_high = []

    # Extrinsics (T_ego_cam) bounds: centered at Stage 1 result, with Stage 1's DE exploration offset
    # This means LS can explore the same range for T_ego_cam as DE did around its center.
    final_ls_xi_bounds_abs_offset = stage1_de_xi_bounds_abs_offset # Use same offset as Stage 1 DE
    bounds_ls_low.extend(list(xi_ego_cam_ls_init - final_ls_xi_bounds_abs_offset))
    bounds_ls_high.extend(list(xi_ego_cam_ls_init + final_ls_xi_bounds_abs_offset))
    logging.info(f"Stage 2 LS: Extrinsic (xi) bounds centered at Stage 1 DE result +/- {final_ls_xi_bounds_abs_offset}")

    # Intrinsics (K, D) bounds: centered at *initial config values*, using `final_ls_intrinsic_bounds_from_config`
    if final_ls_intrinsic_bounds_from_config is None:
        logging.warning("`final_ls_intrinsic_bounds_from_config` not provided. Using wider default intrinsic bounds for LS.")
        # Fallback to wider default K bounds (e.g., +/- 10% for f, +/- 50px for c)
        k_ls_b_low_default = [k_params_ls_init[0]*0.9, k_params_ls_init[1]*0.9, k_params_ls_init[2]-50, k_params_ls_init[3]-50]
        k_ls_b_high_default = [k_params_ls_init[0]*1.1, k_params_ls_init[1]*1.1, k_params_ls_init[2]+50, k_params_ls_init[3]+50]
        bounds_ls_low.extend(k_ls_b_low_default)
        bounds_ls_high.extend(k_ls_b_high_default)
        if num_d_params > 0: # Fallback to wider D bounds
            d_abs_init = np.abs(d_params_ls_init)
            bounds_ls_low.extend(list(d_params_ls_init - 0.3 * d_abs_init - 0.02))
            bounds_ls_high.extend(list(d_params_ls_init + 0.3 * d_abs_init + 0.02))
    else:
        cfg_bounds_ls = final_ls_intrinsic_bounds_from_config
        k_bounds_ls_low_cfg = cfg_bounds_ls.get('k_low', k_params_ls_init * 0.98) # Default to init +/- 2%
        k_bounds_ls_high_cfg = cfg_bounds_ls.get('k_high', k_params_ls_init * 1.02)
        d_bounds_ls_low_cfg = cfg_bounds_ls.get('d_low', d_params_ls_init - 0.01) if num_d_params > 0 else []
        d_bounds_ls_high_cfg = cfg_bounds_ls.get('d_high', d_params_ls_init + 0.01) if num_d_params > 0 else []
        
        # Ensure these bounds from config are arrays of correct length
        # (k_params_ls_init is length NUM_K_PARAMS, d_params_ls_init is length num_d_params)
        # This assumes cfg_intrinsic_bounds provides flat lists/arrays for k_low/high and d_low/high
        if len(k_bounds_ls_low_cfg) == NUM_K_PARAMS: bounds_ls_low.extend(k_bounds_ls_low_cfg)
        else: bounds_ls_low.extend(k_params_ls_init * 0.98) # Fallback
        if len(k_bounds_ls_high_cfg) == NUM_K_PARAMS: bounds_ls_high.extend(k_bounds_ls_high_cfg)
        else: bounds_ls_high.extend(k_params_ls_init * 1.02) # Fallback

        if num_d_params > 0:
            if len(d_bounds_ls_low_cfg) == num_d_params: bounds_ls_low.extend(d_bounds_ls_low_cfg)
            else: bounds_ls_low.extend(d_params_ls_init - 0.01) # Fallback
            if len(d_bounds_ls_high_cfg) == num_d_params: bounds_ls_high.extend(d_bounds_ls_high_cfg)
            else: bounds_ls_high.extend(d_params_ls_init + 0.01) # Fallback
        logging.info(f"Stage 2 LS: Using intrinsic bounds from config (relative to initial K/D).")

    # dt bounds (same as before)
    bounds_ls_low.extend([final_ls_dt_bounds_seconds[0]] * opt_data_final_ls.num_opt_images)
    bounds_ls_high.extend([final_ls_dt_bounds_seconds[1]] * opt_data_final_ls.num_opt_images)
    bounds_final_ls_staged = (bounds_ls_low, bounds_ls_high)

    # (f_scale_final_ls, initial cost calculation, least_squares call, and result unpacking remain the same
    #  as in the previous refine_parameters_de_extr_then_ls_joint)
    f_scale_final_ls = de_robust_loss_scale
    try:
        initial_final_ls_residuals = compute_residuals(x0_final_ls, opt_data_final_ls)
        initial_final_ls_cost_robust = calculate_robust_cost_for_de(
            initial_final_ls_residuals, final_ls_loss_function, f_scale_final_ls
        )
        logging.info(f"Stage 2 LS: Initial cost (robust '{final_ls_loss_function}' eval, scale={f_scale_final_ls}): {initial_final_ls_cost_robust:.4e}")
    except Exception as e_ls_init_cost:
        logging.error(f"Failed to compute initial cost for Stage 2 LS: {e_ls_init_cost}. Returning Stage 1 T_ego_cam & initial K/D.", exc_info=True)
        return T_ego_cam_from_stage1, K_initial_sensor, initial_D_sensor_coeffs, dt_init_for_final_ls_map, None

    final_ls_result_obj = None
    try:
        final_ls_result_obj = least_squares(
            compute_residuals, x0_final_ls, jac='2-point', bounds=bounds_final_ls_staged, method='trf',
            ftol=1e-8, xtol=1e-8, gtol=1e-8,
            loss=final_ls_loss_function, f_scale=f_scale_final_ls,
            verbose=final_ls_verbose,
            max_nfev=2000 * len(x0_final_ls), 
            args=(opt_data_final_ls,)
        )
        logging.info(f"Stage 2 LS optimization finished. Status: {final_ls_result_obj.status} ({final_ls_result_obj.message})")
        logging.info(f"Stage 2 LS cost: {final_ls_result_obj.cost:.4e}. Optimality: {final_ls_result_obj.optimality:.4e}")
    except Exception as e_ls_final:
        logging.error(f"Stage 2 LS optimization crashed: {e_ls_final}. Returning Stage 1 T_ego_cam & initial K/D.", exc_info=True)
        return T_ego_cam_from_stage1, K_initial_sensor, initial_D_sensor_coeffs, dt_init_for_final_ls_map, None

    T_ego_cam_final_refined = T_ego_cam_from_stage1
    K_sensor_final_refined = K_initial_sensor
    D_sensor_final_refined = initial_D_sensor_coeffs
    final_refined_delta_t_map = dt_init_for_final_ls_map

    if final_ls_result_obj and (final_ls_result_obj.success or final_ls_result_obj.cost < initial_final_ls_cost_robust * 0.999):
        X_ls_final = final_ls_result_obj.x
        T_ego_cam_final_refined = se3_to_SE3(X_ls_final[:6])
        k_final_params = X_ls_final[6:6+NUM_K_PARAMS]
        K_sensor_final_refined = np.array([[k_final_params[0], 0, k_final_params[2]],
                                           [0, k_final_params[1], k_final_params[3]],
                                           [0,0,1]])
        D_sensor_final_refined_flat = X_ls_final[6+NUM_K_PARAMS : 6+NUM_K_PARAMS+num_d_params]
        D_sensor_final_refined = D_sensor_final_refined_flat
        
        dt_final_seconds_flat = X_ls_final[6+NUM_K_PARAMS+num_d_params:]
        for i, img_idx_in_opt_data in enumerate(opt_data_final_ls.image_indices):
            if img_idx_in_opt_data in final_refined_delta_t_map:
                 final_refined_delta_t_map[img_idx_in_opt_data] = dt_final_seconds_flat[i]
        logging.info("Successfully refined all parameters using 2-Stage optimization.")
    else:
        logging.warning("Stage 2 LS did not improve significantly or failed. Returning Stage 1 T_ego_cam, initial K/D, and pre-LS dt map.")

    logging.info(f"Final Refined T_ego_cam (2-Stage):\n{np.round(T_ego_cam_final_refined, 6)}")
    logging.info(f"Final Refined K_sensor (2-Stage):\n{np.round(K_sensor_final_refined, 4)}")
    if D_sensor_final_refined is not None and D_sensor_final_refined.size > 0:
        logging.info(f"Final Refined D_sensor (2-Stage):\n{np.round(D_sensor_final_refined, 7)}")

    return T_ego_cam_final_refined, K_sensor_final_refined, D_sensor_final_refined, final_refined_delta_t_map, final_ls_result_obj

def refine_all_parameters_hybrid(
    # Config and Initial State
    initial_T_ego_cam_from_config: np.ndarray,
    pnp_derived_T_ego_cam_guess: np.ndarray,
    # Data: PnP inliers (ideal plane) and corresponding 3D world points
    # This map contains ALL PnP successful matches before spatial distribution.
    inlier_matches_map_undistorted_ideal_ALL_PNP_SUCCESS: dict,
    ego_timestamps_us: np.ndarray,
    ego_poses: list,
    # query_indices_for_opt_ALL_PNP_SUCCESS: List of image indices that had *any* PnP success
    query_indices_for_opt_ALL_PNP_SUCCESS: list,
    query_timestamps_rec_us: dict, # Full map {img_idx: t_rec_us}
    # Sensor model parameters
    K_ideal_plane_definition: np.ndarray, # K defining the ideal undistorted plane
    K_initial_sensor_cfg: np.ndarray, # Initial K of the physical sensor (from config)
    initial_D_sensor_coeffs_cfg: np.ndarray, # Initial D of the physical sensor (from config)
    model_type: str,
    img_width: int, img_height: int,
    # DE Parameters
    dt_bounds_seconds_de_inner_opt: tuple[float, float] = (-0.02, 0.02),
    de_robust_loss_type: str = 'cauchy',
    de_robust_loss_scale: float = 5.0,
    de_popsize_factor: int = 15,
    de_maxiter: int = 150,
    de_workers: int = -1,
    # Final LS Parameters
    dt_bounds_seconds_final_ls: tuple[float, float] = (-0.05, 0.05),
    loss_function_final_ls: str = 'cauchy',
    final_ls_verbose: int = 1,
    # Bounds Configuration (passed via intrinsic_bounds_config)
    interpolation_tolerance_us: int = 1,
    intrinsic_bounds_config: dict = None, # Contains DE and LS bounds details
    # Spatial Distribution Parameters
    spatial_distribution_grid_size: int = 10,
    max_points_per_grid_cell: int = 2,
    min_total_distributed_points: int = 100,
    # Debug Visualization Parameters
    debug_opt_data_vis_params: dict = None
):
    logging.info("--- Starting HYBRID Extrinsics, Intrinsics, and Timestamp Refinement (Global Spatial Distribution) ---")
    logging.info(f"DE Robust Loss: Type='{de_robust_loss_type}', Scale={de_robust_loss_scale}")
    logging.info(f"Final LS Robust Loss: Type='{loss_function_final_ls}'")

    global de_eval_count_global
    de_eval_count_global = 0

    if not query_indices_for_opt_ALL_PNP_SUCCESS:
        logging.error("No PnP successful query images provided for hybrid refinement.")
        return None, None, None, None, None

    num_d_params = 0
    flat_initial_D_sensor_cfg = np.array([])
    if initial_D_sensor_coeffs_cfg is not None:
        flat_initial_D_sensor_cfg = initial_D_sensor_coeffs_cfg.flatten()
        num_d_params = len(flat_initial_D_sensor_cfg)

    T_ego_cam_for_de_center = initial_T_ego_cam_from_config
    # if pnp_derived_T_ego_cam_guess is not None:
    #     try:
    #         if np.isclose(np.linalg.det(pnp_derived_T_ego_cam_guess[:3,:3]), 1.0) and \
    #            np.all(np.isfinite(pnp_derived_T_ego_cam_guess)):
    #             T_ego_cam_for_de_center = pnp_derived_T_ego_cam_guess
    #             logging.info("Using PnP-derived T_ego_cam as center for DE search.")
    #     except Exception as e_pnp_val:
    #         logging.warning(f"Error validating PnP T_ego_cam guess ({e_pnp_val}), using config T_ego_cam for DE center.")

    try:
        xi_ego_cam_de_initial_center = SE3_to_se3(T_ego_cam_for_de_center)
    except Exception as e_se3:
        logging.error(f"Failed to convert T_ego_cam_for_de_center to se3: {e_se3}. Using identity for xi initial.")
        xi_ego_cam_de_initial_center = np.zeros(6)

    # K_sensor for DE initial guess (can be same as K_initial_sensor_cfg or K_ideal_plane_definition)
    k_params_init_sensor_for_DE = np.array([
        K_initial_sensor_cfg[0, 0], K_initial_sensor_cfg[1, 1],
        K_initial_sensor_cfg[0, 2], K_initial_sensor_cfg[1, 2]
    ])

    # --- 1. Pre-calculate GLOBAL fixed target distorted points for ALL PnP successful input points ---
    # These targets are based on K_initial_sensor_cfg and initial_D_sensor_coeffs_cfg.
    precalculated_target_distorted_points_map_GLOBAL_ALL_PNP = {}
    logging.info("Pre-calculating GLOBAL fixed target distorted points for ALL input PnP-successful points...")

    # Debug visualization for these global targets
    global_target_vis_dir = None
    vis_image_indices_global_targets = []
    if debug_opt_data_vis_params and debug_opt_data_vis_params.get('output_vis_dir_base'):
        global_target_vis_dir = Path(debug_opt_data_vis_params['output_vis_dir_base']) / "GLOBAL_fixed_targets_debug"
        global_target_vis_dir.mkdir(parents=True, exist_ok=True)
        vis_image_indices_global_targets = debug_opt_data_vis_params.get('vis_image_indices_global_targets', [])

    if not inlier_matches_map_undistorted_ideal_ALL_PNP_SUCCESS:
        logging.error("Input 'inlier_matches_map_undistorted_ideal_ALL_PNP_SUCCESS' is empty. Cannot proceed.")
        return initial_T_ego_cam_from_config, K_initial_sensor_cfg, initial_D_sensor_coeffs_cfg, \
               {idx: 0.0 for idx in query_indices_for_opt_ALL_PNP_SUCCESS}, None

    for img_idx in query_indices_for_opt_ALL_PNP_SUCCESS:
        if img_idx not in inlier_matches_map_undistorted_ideal_ALL_PNP_SUCCESS:
            logging.warning(f"Image index {img_idx} listed in query_indices_for_opt_ALL_PNP_SUCCESS but not found in input map. Skipping.")
            continue

        p2d_undist_ideal_kps_img, _ = inlier_matches_map_undistorted_ideal_ALL_PNP_SUCCESS[img_idx]
        if p2d_undist_ideal_kps_img.shape[0] == 0:
            precalculated_target_distorted_points_map_GLOBAL_ALL_PNP[img_idx] = np.empty((0,2), dtype=np.float64)
            continue

        targets = generate_target_distorted_points(
            p2d_undistorted_pixels=p2d_undist_ideal_kps_img,
            K_sensor_current=K_initial_sensor_cfg, # Use initial configured sensor K
            D_sensor_current=initial_D_sensor_coeffs_cfg, # Use initial configured sensor D
            model_type=model_type,
            K_ideal_fixed_for_normalization=K_ideal_plane_definition, # K defining the ideal plane
            img_width=img_width,
            img_height=img_height
        )
        precalculated_target_distorted_points_map_GLOBAL_ALL_PNP[img_idx] = targets

        if global_target_vis_dir and img_idx in vis_image_indices_global_targets:
            image_filename = debug_opt_data_vis_params['image_idx_to_name_map'].get(img_idx)
            if image_filename:
                original_image_file_path_str = str(Path(debug_opt_data_vis_params['query_image_dir']) / image_filename)
                if Path(original_image_file_path_str).exists():
                    vis_path_str = str(global_target_vis_dir / f"GLOBAL_target_img_{img_idx}_{Path(image_filename).stem}.png")
                    logging.debug(f"DEBUG VIS (Global Target): Visualizing for img_idx {img_idx} to {vis_path_str}")
                    visualize_2d_points_on_image(
                        original_image_path=original_image_file_path_str, points_2d_to_draw=targets.copy(),
                        output_path=vis_path_str, label=f"GLOBAL Fixed Target (idx {img_idx})", point_color=(0, 128, 255)
                    )
    logging.info("Finished pre-calculating GLOBAL fixed targets.")

    # --- 2. Grid-Based Spatial Distribution and Subsampling (MODIFIED SECTION V2 - Nearest to Cell Center) ---
    logging.info(f"Applying grid-based spatial distribution (Grid: {spatial_distribution_grid_size}x{spatial_distribution_grid_size}, Max/Cell: {max_points_per_grid_cell}) by nearest to cell center...")
    grid_cells_data = [[[] for _ in range(spatial_distribution_grid_size)] for _ in range(spatial_distribution_grid_size)]
    cell_pixel_w = img_width / float(spatial_distribution_grid_size)
    cell_pixel_h = img_height / float(spatial_distribution_grid_size)
    total_pts_before_sampling = 0

    for img_idx in query_indices_for_opt_ALL_PNP_SUCCESS:
        if img_idx not in inlier_matches_map_undistorted_ideal_ALL_PNP_SUCCESS or \
           img_idx not in precalculated_target_distorted_points_map_GLOBAL_ALL_PNP:
            continue

        p2d_ideals_img, P3Ds_world_img = inlier_matches_map_undistorted_ideal_ALL_PNP_SUCCESS[img_idx]
        targets_distorted_img = precalculated_target_distorted_points_map_GLOBAL_ALL_PNP.get(img_idx)

        if targets_distorted_img is None or \
           p2d_ideals_img.shape[0] != P3Ds_world_img.shape[0] or \
           p2d_ideals_img.shape[0] != targets_distorted_img.shape[0] or \
           p2d_ideals_img.shape[0] == 0:
            if p2d_ideals_img.shape[0] > 0:
                logging.warning(f"Data mismatch or missing targets for img_idx {img_idx} during spatial distribution. Skipping its points.")
            continue
        
        total_pts_before_sampling += p2d_ideals_img.shape[0]

        for i in range(p2d_ideals_img.shape[0]):
            p2d_ideal = p2d_ideals_img[i] # This is the 2D point in the K_ideal_plane_definition frame

            # Determine grid cell
            col_idx = int(p2d_ideal[0] / cell_pixel_w)
            row_idx = int(p2d_ideal[1] / cell_pixel_h)
            col_idx = max(0, min(col_idx, spatial_distribution_grid_size - 1))
            row_idx = max(0, min(row_idx, spatial_distribution_grid_size - 1))

            # Calculate center of this grid cell
            cell_center_x = (col_idx + 0.5) * cell_pixel_w
            cell_center_y = (row_idx + 0.5) * cell_pixel_h
            
            # Calculate distance from p2d_ideal to its cell center
            distance_to_cell_center = np.linalg.norm(p2d_ideal - np.array([cell_center_x, cell_center_y]))
            
            grid_cells_data[row_idx][col_idx].append({
                "p2d_ideal": p2d_ideal, 
                "P3d_world": P3Ds_world_img[i],
                "img_idx": img_idx, 
                "target_distorted": targets_distorted_img[i],
                "distance_to_cell_center": distance_to_cell_center # Store the distance
            })

    selected_points_for_de_and_ls = []
    for r_idx in range(spatial_distribution_grid_size):
        for c_idx in range(spatial_distribution_grid_size):
            cell_pts_with_dist = grid_cells_data[r_idx][c_idx]
            if not cell_pts_with_dist: 
                continue
            
            # Sort points in the cell by their distance to the cell center (ascending)
            cell_pts_with_dist.sort(key=lambda x: x["distance_to_cell_center"])
            
            # Select up to max_points_per_grid_cell
            selected_points_for_de_and_ls.extend(cell_pts_with_dist[:max_points_per_grid_cell])

    logging.info(f"Collected {total_pts_before_sampling} points before spatial sampling.")
    logging.info(f"Selected {len(selected_points_for_de_and_ls)} points after grid-based spatial distribution (nearest to cell center) for DE/LS.")
    # --- END OF MODIFIED SECTION V2 ---

    if len(selected_points_for_de_and_ls) < min_total_distributed_points:
        logging.warning(f"Number of spatially distributed points ({len(selected_points_for_de_and_ls)}) "
                        f"is less than threshold ({min_total_distributed_points}). "
                        "Optimization might be unstable. Consider adjusting grid/sampling parameters "
                        "or using all PnP-successful points as a fallback (not implemented here for brevity).")
        # Fallback: return initial parameters
        dt_map_fallback = {idx: 0.0 for idx in query_indices_for_opt_ALL_PNP_SUCCESS}
        return initial_T_ego_cam_from_config, K_initial_sensor_cfg, initial_D_sensor_coeffs_cfg, dt_map_fallback, None

    # --- 3. Reconstruct Data for DE and LS based on selected_points_for_de_and_ls ---
    inlier_matches_map_SAMPLED = {}
    precalculated_target_distorted_points_map_SAMPLED = {}
    for item in selected_points_for_de_and_ls:
        img_idx = item["img_idx"]
        if img_idx not in inlier_matches_map_SAMPLED:
            inlier_matches_map_SAMPLED[img_idx] = ([], [])
            precalculated_target_distorted_points_map_SAMPLED[img_idx] = []
        inlier_matches_map_SAMPLED[img_idx][0].append(item["p2d_ideal"])
        inlier_matches_map_SAMPLED[img_idx][1].append(item["P3d_world"])
        precalculated_target_distorted_points_map_SAMPLED[img_idx].append(item["target_distorted"])

    for img_idx in inlier_matches_map_SAMPLED.keys():
        inlier_matches_map_SAMPLED[img_idx] = \
            (np.array(inlier_matches_map_SAMPLED[img_idx][0]), np.array(inlier_matches_map_SAMPLED[img_idx][1]))
        precalculated_target_distorted_points_map_SAMPLED[img_idx] = \
            np.array(precalculated_target_distorted_points_map_SAMPLED[img_idx])

    query_indices_SAMPLED = sorted(list(inlier_matches_map_SAMPLED.keys()))
    if not query_indices_SAMPLED: # Should be caught by min_total_distributed_points
        logging.error("No points selected after spatial distribution (empty query_indices_SAMPLED). Cannot proceed.")
        dt_map_fallback = {idx: 0.0 for idx in query_indices_for_opt_ALL_PNP_SUCCESS}
        return initial_T_ego_cam_from_config, K_initial_sensor_cfg, initial_D_sensor_coeffs_cfg, dt_map_fallback, None

    # --- 4. DE Optimization (using the spatially distributed sampled data) ---
    # DE Bounds Setup (Using intrinsic_bounds_config)
    cfg_b = intrinsic_bounds_config if intrinsic_bounds_config else {}
    xi_de_abs_offset = np.array(cfg_b.get('de_xi_abs_offset', [0.2,0.2,0.2,0.5,0.5,0.5]))
    b_de_xi_low = xi_ego_cam_de_initial_center - xi_de_abs_offset
    b_de_xi_high = xi_ego_cam_de_initial_center + xi_de_abs_offset
    # K bounds for DE
    k_de_rel_low = cfg_b.get('de_k_rel_offset_low', 0.8); k_de_rel_high = cfg_b.get('de_k_rel_offset_high', 1.2)
    k_de_cxcy_abs_px = cfg_b.get('de_k_cxcy_abs_offset_px', 0.15 * max(img_width, img_height))
    b_de_k_low = [max(10.,k_params_init_sensor_for_DE[0]*k_de_rel_low), max(10.,k_params_init_sensor_for_DE[1]*k_de_rel_low),
                  max(0.,k_params_init_sensor_for_DE[2]-k_de_cxcy_abs_px), max(0.,k_params_init_sensor_for_DE[3]-k_de_cxcy_abs_px)]
    b_de_k_high = [k_params_init_sensor_for_DE[0]*k_de_rel_high, k_params_init_sensor_for_DE[1]*k_de_rel_high,
                   min(float(img_width-1),k_params_init_sensor_for_DE[2]+k_de_cxcy_abs_px), min(float(img_height-1),k_params_init_sensor_for_DE[3]+k_de_cxcy_abs_px)]
    # D bounds for DE
    b_de_d_low, b_de_d_high = [], []
    if num_d_params > 0:
        d_de_abs_offset_val = np.array(cfg_b.get('de_d_abs_offset', np.full(num_d_params, 0.15)))
        if len(d_de_abs_offset_val) != num_d_params: d_de_abs_offset_val = np.resize(np.full(num_d_params,0.15), num_d_params)
        b_de_d_low = list(flat_initial_D_sensor_cfg - d_de_abs_offset_val)
        b_de_d_high = list(flat_initial_D_sensor_cfg + d_de_abs_offset_val)
        for i_d_de in range(num_d_params):
            if b_de_d_low[i_d_de] >= b_de_d_high[i_d_de]: (b_de_d_low[i_d_de], b_de_d_high[i_d_de]) = (b_de_d_high[i_d_de]-0.01, b_de_d_low[i_d_de]+0.01)

    de_bounds = list(zip(np.concatenate((b_de_xi_low, b_de_k_low, b_de_d_low)),
                         np.concatenate((b_de_xi_high, b_de_k_high, b_de_d_high))))

    de_obj_data = {
        "inlier_matches_map_undistorted_ideal": inlier_matches_map_SAMPLED,
        "precalculated_target_distorted_points": precalculated_target_distorted_points_map_SAMPLED,
        "query_indices_for_opt": query_indices_SAMPLED,
        "query_timestamps_rec_us": query_timestamps_rec_us,
        "ego_interpolator_us": functools.partial(get_pose_for_timestamp, timestamps_us=ego_timestamps_us, poses=ego_poses, tolerance_us=interpolation_tolerance_us),
        "model_type": model_type, "img_width": img_width, "img_height": img_height,
        "dt_bounds_seconds_de_inner_opt": dt_bounds_seconds_de_inner_opt,
        "num_d_params": num_d_params,
        "de_robust_loss_type": de_robust_loss_type, "de_robust_loss_scale": de_robust_loss_scale,
        "de_popsize_factor_for_log": de_popsize_factor
    }
    logging.info(f"Running DE (PopFactor={de_popsize_factor}, MaxIter={de_maxiter}, Workers={de_workers}) on {len(query_indices_SAMPLED)} images with SAMPLED points...")
    de_start_t = time.time()
    de_res = differential_evolution(de_objective_func_global, bounds=de_bounds, args=(de_obj_data,),
                                  strategy='best1bin', maxiter=de_maxiter, popsize=de_popsize_factor,
                                  tol=1e-4, polish=False, disp=True, workers=de_workers, updating='deferred')
    logging.info(f"DE finished in {time.time()-de_start_t:.2f}s. Final avg robust cost: {de_res.fun:.6e}")

    if not de_res.success and de_res.fun > 1e5: # High cost threshold indicates severe failure
        logging.error(f"DE failed badly (cost: {de_res.fun:.2e}). Aborting hybrid refinement.")
        dt_map_fallback = {idx: 0.0 for idx in query_indices_for_opt_ALL_PNP_SUCCESS}
        return initial_T_ego_cam_from_config, K_initial_sensor_cfg, initial_D_sensor_coeffs_cfg, dt_map_fallback, de_res

    X_de_best = de_res.x
    xi_ego_cam_best_de = X_de_best[:6]
    k_params_sensor_best_de = X_de_best[6:6+NUM_K_PARAMS]
    d_params_sensor_best_de_flat = X_de_best[6+NUM_K_PARAMS : 6+NUM_K_PARAMS+num_d_params]
    T_ego_cam_best_de = se3_to_SE3(xi_ego_cam_best_de)
    K_sensor_best_de = np.array([[k_params_sensor_best_de[0],0,k_params_sensor_best_de[2]],
                                 [0,k_params_sensor_best_de[1],k_params_sensor_best_de[3]], [0,0,1]])
    logging.info(f"Best T_ego_cam from DE:\n{np.round(T_ego_cam_best_de,5)}")
    logging.info(f"Best K_sensor from DE:\n{np.round(K_sensor_best_de,4)}")
    if d_params_sensor_best_de_flat.size > 0: logging.info(f"Best D_sensor from DE: {np.round(d_params_sensor_best_de_flat,6)}")

    # Calculate dt_ls_init_map_seconds using SAMPLED data and DE's best params
    dt_ls_init_map_seconds = {}
    logging.info("Calculating dt values for LS initialization using DE results and SAMPLED points...")
    for img_idx in query_indices_SAMPLED: # Iterate over indices that contributed to DE
        P3d_world_sampled_img = inlier_matches_map_SAMPLED[img_idx][1]
        target_dist_sampled_img = precalculated_target_distorted_points_map_SAMPLED[img_idx]
        t_rec_us_img = query_timestamps_rec_us[img_idx]

        if P3d_world_sampled_img.shape[0] == 0 or target_dist_sampled_img.shape[0] == 0:
            dt_ls_init_map_seconds[img_idx] = 0.0; continue

        dt_opt_s, _, _, _, _ = refine_timestamp_only_revised_for_de(
            current_T_ego_cam=T_ego_cam_best_de, current_K_matrix_sensor=K_sensor_best_de,
            current_D_coeffs_sensor=d_params_sensor_best_de_flat, current_model_type=model_type,
            current_img_width=img_width, current_img_height=img_height,
            p2d_target_distorted_pixels=target_dist_sampled_img, # Target from sampled data
            P3d_world_points=P3d_world_sampled_img, # 3D from sampled data
            t_rec_us=t_rec_us_img, ego_interpolator_us=de_obj_data["ego_interpolator_us"],
            dt_bounds_seconds=dt_bounds_seconds_de_inner_opt, loss_function='linear'
        )
        dt_ls_init_map_seconds[img_idx] = dt_opt_s if dt_opt_s is not None else 0.0

    # --- 5. Final Least Squares (OptData also uses SAMPLED data, but its targets are fixed original global targets) ---
    logging.info("--- Starting Final least_squares Fine-tuning (on SAMPLED points) ---")
    opt_data_ls_dbg_params = {} # Setup as before if debug needed for LS OptData
    if debug_opt_data_vis_params:
        ls_dbg_vis_dir = Path(debug_opt_data_vis_params.get('output_vis_dir_base', 'vis_debug')) / \
                         debug_opt_data_vis_params.get('output_vis_dir_LS_targets_subdir', 'LS_OptData_debug_HybridSpatial')
        opt_data_ls_dbg_params = {
            'debug_image_idx_to_name_map': debug_opt_data_vis_params.get('image_idx_to_name_map'),
            'debug_query_image_dir': debug_opt_data_vis_params.get('query_image_dir'),
            'debug_output_vis_dir': ls_dbg_vis_dir,
            'debug_vis_image_indices': debug_opt_data_vis_params.get('vis_image_indices_LS_targets', [])
        }

    try:
        # OptData's internal targets are based on K_initial_sensor_cfg and initial_D_sensor_coeffs_cfg.
        # It receives the SAMPLED p2d_ideal points and their P3D points.
        opt_data_final_ls = OptimizationData(
            inlier_matches_map_us=inlier_matches_map_SAMPLED, # SAMPLED p2d_ideal and P3D_world
            ego_interpolator_us=de_obj_data["ego_interpolator_us"],
            K_initial_ideal_for_opt_data_targets=K_initial_sensor_cfg, # Original config K for OptData's targets
            D_initial_sensor_for_opt_data_targets=initial_D_sensor_coeffs_cfg, # Original config D
            model_type=model_type, img_width=img_width, img_height=img_height,
            t_rec_map_us=query_timestamps_rec_us,
            num_images_passed_to_constructor=len(query_indices_SAMPLED),
            K_GLOBAL_IDEAL_FOR_UNDIST_NORMALIZATION=K_ideal_plane_definition,
            **opt_data_ls_dbg_params
        )
    except Exception as e_optdata_ls:
        logging.error(f"Failed to create OptData for final LS: {e_optdata_ls}", exc_info=True)
        dt_map_fallback = {idx: dt_ls_init_map_seconds.get(idx, 0.0) for idx in query_indices_for_opt_ALL_PNP_SUCCESS}
        return T_ego_cam_best_de, K_sensor_best_de, d_params_sensor_best_de_flat, dt_map_fallback, de_res

    if opt_data_final_ls.num_opt_images == 0:
        logging.error("OptData for final LS (HybridSpatial) has zero images. Returning DE results.")
        dt_map_fallback = {idx: dt_ls_init_map_seconds.get(idx, 0.0) for idx in query_indices_for_opt_ALL_PNP_SUCCESS}
        return T_ego_cam_best_de, K_sensor_best_de, d_params_sensor_best_de_flat, dt_map_fallback, de_res

    xi_ls_init = xi_ego_cam_best_de
    k_ls_init = k_params_sensor_best_de
    d_ls_init = d_params_sensor_best_de_flat
    dt_ls_init_vec = np.zeros(opt_data_final_ls.num_opt_images) # OptData determines actual num_dt_params
    for i, img_idx_ls_optdata in enumerate(opt_data_final_ls.image_indices): # image_indices are from query_indices_SAMPLED
        dt_ls_init_vec[i] = dt_ls_init_map_seconds.get(img_idx_ls_optdata, 0.0)
    x0_ls = np.concatenate((xi_ls_init, k_ls_init, d_ls_init, dt_ls_init_vec))

    # LS Bounds (tighter, around DE solution, using intrinsic_bounds_config)
    ls_xi_abs_off = np.array(cfg_b.get('ls_xi_abs_offset', [0.03,0.03,0.03,0.08,0.08,0.08]))
    b_ls_xi_low = xi_ls_init - ls_xi_abs_off; b_ls_xi_high = xi_ls_init + ls_xi_abs_off
    ls_k_rel_low = cfg_b.get('ls_k_rel_offset_low',0.98); ls_k_rel_high = cfg_b.get('ls_k_rel_offset_high',1.02)
    ls_k_cxcy_abs_px = cfg_b.get('ls_k_cxcy_abs_offset_px',0.05*max(img_width,img_height))
    b_ls_k_low = [max(10.,k_ls_init[0]*ls_k_rel_low), max(10.,k_ls_init[1]*ls_k_rel_low),
                  max(0.,k_ls_init[2]-ls_k_cxcy_abs_px), max(0.,k_ls_init[3]-ls_k_cxcy_abs_px)]
    b_ls_k_high = [k_ls_init[0]*ls_k_rel_high, k_ls_init[1]*ls_k_rel_high,
                   min(float(img_width-1),k_ls_init[2]+ls_k_cxcy_abs_px), min(float(img_height-1),k_ls_init[3]+ls_k_cxcy_abs_px)]
    b_ls_d_low, b_ls_d_high = [], []
    if num_d_params > 0:
        ls_d_abs_s = cfg_b.get('ls_d_abs_offset_scale',0.05); ls_d_abs_c = cfg_b.get('ls_d_abs_offset_const',0.008)
        d_abs_ls_init = np.abs(d_ls_init)
        b_ls_d_low = list(d_ls_init - ls_d_abs_s*d_abs_ls_init - ls_d_abs_c)
        b_ls_d_high = list(d_ls_init + ls_d_abs_s*d_abs_ls_init + ls_d_abs_c)
        for i_d_ls in range(num_d_params):
            if b_ls_d_low[i_d_ls] >= b_ls_d_high[i_d_ls]: (b_ls_d_low[i_d_ls], b_ls_d_high[i_d_ls]) = (b_ls_d_high[i_d_ls]-1e-4, b_ls_d_low[i_d_ls]+1e-4)
    bounds_ls = (np.concatenate((b_ls_xi_low,b_ls_k_low,b_ls_d_low, [dt_bounds_seconds_final_ls[0]]*opt_data_final_ls.num_opt_images)),
                 np.concatenate((b_ls_xi_high,b_ls_k_high,b_ls_d_high, [dt_bounds_seconds_final_ls[1]]*opt_data_final_ls.num_opt_images)))

    f_scale_ls = de_robust_loss_scale # Can be tuned separately
    try:
        initial_ls_res = compute_residuals(x0_ls, opt_data_final_ls)
        initial_ls_cost_rob = calculate_robust_cost_for_de(initial_ls_res, loss_function_final_ls, f_scale_ls)
        logging.info(f"Final LS: Initial cost (from DE best, robust '{loss_function_final_ls}', scale={f_scale_ls}): {initial_ls_cost_rob:.4e}")
    except Exception as e_ls_icost:
        logging.error(f"Failed to compute initial cost for final LS: {e_ls_icost}. Returning DE results.", exc_info=True)
        dt_map_fallback = {idx: dt_ls_init_map_seconds.get(idx, 0.0) for idx in query_indices_for_opt_ALL_PNP_SUCCESS}
        return T_ego_cam_best_de, K_sensor_best_de, d_params_sensor_best_de_flat, dt_map_fallback, de_res

    final_ls_res_obj = None
    try:
        final_ls_res_obj = least_squares(
            compute_residuals, x0_ls, jac='2-point', bounds=bounds_ls, method='trf',
            ftol=1e-9, xtol=1e-9, gtol=1e-9, loss=loss_function_final_ls, f_scale=f_scale_ls,
            verbose=final_ls_verbose, max_nfev=2500 * len(x0_ls), args=(opt_data_final_ls,)
        )
        logging.info(f"Final LS opt finished. Status: {final_ls_res_obj.status} ({final_ls_res_obj.message})")
        logging.info(f"Final LS cost (robust '{loss_function_final_ls}', scale={f_scale_ls}): {final_ls_res_obj.cost:.4e}. Opt: {final_ls_res_obj.optimality:.4e}")
    except Exception as e_ls_final_run:
        logging.error(f"Final LS optimization crashed: {e_ls_final_run}. Returning DE results.", exc_info=True)
        dt_map_fallback = {idx: dt_ls_init_map_seconds.get(idx, 0.0) for idx in query_indices_for_opt_ALL_PNP_SUCCESS}
        return T_ego_cam_best_de, K_sensor_best_de, d_params_sensor_best_de_flat, dt_map_fallback, de_res

    # Unpack final results, defaulting to DE if LS fails or doesn't improve
    T_ego_cam_final = T_ego_cam_best_de
    K_sensor_final = K_sensor_best_de
    D_sensor_final_flat = d_params_sensor_best_de_flat
    # Initialize final_dt_map for ALL original PnP successful images
    final_refined_delta_t_map_seconds = {idx: 0.0 for idx in query_indices_for_opt_ALL_PNP_SUCCESS}
    # Populate with dt values from DE-based initialization (for those in SAMPLED set)
    for idx_sampled_dt, dt_val_sampled in dt_ls_init_map_seconds.items():
        if idx_sampled_dt in final_refined_delta_t_map_seconds:
            final_refined_delta_t_map_seconds[idx_sampled_dt] = dt_val_sampled

    improvement_thresh_factor = 0.999 # Allow slight cost increase if success flag is True
    if final_ls_res_obj and (final_ls_res_obj.success or final_ls_res_obj.cost < initial_ls_cost_rob * improvement_thresh_factor):
        X_ls_final = final_ls_res_obj.x
        T_ego_cam_final = se3_to_SE3(X_ls_final[:6])
        k_final_ls = X_ls_final[6:6+NUM_K_PARAMS]
        K_sensor_final = np.array([[k_final_ls[0],0,k_final_ls[2]], [0,k_final_ls[1],k_final_ls[3]], [0,0,1]])
        D_sensor_final_flat = X_ls_final[6+NUM_K_PARAMS : 6+NUM_K_PARAMS+num_d_params]
        dt_final_flat_ls = X_ls_final[6+NUM_K_PARAMS+num_d_params:]
        # Update dt map with LS results for images processed by LS's OptData
        for i, img_idx_ls_optdata_final in enumerate(opt_data_final_ls.image_indices):
            if img_idx_ls_optdata_final in final_refined_delta_t_map_seconds:
                 final_refined_delta_t_map_seconds[img_idx_ls_optdata_final] = dt_final_flat_ls[i]
        logging.info("Successfully refined all parameters using DE (spatially sampled) + LS (spatially sampled).")
    else:
        logging.warning(f"Final LS did not improve significantly or failed. Returning DE best parameters with DE-initialized dt map.")

    logging.info(f"Final Refined T_ego_cam (HybridSpatial):\n{np.round(T_ego_cam_final, 6)}")
    logging.info(f"Final Refined K_sensor (HybridSpatial):\n{np.round(K_sensor_final, 4)}")
    if D_sensor_final_flat.size > 0: logging.info(f"Final Refined D_sensor (HybridSpatial):\n{np.round(D_sensor_final_flat, 7)}")

    return T_ego_cam_final, K_sensor_final, D_sensor_final_flat, final_refined_delta_t_map_seconds, final_ls_res_obj

def refine_all_parameters(
    initial_T_ego_cam: np.ndarray,
    inlier_matches_map: dict,
    ego_timestamps_us: np.ndarray,
    ego_poses: list,
    query_indices: list,
    query_timestamps_rec_us: dict,
    initial_K_matrix: np.ndarray,
    initial_D_coeffs: np.ndarray,
    model_type: str,
    img_width: int,
    img_height: int,
    dt_bounds_seconds=(-0.1, 0.1),
    loss_function='cauchy',
    verbose=1,
    interpolation_tolerance_us: int = 1,
    intrinsic_bounds_config: dict = None,
    # --- DEBUG VISUALIZATION PARAMETERS FOR OPT_DATA ---
    debug_opt_data_vis_params: dict = None # New dictionary to hold debug params
):
    logging.info("--- Starting Joint Extrinsics, Intrinsics, and Timestamp Refinement ---")

    num_total_query_images_in_batch = len(query_indices)
    if num_total_query_images_in_batch == 0:
        logging.error("No query images provided for refinement.")
        return None, None, None, None, None # T_ego_cam, K, D, delta_t_map, result_obj

    # Filter inlier_matches_map to only include those in query_indices that have actual points
    # AND have a corresponding recorded timestamp. These will be used for OptimizationData.
    valid_inlier_matches_for_opt_data = {}
    valid_query_timestamps_rec_us_for_opt_data = {}
    # valid_query_indices_for_opt_data will be derived from keys of valid_inlier_matches_for_opt_data

    for idx in query_indices: # Iterate through all originally intended images
        if idx in inlier_matches_map and \
           isinstance(inlier_matches_map[idx], tuple) and \
           len(inlier_matches_map[idx]) == 2 and \
           isinstance(inlier_matches_map[idx][0], np.ndarray) and \
           inlier_matches_map[idx][0].shape[0] > 0: # Check for valid match data
            if idx in query_timestamps_rec_us:
                valid_inlier_matches_for_opt_data[idx] = inlier_matches_map[idx]
                valid_query_timestamps_rec_us_for_opt_data[idx] = query_timestamps_rec_us[idx]
            else:
                logging.warning(f"Image index {idx} has matches but no recorded timestamp. Skipping for opt.")
        # else: image idx not in inlier_matches_map, or has no points, or no timestamp, or data malformed.

    num_images_actually_in_opt_data = len(valid_inlier_matches_for_opt_data)
    if num_images_actually_in_opt_data == 0:
        logging.error("No images with valid inlier matches AND timestamps found for optimization.")
        # Return initial values, and a dt_map for all originally intended images
        final_dt_map_for_return = {idx: 0.0 for idx in query_indices}
        return initial_T_ego_cam, initial_K_matrix, initial_D_coeffs, \
               final_dt_map_for_return, None

    logging.info(f"Preparing OptimizationData with {num_images_actually_in_opt_data} images "
                 f"(out of {num_total_query_images_in_batch} initially specified for this batch).")

    try:
        xi_ego_cam_init = SE3_to_se3(initial_T_ego_cam)
    except Exception as e:
        logging.error(f"Failed to convert initial T_ego_cam to se(3): {e}", exc_info=True)
        return None, None, None, None, None

    # --- Prepare Initial Guess x0 ---
    k_params_init = np.array([
        initial_K_matrix[0, 0], initial_K_matrix[1, 1], # fx, fy
        initial_K_matrix[0, 2], initial_K_matrix[1, 2]  # cx, cy
    ])
    d_params_init = initial_D_coeffs.flatten() if initial_D_coeffs is not None else np.array([])
    num_d_params = len(d_params_init)


    # Create OptimizationData instance. This will determine the actual number of
    # dt parameters needed based on how many images it successfully processes.
    opt_data_debug_kwargs = {}
    if debug_opt_data_vis_params:
        opt_data_debug_kwargs['debug_image_idx_to_name_map'] = debug_opt_data_vis_params.get('image_idx_to_name_map')
        opt_data_debug_kwargs['debug_query_image_dir'] = debug_opt_data_vis_params.get('query_image_dir')
        opt_data_debug_kwargs['debug_output_vis_dir'] = debug_opt_data_vis_params.get('output_vis_dir')
        opt_data_debug_kwargs['debug_vis_image_indices'] = debug_opt_data_vis_params.get('vis_image_indices')

    try:
        opt_data = OptimizationData(
            valid_inlier_matches_for_opt_data,
            functools.partial(get_pose_for_timestamp, timestamps_us=ego_timestamps_us, poses=ego_poses, tolerance_us=interpolation_tolerance_us),
            initial_K_matrix, initial_D_coeffs, model_type,
            img_width, img_height,
            valid_query_timestamps_rec_us_for_opt_data,
            num_images_actually_in_opt_data,
            **opt_data_debug_kwargs # Pass the debug params here
        )
    except Exception as e:
        logging.error(f"Failed to create OptimizationData: {e}", exc_info=True)
        # traceback.print_exc() # For more detailed debugging
        return None, None, None, None, None
    
    # After opt_data is created, opt_data.num_opt_images is the count of images
    # that will actually have dt parameters in the optimization vector X.
    # This might be less than num_images_actually_in_opt_data if OptData filters further.
    num_dt_params_in_X = opt_data.num_opt_images

    if num_dt_params_in_X == 0 and opt_data.total_residuals == 0 : # Check if OptData ended up with nothing
        logging.error("OptimizationData processed zero images or zero residuals. Cannot optimize.")
        final_dt_map_for_return = {idx: 0.0 for idx in query_indices}
        return initial_T_ego_cam, initial_K_matrix, initial_D_coeffs, \
               final_dt_map_for_return, None

    dt_init_seconds = np.zeros(num_dt_params_in_X) # Size based on opt_data processing

    x0 = np.concatenate((xi_ego_cam_init, k_params_init, d_params_init, dt_init_seconds))
    # --- End Prepare Initial Guess x0 ---

    # --- Prepare Bounds ---
    bounds_low = [-np.inf] * 6  # For xi_ego_cam
    bounds_high = [np.inf] * 6

    # Default K bounds
    k_bounds_low_default = [ max(10.0, initial_K_matrix[0,0] * 0.7), max(10.0, initial_K_matrix[1,1] * 0.7), 0.0, 0.0, ]
    k_bounds_high_default = [ initial_K_matrix[0,0] * 1.3, initial_K_matrix[1,1] * 1.3, float(img_width -1), float(img_height -1), ]
    
    # Default D bounds
    if model_type == "KANNALA_BRANDT":
        d_b_low_def = [-0.5, -0.2, -0.1, -0.1] 
        d_b_high_def = [0.5,  0.2,  0.1,  0.1]
    elif model_type == "PINHOLE":
        d_b_low_def = [-0.8, -0.5, -0.01, -0.01, -0.5] # k1,k2,p1,p2,k3
        d_b_high_def = [0.8,  0.5,  0.01,  0.01,  0.5]
        if num_d_params != 5 and num_d_params > 0:
            d_b_low_def = [-1.0] * num_d_params
            d_b_high_def = [1.0] * num_d_params
    else:
        d_b_low_def = [-1.0] * num_d_params if num_d_params > 0 else []
        d_b_high_def = [1.0] * num_d_params if num_d_params > 0 else []

    cfg_bounds = intrinsic_bounds_config if intrinsic_bounds_config else {}
    k_bounds_low = cfg_bounds.get('k_low', k_bounds_low_default)
    k_bounds_high = cfg_bounds.get('k_high', k_bounds_high_default)
    d_bounds_low = cfg_bounds.get('d_low', d_b_low_def)
    d_bounds_high = cfg_bounds.get('d_high', d_b_high_def)

    if len(k_bounds_low) != NUM_K_PARAMS or len(k_bounds_high) != NUM_K_PARAMS:
        logging.warning(f"K bounds length mismatch. Using defaults.")
        k_bounds_low, k_bounds_high = k_bounds_low_default, k_bounds_high_default
    if num_d_params > 0 and (len(d_bounds_low) != num_d_params or len(d_bounds_high) != num_d_params) :
        logging.warning(f"D bounds length mismatch. Using defaults or wide bounds.")
        d_bounds_low = [-1.0] * num_d_params if len(d_b_low_def) != num_d_params else d_b_low_def
        d_bounds_high = [1.0] * num_d_params if len(d_b_high_def) != num_d_params else d_b_high_def


    bounds_low.extend(k_bounds_low)
    bounds_high.extend(k_bounds_high)
    if num_d_params > 0:
        bounds_low.extend(d_bounds_low)
        bounds_high.extend(d_bounds_high)

    bounds_low.extend([dt_bounds_seconds[0]] * num_dt_params_in_X) # Use num_dt_params_in_X
    bounds_high.extend([dt_bounds_seconds[1]] * num_dt_params_in_X) # Use num_dt_params_in_X
    bounds = (bounds_low, bounds_high)
    # --- End Prepare Bounds ---
    
    logging.info(f"Total number of parameters in X: {len(x0)}")
    logging.info(f"  Extrinsic (T_ego_cam): 6")
    logging.info(f"  Intrinsic K (fx,fy,cx,cy): {NUM_K_PARAMS}")
    logging.info(f"  Intrinsic D: {num_d_params}")
    logging.info(f"  Delta_t (per image processed by OptData): {num_dt_params_in_X}")
    logging.info(f"Total number of residuals: {opt_data.total_residuals}")

    try:
        initial_residuals = compute_residuals(x0, opt_data) # compute_residuals uses opt_data
        initial_cost = 0.5 * np.sum(initial_residuals**2)
        logging.info(f"Calculated Initial cost: {initial_cost:.4e}")
        if not np.isfinite(initial_cost) or initial_cost > 1e12: # Increased threshold for very bad initializations
            logging.error(f"Initial cost is non-finite or extremely large ({initial_cost:.2e})! Check inputs/bounds.")
            final_dt_map_for_return = {idx: 0.0 for idx in query_indices}
            return initial_T_ego_cam, initial_K_matrix, initial_D_coeffs, final_dt_map_for_return, None
    except Exception as e:
        logging.error(f"Failed to compute initial residuals/cost: {e}", exc_info=True)
        return None, None, None, None, None

    start_time = time.time()
    result_obj = None
    try:
        result_obj = least_squares(
            compute_residuals, x0, jac='2-point', bounds=bounds, method='trf',
            ftol=1e-8, xtol=1e-8, gtol=1e-8, loss=loss_function, verbose=verbose,
            max_nfev=1500 * len(x0), # Allow more evaluations
            args=(opt_data,) # Pass opt_data here
        )
        duration = time.time() - start_time
        logging.info(f"Optimization finished in {duration:.2f} seconds. Status: {result_obj.status} ({result_obj.message})")
        logging.info(f"Final cost: {result_obj.cost:.4e}. Optimality: {result_obj.optimality:.4e}")
    except Exception as e:
        logging.error(f"Optimization crashed: {e}", exc_info=True)
        final_dt_map_for_return = {idx: 0.0 for idx in query_indices}
        return initial_T_ego_cam, initial_K_matrix, initial_D_coeffs, final_dt_map_for_return, None

    refined_T_ego_cam = initial_T_ego_cam # Default to initial if something goes wrong with unpacking
    refined_K = initial_K_matrix
    refined_D = initial_D_coeffs

    if not result_obj.success and result_obj.status <= 0:
        logging.warning(f"Optimization reported failure or did not converge (status: {result_obj.status}).")
        if result_obj.cost > initial_cost * 1.5 and initial_cost > 1e-3:
            logging.error("Optimization significantly increased cost or failed badly. Reverting to initial parameters.")
            final_dt_map_for_return = {idx: 0.0 for idx in query_indices}
            return initial_T_ego_cam, initial_K_matrix, initial_D_coeffs, final_dt_map_for_return, result_obj
        else: # If failed but cost is not much worse, still use the (potentially bad) result.x
             logging.warning("Using result.x despite optimization failure/non-convergence.")


    # --- Unpack Refined Parameters ---
    refined_x = result_obj.x
    xi_ego_cam_refined = refined_x[:6]

    fx_r = refined_x[6]; fy_r = refined_x[7]; cx_r = refined_x[8]; cy_r = refined_x[9]
    refined_K = np.array([[fx_r, 0, cx_r], [0, fy_r, cy_r], [0, 0, 1]])

    idx_start_D_res = 6 + NUM_K_PARAMS
    idx_end_D_res = idx_start_D_res + num_d_params
    refined_D_flat = refined_x[idx_start_D_res:idx_end_D_res]
    refined_D = refined_D_flat

    idx_start_dt_res = idx_end_D_res
    # The number of dt parameters in refined_x is num_dt_params_in_X
    all_dt_refined_seconds_for_opt_batch = refined_x[idx_start_dt_res : idx_start_dt_res + num_dt_params_in_X]
    # --- End Unpack Refined Parameters ---

    try:
        refined_T_ego_cam = se3_to_SE3(xi_ego_cam_refined)
    except Exception as e:
        logging.error(f"Failed to convert refined xi_ego_cam to SE(3): {e}", exc_info=True)
        refined_T_ego_cam = initial_T_ego_cam # Fallback

    # Create the delta_t map for ALL original query_indices.
    final_refined_delta_t_map_seconds = {idx: 0.0 for idx in query_indices}
    
    # opt_data.image_indices contains the sorted list of image indices that were ACTUALLY processed by OptimizationData
    # opt_data.image_idx_to_dt_param_idx_map maps these image_indices to their parameter index in all_dt_refined_seconds_for_opt_batch
    for original_img_idx_processed in opt_data.image_indices: # Iterate through images opt_data knows about
        if original_img_idx_processed in opt_data.image_idx_to_dt_param_idx_map:
            dt_param_idx = opt_data.image_idx_to_dt_param_idx_map[original_img_idx_processed]
            if dt_param_idx < len(all_dt_refined_seconds_for_opt_batch):
                final_refined_delta_t_map_seconds[original_img_idx_processed] = all_dt_refined_seconds_for_opt_batch[dt_param_idx]
            else:
                logging.error(f"dt_param_idx {dt_param_idx} for image {original_img_idx_processed} is out of bounds for refined dt array (len {len(all_dt_refined_seconds_for_opt_batch)}). This indicates a logic error.")
        else:
             # This should not happen if image_indices are keys of image_idx_to_dt_param_idx_map
             logging.error(f"Image index {original_img_idx_processed} from opt_data.image_indices was not found in its own dt_param_map. Logic error.")


    logging.info(f"Refined T_ego_cam:\n{np.round(refined_T_ego_cam, 6)}")
    logging.info(f"Refined K_matrix:\n{np.round(refined_K, 4)}")
    if num_d_params > 0:
        logging.info(f"Refined D_coeffs (flat):\n{np.round(refined_D, 7)}")
    else:
        logging.info("No D_coeffs were optimized.")

    dt_values_for_log = [
        dt_val for idx, dt_val in final_refined_delta_t_map_seconds.items()
        if idx in opt_data.image_idx_to_dt_param_idx_map # Log stats only for those actually optimized
    ]
    if dt_values_for_log:
        logging.info(f"Refined Delta_t (seconds) stats (for {len(dt_values_for_log)} optimized images in this batch): "
                     f"Mean={np.mean(dt_values_for_log):.6f}s, Std={np.std(dt_values_for_log):.6f}s, "
                     f"Min={np.min(dt_values_for_log):.6f}s, Max={np.max(dt_values_for_log):.6f}s")

    optimality_threshold = 1e-1
    if result_obj and result_obj.optimality > optimality_threshold: # Check if result_obj exists
        logging.warning(f"High first-order optimality ({result_obj.optimality:.2e}) suggests result might be "
                        f"suboptimal or near a constraint boundary.")

    return refined_T_ego_cam, refined_K, refined_D, final_refined_delta_t_map_seconds, result_obj

def save_poses_to_colmap_format(poses_cam_map: dict, output_file: Path):
    logging.info(f"Saving {len(poses_cam_map)} poses to {output_file} in COLMAP format...")
    num_saved = 0
    num_skipped = 0
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open('w') as f:
            sorted_image_names = sorted(poses_cam_map.keys())
            for img_name in sorted_image_names:
                T_cam_map_mat = poses_cam_map.get(img_name)
                if T_cam_map_mat is None:
                    logging.warning(f"Pose for {img_name} is None. Skipping save.")
                    num_skipped += 1
                    continue
                try:
                    if not isinstance(T_cam_map_mat, np.ndarray) or T_cam_map_mat.shape != (4, 4):
                        raise ValueError("Pose matrix is not a 4x4 NumPy array")
                    R_mat = T_cam_map_mat[:3, :3]
                    if not np.allclose(R_mat.T @ R_mat, np.eye(3), atol=1e-4):
                         logging.warning(f"Rotation matrix for {img_name} is not orthogonal. Quaternion might be inaccurate. Skipping save.")
                         num_skipped += 1
                         continue
                    if not np.isclose(np.linalg.det(R_mat), 1.0, atol=1e-4):
                         logging.warning(f"Rotation matrix for {img_name} has determinant != 1 ({np.linalg.det(R_mat):.4f}). Quaternion might be inaccurate. Skipping save.")
                         num_skipped += 1
                         continue
                    q = R.from_matrix(R_mat).as_quat() 
                    qw, qx, qy, qz = q[3], q[0], q[1], q[2] 
                    t = T_cam_map_mat[:3, 3]
                    tx, ty, tz = t[0], t[1], t[2]
                    f.write(f"{img_name} {qw} {qx} {qy} {qz} {tx} {ty} {tz}\n")
                    num_saved += 1
                except ValueError as e_val:
                     logging.error(f"Data validation error for {img_name}: {e_val}. Skipping save.")
                     num_skipped += 1
                except Exception as e_save:
                    logging.error(f"Error formatting/saving pose for {img_name}: {e_save}", exc_info=True)
                    num_skipped += 1
        logging.info(f"Poses saved: {num_saved} successful, {num_skipped} skipped.")
    except IOError as e:
        logging.error(f"Error writing results file {output_file}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during results saving: {e}", exc_info=True)

def parse_value(value_str):
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
    configs = []
    current_config_data = None
    dict_stack = [] 
    lines = config_text.strip().split('\n')
    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#'): continue
        if line == 'config {':
            if dict_stack: 
                logging.warning(f"Found 'config {{' while already inside a block at line {line_num + 1}. Ignoring previous state.")
            current_config_data = {}
            dict_stack = [current_config_data] 
            continue
        if line == '}':
            if not dict_stack:
                logging.warning(f"Found unexpected '}}' at line {line_num + 1}. Ignoring.")
                continue
            dict_stack.pop() 
            if not dict_stack and current_config_data is not None:
                 configs.append(current_config_data)
                 current_config_data = None 
            continue 
        if not dict_stack:
            logging.warning(f"Found data outside expected block structure at line {line_num + 1}: '{line}'. Ignoring.")
            continue
        match_block = re.match(r'^(\w+)\s*\{$', line)
        if match_block:
            key = match_block.group(1)
            new_dict = {}
            parent_dict = dict_stack[-1] 
            if key in parent_dict:
                 logging.error(f"Duplicate block key '{key}' found at line {line_num + 1} within the same parent block. Parsing may be incorrect.")
            parent_dict[key] = new_dict   
            dict_stack.append(new_dict)   
            continue
        match_kv = re.match(r'^(\w+)\s*:\s*(.+)$', line)
        if match_kv:
            key = match_kv.group(1)
            value_str = match_kv.group(2).strip()
            value = parse_value(value_str)
            current_dict = dict_stack[-1] 
            if key in current_dict:
                 logging.warning(f"Duplicate key '{key}' found at line {line_num + 1}. Overwriting previous value '{current_dict[key]}'.")
            current_dict[key] = value     
            continue
        logging.warning(f"Could not parse line {line_num + 1}: '{line}'. Ignoring.")
    if dict_stack:
         logging.warning(f"Reached end of input but parsing stack is not empty: {len(dict_stack)} levels deep. Config file may be truncated or have missing braces.")
         if current_config_data is not None and dict_stack[0] is current_config_data:
              logging.warning("Appending potentially incomplete last config block.")
              configs.append(current_config_data)
    return configs

def get_camera_params_from_parsed(all_configs, target_camera_dev):
    target_config = None
    for config in all_configs:
        if config.get("camera_dev") == target_camera_dev:
            target_config = config
            break
    if target_config is None:
        logging.error(f"Camera '{target_camera_dev}' not found in the parsed configuration list.")
        return None
    logging.info(f"Found configuration for camera '{target_camera_dev}'. Extracting parameters...")
    try:
        params = target_config.get("parameters", {})
        extrinsic = params.get("extrinsic", {}).get("sensor_to_cam", {})
        pos = extrinsic.get("position", {})
        orient = extrinsic.get("orientation", {})
        intrinsic = params.get("intrinsic", {})
        pos_x = pos.get("x")
        pos_y = pos.get("y")
        pos_z = pos.get("z")
        quat_qx = orient.get("qx")
        quat_qy = orient.get("qy")
        quat_qz = orient.get("qz")
        quat_qw = orient.get("qw")
        img_width = intrinsic.get("img_width")
        img_height = intrinsic.get("img_height")
        f_x = intrinsic.get("f_x")
        f_y = intrinsic.get("f_y")
        o_x = intrinsic.get("o_x")
        o_y = intrinsic.get("o_y")
        k_1 = intrinsic.get("k_1")
        k_2 = intrinsic.get("k_2")
        k_3 = intrinsic.get("k_3") 
        k_4 = intrinsic.get("k_4") 
        p_1 = intrinsic.get("p_1") 
        p_2 = intrinsic.get("p_2") 
        model_type = intrinsic.get("model_type")
        required_base_vals = [pos_x, pos_y, pos_z, quat_qx, quat_qy, quat_qz, quat_qw,
                              img_width, img_height, f_x, f_y, o_x, o_y, model_type]
        required_dist_vals = []
        if model_type == "KANNALA_BRANDT":
             required_dist_vals = [k_1, k_2, k_3, k_4]
             if p_1 is None: p_1 = 0.0 
             if p_2 is None: p_2 = 0.0
        elif model_type == "PINHOLE":
             required_dist_vals = [k_1, k_2, p_1, p_2] # k3 is often part of this for OpenCV
             if k_3 is None: k_3 = 0.0 
             if k_4 is None: k_4 = 0.0
        elif model_type is None:
            logging.error(f"Missing 'model_type' in intrinsic parameters for {target_camera_dev}.")
            return None
        else:
            logging.warning(f"Unknown camera model '{model_type}'. Assuming KANNALA_BRANDT K1-K4 like validation for distortion.")
            required_dist_vals = [k_1, k_2, k_3, k_4]
            if p_1 is None: p_1 = 0.0
            if p_2 is None: p_2 = 0.0

        if any(v is None for v in required_base_vals + required_dist_vals):
             missing_details = {
                 "pos": (pos_x, pos_y, pos_z), "quat": (quat_qx, quat_qy, quat_qz, quat_qw),
                 "dims": (img_width, img_height), "focal": (f_x, f_y), "principal": (o_x, o_y),
                 "model": model_type, "k1k2": (k_1, k_2), "k3k4": (k_3, k_4), "p1p2": (p_1, p_2)
             }
             logging.error(f"Missing one or more required parameters for '{target_camera_dev}'. Check parsed data structure and required fields for model '{model_type}'.")
             logging.error(f"Extracted values (some may be None): {missing_details}")
             return None
        rotation = R.from_quat([quat_qx, quat_qy, quat_qz, quat_qw])
        R_mat = rotation.as_matrix()
        t_vec = np.array([pos_x, pos_y, pos_z])
        T_sensor_cam = np.eye(4)
        T_sensor_cam[:3, :3] = R_mat
        T_sensor_cam[:3, 3] = t_vec
        K_mat = np.array([[f_x, 0, o_x], [0, f_y, o_y], [0, 0, 1]]) # Renamed K to K_mat
        D_arr = [] # Renamed D to D_arr
        if model_type == "KANNALA_BRANDT":
             D_arr = np.array([k_1, k_2, k_3, k_4], dtype=np.float64)
        elif model_type == "PINHOLE":
             # OpenCV Pinhole often uses [k1, k2, p1, p2, k3, (k4, k5, k6 optional)]
             # Assuming up to k3 for now based on typical simple pinhole.
             D_arr = np.array([k_1, k_2, p_1, p_2, k_3], dtype=np.float64)
        else: 
             D_arr = np.array([k_1, k_2, k_3, k_4], dtype=np.float64) # Fallback, may not be correct
        # D_arr = D_arr.reshape(-1, 1) # Distortion coefficients are usually 1D array (1xN or Nx1)
                                      # OpenCV functions usually expect (N,) or (1,N) or (N,1)
        logging.info(f"Successfully extracted and formatted parameters for {target_camera_dev}.")
        return {
            'T_ego_cam': T_sensor_cam,
            'img_width': int(img_width), 
            'img_height': int(img_height), 
            'K': K_mat, # Use K_mat
            'D': D_arr,   # Use D_arr
            'model_type': model_type
        }
    except Exception as e:
        logging.error(f"Error formatting parameters for '{target_camera_dev}' after parsing: {e}")
        logging.error(traceback.format_exc())
        return None

@dataclass
class PnPResult:
    query_idx: int
    query_name: str
    success: bool
    num_inliers: int
    p2d_inliers: np.ndarray = None
    P3d_inliers: np.ndarray = None
    initial_T_map_cam: np.ndarray = None 

def run_pipeline(
    arg_config, # Contains args.steps, args.camera_name, args.num_joint_opt
    lidar_map_file: Path,
    query_image_dir: Path,
    query_image_list_file: Path,
    output_dir: Path,
    render_poses_list, # Initial poses for rendering (T_map_cam)
    ego_pose_file: Path,
    initial_T_ego_cam_guess: np.ndarray, # User-provided/fallback T_sensor_cam
    camera_intrinsics_matrix_ideal: np.ndarray, # K_ideal (for undistorted images)
    camera_distortion_array_sensor: np.ndarray, # D_sensor (physical distortion)
    image_width: int, image_height: int,
    camera_model_type: str, # Added this parameter
    camera_name_in_list: str,
    min_height: float = -2.0, voxel_size: float = 0.03, normal_radius: float = 0.15,
    normal_max_nn: int = 50, device: str = "auto",
    render_shading_mode: str = 'normal', render_point_size: float = 2,
    intensity_highlight_threshold: float = None,
    feature_conf='superpoint_aachen', matcher_conf='superglue',
    distance_threshold_px: float = 30.0,
    pnp_iterations: int = 500, pnp_reprojection_error: float = 5.0,
    pnp_confidence: float = 0.999999, pnp_min_inliers: int = 15,
    dt_bounds_de_inner_opt: tuple[float, float] = (-0.02, 0.02), # For DE inner loop
    dt_bounds_final_ls: tuple[float, float] = (-0.05, 0.05),     # For final LS
    final_ls_opt_verbose: int = 1, # Verbosity for final LS
    visualize_steps: bool = True,
    num_images_to_visualize: int = 3,
    visualize_map_point_size: float = 1,
    de_robust_loss_type: str = 'cauchy',
    de_robust_loss_scale: float = 5.0,
    loss_function_final_ls: str = 'cauchy', # For final LS
    intrinsic_bounds_config: dict = None,
    visualize_distortpoints_debug: bool = True,
    num_distortpoints_debug_images: int = 1,
    de_popsize_factor: int = 15, # DE param
    de_maxiter: int = 150,       # DE param
    de_workers: int = -1,         # DE param
    spatial_grid_size_hybrid: int = 25,
    max_pts_per_cell_hybrid: int = 1,
    min_total_pts_hybrid: int = 100 # Reduced min threshold for testing
):
    # ... (Setup output_dir, hloc_out_dir etc. - REMAINS THE SAME) ...
    output_dir = Path(output_dir)
    hloc_out_dir = output_dir / 'hloc'
    query_image_dir_undistorted = hloc_out_dir / 'query_images_undistorted'
    renders_out_dir = output_dir / 'renders'
    render_image_list_path = renders_out_dir / "render_list.txt"
    refined_poses_file = output_dir / f'refined_poses_{camera_name_in_list}.txt'
    refined_extrinsics_file = output_dir / f'refined_extrinsics_{camera_name_in_list}.txt'
    refined_delta_t_file = output_dir / f'refined_delta_t_{camera_name_in_list}.csv'
    vis_base_output_dir = hloc_out_dir / 'visualizations'
    mask_suffix = "_mask.png"
    mid_data_dir = output_dir / "mid_data_cache"

    query_image_dir = Path(query_image_dir)
    query_image_list_file = Path(query_image_list_file)
    lidar_map_file = Path(lidar_map_file)
    ego_pose_file = Path(ego_pose_file)

    output_dir.mkdir(parents=True, exist_ok=True)
    hloc_out_dir.mkdir(parents=True, exist_ok=True)
    query_image_dir_undistorted.mkdir(parents=True, exist_ok=True)
    renders_out_dir.mkdir(parents=True, exist_ok=True)
    vis_base_output_dir.mkdir(parents=True, exist_ok=True)
    mid_data_dir.mkdir(parents=True, exist_ok=True)

    feature_output_base_name = 'feats-superpoint-n4096-r1024' # Example
    features_filename = f"{feature_output_base_name}.h5"
    features_path = hloc_out_dir / features_filename
    matches_output_path = hloc_out_dir / 'distance_matches.h5'
    # masked_render_features_path = features_path # If applying masks modifies the original file
    vis_pnp_output_dir = vis_base_output_dir / 'pnp'
    vis_pnp_output_dir.mkdir(parents=True, exist_ok=True)


    logging.info("--- 0. Loading Ego Vehicle Poses ---")
    ego_timestamps_us, ego_poses_list = load_and_prepare_ego_poses(ego_pose_file)
    if ego_timestamps_us is None or ego_poses_list is None:
        logging.error(f"Failed to load ego poses from {ego_pose_file}. Aborting.")
        return
    # Ego interpolator (used by PnP initial guess and refinement)
    ego_interpolator_us_func = functools.partial(get_pose_for_timestamp, 
                                             timestamps_us=ego_timestamps_us, 
                                             poses=ego_poses_list, 
                                             tolerance_us=1) # Tight tolerance for interpolation

    # --- Initializations for variables used across steps ---
    processed_lidar_data_loaded = None
    rendered_views_info_loaded = None
    query_image_names = [] # List of image basenames (e.g., "12345.png")
    query_idx_to_name = {} # {0: "12345.png", 1: "12346.png", ...}
    query_name_to_idx = {} # {"12345.png": 0, ...}
    # query_timestamps_rec_us: {image_basename: timestamp_us} - from filename
    # query_timestamps_rec_us_indexed: {query_idx: timestamp_us}
    query_timestamps_rec_us, query_timestamps_rec_us_indexed = {}, {}
    
    # Final results to be populated
    final_refined_T_ego_cam = np.copy(initial_T_ego_cam_guess)
    final_refined_K_sensor = np.copy(camera_intrinsics_matrix_ideal) # Start K_sensor as K_ideal
    final_refined_D_sensor = np.copy(camera_distortion_array_sensor) if camera_distortion_array_sensor is not None else np.array([])
    final_delta_t_map_seconds = {} # {query_idx: dt_seconds}
    final_refined_poses_cam_map = {} # {image_basename: T_cam_map}

    # --- Step 1: Preprocessing & Rendering ---
    if not arg_config.steps or 1 in arg_config.steps:
        logging.info("--- Running Step 1: Preprocessing & Rendering ---")
        try:
            pcd_tensor, processed_lidar_data_runtime = preprocess_lidar_map(
                lidar_map_file, min_height, normal_radius, normal_max_nn, voxel_size, device
            )
            if pcd_tensor is None: raise RuntimeError("Preprocessing failed.")
        except Exception as e:
            logging.error(f"FATAL: Preprocessing failed: {e}", exc_info=True); return

        rendered_views_info_runtime = []
        if not render_poses_list: # render_poses_list are T_map_cam
             logging.warning("Render poses list is empty. Skipping rendering step.")
        else:
            with open(render_image_list_path, 'w') as f_list:
                for i, pose_map_cam_render in enumerate(render_poses_list):
                    render_name = f"render_{i:05d}"
                    logging.info(f"Rendering view {i+1}/{len(render_poses_list)} ({render_name})...")
                    # Render with K_ideal and D=None (since we render to an ideal pinhole view)
                    render_output = render_geometric_viewpoint_open3d(
                        pcd_tensor, processed_lidar_data_runtime, 
                        pose_map_cam_render, # T_map_cam
                        camera_intrinsics_matrix_ideal, # K_ideal
                        image_width, image_height, 
                        shading_mode=render_shading_mode,
                        point_size=render_point_size, 
                        intensity_highlight_threshold=intensity_highlight_threshold
                    )
                    if render_output:
                        # ... (save rendered outputs - geom, depth, mask) ...
                        geom_img_path = renders_out_dir / f"{render_name}.png"
                        depth_map_path = renders_out_dir / f"{render_name}_depth.npy"
                        mask_path = renders_out_dir / f"{render_name}_mask.png"
                        cv2.imwrite(str(geom_img_path), render_output['geometric_image'])
                        cv2.imwrite(str(mask_path), render_output['render_mask'])
                        np.save(str(depth_map_path), render_output['depth'])
                        f_list.write(f"{geom_img_path.name}\n")
                        rendered_views_info_runtime.append({
                            'name': render_name, 'geometric_image_path': geom_img_path,
                            'depth_map_path': depth_map_path, 'mask_path': mask_path,
                            'pose': render_output['pose'] # This is T_map_cam_render
                        })
                    else: logging.warning(f"Failed to render view {i} ({render_name})")
            if not rendered_views_info_runtime and render_poses_list:
                logging.error("FATAL: No views rendered successfully. Aborting."); return
        
        save_ok = save_processed_data(
            output_dir=mid_data_dir, 
            processed_lidar_data=processed_lidar_data_runtime, 
            rendered_views_info=rendered_views_info_runtime
        )
        if not save_ok: logging.warning("Failed to save intermediate processed data.")
        # Update loaded variables for subsequent steps if this step ran
        processed_lidar_data_loaded = processed_lidar_data_runtime
        rendered_views_info_loaded = rendered_views_info_runtime
    
    # --- Step 2: Feature Extraction & Masking ---
    if not arg_config.steps or 2 in arg_config.steps:

        if processed_lidar_data_loaded is None or rendered_views_info_loaded is None:
             logging.info("Loading processed LiDAR data and Rendered Views Info for Step 3...")
             processed_lidar_data_loaded, rendered_views_info_loaded = load_processed_data( output_dir=mid_data_dir, rebuild_kdtree=True )
             if processed_lidar_data_loaded is None: logging.error("Failed to load LiDAR data for PnP."); return
             # rendered_views_info_loaded can be None if no renders were made, this is handled next.
             if rendered_views_info_loaded is None and Path(render_image_list_path).exists():
                 logging.error("Render info file seems to be missing, but render list exists. Data inconsistency.") # Render list might be empty though
             elif rendered_views_info_loaded is None:
                 logging.warning("No rendered views info loaded. Distance matching and depth linking will be skipped.")

        logging.info("--- Running Step 2: Feature Extraction & Masking ---")
        # Undistort query images to an ideal pinhole view (using K_ideal, original D_sensor)
        # Features will be extracted from these undistorted images.
        logging.info(f"Undistorting query images (Model: {camera_model_type}). Output to: {query_image_dir_undistorted}")
        query_undistortion_ok = False
        if camera_model_type == "KANNALA_BRANDT":
            query_undistortion_ok = undistort_images_fisheye(
                image_list_path=query_image_list_file,
                original_image_dir=query_image_dir, # Original distorted images
                output_image_dir=query_image_dir_undistorted, # Output ideal images
                K=camera_intrinsics_matrix_ideal, # K of the sensor (approx K_ideal)
                D=camera_distortion_array_sensor, # D of the sensor
                new_K=camera_intrinsics_matrix_ideal, # Target K for undistorted (K_ideal)
                new_size=(image_width, image_height)
            )
        elif camera_model_type == "PINHOLE":
            query_undistortion_ok = True # Assume true, loop will set false on error
            try:
                raw_q_names = [line.strip() for line in query_image_list_file.read_text().splitlines() if line.strip()]
                for q_name in raw_q_names:
                    src_img_path = query_image_dir / q_name
                    dst_img_path = query_image_dir_undistorted / q_name
                    img = cv2.imread(str(src_img_path))
                    if img is None: query_undistortion_ok = False; continue
                    
                    # Undistort from K_ideal, D_sensor To K_ideal, D_none
                    # Note: For Pinhole, camera_intrinsics_matrix_ideal is K of sensor if D_sensor is also for it.
                    # If D_sensor is None/empty, it's already ideal.
                    undist_img = cv2.undistort(img, camera_intrinsics_matrix_ideal, 
                                               camera_distortion_array_sensor, # D_sensor
                                               None, camera_intrinsics_matrix_ideal) # Target K_ideal
                    cv2.imwrite(str(dst_img_path), undist_img)
            except Exception as e_undist: query_undistortion_ok = False; logging.error(f"Pinhole undistortion error: {e_undist}")
        
        if not query_undistortion_ok: logging.error("Query image undistortion failed. Aborting."); return
        
        # Extract features (Query from undistorted, Renders from their ideal renders)
        # (Feature extraction logic - REMAINS THE SAME)
        if features_path.exists(): logging.warning(f"Deleting existing features file: {features_path}"); features_path.unlink(missing_ok=True) # Py3.8+
        
        logging.info("Extracting features for UNDISTORTED Query Images...")
        conf_q = extract_features.confs[str(feature_conf)]
        extract_features.main(conf=conf_q, image_dir=query_image_dir_undistorted, image_list=query_image_list_file, export_dir=hloc_out_dir, feature_path=features_path )
        
        logging.info("Extracting features for Rendered Images...")
        if not render_image_list_path.exists() and rendered_views_info_loaded: # If renders were made
             logging.error(f"Render image list {render_image_list_path} not found, but renders exist. Aborting."); return
        elif not rendered_views_info_loaded: # No renders were made in step 1
             logging.warning("No rendered views from Step 1. Skipping render feature extraction and masking.")
        else: # Renders exist and list should exist
            conf_r = extract_features.confs[str(feature_conf)]
            extract_features.main(conf=conf_r, image_dir=renders_out_dir, image_list=render_image_list_path, export_dir=hloc_out_dir, feature_path=features_path )
            logging.info("Applying masks to rendered features...")
            apply_masks_to_features( feature_file_path=features_path, image_list_path=render_image_list_path, image_base_dir=renders_out_dir, mask_suffix=mask_suffix, neighborhood_size=2 )

        logging.info("Checking and fixing features format...")
        check_and_fix_features(features_path, conf_q.get('descriptor_dim', 256)) # Get dim from conf
        
        if visualize_steps:
            # (Visualization logic - REMAINS THE SAME)
            visualize_features( h5_feature_path=features_path, image_list_path=query_image_list_file, image_base_dir=query_image_dir_undistorted, vis_output_dir=vis_base_output_dir / 'query_undistorted', num_to_vis=num_images_to_visualize, prefix="query_undistorted_vis" )
            if rendered_views_info_loaded: # Only visualize if renders were made
                visualize_features( h5_feature_path=features_path, image_list_path=render_image_list_path, image_base_dir=renders_out_dir, vis_output_dir=vis_base_output_dir / 'render_masked', num_to_vis=num_images_to_visualize, prefix="render_masked_vis" )

    # --- Step 3: Matching, PnP & Hybrid Refinement ---
    if not arg_config.steps or 3 in arg_config.steps:
        logging.info("--- Running Step 3: Matching, PnP & Hybrid Refinement ---")
        
        # --- 3.1 Load Data (if not loaded by step 1) ---
        if processed_lidar_data_loaded is None or rendered_views_info_loaded is None:
             logging.info("Loading processed LiDAR data and Rendered Views Info for Step 3...")
             processed_lidar_data_loaded, rendered_views_info_loaded = load_processed_data( output_dir=mid_data_dir, rebuild_kdtree=True )
             if processed_lidar_data_loaded is None: logging.error("Failed to load LiDAR data for PnP."); return
             # rendered_views_info_loaded can be None if no renders were made, this is handled next.
             if rendered_views_info_loaded is None and Path(render_image_list_path).exists():
                 logging.error("Render info file seems to be missing, but render list exists. Data inconsistency.") # Render list might be empty though
             elif rendered_views_info_loaded is None:
                 logging.warning("No rendered views info loaded. Distance matching and depth linking will be skipped.")


        # --- 3.2 Matching (if renders exist) ---
        if rendered_views_info_loaded and Path(render_image_list_path).exists():
            logging.info("Running Distance-Based Feature Matching...")
            if not features_path.exists(): logging.error(f"Feature file missing: {features_path}. Cannot match."); return
            if matches_output_path.exists(): matches_output_path.unlink(missing_ok=True)
            
            match_by_distance( features_path=features_path, query_image_list_file=query_image_list_file, 
                               render_image_list_file=render_image_list_path, 
                               matches_output_path=matches_output_path, 
                               distance_threshold_px=distance_threshold_px )
        elif not rendered_views_info_loaded:
            logging.warning("No rendered views available. Skipping feature matching and depth linking. PnP will not be possible.")
            # If matching is essential, you might want to `return` here.
        else: # Render list path missing but info loaded implies an issue earlier
            logging.error(f"Render image list {render_image_list_path} not found, cannot match. Aborting PnP stage."); return

        # --- 3.3 Prepare Query Info (Names, Timestamps) ---
        # (REMAINS THE SAME as original script)
        logging.info("Reading query image list and parsing timestamps...")
        try:
            query_image_names.clear(); query_idx_to_name.clear(); query_name_to_idx.clear()
            query_timestamps_rec_us.clear(); query_timestamps_rec_us_indexed.clear()
            
            raw_query_names_from_list = [line.strip() for line in query_image_list_file.read_text().splitlines() if line.strip()]
            for i, name_from_list in enumerate(raw_query_names_from_list):
                try: 
                    ts_us = int(Path(name_from_list).stem)
                    query_image_names.append(name_from_list) # Keep order from list
                    query_idx_to_name[i] = name_from_list
                    query_name_to_idx[name_from_list] = i
                    query_timestamps_rec_us[name_from_list] = ts_us
                    query_timestamps_rec_us_indexed[i] = ts_us
                except ValueError: logging.warning(f"Could not parse timestamp from {name_from_list}. Skipping.")
            
            if not query_image_names: logging.error("No query images with parseable timestamps found."); return
        except Exception as e_qlist: logging.error(f"Error processing query list: {e_qlist}", exc_info=True); return


        # --- 3.4 Initial Visualizations (Map Proj using initial T_map_cam from render_poses_list) ---
        if visualize_steps and render_poses_list:
            logging.info(f"Generating Initial Map Projection Visualizations ({num_images_to_visualize} images)...")
            vis_count = 0
            # Iterate using query_idx_to_name to ensure we match render_poses_list indices
            for q_idx, q_name in query_idx_to_name.items():
                if vis_count >= num_images_to_visualize: break
                if q_idx >= len(render_poses_list): continue # Safety check

                T_map_cam_initial_render = render_poses_list[q_idx] # T_map_cam
                if T_map_cam_initial_render is None: continue
                try: T_cam_map_initial_render = np.linalg.inv(T_map_cam_initial_render)
                except np.linalg.LinAlgError: continue
                
                img_path_original_distorted = query_image_dir / q_name
                if not img_path_original_distorted.exists(): continue
                
                vis_proj_path = vis_pnp_output_dir / f"{Path(q_name).stem}_{camera_name_in_list}_initial_map_proj.jpg"
                visualize_map_projection(
                     str(img_path_original_distorted), processed_lidar_data_loaded,
                     camera_intrinsics_matrix_ideal, # Using K_ideal for sensor
                     camera_distortion_array_sensor, # Using D_sensor
                     camera_model_type, T_cam_map_initial_render, str(vis_proj_path),
                     point_size=int(visualize_map_point_size), filter_distance=50.0
                )
                vis_count +=1
            logging.info(f"Finished generating {vis_count} initial map projection visualizations.")

        # Determine num_d_params based on the initial D_sensor from config
        flat_initial_D_sensor_coeffs_cfg = pipeline_D_sensor.flatten() if pipeline_D_sensor is not None else np.array([])
        num_d_params_cfg = len(flat_initial_D_sensor_coeffs_cfg)

        # --- 3.5 PnP Loop (Requires successful matching and depth linking) ---
        all_pnp_results: list[PnPResult] = []
        if rendered_views_info_loaded and matches_output_path.exists(): # Only run if matching was done
            logging.info(f"Running PnP RANSAC (min_inliers: {pnp_min_inliers})...")
            nn_link_dist_thresh = voxel_size * 2.5 # Slightly increased
            
            for q_name in query_image_names: # Iterate through query images that had timestamps
                q_idx = query_name_to_idx[q_name]
                
                # Link matches for this query_name to 3D points
                # K_ideal is used because features/matches were on undistorted images / ideal renders
                query_kps_2d_undist, map_points_3d = link_matches_via_depth(
                    query_image_name=q_name, features_path=features_path, 
                    matches_path=matches_output_path, 
                    rendered_views_info=rendered_views_info_loaded, 
                    processed_lidar_data=processed_lidar_data_loaded, 
                    camera_intrinsics=camera_intrinsics_matrix_ideal, # K_ideal for linking
                    nn_distance_threshold=nn_link_dist_thresh,
                    max_depth_value=100.0
                )

                if query_kps_2d_undist.shape[0] < pnp_min_inliers:
                    all_pnp_results.append(PnPResult(q_idx, q_name, False, query_kps_2d_undist.shape[0]))
                    continue
                
                # PnP uses K_ideal and D=None (implicitly, as points are already on ideal image)
                try:
                    success_pnp, rvec, tvec, inliers_pnp_indices = cv2.solvePnPRansac(
                        map_points_3d.astype(np.float32), 
                        query_kps_2d_undist.astype(np.float32), 
                        camera_intrinsics_matrix_ideal, # K_ideal for PnP
                        None, # D=None for PnP on undistorted points
                        iterationsCount=pnp_iterations, 
                        reprojectionError=pnp_reprojection_error, 
                        confidence=pnp_confidence, 
                        flags=getattr(cv2, 'SOLVEPNP_SQPNP', 0)
                    )
                    num_found_inliers = len(inliers_pnp_indices) if inliers_pnp_indices is not None else 0
                    
                    if success_pnp and num_found_inliers >= pnp_min_inliers:
                        p2d_inliers_undist = query_kps_2d_undist[inliers_pnp_indices.flatten()]
                        P3d_inliers_map = map_points_3d[inliers_pnp_indices.flatten()]
                        
                        R_mat_cam_map, _ = cv2.Rodrigues(rvec)
                        T_cam_map_pnp_mat = np.eye(4)
                        T_cam_map_pnp_mat[:3,:3] = R_mat_cam_map
                        T_cam_map_pnp_mat[:3,3] = tvec.flatten()
                        T_map_cam_pnp_mat = np.linalg.inv(T_cam_map_pnp_mat) # This is T_map_cam

                        all_pnp_results.append(PnPResult(q_idx, q_name, True, num_found_inliers, 
                                                         p2d_inliers_undist, P3d_inliers_map, 
                                                         T_map_cam_pnp_mat)) # Store T_map_cam
                    else:
                        all_pnp_results.append(PnPResult(q_idx, q_name, False, num_found_inliers))
                except Exception as e_pnp:
                    logging.error(f"Error during PnP for {q_name}: {e_pnp}", exc_info=True)
                    all_pnp_results.append(PnPResult(q_idx, q_name, False, 0))
        else:
            logging.warning("Skipping PnP loop as prerequisites (rendered_views_info or matches_output) are missing.")


        successful_pnp_results = [res for res in all_pnp_results if res.success]
        if not successful_pnp_results:
            logging.error("PnP failed for ALL images (or prerequisites missing). Cannot proceed with refinement. Will save initial guesses.")
            # Populate final_delta_t_map with 0s for all query images
            for q_idx_fail in query_idx_to_name.keys(): final_delta_t_map_seconds[q_idx_fail] = 0.0
             # Populate final_refined_poses_cam_map with initial render poses (if available)
            if render_poses_list:
                for q_idx_fail, q_name_fail in query_idx_to_name.items():
                    if q_idx_fail < len(render_poses_list) and render_poses_list[q_idx_fail] is not None:
                        try: final_refined_poses_cam_map[q_name_fail] = np.linalg.inv(render_poses_list[q_idx_fail])
                        except: pass
            # Skip to saving results (Step 4)
        else: # PnP was successful for at least one image
            logging.info(f"PnP successful for {len(successful_pnp_results)} / {len(query_image_names)} images.")

            # --- 3.6 Hybrid Refinement ---
            # Use PnP results to get a better initial T_ego_cam for DE
            pnp_derived_T_ego_cam = get_pnp_derived_initial_T_ego_cam(
                successful_pnp_results, query_timestamps_rec_us_indexed, 
                ego_interpolator_us_func, initial_T_ego_cam_guess
            )
            
            # Prepare inlier_matches_map_undistorted_ideal for refinement function
            # {query_idx: (p2d_UNDISTORTED_ideal_kps, P3d_world)}
            # These p2d_inliers are from PnP, already in the K_ideal frame.
            refinement_inlier_matches = {
                res.query_idx: (res.p2d_inliers, res.P3d_inliers) 
                for res in successful_pnp_results
            }
            refinement_query_indices = [res.query_idx for res in successful_pnp_results]
            # Filter query_timestamps_rec_us_indexed for only those in refinement_query_indices
            refinement_query_timestamps_rec_us = {
                idx: ts for idx, ts in query_timestamps_rec_us_indexed.items() if idx in refinement_query_indices
            }
            
            # Prepare debug visualization parameters for the final LS part of hybrid refinement
            opt_data_debug_params_for_ls = None
            if visualize_distortpoints_debug and query_idx_to_name and successful_pnp_results:
                indices_for_ls_opt_data_vis = [res.query_idx for res in successful_pnp_results[:num_distortpoints_debug_images]]
                if indices_for_ls_opt_data_vis:
                    opt_data_debug_params_for_ls = {
                        'image_idx_to_name_map': query_idx_to_name,
                        'query_image_dir': query_image_dir, # Original distorted images
                        'output_vis_dir': vis_base_output_dir / 'hybrid_opt_data_LS_debug',
                        'vis_image_indices': indices_for_ls_opt_data_vis
                    }
                    (vis_base_output_dir / 'hybrid_opt_data_LS_debug').mkdir(parents=True, exist_ok=True)

            if refinement_query_indices:
                logging.info(f"--- Running Hybrid Refinement on {len(refinement_query_indices)} PnP-successful Images ---")
                
                # Prepare debug viz params for OptData creation within hybrid refinement's LS stage
                # AND for DE target pre-calculation step
                debug_params_for_hybrid_refinement = {}
                if visualize_distortpoints_debug and query_idx_to_name and successful_pnp_results:
                    # General params
                    debug_params_for_hybrid_refinement['image_idx_to_name_map'] = query_idx_to_name
                    debug_params_for_hybrid_refinement['query_image_dir'] = query_image_dir

                    # For DE pre-calculated target visualization
                    de_targets_vis_dir_name = 'DE_precalculated_targets_debug'
                    de_targets_vis_dir_path = vis_base_output_dir / de_targets_vis_dir_name
                    de_targets_vis_indices = [res.query_idx for res in successful_pnp_results[:num_distortpoints_debug_images]]
                    debug_params_for_hybrid_refinement['output_vis_dir_DE_targets'] = de_targets_vis_dir_path
                    debug_params_for_hybrid_refinement['vis_image_indices_DE_targets'] = de_targets_vis_indices
                    
                    # For LS OptData internal target visualization
                    ls_targets_vis_dir_name = 'LS_OptData_targets_debug'
                    ls_targets_vis_dir_path = vis_base_output_dir / ls_targets_vis_dir_name
                    ls_targets_vis_indices = [res.query_idx for res in successful_pnp_results[:num_distortpoints_debug_images]] # Can be same or different
                    debug_params_for_hybrid_refinement['output_vis_dir_LS_targets_subdir'] = ls_targets_vis_dir_name # Subdir name for OptData
                    debug_params_for_hybrid_refinement['output_vis_dir_base'] = vis_base_output_dir # Base for OptData path construction
                    debug_params_for_hybrid_refinement['vis_image_indices_LS_targets'] = ls_targets_vis_indices


                opt_T_ego_cam, opt_K_sensor, opt_D_sensor, opt_delta_t_map_s, _ = refine_all_parameters_hybrid(
                    initial_T_ego_cam_from_config=pipeline_initial_T_ego_cam,
                    pnp_derived_T_ego_cam_guess=pnp_derived_T_ego_cam,
                    inlier_matches_map_undistorted_ideal_ALL_PNP_SUCCESS=refinement_inlier_matches, # Pass PnP success here
                    ego_timestamps_us=ego_timestamps_us, ego_poses=ego_poses_list,
                    query_indices_for_opt_ALL_PNP_SUCCESS=refinement_query_indices, # Use PnP success indices
                    query_timestamps_rec_us=refinement_query_timestamps_rec_us, # Pass indexed map
                    K_ideal_plane_definition=pipeline_K_ideal,
                    K_initial_sensor_cfg=pipeline_K_ideal, # Initial K_sensor is K_ideal from config
                    initial_D_sensor_coeffs_cfg=pipeline_D_sensor,
                    model_type=camera_model_type, img_width=image_width, img_height=image_height,
                    dt_bounds_seconds_de_inner_opt=dt_bounds_de_inner_opt,
                    de_robust_loss_type=de_robust_loss_type, de_robust_loss_scale=de_robust_loss_scale,
                    de_popsize_factor=de_popsize_factor, de_maxiter=de_maxiter, de_workers=de_workers,
                    dt_bounds_seconds_final_ls=dt_bounds_final_ls, loss_function_final_ls=loss_function_final_ls,
                    final_ls_verbose=final_ls_opt_verbose,
                    interpolation_tolerance_us=1, intrinsic_bounds_config=intrinsic_bounds_config,
                    spatial_distribution_grid_size=spatial_grid_size_hybrid,
                    max_points_per_grid_cell=max_pts_per_cell_hybrid,
                    min_total_distributed_points=min_total_pts_hybrid,
                    debug_opt_data_vis_params=debug_params_for_hybrid_refinement
                )

                # debug_params_for_staged_refinement = {}
                # if visualize_distortpoints_debug and query_idx_to_name and successful_pnp_results:
                #     # ... (setup debug_params_for_staged_refinement as before) ...
                #     debug_params_for_staged_refinement['image_idx_to_name_map'] = query_idx_to_name
                #     debug_params_for_staged_refinement['query_image_dir'] = query_image_dir
                #     base_vis_dir_for_staged = vis_base_output_dir / "staged_refinement_debug"
                #     debug_params_for_staged_refinement['output_vis_dir_base'] = base_vis_dir_for_staged
                #     global_targets_vis_indices = [res.query_idx for res in successful_pnp_results[:num_distortpoints_debug_images]]
                #     debug_params_for_staged_refinement['vis_image_indices_global_targets'] = global_targets_vis_indices
                #     final_ls_optdata_vis_indices = [res.query_idx for res in successful_pnp_results[:num_distortpoints_debug_images]]
                #     debug_params_for_staged_refinement['vis_image_indices_final_ls_optdata'] = final_ls_optdata_vis_indices


                # # Corrected call to refine_all_parameters_staged
                # opt_T_ego_cam, opt_K_sensor, opt_D_sensor, opt_delta_t_map_s, _ = refine_all_parameters_staged(
                #     initial_T_ego_cam_from_config=pipeline_initial_T_ego_cam, 
                #     pnp_derived_T_ego_cam_guess=pnp_derived_T_ego_cam, 
                #     inlier_matches_map_undistorted_ideal=refinement_inlier_matches,
                #     ego_timestamps_us=ego_timestamps_us,
                #     ego_poses=ego_poses_list,
                #     query_indices_for_opt=refinement_query_indices,
                #     query_timestamps_rec_us=refinement_query_timestamps_rec_us,
                    
                #     # Parameters defining the input data and initial sensor config:
                #     K_ideal_plane_definition=pipeline_K_ideal,        # K of the undistorted plane (where PnP points live)
                #     K_initial_sensor_cfg=pipeline_K_ideal,            # Initial K of the physical sensor (from cameras.cfg, often same as K_ideal if D is for it)
                #     initial_D_sensor_coeffs_cfg=pipeline_D_sensor,    # Initial D of the physical sensor (from cameras.cfg)
                    
                #     model_type=pipeline_model_type,
                #     img_width=pipeline_img_w, img_height=pipeline_img_h,
                    
                #     # Stage 1 DE Extrinsics Params
                #     stage1_de_xi_bounds_abs_offset=np.array(cfg_intrinsic_bounds.get('de_xi_abs_offset_stage1', [0.3, 0.3, 0.3, 0.5, 0.5, 0.5])),
                #     stage1_de_popsize_factor=de_popsize_factor, # from args
                #     stage1_de_maxiter=de_maxiter,               # from args
                    
                #     # Stage 2 DE Intrinsics Params
                #     stage2_de_k_bounds_rel_offset=tuple(cfg_intrinsic_bounds.get('de_k_rel_offset_stage2', (0.99, 1.01))),
                #     stage2_de_cxcy_bounds_abs_offset_px=float(cfg_intrinsic_bounds.get('de_k_cxcy_abs_offset_px_stage2', 5.0)),
                #     # Ensure stage2_de_d_bounds_abs_offset is sized correctly using num_d_params_cfg
                #     stage2_de_d_bounds_abs_offset=np.array(cfg_intrinsic_bounds.get('de_d_abs_offset_stage2', [0.05, 0.03, 0.001, 0.001, 0.01])[:num_d_params_cfg] if num_d_params_cfg > 0 else []),
                #     stage2_de_popsize_factor=de_popsize_factor, # from args
                #     stage2_de_maxiter=de_maxiter,               # from args

                #     # Shared DE params
                #     de_robust_loss_type=de_robust_loss_type,
                #     de_robust_loss_scale=de_robust_loss_scale,
                #     de_workers=de_workers,                      # from args

                #     # Stage 3 (Final LS) params
                #     dt_bounds_seconds_final_ls=dt_bounds_final_ls,
                #     loss_function_final_ls=loss_function_final_ls,
                #     final_ls_verbose=final_ls_opt_verbose,
                #     final_ls_xi_bounds_abs_offset=np.array(cfg_intrinsic_bounds.get('ls_xi_abs_offset', [0.3, 0.3, 0.3, 0.5, 0.5, 0.5])),
                #     final_ls_k_bounds_rel_offset=tuple(cfg_intrinsic_bounds.get('ls_k_rel_offset', (0.9, 1.1))),
                #     final_ls_cxcy_bounds_abs_offset_px=float(cfg_intrinsic_bounds.get('ls_k_cxcy_abs_offset_px', 30.0)),
                #     final_ls_d_bounds_abs_offset_scale=float(cfg_intrinsic_bounds.get('ls_d_abs_offset_scale', 0.05)),
                #     final_ls_d_bounds_abs_offset_const=float(cfg_intrinsic_bounds.get('ls_d_abs_offset_const', 0.01)),
                    
                #     interpolation_tolerance_us=1, 
                #     intrinsic_bounds_config=intrinsic_bounds_config, 
                #     debug_opt_data_vis_params=debug_params_for_staged_refinement
                # )
                
                if opt_T_ego_cam is not None:
                    final_refined_T_ego_cam = opt_T_ego_cam
                    final_refined_K_sensor = opt_K_sensor
                    final_refined_D_sensor = opt_D_sensor 
                    for q_idx, dt_val in opt_delta_t_map_s.items():
                        final_delta_t_map_seconds[q_idx] = dt_val 
                else:
                    logging.error("Hybrid refinement FAILED. Using initial guesses for final T_ego_cam, K_sensor, D_sensor.")
                    # K_sensor and D_sensor remain pipeline_K_ideal, pipeline_D_sensor
                    for q_idx_fail in refinement_query_indices:
                        final_delta_t_map_seconds[q_idx_fail] = 0.0 # No dt refinement
            else: # No images for refinement (e.g. all PnP failed)
                 logging.warning("No images available for hybrid refinement after PnP. Using initial guesses.")
                 # K_sensor and D_sensor remain pipeline_K_ideal, pipeline_D_sensor
                 for q_idx_fail_all in query_idx_to_name.keys(): # Set dt=0 for all if no PnP success at all for refinement
                    final_delta_t_map_seconds[q_idx_fail_all] = 0.0
        
        # --- 3.7 Post-Refinement: Set dt=0 for PnP-failed images, Calculate Final Poses ---
        for pnp_res in all_pnp_results: # Iterate all PnP attempts
            if pnp_res.query_idx not in final_delta_t_map_seconds:
                # This applies to PnP-failed images, or if refinement itself failed for PnP-successful ones
                final_delta_t_map_seconds[pnp_res.query_idx] = 0.0
        
        # Calculate final poses T_cam_map using final_refined_T_ego_cam and final_delta_t_map_seconds
        logging.info(f"Calculating Final Poses using refined T_ego_cam, K_sensor, D_sensor and delta_t map...")
        num_final_poses_calc = 0
        for q_idx, q_name in query_idx_to_name.items():
            rec_ts_us = query_timestamps_rec_us_indexed.get(q_idx)
            delta_t_s = final_delta_t_map_seconds.get(q_idx)

            if rec_ts_us is None or delta_t_s is None:
                logging.warning(f"Missing timestamp or delta_t for {q_name}. Cannot calculate final pose."); continue
            
            true_ts_us = float(rec_ts_us) + (delta_t_s * 1_000_000.0)
            T_map_ego_final = ego_interpolator_us_func(true_ts_us)
            if T_map_ego_final is None:
                logging.warning(f"Could not interpolate final ego pose for {q_name}."); continue
            
            T_map_cam_final = T_map_ego_final @ final_refined_T_ego_cam # final_refined_T_ego_cam is T_sensor_cam
            try:
                final_refined_poses_cam_map[q_name] = np.linalg.inv(T_map_cam_final) # Store T_cam_map
                num_final_poses_calc += 1
            except np.linalg.LinAlgError:
                logging.warning(f"Could not invert final pose T_map_cam for {q_name}.")
        logging.info(f"Calculated {num_final_poses_calc} final refined poses (T_cam_map).")


        # --- Saving Results & Final Visualizations ---
        logging.info("--- Running Step 4: Saving Final Results & Visualizations ---")
        
        # Save refined T_ego_cam (Sensor to Camera Transformation)
        logging.info(f"Saving refined T_ego_cam to {refined_extrinsics_file}")
        logging.info(f"Refined T_ego_cam value:\n{np.round(final_refined_T_ego_cam, 6)}")
        with open(refined_extrinsics_file, 'w') as f:
            f.write("# Refined T_ego_cam (Sensor to Camera Transformation) - Row Major\n")
            for row in final_refined_T_ego_cam: f.write(" ".join(map(str, row)) + "\n")

        # Save refined K_sensor and D_sensor
        refined_K_file = output_dir / f'refined_intrinsics_K_sensor_{camera_name_in_list}.txt'
        refined_D_file = output_dir / f'refined_intrinsics_D_sensor_{camera_name_in_list}.txt'
        logging.info(f"Saving refined K_sensor to {refined_K_file}:\n{np.round(final_refined_K_sensor,4)}")
        np.savetxt(refined_K_file, final_refined_K_sensor, fmt='%.18e')
        logging.info(f"Saving refined D_sensor to {refined_D_file}: {np.round(final_refined_D_sensor,6)}")
        np.savetxt(refined_D_file, final_refined_D_sensor, fmt='%.18e')
        with open(refined_D_file, 'a') as f: f.write(f"\n# model_type: {camera_model_type}\n")

        # Save refined delta_t map
        if final_delta_t_map_seconds:
            # (Saving logic for delta_t - REMAINS THE SAME)
            logging.info(f"Saving refined delta_t map ({len(final_delta_t_map_seconds)} entries) to {refined_delta_t_file}")
            try:
                with open(refined_delta_t_file, 'w', newline='') as f_dt:
                    writer_dt = csv.writer(f_dt); writer_dt.writerow(['image_name', 'delta_t_seconds'])
                    sorted_indices_dt = sorted(final_delta_t_map_seconds.keys(), key=lambda idx_dt: query_idx_to_name.get(idx_dt, str(idx_dt)))
                    for q_idx_dt in sorted_indices_dt:
                        writer_dt.writerow([query_idx_to_name.get(q_idx_dt, f"idx_{q_idx_dt}"), f"{final_delta_t_map_seconds[q_idx_dt]:.9f}"])
            except Exception as e_dt_save: logging.error(f"Failed to save delta_t map: {e_dt_save}", exc_info=True)
        else: logging.warning("Final delta_t map is empty. Cannot save delta_t CSV.")

        # Save final refined poses (T_cam_map)
        logging.info(f"Saving {len(final_refined_poses_cam_map)} final refined poses (T_cam_map) to {refined_poses_file}")
        if final_refined_poses_cam_map:
            save_poses_to_colmap_format(final_refined_poses_cam_map, refined_poses_file)
        else: logging.warning("No refined poses (T_cam_map) to save.")

        # Final Visualizations (Map Projection using final_refined_K_sensor, D_sensor, T_cam_map)
        if visualize_steps and final_refined_poses_cam_map and processed_lidar_data_loaded:
            # (Visualization logic - REMAINS THE SAME, but uses final_refined_K_sensor and final_refined_D_sensor)
            logging.info(f"Generating Final Map Projection Visualizations ({num_images_to_visualize} images)...")
            vis_count = 0
            sorted_final_q_names = sorted(final_refined_poses_cam_map.keys())
            for q_name_final_vis in sorted_final_q_names:
                 if vis_count >= num_images_to_visualize: break
                 T_cam_map_final_vis = final_refined_poses_cam_map.get(q_name_final_vis)
                 if T_cam_map_final_vis is None: continue
                 
                 img_path_original_distorted_vis = query_image_dir / q_name_final_vis
                 if not img_path_original_distorted_vis.exists(): continue
                 
                 vis_proj_path_final = vis_pnp_output_dir / f"{Path(q_name_final_vis).stem}_{camera_name_in_list}_refined_map_proj.jpg"
                 visualize_map_projection(
                     str(img_path_original_distorted_vis), processed_lidar_data_loaded,
                     final_refined_K_sensor, final_refined_D_sensor,
                     camera_model_type, T_cam_map_final_vis, str(vis_proj_path_final),
                     point_size=int(visualize_map_point_size), filter_distance=50.0
                 )
                 vis_count+=1
            logging.info(f"Finished generating {vis_count} final map projection visualizations.")
        elif visualize_steps:
            logging.warning("Skipping final visualizations due to missing refined poses or LiDAR data.")

    logging.info(f"Pipeline execution finished for camera {camera_name_in_list}.")

# ==============================================
# Example Main Block to Run the Pipeline
# ==============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hybrid camera extrinsic and time offset refinement pipeline.')
    parser.add_argument('-s', '--steps', nargs='*', type=int, default=None, help='List of steps to run (e.g., 1 2 3)')
    parser.add_argument('-c', '--camera_name', type=str, default=None, required=True, help='Camera name to be processed (must match a name in cameras.cfg).')
    # DE specific parameters
    parser.add_argument('--de_popsize_factor', type=int, default=20, help='Differential Evolution popsize factor (popsize = factor * num_params).')
    parser.add_argument('--de_maxiter', type=int, default=500, help='Differential Evolution maximum iterations.')
    parser.add_argument('--de_workers', type=int, default=-1, help='Differential Evolution workers (-1 for all cores).')
    # Add other general args if needed (e.g., paths, already handled by fixed vars below)
    args = parser.parse_args()

    # --- Configuration (mostly fixed paths for this example) ---
    CAMERA_NAME_ARG = args.camera_name # From command line
    INPUT_PATH = Path("input_tmp") # Example base input path
    LIDAR_MAP_PATH = INPUT_PATH / "whole_map.pcd"
    QUERY_IMG_DIR_BASE = INPUT_PATH # Query images are in INPUT_PATH / CAMERA_NAME_ARG
    QUERY_IMG_LIST_TPL = INPUT_PATH / "query_image_list_{camera_name}.txt"
    OUTPUT_DIR_BASE = Path("output") # New output directory for hybrid results
    
    # Initial poses for rendering (T_map_cam)
    INIT_POSE_CSV_PATH = INPUT_PATH / "null_0_0_0_local2global_cam_pose.csv" # CSV for T_map_cam
    # Ego vehicle poses (T_map_ego)
    EGO_POSE_CSV_PATH = INPUT_PATH / "null_0_0_0_local2global_pose.csv"
    # Camera configuration file
    CAM_CONFIG_FILE_PATH = INPUT_PATH / "cameras.cfg"

    # --- Parse Camera Configuration ---
    try:
        with open(CAM_CONFIG_FILE_PATH, "r") as f_cfg: config_text = f_cfg.read()
    except FileNotFoundError: logging.error(f"Camera config file {CAM_CONFIG_FILE_PATH} not found!"); exit(1)
    
    all_parsed_configs = parse_camera_configs(config_text)
    if not all_parsed_configs: logging.error("Failed to parse camera configurations."); exit(1)

    cam_params = get_camera_params_from_parsed(all_parsed_configs, CAMERA_NAME_ARG)
    if not cam_params: logging.error(f"Could not get parameters for camera {CAMERA_NAME_ARG}."); exit(1)

    # Assign parsed parameters
    # T_sensor_cam (or T_ego_cam if sensor frame = ego frame)
    # This is the initial guess for the extrinsic transformation
    # For the pipeline, it's called initial_T_ego_cam_guess
    pipeline_initial_T_ego_cam = cam_params['T_ego_cam']
    pipeline_img_w = cam_params['img_width']
    pipeline_img_h = cam_params['img_height']
    # K_ideal: The K matrix for the ideal, undistorted camera model.
    # We assume the parsed 'K' from config is this K_ideal.
    pipeline_K_ideal = cam_params['K']
    # D_sensor: The distortion parameters of the physical sensor.
    pipeline_D_sensor = cam_params['D']
    pipeline_model_type = cam_params['model_type']

    print("\n--- Parsed Values ---")
    print(f"Camera Model: {pipeline_model_type}")
    print(f"IMG_W: {pipeline_img_w}, IMG_H: {pipeline_img_h}")
    print(f"K_MATRIX:\n{pipeline_K_ideal}")
    print(f"D_FISHEYE:\n{pipeline_D_sensor}")
    print(f"INITIAL_T_EGO_CAM_GUESS (T_sensor_cam):\n{pipeline_initial_T_ego_cam}")

    # --- Prepare query image list path ---
    current_query_img_list_path = Path(str(QUERY_IMG_LIST_TPL).format(camera_name=CAMERA_NAME_ARG))
    current_query_img_dir = QUERY_IMG_DIR_BASE / CAMERA_NAME_ARG
    current_output_dir = OUTPUT_DIR_BASE / CAMERA_NAME_ARG

    # --- Get Initial Poses for Rendering (T_map_cam) ---
    # These are often ground truth or from another localization system
    initial_render_poses_map_cam = get_init_poses(INIT_POSE_CSV_PATH, CAMERA_NAME_ARG, current_query_img_list_path)
    if not initial_render_poses_map_cam:
        logging.warning(f"No initial render poses found for {CAMERA_NAME_ARG}. Rendering step will be skipped if it relies on this.")
        initial_render_poses_map_cam = [] # Ensure it's a list

    # --- File/Directory Checks ---
    if not LIDAR_MAP_PATH.exists(): raise FileNotFoundError(f"LiDAR map not found: {LIDAR_MAP_PATH}")
    if not current_query_img_dir.is_dir(): raise NotADirectoryError(f"Query image directory not found: {current_query_img_dir}")
    if not current_query_img_list_path.exists(): raise FileNotFoundError(f"Query image list not found: {current_query_img_list_path}")
    if not EGO_POSE_CSV_PATH.exists(): raise FileNotFoundError(f"Ego pose CSV not found: {EGO_POSE_CSV_PATH}")
    if not INIT_POSE_CSV_PATH.exists(): logging.warning(f"Initial pose CSV for rendering not found: {INIT_POSE_CSV_PATH}")


    # --- Define Intrinsic Bounds Configuration (example) ---
    # This should be adapted based on expected model_type and D_sensor length
    cfg_intrinsic_bounds = {}
    cfg_intrinsic_bounds['de_xi_abs_offset'] = [0.785, 0.785, 0.785, 0.8, 0.8, 0.8]
    cfg_intrinsic_bounds['de_k_rel_offset_low'] = 0.999
    cfg_intrinsic_bounds['de_k_rel_offset_high'] = 1.001
    cfg_intrinsic_bounds['de_k_cxcy_abs_offset_px'] = 1e-5 * max(pipeline_img_w, pipeline_img_h)
    default_d_offset_val = 0.15 # Default offset if model specific not found
    d_initial_flat_sensor = pipeline_D_sensor.flatten() if pipeline_D_sensor is not None else np.array([])
    num_d_params = len(d_initial_flat_sensor)
    d_offset_de_specific = { # num_d_params as part of key
        f"KANNALA_BRANDT_{4}": np.array([1e-5, 1e-5, 1e-5, 1e-5]),
        f"PINHOLE_{4}": np.array([0.25, 0.15, 0.008, 0.008]), 
        f"PINHOLE_{5}": np.array([0.25, 0.15, 0.008, 0.008, 0.15]) 
    }.get(f"{pipeline_model_type}_{num_d_params}", np.full(num_d_params, default_d_offset_val))
    cfg_intrinsic_bounds['de_d_abs_offset'] = d_offset_de_specific

    cfg_intrinsic_bounds['ls_xi_abs_offset'] = cfg_intrinsic_bounds['de_xi_abs_offset']
    cfg_intrinsic_bounds['ls_k_rel_offset_low'] = cfg_intrinsic_bounds['de_k_rel_offset_low']
    cfg_intrinsic_bounds['ls_k_rel_offset_high'] = cfg_intrinsic_bounds['de_k_rel_offset_high']
    cfg_intrinsic_bounds['ls_k_cxcy_abs_offset_px'] = cfg_intrinsic_bounds['de_k_cxcy_abs_offset_px']
    cfg_intrinsic_bounds['ls_d_abs_offset_scale'] = 1e-5
    cfg_intrinsic_bounds['ls_d_abs_offset_const'] = 1e-5
    
    # Dynamic point sizes and thresholds based on image resolution (example)
    reso_ratio = float(pipeline_img_w) / 1920.0 # Assuming 1920px width is baseline
    dynamic_render_point_size = max(1.0, 4.0 * reso_ratio)
    dynamic_dist_thresh_px = max(5.0, 60.0 * reso_ratio)
    dynamic_pnp_reproj_err = max(2.0, 4.0 * reso_ratio)
    dynamic_vis_map_pt_size = max(1.0, 1.0 * reso_ratio)


    # --- Run Pipeline ---
    run_pipeline(
        arg_config=args, # Pass parsed args (contains steps, camera_name, de_*)
        lidar_map_file=LIDAR_MAP_PATH,
        query_image_dir=current_query_img_dir,
        query_image_list_file=current_query_img_list_path,
        output_dir=current_output_dir,
        render_poses_list=initial_render_poses_map_cam, # T_map_cam for rendering
        ego_pose_file=EGO_POSE_CSV_PATH,
        initial_T_ego_cam_guess=pipeline_initial_T_ego_cam, # T_sensor_cam
        camera_intrinsics_matrix_ideal=pipeline_K_ideal,   # K_ideal
        camera_distortion_array_sensor=pipeline_D_sensor, # D_sensor
        image_width=pipeline_img_w, image_height=pipeline_img_h,
        camera_model_type=pipeline_model_type, # Pass model_type
        camera_name_in_list=CAMERA_NAME_ARG,
        # Optional params:
        voxel_size=0.03, min_height=-100.0, device="auto",
        render_shading_mode='normal', render_point_size=dynamic_render_point_size, intensity_highlight_threshold=0.1,
        feature_conf='superpoint_aachen', matcher_conf='superpoint+lightglue', # matcher_conf not directly used by dist match
        distance_threshold_px=dynamic_dist_thresh_px,
        pnp_min_inliers=5, pnp_reprojection_error=dynamic_pnp_reproj_err, pnp_iterations=500, pnp_confidence=0.999999,
        dt_bounds_de_inner_opt=(-0.03, 0.03), # For DE inner loop dt opt
        dt_bounds_final_ls=(-0.06, 0.06),     # For final LS dt opt
        final_ls_opt_verbose=1, # Verbosity for final LS
        de_robust_loss_type='cauchy', # Or 'huber', 'linear'
        de_robust_loss_scale=1.0, # Example: use PnP reproj error as scale (adjust as needed)
        loss_function_final_ls='cauchy',
        visualize_steps=True, num_images_to_visualize=(len(initial_render_poses_map_cam) if initial_render_poses_map_cam else 1),
        visualize_map_point_size=dynamic_vis_map_pt_size,
        intrinsic_bounds_config=cfg_intrinsic_bounds,
        visualize_distortpoints_debug=True, # Enable debug for generate_target_distorted_points
        num_distortpoints_debug_images=2,
        de_popsize_factor=args.de_popsize_factor, # From argparse
        de_maxiter=args.de_maxiter,             # From argparse
        de_workers=args.de_workers,             # From argparse
        spatial_grid_size_hybrid=30,
        max_pts_per_cell_hybrid=1,
        min_total_pts_hybrid=50
    )

