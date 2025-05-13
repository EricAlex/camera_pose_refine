
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
from scipy.optimize import least_squares
import functools
from dataclasses import dataclass
import argparse
import csv
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    query_image_path: str,
    processed_lidar_data: dict, # Expects dict with 'points', optionally 'intensities'
    camera_intrinsics: np.ndarray,
    pose_cam_from_map: np.ndarray, # T_cam_map (4x4)
    output_path: str,
    dist_coeffs: np.ndarray = None,
    point_size: int = 1,
    max_vis_points: int = 500000, 
    color_map: str = 'jet',
    filter_distance: float = 50.0, 
    color_by: str = 'intensity', # 'depth' or 'intensity'
    intensity_stretch_percentiles: tuple[float, float] = (10.0, 80.0)
    ):
    """
    Projects LiDAR map points onto an image using a given pose and saves the visualization.
    Points can be colored by depth or by intensity (with optional contrast stretching).

    Args:
        query_image_path: Path to the input query image.
        processed_lidar_data: Dictionary containing processed LiDAR data.
                              Must have 'points'. If color_by='intensity', must also have 'intensities'.
        camera_intrinsics: 3x3 camera intrinsic matrix.
        pose_cam_from_map: 4x4 transformation matrix (Camera from Map, T_cam_map).
        output_path: Path to save the output visualization image.
        dist_coeffs: Optional camera distortion coefficients.
        point_size: Size of the projected points.
        max_vis_points: Maximum number of points to sample for visualization (None for all).
        color_map: Matplotlib colormap name for coloring.
        filter_distance: Maximum distance from camera origin for points to be projected.
                         Set to None to disable distance filtering.
        color_by: How to color points: 'depth' (default) or 'intensity'.
        intensity_stretch_percentiles: Tuple (min_percentile, max_percentile) for contrast
                                       stretching of intensities if color_by='intensity'.
                                       Values are percentages (e.g., 1.0 for 1st percentile,
                                       99.0 for 99th percentile). If None, no stretching is applied.
    """
    logging.debug(f"Visualizing map projection for {Path(query_image_path).name} -> {Path(output_path).name}")
    logging.debug(f"Coloring by: {color_by}, Filter distance: {filter_distance}, Stretch: {intensity_stretch_percentiles}")
    try:
        image = cv2.imread(query_image_path)
        if image is None:
            logging.error(f"Could not load image for visualization: {query_image_path}")
            return

        points3D_map_original = processed_lidar_data.get('points')
        intensities_map_original = None
        if color_by == 'intensity':
            intensities_map_original = processed_lidar_data.get('intensities')
            if intensities_map_original is None:
                logging.warning("Requested coloring by intensity, but 'intensities' not found in processed_lidar_data. Falling back to depth.")
                color_by = 'depth' # Fallback
            elif points3D_map_original is not None and intensities_map_original.shape[0] != points3D_map_original.shape[0]:
                logging.warning("Mismatch between number of points and intensities. Falling back to depth.")
                color_by = 'depth' # Fallback
                intensities_map_original = None


        if points3D_map_original is None or points3D_map_original.shape[0] == 0:
            logging.warning("No points found in processed_lidar_data for visualization.")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, image)
            return

        if pose_cam_from_map is None or pose_cam_from_map.shape != (4, 4):
            logging.error(f"Invalid pose_cam_from_map provided for {Path(query_image_path).name}")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True) # Still try to save original image
            cv2.imwrite(output_path, image)
            return

        # --- Filtering and Subsampling ---
        points3D_map_current = points3D_map_original
        intensities_map_current = intensities_map_original

        points3D_map_h = np.hstack((points3D_map_current, np.ones((points3D_map_current.shape[0], 1))))
        points_cam_h = (pose_cam_from_map @ points3D_map_h.T).T

        # current_indices = np.arange(points3D_map_current.shape[0]) # Not strictly needed for current logic
        if filter_distance is not None:
            distances_cam = np.linalg.norm(points_cam_h[:, :3], axis=1)
            keep_mask_dist = distances_cam < filter_distance
            
            points3D_map_current = points3D_map_current[keep_mask_dist]
            points_cam_h = points_cam_h[keep_mask_dist]
            if intensities_map_current is not None:
                intensities_map_current = intensities_map_current[keep_mask_dist]
            # current_indices = current_indices[keep_mask_dist] 

            logging.debug(f"Filtered points by distance (<{filter_distance}m): {points3D_map_original.shape[0]} -> {points3D_map_current.shape[0]}")
            if points3D_map_current.shape[0] == 0:
                 logging.warning("No points left after distance filtering.")
                 Path(output_path).parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(output_path, image); return
        
        points3D_to_project = points3D_map_current
        intensities_to_project = intensities_map_current
        points_cam_h_to_process = points_cam_h


        num_points = points3D_to_project.shape[0]
        if max_vis_points is not None and num_points > max_vis_points:
            sample_indices = np.random.choice(num_points, max_vis_points, replace=False)
            points3D_to_project = points3D_to_project[sample_indices]
            points_cam_h_to_process = points_cam_h_to_process[sample_indices]
            if intensities_to_project is not None:
                intensities_to_project = intensities_to_project[sample_indices]
            logging.debug(f"Subsampled points for visualization: {num_points} -> {points3D_to_project.shape[0]}")

        if points3D_to_project.shape[0] == 0:
             logging.warning("No points to project after filtering/sampling.")
             Path(output_path).parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(output_path, image); return

        R_cam_map = pose_cam_from_map[:3, :3]
        t_cam_map = pose_cam_from_map[:3, 3]
        try:
            if not np.allclose(R_cam_map.T @ R_cam_map, np.eye(3), atol=1e-4):
                 logging.warning("Rotation matrix non-orthogonal in visualize_map_projection. Projection might be inaccurate.")
            rvec, _ = cv2.Rodrigues(R_cam_map)
        except cv2.error as e_rot:
             logging.error(f"cv2.Rodrigues conversion failed: {e_rot}. Pose: {pose_cam_from_map}")
             Path(output_path).parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(output_path, image); return

        tvec = t_cam_map.reshape(3, 1)
        image_points, _ = cv2.projectPoints(points3D_to_project, rvec, tvec, camera_intrinsics, distCoeffs=dist_coeffs)

        if image_points is None:
             logging.warning("cv2.projectPoints returned None.")
             Path(output_path).parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(output_path, image); return
        image_points = image_points.reshape(-1, 2)

        h_img, w_img = image.shape[:2]
        depths_cam = points_cam_h_to_process[:, 2]

        mismatched_shapes = False
        if image_points.shape[0] != depths_cam.shape[0]: mismatched_shapes = True
        if intensities_to_project is not None and image_points.shape[0] != intensities_to_project.shape[0]: mismatched_shapes = True
        if mismatched_shapes:
             logging.error(f"Shape mismatch: image_points {image_points.shape}, depths_cam {depths_cam.shape}, intensities {getattr(intensities_to_project, 'shape', 'N/A')}. Cannot proceed.")
             Path(output_path).parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(output_path, image); return


        valid_mask_proj = (depths_cam > 0.1) & \
                          (image_points[:, 0] >= 0) & (image_points[:, 0] < w_img) & \
                          (image_points[:, 1] >= 0) & (image_points[:, 1] < h_img)

        image_points_valid = image_points[valid_mask_proj].astype(int)
        
        if image_points_valid.shape[0] == 0:
             logging.warning("No valid projected points within image bounds and in front of camera.")
             Path(output_path).parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(output_path, image); return

        norm_values_for_cmap = None
        if color_by == 'intensity' and intensities_to_project is not None:
            intensities_valid = intensities_to_project[valid_mask_proj]
            if intensities_valid.shape[0] > 0:
                if intensity_stretch_percentiles is not None and len(intensity_stretch_percentiles) == 2:
                    min_p, max_p = intensity_stretch_percentiles
                    # Ensure intensities_valid is 1D for percentile calculation
                    val_min = np.percentile(intensities_valid.flatten(), min_p)
                    val_max = np.percentile(intensities_valid.flatten(), max_p)
                    logging.debug(f"Intensity stretching: min_perc ({min_p}%): {val_min:.3f}, max_perc ({max_p}%): {val_max:.3f}")
                    if val_max - val_min < 1e-6:
                        logging.warning("Intensity range for stretching is too small or zero. Using mid-value.")
                        norm_values_for_cmap = np.full_like(intensities_valid.flatten(), 0.5, dtype=np.float32)
                    else:
                        norm_values_for_cmap = (intensities_valid.flatten() - val_min) / (val_max - val_min)
                else: # No stretching
                    norm_values_for_cmap = intensities_valid.flatten().astype(np.float32)
                norm_values_for_cmap = np.clip(norm_values_for_cmap, 0, 1)
            else: # No valid intensities after masking
                logging.warning("No valid intensities to color by after masking. Falling back to depth.")
                color_by = 'depth' 
        
        # If color_by is 'depth' (either originally or as fallback)
        if color_by == 'depth':
            depths_valid_for_color = depths_cam[valid_mask_proj]
            if depths_valid_for_color.shape[0] > 0:
                min_depth, max_depth = np.min(depths_valid_for_color), np.max(depths_valid_for_color)
                if max_depth - min_depth < 1e-6:
                     norm_values_for_cmap = np.full_like(depths_valid_for_color, 0.5, dtype=np.float32)
                else:
                     norm_values_for_cmap = (depths_valid_for_color - min_depth) / (max_depth - min_depth)
                norm_values_for_cmap = np.clip(norm_values_for_cmap, 0, 1)
            # If depths_valid_for_color is empty, norm_values_for_cmap remains None

        # --- Get colors from colormap ---
        colors_bgr = None
        if norm_values_for_cmap is not None and norm_values_for_cmap.shape[0] > 0:
            try:
                # --- CRITICAL FIX: Ensure norm_values_for_cmap is 1D ---
                norm_values_for_cmap_1d = norm_values_for_cmap.squeeze() # Use squeeze or flatten
                if norm_values_for_cmap_1d.ndim == 0: # Handle scalar case (single point)
                    norm_values_for_cmap_1d = np.array([norm_values_for_cmap_1d.item()])
                elif norm_values_for_cmap_1d.ndim > 1: # Should not happen if squeeze worked, but defensive
                    norm_values_for_cmap_1d = norm_values_for_cmap_1d.flatten()
                # --- END CRITICAL FIX ---

                cmap_obj = plt.get_cmap(color_map)
                colors_rgba = cmap_obj(norm_values_for_cmap_1d) # Expected (M, 4) or (M, 3)

                if not isinstance(colors_rgba, np.ndarray) or colors_rgba.ndim != 2:
                    # If colors_rgba is (M,1,4) or similar, squeeze it.
                    if isinstance(colors_rgba, np.ndarray) and colors_rgba.ndim == 3 and colors_rgba.shape[1] == 1:
                        colors_rgba = colors_rgba.squeeze(axis=1)
                        logging.debug(f"Squeezed colormap output from {colors_rgba.ndim+1}D to 2D. New shape: {colors_rgba.shape}")
                    else:
                        raise ValueError(f"Colormap output is not a 2D array even after attempting squeeze. Shape: {getattr(colors_rgba, 'shape', 'N/A')}, Type: {type(colors_rgba)}")

                num_channels = colors_rgba.shape[1]
                if num_channels == 4: # RGBA
                    colors_bgr_float = colors_rgba[:, [2, 1, 0]] # Select B, G, R
                elif num_channels == 3: # RGB
                    colors_bgr_float = colors_rgba[:, [2, 1, 0]] # Select B, G, R
                elif num_channels == 1: # Grayscale
                    logging.warning(f"Colormap '{color_map}' returned single-channel data (shape {colors_rgba.shape}). Interpreting as grayscale.")
                    gray_channel = colors_rgba[:, 0]
                    colors_bgr_float = np.stack((gray_channel, gray_channel, gray_channel), axis=-1)
                else:
                    raise ValueError(f"Colormap returned unexpected number of channels: {num_channels}. Shape: {colors_rgba.shape}")
                
                colors_bgr = (colors_bgr_float * 255).astype(np.uint8)

            except Exception as e_cmap_proc:
                logging.error(f"Error processing colormap '{color_map}': {e_cmap_proc}. Using fallback gray color.", exc_info=True)
                num_pts_to_color = norm_values_for_cmap.shape[0]
                colors_bgr = np.full((num_pts_to_color, 3), (200, 200, 200), dtype=np.uint8)

        elif image_points_valid.shape[0] > 0 : # No norm_values, but there are points to draw
            logging.warning("Normalized values for colormap are missing or empty, but valid points exist. Using fixed color.")
            colors_bgr = np.full((image_points_valid.shape[0], 3), (200, 200, 200), dtype=np.uint8)
        
        # --- Draw Points ---
        radius = max(1, int(point_size))
        if colors_bgr is not None and image_points_valid.shape[0] == colors_bgr.shape[0]:
            for i in range(image_points_valid.shape[0]):
                center = tuple(image_points_valid[i])
                color = tuple(map(int, colors_bgr[i]))
                cv2.circle(image, center, radius, color, thickness=-1)
        elif image_points_valid.shape[0] > 0:
             logging.warning("Projected points exist, but colors could not be determined. Points will not be drawn over the image.")


        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save_success = cv2.imwrite(output_path, image)
        if save_success:
            logging.debug(f"Saved map projection visualization to {output_path}")
        else:
             logging.error(f"Failed to save map projection visualization to {output_path}")

    except ImportError:
         logging.error("Matplotlib is required for colormaps. Run: pip install matplotlib")
         if 'image' in locals() and image is not None and 'output_path' in locals():
            try:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(output_path, image)
            except Exception as e_save_fallback:
                logging.error(f"Could not save fallback image: {e_save_fallback}")

    except cv2.error as e:
        logging.error(f"OpenCV error during visualization: {e} for {query_image_path}")
    except Exception as e:
        logging.error(f"Error during map projection visualization for {query_image_path}: {e}", exc_info=True)

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

class OptimizationData:
    def __init__(self, inlier_matches_map_us, ego_interpolator_us, K, t_rec_map_us, num_images_with_matches):
        self.inlier_matches = inlier_matches_map_us
        self.ego_interpolator = ego_interpolator_us # Expects query_ts in MICROSECONDS
        self.K = K
        self.t_rec_map = t_rec_map_us # Stores recorded timestamps in MICROSECONDS
        self.image_indices = sorted(inlier_matches_map_us.keys()) 
        self.num_opt_images = len(self.image_indices)

        self.all_p2d = []
        self.all_P3d = []
        self.image_idx_to_dt_param_idx_map = {} 

        current_dt_param_idx = 0
        for original_img_idx in self.image_indices: 
            p2d, P3d = self.inlier_matches[original_img_idx]
            self.all_p2d.append(p2d)
            self.all_P3d.append(P3d)
            self.image_idx_to_dt_param_idx_map[original_img_idx] = current_dt_param_idx
            current_dt_param_idx += 1

        if self.all_p2d: 
            self.all_p2d = np.concatenate(self.all_p2d, axis=0)
            self.all_P3d = np.concatenate(self.all_P3d, axis=0)
            self.total_residuals = self.all_p2d.shape[0] * 2
        else:
            self.total_residuals = 0

def compute_residuals(X, opt_data: OptimizationData):
    residuals = np.zeros(opt_data.total_residuals, dtype=np.float64)
    if opt_data.total_residuals == 0: return residuals 

    try:
        xi_ego_cam = X[:6]
        all_dt_seconds = X[6:] 

        T_ego_cam = se3_to_SE3(xi_ego_cam)
        current_residual_offset = 0

        for i in range(opt_data.num_opt_images):
            original_img_idx = opt_data.image_indices[i] 
            dt_i_seconds = all_dt_seconds[i] 

            p2d_inliers, P3d_inliers = opt_data.inlier_matches[original_img_idx]
            num_matches = p2d_inliers.shape[0]
            if num_matches == 0: continue 

            t_rec_i_us = opt_data.t_rec_map[original_img_idx] # This is in MICROSECONDS

            # --- SCALE dt TO MICROSECONDS ---
            dt_i_us = dt_i_seconds * 1_000_000.0 # Corrected scaling
            t_true_i_us = float(t_rec_i_us) + dt_i_us # Result is in microseconds

            # Interpolate ego pose using the microsecond timestamp
            T_map_ego_i = opt_data.ego_interpolator(t_true_i_us) # ego_interpolator expects microseconds
            
            if T_map_ego_i is None:
                logging.warning(f"Ego pose interpolation failed for img_idx {original_img_idx} at t_true_us={t_true_i_us:.0f} (orig t_rec_us={t_rec_i_us}, dt_s={dt_i_seconds:.4f})")
                residuals[current_residual_offset : current_residual_offset + num_matches * 2] = 1e6
                current_residual_offset += num_matches * 2
                continue

            T_map_cam_i = T_map_ego_i @ T_ego_cam
            try:
                T_cam_map_i = np.linalg.inv(T_map_cam_i)
            except np.linalg.LinAlgError:
                logging.warning(f"Pose inversion failed for img_idx {original_img_idx}")
                residuals[current_residual_offset : current_residual_offset + num_matches * 2] = 1e6
                current_residual_offset += num_matches * 2
                continue

            P3d_h = np.hstack((P3d_inliers, np.ones((num_matches, 1))))
            P_cam_h = (T_cam_map_i @ P3d_h.T).T
            valid_depth_mask = P_cam_h[:, 2] > 1e-3
            
            p_proj_2d_slice = np.full((num_matches, 2), 1e6) 

            if np.any(valid_depth_mask):
                P_cam_h_valid = P_cam_h[valid_depth_mask, :3]
                p_proj_h_valid = (opt_data.K @ P_cam_h_valid.T).T
                
                valid_proj_mask_local = np.abs(p_proj_h_valid[:, 2]) > 1e-6
                
                if np.any(valid_proj_mask_local):
                    p_proj_2d_final_valid = np.zeros((np.sum(valid_proj_mask_local), 2))
                    p_proj_2d_final_valid[:, 0] = p_proj_h_valid[valid_proj_mask_local, 0] / p_proj_h_valid[valid_proj_mask_local, 2]
                    p_proj_2d_final_valid[:, 1] = p_proj_h_valid[valid_proj_mask_local, 1] / p_proj_h_valid[valid_proj_mask_local, 2]
                    
                    original_indices_valid_depth = np.where(valid_depth_mask)[0]
                    original_indices_final_valid = original_indices_valid_depth[valid_proj_mask_local]
                    
                    p_proj_2d_slice[original_indices_final_valid] = p_proj_2d_final_valid

            r_i_slice = p2d_inliers - p_proj_2d_slice
            residuals[current_residual_offset : current_residual_offset + num_matches * 2] = r_i_slice.flatten()
            current_residual_offset += num_matches * 2

    except Exception as e:
        logging.error(f"Error in compute_residuals: {e}\n{traceback.format_exc()}")
        return np.full(opt_data.total_residuals if opt_data.total_residuals > 0 else 1, 1e6, dtype=np.float64)

    if not np.all(np.isfinite(residuals)):
        logging.error("Non-finite values detected in residuals! Replacing with 1e6.")
        residuals[~np.isfinite(residuals)] = 1e6
    return residuals

def compute_residuals_timestamp_only(dt_scalar_seconds, fixed_T_ego_cam, p2d_inliers, P3d_inliers,
                                     t_rec_us, ego_interpolator_us, K): # Suffixes changed
    num_matches = p2d_inliers.shape[0]
    residuals_flat = np.zeros(num_matches * 2, dtype=np.float64) 

    try:
        dt_value_seconds = dt_scalar_seconds[0] if isinstance(dt_scalar_seconds, np.ndarray) and dt_scalar_seconds.size == 1 else dt_scalar_seconds
        
        # --- SCALE dt TO MICROSECONDS ---
        dt_value_us = dt_value_seconds * 1_000_000.0 # Corrected scaling
        t_true_i_us = float(t_rec_us) + dt_value_us # Result is in microseconds

        T_map_ego_i = ego_interpolator_us(t_true_i_us) # ego_interpolator_us expects microseconds
        if T_map_ego_i is None:
            logging.warning(f"TS_ONLY: Ego pose interpolation failed for t_true_us={t_true_i_us:.0f} (orig t_rec_us={t_rec_us}, dt_s={dt_value_seconds:.4f})")
            return np.full(num_matches * 2, 1e6, dtype=np.float64)

        T_map_cam_i = T_map_ego_i @ fixed_T_ego_cam
        try:
            T_cam_map_i = np.linalg.inv(T_map_cam_i)
        except np.linalg.LinAlgError:
            logging.warning("TS_ONLY: Pose inversion failed.")
            return np.full(num_matches * 2, 1e6, dtype=np.float64)

        P3d_h = np.hstack((P3d_inliers, np.ones((num_matches, 1))))
        P_cam_h = (T_cam_map_i @ P3d_h.T).T
        valid_depth_mask = P_cam_h[:, 2] > 1e-3 

        p_proj_2d = np.full((num_matches, 2), 1e6) 

        if np.any(valid_depth_mask):
            P_cam_h_valid = P_cam_h[valid_depth_mask, :3]
            p_proj_h_valid = (K @ P_cam_h_valid.T).T 

            valid_proj_mask_local = np.abs(p_proj_h_valid[:, 2]) > 1e-6 

            if np.any(valid_proj_mask_local):
                p_proj_2d_final_valid = np.zeros((np.sum(valid_proj_mask_local), 2))
                p_proj_2d_final_valid[:, 0] = p_proj_h_valid[valid_proj_mask_local, 0] / p_proj_h_valid[valid_proj_mask_local, 2]
                p_proj_2d_final_valid[:, 1] = p_proj_h_valid[valid_proj_mask_local, 1] / p_proj_h_valid[valid_proj_mask_local, 2]
                
                original_indices_valid_depth = np.where(valid_depth_mask)[0]
                original_indices_final_valid = original_indices_valid_depth[valid_proj_mask_local]
                p_proj_2d[original_indices_final_valid] = p_proj_2d_final_valid
        
        r_i = p2d_inliers - p_proj_2d 
        residuals_flat = r_i.flatten()

    except Exception as e:
        logging.error(f"Error in compute_residuals_timestamp_only: {e}\n{traceback.format_exc()}")
        return np.full(num_matches * 2, 1e6, dtype=np.float64)

    if not np.all(np.isfinite(residuals_flat)):
        logging.error("Non-finite values detected in timestamp-only residuals! Replacing with 1e6.")
        residuals_flat[~np.isfinite(residuals_flat)] = 1e6
    return residuals_flat

def refine_timestamp_only(fixed_T_ego_cam, p2d_inliers, P3d_inliers,
                          t_rec_us, ego_interpolator_us, K,
                          dt_bounds_seconds=(-0.1, 0.1),
                          loss_function='cauchy', verbose=0):
    if p2d_inliers.shape[0] == 0:
        logging.warning("No inliers provided for timestamp-only refinement.")
        # Return values consistent with expected tuple: (dt, status, message, optimality)
        return None, -1, "No inliers", np.inf

    x0_dt_seconds = [0.0]
    bounds_dt_seconds = ([dt_bounds_seconds[0]], [dt_bounds_seconds[1]])

    residual_func_partial = functools.partial(
        compute_residuals_timestamp_only,
        fixed_T_ego_cam=fixed_T_ego_cam,
        p2d_inliers=p2d_inliers,
        P3d_inliers=P3d_inliers,
        t_rec_us=t_rec_us,
        ego_interpolator_us=ego_interpolator_us,
        K=K
    )

    dt_val = None
    status_val = -1 # Default status for pre-optimization
    message_val = "Optimization not run"
    optimality_val = np.inf

    try:
        # Calculate initial cost if verbose
        if verbose > 0:
            initial_residuals_ts_only = residual_func_partial(x0_dt_seconds)
            if initial_residuals_ts_only is not None and initial_residuals_ts_only.size > 0:
                 initial_cost_ts_only = 0.5 * np.sum(initial_residuals_ts_only**2)
                 logging.info(f"TS-Only for t_rec_us={t_rec_us}: Initial cost (dt=0): {initial_cost_ts_only:.4e}")
            else:
                 logging.warning(f"TS-Only for t_rec_us={t_rec_us}: Could not compute initial residuals.")


        result = least_squares(
            residual_func_partial,
            x0_dt_seconds,
            jac='2-point',
            bounds=bounds_dt_seconds,
            method='trf',
            ftol=1e-9, xtol=1e-9, gtol=1e-9, # Slightly tighter tols for 1D
            loss=loss_function,
            verbose=verbose, # This is the verbosity for least_squares iterations
            max_nfev=300 # More NFEV for 1D opt
        )

        status_val = result.status
        message_val = result.message
        optimality_val = result.optimality

        # Calculate final cost if verbose
        if verbose > 0 :
            final_residuals_ts_only = residual_func_partial(result.x)
            if final_residuals_ts_only is not None and final_residuals_ts_only.size > 0 :
                final_cost_ts_only = 0.5 * np.sum(final_residuals_ts_only**2) # or result.cost
                logging.info(f"TS-Only for t_rec_us={t_rec_us}: Final cost (dt={result.x[0]:.7f}): {final_cost_ts_only:.4e}. Optimality: {result.optimality:.2e}")
            else:
                logging.warning(f"TS-Only for t_rec_us={t_rec_us}: Could not compute final residuals for dt={result.x[0]:.7f}")


        if result.success:
            dt_val = result.x[0]
            # This warning is already good
            if result.optimality > 1e-2 and verbose > 0:
                 logging.warning(f"TS-Only opt for t_rec_us={t_rec_us} (dt={dt_val:.7f}s) finished with high optimality: {result.optimality:.2e}")
        else:
            # dt_val remains None
            logging.warning(f"TS-Only optimization FAILED for t_rec_us={t_rec_us}. Status: {result.status} ('{result.message}'), Optimality: {result.optimality:.2e}. Resulting dt (if any): {result.x[0]:.7f}")
            if np.isclose(result.x[0], 0.0) and result.status != 0 : # If it failed and result is 0.0
                 logging.warning(f" -> TS-Only for t_rec_us={t_rec_us} failed and dt is 0.0. Cost may not have improved from initial.")


    except Exception as e:
        logging.error(f"TS-Only optimization CRASHED for t_rec_us={t_rec_us}: {e}\n{traceback.format_exc()}")
        # dt_val remains None
        status_val = -100 # Custom status for crash
        message_val = str(e)
        optimality_val = np.inf

    return dt_val, status_val, message_val, optimality_val

def refine_extrinsics_and_timestamps(
    initial_T_ego_cam: np.ndarray,
    inlier_matches_map: dict, 
    ego_timestamps_us: np.ndarray, # RENAMED: Expects microsecond timestamps
    ego_poses: list, 
    query_indices: list, 
    query_timestamps_rec_us: dict, # RENAMED: Expects {img_idx: recorded_timestamp_us}
    camera_intrinsics: np.ndarray,
    dt_bounds_seconds=(-0.1, 0.1), 
    loss_function='cauchy', 
    verbose=1,
    interpolation_tolerance_us: int = 1 # RENAMED: Default to 1 microsecond tolerance
):
    logging.info("--- Starting Extrinsics and Timestamp Refinement (Timestamps in Microseconds) ---")

    num_images_total = len(query_indices)
    if num_images_total == 0:
        logging.error("No query images provided.")
        return None, None, None

    valid_inlier_matches = {
        idx: data for idx, data in inlier_matches_map.items()
        if idx in query_indices and data[0].shape[0] > 0
    }
    num_images_with_matches = len(valid_inlier_matches)
    if num_images_with_matches == 0: 
        logging.error("No images with valid inlier matches found for optimization.")
        return None, None, None
    logging.info(f"Optimizing using {num_images_with_matches} images with inlier matches out of {num_images_total} specified.")

    try:
        xi_ego_cam_init = SE3_to_se3(initial_T_ego_cam)
    except Exception as e:
        logging.error(f"Failed to convert initial T_ego_cam to se(3): {e}", exc_info=True)
        return None, None, None

    dt_init_seconds = np.zeros(num_images_with_matches) 
    x0 = np.concatenate((xi_ego_cam_init, dt_init_seconds))

    bounds_low = [-np.inf] * 6 + [dt_bounds_seconds[0]] * num_images_with_matches
    bounds_high = [np.inf] * 6 + [dt_bounds_seconds[1]] * num_images_with_matches
    bounds = (bounds_low, bounds_high)

    # --- Create Ego Pose Interpolator Function (Microsecond Aware) ---
    def interpolate_ego_pose_for_opt(query_ts_us_float): # Renamed arg
        # get_pose_for_timestamp expects microsecond query_ts
        return get_pose_for_timestamp(query_ts_us_float, ego_timestamps_us, ego_poses, 
                                      tolerance_us=interpolation_tolerance_us) # Pass us tolerance

    filtered_t_rec_map_us = { # Renamed
        idx: ts for idx, ts in query_timestamps_rec_us.items() 
        if idx in valid_inlier_matches
    }

    try:
        opt_data = OptimizationData(
            valid_inlier_matches,      
            interpolate_ego_pose_for_opt, # Microsecond-aware interpolator
            camera_intrinsics,
            filtered_t_rec_map_us,      # Microsecond timestamps
            num_images_with_matches    
        )
    except Exception as e:
         logging.error(f"Failed to create OptimizationData: {e}", exc_info=True)
         return None, None, None
    
    if opt_data.total_residuals == 0:
        logging.error("OptimizationData reports zero total residuals.")
        return initial_T_ego_cam, {idx: 0.0 for idx in query_indices}, None 

    logging.info(f"Total number of parameters: {len(x0)} (6 for T_ego_cam, {num_images_with_matches} for dt in seconds)")
    logging.info(f"Total number of residuals: {opt_data.total_residuals}")

    try:
        initial_residuals = compute_residuals(x0, opt_data)
        initial_cost = 0.5 * np.sum(initial_residuals**2)
        logging.info(f"Calculated Initial cost: {initial_cost:.4e}")
        if not np.isfinite(initial_cost) or initial_cost > 1e10:
            logging.error(f"Initial cost is non-finite or extremely large ({initial_cost:.2e})! Check inputs.")
            return None, None, None
    except Exception as e:
        logging.error(f"Failed to compute initial residuals/cost: {e}", exc_info=True)
        return None, None, None

    start_time = time.time()
    try:
        result = least_squares(
            compute_residuals, x0, jac='2-point', bounds=bounds, method='trf',
            ftol=1e-8, xtol=1e-8, gtol=1e-8, loss=loss_function, verbose=verbose,
            max_nfev=1000 * len(x0), 
            args=(opt_data,)
        )
        duration = time.time() - start_time
        logging.info(f"Optimization finished in {duration:.2f} seconds. Status: {result.status} ({result.message})")
        logging.info(f"Final cost: {result.cost:.4e}. Optimality: {result.optimality:.4e}")
    except Exception as e:
        logging.error(f"Optimization crashed: {e}\n{traceback.format_exc()}")
        return None, None, None

    if not result.success and result.status <= 0: 
        logging.warning(f"Optimization reported failure or did not converge (status: {result.status}).")

    xi_ego_cam_refined = result.x[:6]
    all_dt_refined_seconds = result.x[6:] 

    try:
        refined_T_ego_cam = se3_to_SE3(xi_ego_cam_refined)
    except Exception as e:
        logging.error(f"Failed to convert refined xi_ego_cam to SE(3): {e}", exc_info=True)
        return None, None, None 

    refined_delta_t_map_seconds = {}
    for i, original_img_idx in enumerate(opt_data.image_indices): 
         if i < len(all_dt_refined_seconds):
             refined_delta_t_map_seconds[original_img_idx] = all_dt_refined_seconds[i]
         else: 
              logging.error(f"CRITICAL: Mismatch in refined_dt array length. Index {i} out of bounds for len {len(all_dt_refined_seconds)}.")

    for img_idx in query_indices:
        if img_idx not in refined_delta_t_map_seconds:
            refined_delta_t_map_seconds[img_idx] = 0.0

    logging.info(f"Refined T_ego_cam:\n{np.round(refined_T_ego_cam, 4)}")
    
    optimized_dt_values = [refined_delta_t_map_seconds[idx] for idx in opt_data.image_indices if idx in refined_delta_t_map_seconds]
    if optimized_dt_values:
         logging.info(f"Refined Delta_t (seconds) stats (for {len(optimized_dt_values)} optimized images): Mean={np.mean(optimized_dt_values):.4f}s, Std={np.std(optimized_dt_values):.4f}s, Min={np.min(optimized_dt_values):.4f}s, Max={np.max(optimized_dt_values):.4f}s")

    optimality_threshold = 1e-1 
    if result.optimality > optimality_threshold:
        logging.warning(f"High first-order optimality ({result.optimality:.2e}) suggests result might be suboptimal or near constraint boundary.")

    return refined_T_ego_cam, refined_delta_t_map_seconds, result

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
    arg_config,
    lidar_map_file: Path,
    query_image_dir: Path,
    query_image_list_file: Path,
    output_dir: Path,
    render_poses_list,
    ego_pose_file: Path,
    initial_T_ego_cam_guess: np.ndarray,
    camera_intrinsics_matrix: np.ndarray,
    camera_distortion_array: np.ndarray,
    image_width: int, image_height: int,
    camera_name_in_list: str,
    min_height: float = -2.0, voxel_size: float = 0.03, normal_radius: float = 0.15,
    normal_max_nn: int = 50, device: str = "auto",
    render_shading_mode: str = 'normal', render_point_size: float = 2,
    intensity_highlight_threshold: float = None,
    feature_conf='superpoint_aachen', matcher_conf='superglue', # matcher_conf not used directly
    distance_threshold_px: float = 30.0,
    pnp_iterations: int = 500, pnp_reprojection_error: float = 5.0,
    pnp_confidence: float = 0.999999, pnp_min_inliers: int = 15,
    num_top_images_for_joint_opt: int = 0, # MODIFIED: Set to 0 to indicate joint opt on ALL successful PnP
    dt_bounds_joint_opt: tuple[float, float] = (-0.05, 0.05),
    opt_verbose: int = 1,
    visualize_steps: bool = True,
    num_images_to_visualize: int = 3,
    visualize_map_point_size: float = 1,
    loss_function: str = 'cauchy',
):
    global pnp_min_inliers_global_placeholder # For dummy link_matches_via_depth
    pnp_min_inliers_global_placeholder = pnp_min_inliers

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

    feature_output_base_name = 'feats-superpoint-n4096-r1024'
    features_filename = f"{feature_output_base_name}.h5"
    features_path = hloc_out_dir / features_filename
    matches_output_path = hloc_out_dir / 'distance_matches.h5'
    masked_render_features_path = features_path
    vis_pnp_output_dir = vis_base_output_dir / 'pnp'
    vis_pnp_output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("--- 0. Loading Ego Vehicle Poses ---")
    ego_timestamps_us, ego_poses_list = load_and_prepare_ego_poses(ego_pose_file)
    if ego_timestamps_us is None or ego_poses_list is None:
        logging.error(f"Failed to load ego poses from {ego_pose_file}. Aborting.")
        return
    if len(ego_timestamps_us) != len(ego_poses_list):
         logging.error("Mismatch between loaded ego timestamps and poses count. Aborting.")
         return
    logging.info(f"Loaded {len(ego_timestamps_us)} ego poses.")

    def interpolate_ego_pose_for_opt_us(query_ts_us, tolerance_us=1):
        return get_pose_for_timestamp(query_ts_us, ego_timestamps_us, ego_poses_list, tolerance_us=tolerance_us)

    # Initialize variables
    processed_lidar_data_loaded = None
    rendered_views_info_loaded = None
    query_image_names = []
    query_timestamps_rec_us = {}
    query_name_to_idx = {}
    query_idx_to_name = {}
    query_timestamps_rec_us_indexed = {}
    # Initialize refined_T_ego_cam and final_delta_t_map early
    refined_T_ego_cam = np.copy(initial_T_ego_cam_guess)
    final_delta_t_map = {} # Will store {query_idx: dt_seconds}
    final_refined_poses_cam_map = {} # For final results

    if not arg_config.steps or 1 in arg_config.steps:
        # ... (Step 1: Preprocessing & Rendering - code remains the same as previous response) ...
        logging.info("--- Running Step 1: Preprocessing & Rendering ---")
        try:
            pcd_tensor, processed_lidar_data = preprocess_lidar_map(
                lidar_map_file, min_height, normal_radius, normal_max_nn, voxel_size, device
            )
            if pcd_tensor is None: raise RuntimeError("Preprocessing failed.")
        except Exception as e:
            logging.error(f"FATAL: Preprocessing failed: {e}", exc_info=True); return

        rendered_views_info = []
        if not render_poses_list:
             logging.warning("Render poses list is empty. Skipping rendering step.")
        else:
            with open(render_image_list_path, 'w') as f_list:
                for i, pose in enumerate(render_poses_list):
                    render_name = f"render_{i:05d}"
                    logging.info(f"Rendering view {i+1}/{len(render_poses_list)} ({render_name})...")
                    render_output = render_geometric_viewpoint_open3d(
                        pcd_tensor, processed_lidar_data, pose, camera_intrinsics_matrix,
                        image_width, image_height, shading_mode=render_shading_mode,
                        point_size=render_point_size, intensity_highlight_threshold=intensity_highlight_threshold
                    )
                    if render_output:
                        geom_img_path = renders_out_dir / f"{render_name}.png"
                        depth_map_path = renders_out_dir / f"{render_name}_depth.npy"
                        mask_path = renders_out_dir / f"{render_name}_mask.png"
                        try:
                            if cv2: # Check if cv2 is available (dummy or real)
                                cv2.imwrite(str(geom_img_path), render_output['geometric_image'])
                                cv2.imwrite(str(mask_path), render_output['render_mask'])
                        except Exception as e: logging.warning(f"Error saving render image/mask with cv2: {e}")
                        np.save(str(depth_map_path), render_output['depth'])
                        f_list.write(f"{geom_img_path.name}\n")
                        rendered_views_info.append({
                            'name': render_name, 'geometric_image_path': geom_img_path,
                            'depth_map_path': depth_map_path, 'mask_path': mask_path,
                            'pose': render_output['pose']
                        })
                    else:
                        logging.warning(f"Failed to render view {i} ({render_name})")
            if not rendered_views_info and render_poses_list : # Only error if poses were provided but nothing rendered
                logging.error("FATAL: No views were rendered successfully despite having render poses.")
                return
        save_ok = save_processed_data( output_dir=mid_data_dir, processed_lidar_data=processed_lidar_data, rendered_views_info=rendered_views_info )
        if not save_ok: logging.warning("Failed to save intermediate processed data.")
        processed_lidar_data_loaded = processed_lidar_data
        rendered_views_info_loaded = rendered_views_info


    if not arg_config.steps or 2 in arg_config.steps:
        # ... (Step 2: Feature Extraction & Masking - code remains the same) ...
        logging.info("--- Running Step 2: Feature Extraction & Masking ---")
        query_undistortion_ok = undistort_images_fisheye( image_list_path=query_image_list_file, original_image_dir=query_image_dir, output_image_dir=query_image_dir_undistorted, K=camera_intrinsics_matrix, D=camera_distortion_array, new_size=(image_width, image_height) )
        if not query_undistortion_ok: logging.error("Query image undistortion failed. Aborting."); return
        logging.info("Extracting features for UNDISTORTED Query Images...")
        if features_path.exists(): logging.warning(f"Deleting existing features file: {features_path}"); features_path.unlink()
        query_extraction_ok = False
        try:
            conf_dict = extract_features.confs[str(feature_conf)]; extract_features.main( conf=conf_dict, image_dir=query_image_dir_undistorted, image_list=query_image_list_file, export_dir=hloc_out_dir, feature_path=features_path )
            if features_path.exists(): query_extraction_ok = True
            else: logging.error(f"Query feature extraction finished, but output file {features_path} not found!")
        except Exception as e: logging.error(f"ERROR during Query feature extraction:\n{traceback.format_exc()}")
        if not query_extraction_ok: logging.error("Query feature extraction failed."); return
        logging.info("Extracting features for Rendered Images...")
        render_extraction_ok = False
        if not render_image_list_path.exists(): logging.error(f"Render image list {render_image_list_path} not found. Cannot extract render features."); return
        try:
            conf_dict = extract_features.confs[str(feature_conf)]; extract_features.main( conf=conf_dict, image_dir=renders_out_dir, image_list=render_image_list_path, export_dir=hloc_out_dir, feature_path=features_path )
            render_extraction_ok = True
        except Exception as e: logging.error(f"ERROR during Render feature extraction:\n{traceback.format_exc()}")
        if not render_extraction_ok: logging.error("Render feature extraction failed."); return
        logging.info("Applying masks to rendered features...")
        masking_completed_ok = apply_masks_to_features( feature_file_path=masked_render_features_path, image_list_path=render_image_list_path, image_base_dir=renders_out_dir, mask_suffix=mask_suffix, neighborhood_size=2 )
        if not masking_completed_ok: logging.warning("Mask application to features reported failure or errors.")
        logging.info("Checking and fixing features...")
        check_fix_ok = check_and_fix_features(features_path, 256)
        if not check_fix_ok: logging.error("Critical errors found during feature check/fix.")
        if visualize_steps:
            visualize_features( h5_feature_path=features_path, image_list_path=query_image_list_file, image_base_dir=query_image_dir_undistorted, vis_output_dir=vis_base_output_dir / 'query_undistorted', num_to_vis=num_images_to_visualize, prefix="query_undistorted_vis" )
            visualize_features( h5_feature_path=masked_render_features_path, image_list_path=render_image_list_path, image_base_dir=renders_out_dir, vis_output_dir=vis_base_output_dir / 'render_masked', num_to_vis=num_images_to_visualize, prefix="render_masked_vis" )


    if not arg_config.steps or 3 in arg_config.steps:
        logging.info("--- Running Step 3: Matching, PnP & Joint Refinement ---")
        # --- 3.1 Matching ---
        logging.info("Running Distance-Based Feature Matching...")
        if not features_path.exists(): logging.error(f"Feature file missing: {features_path}. Cannot match."); return
        if not render_image_list_path.exists(): logging.error(f"Render image list {render_image_list_path} not found. Cannot match."); return
        if matches_output_path.exists(): logging.warning(f"Deleting existing matches file: {matches_output_path}"); matches_output_path.unlink()
        matching_ok = match_by_distance( features_path=features_path, query_image_list_file=query_image_list_file, render_image_list_file=render_image_list_path, matches_output_path=matches_output_path, distance_threshold_px=distance_threshold_px )
        if not matching_ok: logging.error("Distance matching failed."); return

        # --- 3.2 Load Data (if needed) ---
        if processed_lidar_data_loaded is None or rendered_views_info_loaded is None:
             logging.info("Loading processed data for PnP...")
             processed_lidar_data_loaded, rendered_views_info_loaded = load_processed_data( output_dir=mid_data_dir, rebuild_kdtree=True )
             if processed_lidar_data_loaded is None or rendered_views_info_loaded is None: logging.error("Failed to load processed data for PnP."); return
             if 'kdtree' not in processed_lidar_data_loaded or processed_lidar_data_loaded['kdtree'] is None: logging.error("KDTree missing from loaded data."); return

        # --- 3.3 Prepare Query Info ---
        logging.info("Reading query image list and parsing timestamps...")
        try:
            query_image_names.clear(); query_timestamps_rec_us.clear(); query_name_to_idx.clear(); query_idx_to_name.clear(); query_timestamps_rec_us_indexed.clear()
            raw_query_names = [line.strip() for line in query_image_list_file.read_text().splitlines() if line.strip()]
            for i, name in enumerate(raw_query_names):
                try: ts_us = int(Path(name).stem); query_image_names.append(name); query_timestamps_rec_us[name] = ts_us; query_name_to_idx[name] = i; query_idx_to_name[i] = name
                except ValueError: logging.warning(f"Could not parse timestamp from {name}. Skipping.")
            if not query_image_names: logging.error("No query images with parseable timestamps found."); return
            query_timestamps_rec_us_indexed = { query_name_to_idx[name]: ts_us for name, ts_us in query_timestamps_rec_us.items() if name in query_name_to_idx }
            if len(query_timestamps_rec_us_indexed) != len(query_image_names): logging.warning("Mismatch creating indexed timestamp map.")
        except Exception as e: logging.error(f"Error reading query list/parsing timestamps: {e}", exc_info=True); return

        # --- 3.4 Initial Visualizations (Optional) ---
        if visualize_steps:
             logging.info(f"Generating Initial Map Projection Visualizations ({num_images_to_visualize} images)...")
             vis_count = 0
             # Ensure render_poses_list is available and has entries
             if not render_poses_list:
                 logging.warning("render_poses_list is empty, cannot generate initial visualizations.")
             else:
                 for query_idx in sorted(query_idx_to_name.keys()):
                     if vis_count >= num_images_to_visualize: break
                     query_name = query_idx_to_name.get(query_idx)
                     if not query_name or query_idx >= len(render_poses_list): continue
                     T_map_cam_initial = render_poses_list[query_idx]
                     if T_map_cam_initial is None: continue
                     try: T_cam_map_initial = np.linalg.inv(T_map_cam_initial)
                     except np.linalg.LinAlgError: continue
                     img_path = query_image_dir_undistorted / query_name
                     if not img_path.exists(): continue
                     vis_proj_path = vis_pnp_output_dir / f"{Path(query_name).stem}_{camera_name_in_list}_initial.jpg"
                     try:
                         visualize_map_projection(str(img_path), processed_lidar_data_loaded, camera_intrinsics_matrix, T_cam_map_initial, str(vis_proj_path), None, visualize_map_point_size)
                         vis_count += 1
                     except Exception as e_vis: logging.error(f"Error visualizing initial map projection for {query_name}: {e_vis}", exc_info=True)
                 logging.info(f"Finished generating {vis_count} initial visualizations.")


        # --- 3.5 PnP Loop ---
        logging.info(f"Running PnP RANSAC (min_inliers: {pnp_min_inliers})...")
        all_pnp_results: list[PnPResult] = []
        nn_distance_threshold = voxel_size * 2.0
        for query_name in query_image_names:
            logging.debug(f"PnP for: {query_name}")
            query_idx = query_name_to_idx[query_name]
            # Ensure rendered_views_info_loaded and processed_lidar_data_loaded are valid
            if rendered_views_info_loaded is None or processed_lidar_data_loaded is None:
                logging.error("Rendered views or processed LiDAR data not loaded. Cannot link matches."); break
            query_kps_np, map_points_3d_np = link_matches_via_depth( query_image_name=query_name, features_path=features_path, matches_path=matches_output_path, rendered_views_info=rendered_views_info_loaded, processed_lidar_data=processed_lidar_data_loaded, camera_intrinsics=camera_intrinsics_matrix, nn_distance_threshold=nn_distance_threshold )
            if query_kps_np.shape[0] < pnp_min_inliers:
                all_pnp_results.append(PnPResult(query_idx, query_name, False, query_kps_np.shape[0])); continue
            try:
                # Ensure cv2.solvePnPRansac is available
                if not cv2 or not hasattr(cv2, 'solvePnPRansac'):
                    logging.error("cv2.solvePnPRansac not available. Cannot perform PnP."); break
                success, rvec, tvec, inliers = cv2.solvePnPRansac( map_points_3d_np.astype(np.float32), query_kps_np.astype(np.float32), camera_intrinsics_matrix, None, iterationsCount=pnp_iterations, reprojectionError=pnp_reprojection_error, confidence=pnp_confidence, flags=getattr(cv2, 'SOLVEPNP_SQPNP', 0) )
                num_found_inliers = len(inliers) if inliers is not None else 0
                if success and num_found_inliers >= pnp_min_inliers:
                    # ... (store PnP result as before) ...
                    inlier_indices = inliers.flatten(); p2d_inliers = query_kps_np[inlier_indices]; P3d_inliers = map_points_3d_np[inlier_indices]
                    T_map_cam_pnp = None # Initialize
                    try: R_mat_pnp, _ = cv2.Rodrigues(rvec); T_cam_map_pnp = np.eye(4); T_cam_map_pnp[:3,:3] = R_mat_pnp; T_cam_map_pnp[:3,3] = tvec.flatten(); T_map_cam_pnp = np.linalg.inv(T_cam_map_pnp)
                    except Exception: pass # Catch broad errors during inv/rodrigues
                    all_pnp_results.append(PnPResult(query_idx, query_name, True, num_found_inliers, p2d_inliers, P3d_inliers, T_map_cam_pnp))
                else: all_pnp_results.append(PnPResult(query_idx, query_name, False, num_found_inliers))
            except Exception as e: logging.error(f"Error during PnP for {query_name}: {e}", exc_info=True); all_pnp_results.append(PnPResult(query_idx, query_name, False, 0))
        successful_pnp = [res for res in all_pnp_results if res.success]
        if not successful_pnp: logging.error("PnP failed for ALL images. Cannot proceed with refinement."); return
        logging.info(f"PnP successful for {len(successful_pnp)} out of {len(query_image_names)} images.")

        # --- 3.6 Joint Refinement on ALL PnP-successful images ---
        # `refined_T_ego_cam` initialized with `initial_T_ego_cam_guess` already
        if successful_pnp:
            logging.info(f"--- Running Joint Extrinsic and Timestamp Refinement on ALL {len(successful_pnp)} PnP-successful Images ---")
            joint_opt_inlier_matches = {res.query_idx: (res.p2d_inliers, res.P3d_inliers) for res in successful_pnp}
            joint_opt_indices = [res.query_idx for res in successful_pnp]
            joint_opt_timestamps_rec_us = {idx: query_timestamps_rec_us_indexed[idx] for idx in joint_opt_indices if idx in query_timestamps_rec_us_indexed}

            if len(joint_opt_timestamps_rec_us) != len(joint_opt_indices):
                 logging.warning("Timestamp records missing for some PnP-successful images. Joint opt will use available ones.")
                 valid_joint_opt_indices = list(joint_opt_timestamps_rec_us.keys())
                 joint_opt_inlier_matches = {idx: joint_opt_inlier_matches[idx] for idx in valid_joint_opt_indices}
                 joint_opt_indices = valid_joint_opt_indices

            if joint_opt_indices: # Proceed if there are images for joint opt
                opt_T_ego_cam, opt_delta_t_map, joint_opt_result_obj = refine_extrinsics_and_timestamps(
                    initial_T_ego_cam=initial_T_ego_cam_guess, # Always start from the initial guess
                    inlier_matches_map=joint_opt_inlier_matches,
                    ego_timestamps_us=ego_timestamps_us,
                    ego_poses=ego_poses_list,
                    query_indices=joint_opt_indices, # Pass only indices of images being optimized
                    query_timestamps_rec_us=joint_opt_timestamps_rec_us,
                    camera_intrinsics=camera_intrinsics_matrix,
                    dt_bounds_seconds=dt_bounds_joint_opt,
                    loss_function=loss_function,
                    verbose=opt_verbose,
                    interpolation_tolerance_us=1
                )
                if opt_T_ego_cam is not None:
                    refined_T_ego_cam = opt_T_ego_cam # Update the global T_ego_cam
                    # opt_delta_t_map contains dt for PnP-successful images.
                    # Update final_delta_t_map with these.
                    for q_idx, dt_val in opt_delta_t_map.items():
                        if q_idx in joint_opt_indices: # Ensure it was part of this optimization
                            final_delta_t_map[q_idx] = dt_val
                    logging.info("Joint refinement successful for PnP-successful images.")
                    logging.info(f"Refined T_ego_cam:\n{np.round(refined_T_ego_cam, 6)}")
                    # Log stats for the dts that were actually optimized
                    optimized_dt_values_from_joint = [dt for idx, dt in final_delta_t_map.items() if idx in joint_opt_indices]
                    if optimized_dt_values_from_joint:
                        logging.info(f"Refined Delta_t (seconds) from joint opt (for {len(optimized_dt_values_from_joint)} images): Mean={np.mean(optimized_dt_values_from_joint):.6f}s, Std={np.std(optimized_dt_values_from_joint):.6f}s, Min={np.min(optimized_dt_values_from_joint):.6f}s, Max={np.max(optimized_dt_values_from_joint):.6f}s")

                    # Check optimality from joint_opt_result_obj
                    if joint_opt_result_obj and joint_opt_result_obj.optimality > 1e-1: # Stricter check for final run
                        logging.warning(f"Joint optimization finished with high optimality ({joint_opt_result_obj.optimality:.2e}). Results might be suboptimal.")

                else:
                    logging.error("Joint refinement FAILED. `refined_T_ego_cam` will remain the initial guess, and `dt` for PnP-successful images will effectively be 0 from this stage.")
                    # In this case, refined_T_ego_cam is still initial_T_ego_cam_guess.
                    # final_delta_t_map will not be updated for PnP-successful images here,
                    # they will get 0.0 in the next step.
            else:
                logging.warning("No valid images (with timestamps) for joint optimization among PnP-successful ones. `T_ego_cam` remains initial guess.")
        else:
            logging.warning("No PnP-successful images. Skipping joint refinement. `T_ego_cam` remains initial guess.")

        # --- 3.7 Set dt=0 for PnP-failed images ---
        # Ensure all images (PnP success or fail) have an entry in final_delta_t_map
        for pnp_res_orig in all_pnp_results:
            if pnp_res_orig.query_idx not in final_delta_t_map:
                # This will assign dt=0 to PnP-failed images,
                # and also to PnP-successful images if joint opt failed or they were excluded.
                logging.debug(f"Assigning dt=0 to image {pnp_res_orig.query_name} (idx: {pnp_res_orig.query_idx}).")
                final_delta_t_map[pnp_res_orig.query_idx] = 0.0

        logging.info(f"Final delta_t map populated for {len(final_delta_t_map)} images (PnP-successful from joint opt, others 0.0).")

        # --- 3.8 Calculate Final Poses ---
        if refined_T_ego_cam is None: # Should be initial_T_ego_cam_guess if joint opt failed
             logging.error("`refined_T_ego_cam` is None before final pose calculation, though it should default to initial guess. Critical error. Aborting.")
             return
        if not final_delta_t_map and query_idx_to_name : # If we have query images but no dt map (should not happen)
             logging.error("Final delta_t map is unexpectedly empty. Aborting pose calculation.")
             return

        logging.info(f"Calculating Final Poses using refined T_ego_cam and delta_t map...")
        # final_refined_poses_cam_map already initialized
        num_final_poses = 0; num_calc_errors = 0
        for query_idx, query_name in query_idx_to_name.items():
            if query_idx not in final_delta_t_map:
                 logging.warning(f"Delta_t missing for {query_name} in final map. Skipping pose."); num_calc_errors += 1; continue
            if query_idx not in query_timestamps_rec_us_indexed:
                 logging.warning(f"Recorded timestamp missing for {query_name}. Skipping pose."); num_calc_errors += 1; continue
            rec_ts_us = query_timestamps_rec_us_indexed[query_idx]
            delta_t_seconds = final_delta_t_map[query_idx]
            delta_t_us_float = delta_t_seconds * 1_000_000.0
            true_ts_us_float = float(rec_ts_us) + delta_t_us_float
            T_map_ego_final = interpolate_ego_pose_for_opt_us(true_ts_us_float)
            if T_map_ego_final is None:
                logging.warning(f"Could not interpolate final ego pose for {query_name}."); num_calc_errors += 1; continue
            T_map_cam_final = T_map_ego_final @ refined_T_ego_cam
            try: final_refined_poses_cam_map[query_name] = np.linalg.inv(T_map_cam_final)
            except np.linalg.LinAlgError: logging.warning(f"Could not invert final pose for {query_name}.") # T_cam_map will be missing
            num_final_poses += 1
        logging.info(f"Calculated {num_final_poses} final refined poses ({num_calc_errors} errors/skips).")
        if num_final_poses == 0 and len(query_idx_to_name) > 0 :
            logging.error("No final poses could be calculated successfully. Check logs.")

        # --- Step 4: Saving Results ---
        logging.info("--- Running Step 4: Saving Final Results ---")
        # (Saving logic remains the same as previous response, uses refined_T_ego_cam, final_delta_t_map, final_refined_poses_cam_map)
        if refined_T_ego_cam is not None:
            logging.info(f"Saving refined T_ego_cam to {refined_extrinsics_file}")
            logging.info(f"Refined T_ego_cam value to be saved:\n{np.round(refined_T_ego_cam, 6)}")
            try:
                with open(refined_extrinsics_file, 'w') as f:
                    f.write("# Refined T_ego_cam (Sensor to Camera Transformation) - Row Major\n")
                    for row in refined_T_ego_cam: f.write(" ".join(map(str, row)) + "\n")
                logging.info("Successfully saved refined extrinsics.")
            except Exception as e: logging.error(f"Failed to save refined extrinsics: {e}", exc_info=True)
        else: logging.warning("Refined T_ego_cam is None. Cannot save extrinsics.")

        if final_delta_t_map:
            logging.info(f"Saving refined delta_t map ({len(final_delta_t_map)} entries) to {refined_delta_t_file}")
            all_dt_values_final_map = [dt for dt in final_delta_t_map.values()] # Get all values from final map
            if all_dt_values_final_map: # Check if list is not empty
                logging.info(f" -> Final Delta_t Stats (ALL images): Mean={np.mean(all_dt_values_final_map):.6f}s, Std={np.std(all_dt_values_final_map):.6f}s, Min={np.min(all_dt_values_final_map):.6f}s, Max={np.max(all_dt_values_final_map):.6f}s")
            else: logging.info(" -> Final Delta_t map was populated but values list is empty (should not happen if map not empty).")
            try:
                with open(refined_delta_t_file, 'w', newline='') as f:
                    writer = csv.writer(f); writer.writerow(['image_name', 'delta_t_seconds'])
                    sorted_indices = sorted(final_delta_t_map.keys(), key=lambda idx: query_idx_to_name.get(idx, str(idx)))
                    for query_idx in sorted_indices:
                        writer.writerow([query_idx_to_name.get(query_idx, f"idx_{query_idx}"), f"{final_delta_t_map[query_idx]:.9f}"])
                logging.info("Successfully saved refined delta_t map.")
            except Exception as e: logging.error(f"Failed to save delta_t map: {e}", exc_info=True)
        else: logging.warning("Final delta_t map is empty. Cannot save delta_t CSV.")

        logging.info(f"Saving {len(final_refined_poses_cam_map)} final refined poses (T_cam_map) to {refined_poses_file}")
        if final_refined_poses_cam_map: save_poses_to_colmap_format(final_refined_poses_cam_map, refined_poses_file)
        else: logging.warning("No refined poses (T_cam_map) to save.")

        # --- Final Visualizations (Optional) ---
        if visualize_steps and final_refined_poses_cam_map:
            logging.info(f"Generating Final Map Projection Visualizations ({num_images_to_visualize} images)...")
            # (Same visualization logic as before)
            vis_count = 0
            sorted_query_names = sorted(final_refined_poses_cam_map.keys())
            for query_name in sorted_query_names:
                 if vis_count >= num_images_to_visualize: break
                 T_cam_map_final = final_refined_poses_cam_map.get(query_name)
                 if T_cam_map_final is None: continue # Should not happen if key exists, but safe check
                 img_path = query_image_dir_undistorted / query_name
                 if not img_path.exists(): continue
                 vis_proj_path = vis_pnp_output_dir / f"{Path(query_name).stem}_{camera_name_in_list}_refined.jpg"
                 try:
                     visualize_map_projection(str(img_path), processed_lidar_data_loaded, camera_intrinsics_matrix, T_cam_map_final, str(vis_proj_path), None, visualize_map_point_size)
                     vis_count += 1
                 except Exception as e_vis: logging.error(f"Error visualizing final projection for {query_name}: {e_vis}", exc_info=True)
            logging.info(f"Finished generating {vis_count} final visualizations.")
        elif visualize_steps:
             logging.warning("Skipping final visualizations because no refined poses were available.")

    logging.info("Pipeline execution finished.")

# ==============================================
# Example Main Block to Run the Pipeline
# ==============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Camera extrinsic and time offset refinement pipeline with staged optimization.')
    parser.add_argument('-s', '--steps', nargs='*', type=int, default=None, help='List of steps to run (e.g., 1 2 3)')
    parser.add_argument('-c', '--camera_name', type=str, default=None, help='Camera name to be processed.')
    # Add argument for controlling the number of images in joint optimization
    parser.add_argument('--num_joint_opt', type=int, default=20, help='Number of top images (by inliers) to use for joint extrinsic/time optimization.')
    args = parser.parse_args()

    # --- Configuration ---
    # Make sure the LiDAR map has intensity if you want to use highlighting
    CAMERA_NAME = args.camera_name
    INPUT_PATH = Path("input_tmp")
    LIDAR_MAP_PATH = INPUT_PATH / "whole_map.pcd"
    QUERY_IMG_DIR = INPUT_PATH / CAMERA_NAME 
    QUERY_IMG_LIST = INPUT_PATH / ("query_image_list_" + CAMERA_NAME + ".txt") # Text file, one image name per line (e.g., frame_00101.png)
    OUTPUT_DIR = Path("output")
    INIT_POSE_PATH = INPUT_PATH / "null_0_0_0_local2global_cam_pose.csv"
    EGO_POSE_CSV_PATH = INPUT_PATH / "null_0_0_0_local2global_pose.csv"

    cam_config_file_path = INPUT_PATH / "cameras.cfg"

    try:
        with open(cam_config_file_path, "r") as f:
            config_text = f.read()
        logging.info(f"Read config file {cam_config_file_path}")
    except FileNotFoundError:
        logging.error(f"Error: Config file {cam_config_file_path} not found!")
        exit(1)
    except Exception as e:
         logging.error(f"Error reading config file {cam_config_file_path}: {e}")
         exit(1)

    # Use your parser
    all_parsed_configs = parse_camera_configs(config_text)
    if not all_parsed_configs:
        logging.error("Failed to parse any configurations from the file.")
        exit(1)

    # --- Extract Parameters for the Target Camera ---
    logging.info(f"Extracting parameters for camera '{CAMERA_NAME}'...")
    # Use the new wrapper function
    parsed_data = get_camera_params_from_parsed(all_parsed_configs, CAMERA_NAME)

    if parsed_data:
        # Now you can assign these values to your pipeline variables
        INITIAL_T_EGO_CAM_GUESS = parsed_data['T_ego_cam']
        assert INITIAL_T_EGO_CAM_GUESS.shape == (4,4), "Initial T_ego_cam guess must be a 4x4 matrix."
        IMG_W = parsed_data['img_width']
        IMG_H = parsed_data['img_height']
        K_MATRIX = parsed_data['K']
        D_FISHEYE = parsed_data['D']
        MODEL_TYPE = parsed_data['model_type']

        print("\n--- Parsed Values ---")
        print(f"Camera Model: {MODEL_TYPE}")
        print(f"IMG_W: {IMG_W}, IMG_H: {IMG_H}")
        print(f"K_MATRIX:\n{K_MATRIX}")
        print(f"D_FISHEYE:\n{D_FISHEYE}")
        print(f"INITIAL_T_EGO_CAM_GUESS (T_sensor_cam):\n{INITIAL_T_EGO_CAM_GUESS}")

        RENDER_POSES = get_init_poses(INIT_POSE_PATH, CAMERA_NAME, QUERY_IMG_LIST)

        # --- Check files/folders ---
        if not LIDAR_MAP_PATH.exists(): raise FileNotFoundError(f"LiDAR map not found: {LIDAR_MAP_PATH}")
        if not QUERY_IMG_DIR.is_dir(): raise NotADirectoryError(f"Query image directory not found: {QUERY_IMG_DIR}")
        if not QUERY_IMG_LIST.exists(): raise FileNotFoundError(f"Query image list not found: {QUERY_IMG_LIST}")
        if not EGO_POSE_CSV_PATH.exists(): raise FileNotFoundError(f"Ego pose CSV not found: {EGO_POSE_CSV_PATH}")

        reso_ratio = float(IMG_W)/1920.0
        render_point_size = 4 * reso_ratio
        distance_threshold_px = int(30 * reso_ratio)
        pnp_reprojection_error = int(4 * reso_ratio)
        visualize_map_point_size = 1.0 * reso_ratio

        # --- Run ---
        run_pipeline(
            arg_config=args,
            lidar_map_file=LIDAR_MAP_PATH,
            query_image_dir=QUERY_IMG_DIR,
            query_image_list_file=QUERY_IMG_LIST,
            output_dir=OUTPUT_DIR,
            render_poses_list=RENDER_POSES,
            ego_pose_file=EGO_POSE_CSV_PATH,
            initial_T_ego_cam_guess=INITIAL_T_EGO_CAM_GUESS,
            camera_intrinsics_matrix=K_MATRIX,
            camera_distortion_array=D_FISHEYE,
            image_width=IMG_W, image_height=IMG_H,
            camera_name_in_list=CAMERA_NAME,
            # Optional params:
            voxel_size=0.03, # Adjust voxel size as needed
            min_height=-2.0, # Filter ground points more aggressively if needed
            device="auto", # Use "CUDA:0" or "CPU:0" to force
            render_shading_mode='normal', # 'checkerboard' or 'normal'
            render_point_size=render_point_size, # Tune this based on point density and resolution! Smaller for dense clouds.
            intensity_highlight_threshold=0.1, # Example: Highlight points with intensity > 0.8
            feature_conf='superpoint_aachen', # Or 'superpoint_max', 'superpoint_inloc', etc.
            matcher_conf='superpoint+lightglue', # Or 'NN-superpoint' (if descriptors match), 'superglue-fast'
            distance_threshold_px=distance_threshold_px,
            pnp_min_inliers=5, # Increase if needed
            pnp_reprojection_error=pnp_reprojection_error, # Maybe tighten this
            pnp_iterations=500,
            pnp_confidence=0.999999,
            num_top_images_for_joint_opt=40,
            dt_bounds_joint_opt=(-0.05, 0.05),
            loss_function='cauchy',
            visualize_steps=True,
            num_images_to_visualize=len(RENDER_POSES),
            visualize_map_point_size=visualize_map_point_size,
            opt_verbose=0,
        )

    else:
        print(f"Could not parse configuration for {CAMERA_NAME}.")

