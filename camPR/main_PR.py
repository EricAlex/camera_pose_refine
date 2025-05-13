import open3d as o3d
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
import subprocess # For running hloc commands
from scipy.spatial import KDTree
import logging
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from bisect import bisect_left, bisect_right
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
    logging.info(f"  Depth/Mask Point Size: 1.0") # Explicitly log the fixed size
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
        mat_depth_mask.point_size = 1.0 # Use fixed size 1.0 for accuracy
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

def visualize_map_projection(
    query_image_path: Path,
    processed_lidar_data: dict, # Needs 'points' and 'intensities' (intensities no longer strictly needed but good to have)
    camera_intrinsics: np.ndarray, # 3x3
    pose_cam_from_map: np.ndarray, # 4x4, Camera <- Map transform
    output_path: Path,
    dist_coeffs: np.ndarray = None,
    point_size: int = 1, # Size of projected points
    max_vis_points: int = 1000000, # Limit points projected for performance
    cmap_name: str = 'jet', # Use reversed viridis for depth (closer=brighter/hotter)
    max_color_depth: float = 50.0, # Max depth to map to the 'hot' end of colormap
    min_color_depth: float = 1.0   # Min depth to map to the 'cold' end of colormap
):
    """
    Projects the processed LiDAR point cloud onto the query image using the
    provided camera pose and colors the points by their distance to the camera.
    Points are drawn from far to near for correct occlusion simulation.

    Args:
        query_image_path (Path): Path to the query image.
        processed_lidar_data (dict): Dict containing 'points' (Nx3) from the map.
                                     'intensities' is ignored now.
        camera_intrinsics (np.ndarray): 3x3 camera intrinsics matrix.
        pose_cam_from_map (np.ndarray): 4x4 matrix defining the Camera-from-Map
                                        transformation used for projection.
        output_path (Path): Path to save the visualization image.
        dist_coeffs (np.ndarray, optional): Distortion coefficients. Defaults to None.
        point_size (int): Size of the projected points (radius). Defaults to 1.
        max_vis_points (int): Maximum number of points to sample from the map.
        cmap_name (str): Colormap for depth (e.g., 'viridis_r', 'plasma_r', 'jet').
        max_color_depth (float): Max distance for colormap normalization. Points beyond
                                this will be clamped to the max color.
        min_color_depth (float): Min distance for colormap normalization. Points closer
                                than this will be clamped to the min color.
    """
    logging.info(f"Visualizing depth-colored map projection for {query_image_path.name} -> {output_path.name}")
    start_time_vis = time.time()

    if not query_image_path.is_file(): logging.error(f"Query image not found: {query_image_path}"); return
    if not isinstance(processed_lidar_data, dict) or 'points' not in processed_lidar_data: logging.error("Lidar data missing 'points'."); return
    if not isinstance(pose_cam_from_map, np.ndarray) or pose_cam_from_map.shape != (4, 4): logging.error(f"Invalid pose_cam_from_map."); return

    img = cv2.imread(str(query_image_path), cv2.IMREAD_COLOR)
    if img is None: logging.error(f"Failed to load query image: {query_image_path}"); return
    if img.ndim != 3 or img.shape[2] != 3: logging.error(f"Loaded image is not 3-channel BGR! Shape: {img.shape}."); return
    img_h, img_w = img.shape[:2]

    map_points = processed_lidar_data['points']
    if map_points.shape[0] == 0: logging.warning(f"Map has 0 points."); return

    # --- Sample Points ---
    num_map_points = map_points.shape[0]
    indices_to_process = np.arange(num_map_points)
    if max_vis_points is not None and num_map_points > max_vis_points:
        logging.info(f"Sampling {max_vis_points:,} / {num_map_points:,} map points for projection visualization.")
        indices_to_process = np.random.choice(num_map_points, max_vis_points, replace=False)
    map_points_vis = map_points[indices_to_process]
    num_vis_points = map_points_vis.shape[0]

    # --- Transform Points to Camera Frame & Get Depth (Z coordinate) ---
    map_points_vis_h = np.hstack((map_points_vis, np.ones((num_vis_points, 1)))) # Homogeneous coords (N, 4)
    # Transform using Camera <- Map pose
    points_in_cam_h = (pose_cam_from_map @ map_points_vis_h.T).T # Result is (N, 4)
    # Dehomogenize - not strictly necessary for depth/projection but good practice
    # points_in_cam = points_in_cam_h[:, :3] / points_in_cam_h[:, 3, np.newaxis]
    # Extract depth (Z coordinate in camera frame)
    depths_vis = points_in_cam_h[:, 2] # Shape (N,) - Use Z before potential division

    # Filter points behind the camera
    valid_depth_mask = depths_vis > 0
    if not np.any(valid_depth_mask):
        logging.warning("All sampled map points are behind the camera. Cannot visualize projection.")
        # Save blank image? Or just return?
        cv2.putText(img, "No Points in Front of Camera", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        try: output_path.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(output_path), img)
        except Exception as e_save: logging.error(f"Error saving empty proj vis: {e_save}")
        return

    map_points_vis = map_points_vis[valid_depth_mask]
    depths_vis = depths_vis[valid_depth_mask]
    num_vis_points_f = map_points_vis.shape[0] # Number of points in front of camera

    # --- Prepare Pose for OpenCV Projection ---
    try:
        R_cam_map = pose_cam_from_map[:3, :3]; t_cam_map = pose_cam_from_map[:3, 3]
        rvec, _ = cv2.Rodrigues(R_cam_map); tvec = t_cam_map.reshape(3, 1)
    except Exception as e_conv: logging.error(f"Error converting pose: {e_conv}."); return
    if dist_coeffs is None: dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # --- Project Valid 3D Map Points ---
    try:
        projected_points, _ = cv2.projectPoints(
            map_points_vis.astype(np.float64), # Use points in front of camera
            rvec, tvec, camera_intrinsics, dist_coeffs
        )
        projected_points = projected_points.squeeze() # Shape (M, 2)
        if projected_points.ndim == 1: projected_points = projected_points.reshape(1, 2)
    except Exception as e_proj: logging.error(f"Error projecting map points: {e_proj}", exc_info=True); return

    # --- Calculate Colors based on Depth ---
    try:
        # Normalize depths using specified range for consistent coloring
        norm_depths = (depths_vis - min_color_depth) / (max_color_depth - min_color_depth)
        norm_depths = np.clip(norm_depths, 0.0, 1.0) # Clamp values outside the range

        colormap = cm.get_cmap(cmap_name)
        point_colors_rgba = colormap(norm_depths)
        point_colors_bgr = (point_colors_rgba[:, :3][:, ::-1] * 255).astype(np.uint8)
        if point_colors_bgr.shape[0] != num_vis_points_f: # Sanity check
             raise ValueError("Color array shape mismatch after depth filtering.")
    except Exception as e_cmap:
        logging.error(f"Error applying colormap '{cmap_name}' to depths: {e_cmap}. Using default color.")
        point_colors_bgr = np.full((num_vis_points_f, 3), (0, 255, 0), dtype=np.uint8)

    # --- Sort points by depth (Far to Near) ---
    sort_start_time = time.time()
    # Get indices that would sort depths in descending order (far first)
    sort_indices = np.argsort(depths_vis)[::-1]
    projected_points_sorted = projected_points[sort_indices]
    point_colors_bgr_sorted = point_colors_bgr[sort_indices]
    logging.debug(f"Sorting points by depth took: {time.time() - sort_start_time:.3f}s")

    # --- Filter points outside image bounds (on the *sorted* array) ---
    pixel_coords_sorted = projected_points_sorted.round().astype(int)
    valid_mask = (pixel_coords_sorted[:, 0] >= 0) & (pixel_coords_sorted[:, 0] < img_w) & \
                 (pixel_coords_sorted[:, 1] >= 0) & (pixel_coords_sorted[:, 1] < img_h)

    valid_coords = pixel_coords_sorted[valid_mask]
    valid_colors = point_colors_bgr_sorted[valid_mask]
    valid_projection_count = valid_coords.shape[0]
    logging.info(f"Projected {valid_projection_count} / {num_vis_points_f} points within image bounds after depth sort.")

    # --- Draw Sorted Points ---
    draw_start_time = time.time()
    if point_size <= 1 and valid_projection_count > 0:
         # Direct assignment works because later assignments (closer points) overwrite earlier ones
         img[valid_coords[:, 1], valid_coords[:, 0]] = valid_colors
    elif valid_projection_count > 0:
        for i in range(valid_projection_count): # Loop explicitly respects sorted order
            u, v = valid_coords[i]
            color = tuple(valid_colors[i].tolist())
            cv2.circle(img, (u, v), radius=point_size, color=color, thickness=-1, lineType=cv2.LINE_AA)
    logging.debug(f"Drawing points took: {time.time() - draw_start_time:.3f}s")

    # --- Add Colorbar (Optional) ---
    # This is tricky to overlay directly on the image with cv2.
    # Consider saving a separate colorbar image using matplotlib if needed.
    # Or draw a simple gradient rect + text on the image itself.

    # --- Save Image ---
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(output_path), img)
        if success: logging.info(f"Saved depth-colored map projection to {output_path}")
        else: logging.error(f"Failed to save depth projection image to {output_path}")
    except Exception as e: logging.error(f"Error saving depth projection image {output_path}: {e}", exc_info=True)

    logging.debug(f"Total map projection visualization time: {time.time() - start_time_vis:.3f}s")

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


# --- Undistortion Function ---
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

# --- Main Pipeline Orchestration (MODIFIED) ---
def run_pipeline(
    lidar_map_file,
    query_image_dir,
    query_image_list_file, # Text file listing query image names
    output_dir,
    render_poses_list, # List of 4x4 numpy arrays for rendering viewpoints
    camera_intrinsics_matrix, # 3x3 numpy array
    camera_distortion_array, # Shape (4, 1)
    image_width, image_height,
    # Preprocessing Params
    min_height=-2, voxel_size=0.03, normal_radius=0.15, normal_max_nn=50, device="auto",
    # Rendering Params
    render_shading_mode='normal', render_point_size=2, intensity_highlight_threshold=None, # Added intensity threshold
    # Hloc Params
    feature_conf='superpoint_aachen', matcher_conf='superglue',
    # PnP Params (within estimate_final_pose)
):
    # --- Setup ---
    output_dir = Path(output_dir) # Ensure output_dir is a Path object
    hloc_out_dir = output_dir / 'hloc'
    query_image_dir_undistorted = hloc_out_dir / 'query_images_undistorted'
    renders_out_dir = output_dir / 'renders'
    render_image_list_path = renders_out_dir / "render_list.txt"
    results_file = output_dir / 'refined_poses.txt'
    vis_base_output_dir = hloc_out_dir / 'visualizations'
    num_images_to_visualize = 3
    mask_suffix = "_mask.png"
    mid_data_dir = output_dir / "mid_data_cache"

    # Ensure input paths are Path objects for consistency
    query_image_dir = Path(query_image_dir)
    query_image_list_file = Path(query_image_list_file)
    lidar_map_file = Path(lidar_map_file)

    # Use Path.mkdir()
    output_dir.mkdir(parents=True, exist_ok=True)
    hloc_out_dir.mkdir(parents=True, exist_ok=True)
    renders_out_dir.mkdir(parents=True, exist_ok=True)
    vis_base_output_dir.mkdir(parents=True, exist_ok=True)

    feature_output_base_name = 'feats-superpoint-n4096-r1024' # Base name from config
    # *** Define SEPARATE feature file paths using Pathlib ***
    features_filename = f"{feature_output_base_name}.h5"
    features_path = hloc_out_dir / features_filename
    matches_output_path = hloc_out_dir / 'distance_matches.h5'
    # The path for masked render features (could be same as render_features_path if modified in-place)
    masked_render_features_path = features_path # Still a Path object, modifying in place
    vis_pnp_output_dir = vis_base_output_dir / 'pnp'
    vis_pnp_output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Preprocessing ---
    try:
        # Use 'orig' suffix to clarify these are the original processed points
        # Pass lidar_map_file as Path (or str if the function requires it)
        pcd_tensor, processed_lidar_data = preprocess_lidar_map(
            lidar_map_file, min_height, normal_radius, normal_max_nn, voxel_size, device
        )
        if pcd_tensor is None: raise RuntimeError("Preprocessing failed.")
    except Exception as e:
        print(f"FATAL: Preprocessing failed: {e}")
        return # Assuming this is inside a function

    # --- 2. Meshing & Sampling ---
    # SKIPPED - We will render pcd_tensor_orig directly

    # --- 3. Rendering (Using Original Points) ---
    rendered_views_info = []
    # Use Path object with open() - it's supported
    with open(render_image_list_path, 'w') as f_list:
        for i, pose in enumerate(render_poses_list):
            render_name = f"render_{i:05d}"
            print(f"\n--- Rendering view {i+1}/{len(render_poses_list)} ({render_name})---")
            render_output = render_geometric_viewpoint_open3d( # Call the modified function
                pcd_tensor,
                processed_lidar_data,
                pose, camera_intrinsics_matrix, image_width, image_height,
                shading_mode=render_shading_mode,
                # Set point size to 1 for depth accuracy, adjust geom render later if needed visually
                point_size=2.0, # <--- Use 1.0 for accurate depth
                intensity_highlight_threshold=intensity_highlight_threshold
            )
            if render_output:
                geom_img_path = renders_out_dir / f"{render_name}.png"
                depth_map_path = renders_out_dir / f"{render_name}_depth.npy" # <--- Depth map path
                mask_path = renders_out_dir / f"{render_name}_mask.png"

                cv2.imwrite(str(geom_img_path), render_output['geometric_image'])
                np.save(str(depth_map_path), render_output['depth']) # <--- Save depth map
                cv2.imwrite(str(mask_path), render_output['render_mask'])

                f_list.write(f"{geom_img_path.name}\n") # Write geom image name to list

                rendered_views_info.append({
                    'name': render_name,
                    'geometric_image_path': geom_img_path,
                    'depth_map_path': depth_map_path, # <--- Store depth map path
                    'mask_path': mask_path,
                    'pose': render_output['pose']
                })
            else:
                print(f"Warning: Failed to render view {i} ({render_name})")


    if not rendered_views_info:
        print("FATAL: No views were rendered successfully.")
        return # Assuming this is inside a function

    save_ok = save_processed_data(
        output_dir=mid_data_dir,
        processed_lidar_data=processed_lidar_data, # KDTree will be lost here if not saved
        rendered_views_info=rendered_views_info
    )
    if not save_ok: logging.warning("Failed to save intermediate processed data.")

    # --- 4. Hloc Feature Extraction ---
    # --- 4.0. Undistort Query Images ---
    # Pass Path objects directly to the function.
    # Assumes undistort_images_fisheye can handle Path objects. If not, wrap args with str().
    query_undistortion_ok = undistort_images_fisheye(
        image_list_path=query_image_list_file,        # Path object
        original_image_dir=query_image_dir,           # Path object
        output_image_dir=query_image_dir_undistorted, # Path object
        K=camera_intrinsics_matrix,
        D=camera_distortion_array,
        new_size=(image_width, image_height)
    )

    if not query_undistortion_ok:
        logging.error("Query image undistortion failed. Aborting feature extraction.")
        return

    # --- 4.1. Query Feature Extraction (Using UNDISTORTED images) ---
    logging.info("\n--- Running Hloc Feature Extraction for UNDISTORTED Query Images ---")

    # Use Path.exists() to check if file exists
    if features_path.exists():
        logging.warning(f"Deleting existing query features file: {features_path}")
        # Use Path.unlink() to remove the file
        features_path.unlink()

    # *** Check if undistortion was successful before proceeding ***
    if query_undistortion_ok:
        logging.info("\n--- Running Hloc Feature Extraction for UNDISTORTED Query Images (direct call) ---")
        query_extraction_ok = False
        start_time = time.time()
        try:
            # --- Prepare arguments for extract_features.main ---
            # Assume feature_conf holds the string name like "superpoint_aachen"
            conf_name = str(feature_conf) # Ensure it's a string
            if conf_name not in extract_features.confs:
                # Handle case where the config name isn't found
                raise ValueError(f"Configuration '{conf_name}' not found in hloc.extract_features.confs")
            conf_dict = extract_features.confs[conf_name]

            logging.info(f"Calling hloc.extract_features.main with config: {conf_name}")
            logging.info(f"  image_dir: {query_image_dir_undistorted}")
            logging.info(f"  image_list: {query_image_list_file}")
            logging.info(f"  export_dir: {hloc_out_dir}")
            logging.info(f"  feature_path: {features_path}")

            # --- Call the main function directly ---
            extract_features.main(
                conf=conf_dict,                     # Pass the loaded config dictionary
                image_dir=query_image_dir_undistorted, # Pass Path object
                image_list=query_image_list_file,    # Pass Path object to the list file
                export_dir=hloc_out_dir,           # Pass Path object
                feature_path=features_path         # Pass Path object
            )

            # --- Check for success (output file exists) ---
            if features_path.exists():
                query_extraction_ok = True
                logging.info(f"hloc.extract_features.main completed successfully.")
            else:
                # This might happen if main runs but fails internally without exception
                logging.error(f"hloc.extract_features.main finished, but output file {features_path} was not found!")

        except Exception as e:
            # Catch general exceptions from the direct call
            logging.error(f"ERROR during direct call to hloc.extract_features.main for Query:")
            logging.error(traceback.format_exc()) # Log the full traceback

        logging.info(f"Query extraction finished in {time.time() - start_time:.2f} seconds. Success: {query_extraction_ok}")
    else:
        logging.error("Skipping query feature extraction because undistortion failed.")
        query_extraction_ok = False # Ensure flag is false

    # --- 4.2. Query Feature Visualization (Using UNDISTORTED images) ---
    if query_extraction_ok:
        # Pass Path objects to the function (ensure it supports Path or convert to str if needed)
        visualize_features(
            h5_feature_path=features_path,
            image_list_path=query_image_list_file,
            # *** Use the UNDISTORTED image directory for visualization ***
            image_base_dir=query_image_dir_undistorted,
            vis_output_dir=vis_base_output_dir / 'query_undistorted', # Use Path / operator
            num_to_vis=num_images_to_visualize,
            prefix="query_undistorted_vis" # Changed prefix
        )
    else:
        logging.warning("Skipping query visualization because extraction failed or was skipped.")

    # --- 4.3. Rendered Feature Extraction ---
    logging.info("\n--- Running Hloc Feature Extraction for Rendered Images (NO MASKING YET - direct call) ---")
    render_extraction_ok = False
    start_time = time.time()
    try:
        # --- Prepare arguments for extract_features.main ---
        # Reuse the config name/dict from above if it's the same
        conf_name = str(feature_conf)
        if conf_name not in extract_features.confs:
            raise ValueError(f"Configuration '{conf_name}' not found in hloc.extract_features.confs")
        conf_dict = extract_features.confs[conf_name]

        logging.info(f"Calling hloc.extract_features.main with config: {conf_name}")
        logging.info(f"  image_dir: {renders_out_dir}")
        logging.info(f"  image_list: {render_image_list_path}")
        logging.info(f"  export_dir: {hloc_out_dir}")
        logging.info(f"  feature_path: {features_path}") # Output to the same file

        # --- Call the main function directly ---
        extract_features.main(
            conf=conf_dict,
            image_dir=renders_out_dir,         # Use render directory Path
            image_list=render_image_list_path, # Use render list Path
            export_dir=hloc_out_dir,           # Use hloc dir Path
            feature_path=features_path         # Use feature file Path
        )

        # --- Check for success (output file exists/updated) ---
        # Note: Since we write to the same file, checking existence alone might not
        # be sufficient if query extraction failed but created the file.
        # Checking modification time or assuming success if no exception occurred might be better.
        # For simplicity, we'll keep the existence check, assuming main() would raise an error on failure.
        if features_path.exists():
            render_extraction_ok = True
            logging.info(f"hloc.extract_features.main completed successfully.")
        else:
            logging.error(f"hloc.extract_features.main finished, but output file {features_path} was somehow not found!")

    except Exception as e:
        # Catch general exceptions
        logging.error(f"ERROR during direct call to hloc.extract_features.main for Renders:")
        logging.error(traceback.format_exc()) # Log the full traceback

    logging.info(f"Render extraction finished in {time.time() - start_time:.2f} seconds. Success: {render_extraction_ok}")

    # --- 4.4. Apply Masks to Rendered Features (Post-Extraction) ---
    masking_completed_ok = False
    if render_extraction_ok: # Only proceed if extraction was definitely successful
        # Pass Path objects directly to the masking function.
        # Assumes apply_masks_to_features can handle Path objects. If not, wrap args with str().
        masking_completed_ok = apply_masks_to_features(
            feature_file_path=masked_render_features_path, # Path object
            image_list_path=render_image_list_path,        # Path object
            image_base_dir=renders_out_dir,                # Path object
            mask_suffix=mask_suffix,
            neighborhood_size=1 # Example parameter
        )
    else:
        logging.error("Skipping mask application because render extraction failed or feature file potentially missing/incomplete.")

    # --- 4.4.5. Check and Fix Features
    check_fix_ok = False
    if masking_completed_ok:
        # Define the expected descriptor dimension based on the feature type used
        # For SuperPoint, it's typically 256
        expected_dim = 256 # Adjust if using a different feature extractor
        check_fix_ok = check_and_fix_features(features_path, expected_dim)
        if not check_fix_ok:
            logging.error("Critical errors found during query feature check/fix. Subsequent steps might fail.")
            # Decide how to proceed: maybe halt execution, or just log the error and continue cautiously.
            # For now, we just log and rely on subsequent steps potentially failing.
    else:
        logging.warning("Skipping feature check/fix because mask application failed or was skipped.")

    # --- 4.5. Rendered Feature Visualization (After Masking) ---
    if masking_completed_ok: # Visualize only if masking function returned True
        # Pass Path objects directly to the visualization function.
        # Assumes visualize_features can handle Path objects. If not, wrap args with str().
        visualize_features(
            h5_feature_path=masked_render_features_path, # Path object
            image_list_path=render_image_list_path,      # Path object
            image_base_dir=renders_out_dir,              # Path object
            # Use Path / operator for joining paths
            vis_output_dir=vis_base_output_dir / 'render_masked', # Path object
            num_to_vis=num_images_to_visualize,
            prefix="render_masked_vis"
        )
    elif render_extraction_ok: # If extraction worked but masking failed
        logging.warning("Skipping visualization of masked render features because masking step failed or reported errors.")
        # Optionally visualize the *unmasked* renders here if needed for debug:
        visualize_features(
            h5_feature_path=features_path, # Use original unmasked path
            image_list_path=render_image_list_path,
            image_base_dir=renders_out_dir,
            vis_output_dir=vis_base_output_dir / 'render_unmasked', # Different output dir
            num_to_vis=num_images_to_visualize,
            prefix="render_unmasked_vis"
        )
    else: # If extraction didn't even work
        logging.warning("Skipping render visualization because feature file was not created or extraction failed.")


    # --- 5. Hloc Feature Matching (REPLACED with Distance Matching) ---
    print("\n--- Running Simple Distance-Based Feature Matching ---")

    # Check if features file exists
    if not features_path.exists():
        logging.error(f"Feature file missing: {features_path}. Cannot perform distance matching.")
        # return or exit
        exit()

    # Define distance threshold
    distance_threshold = 30.0 # Pixels - ADJUST AS NEEDED

    # Delete existing matches file if needed
    if matches_output_path.exists():
        logging.warning(f"Deleting existing distance matches file: {matches_output_path}")
        matches_output_path.unlink()

    # --- Call the Distance Matching Function ---
    matching_ok = match_by_distance(
        features_path=features_path,
        query_image_list_file=query_image_list_file,
        render_image_list_file=render_image_list_path,
        matches_output_path=matches_output_path,
        distance_threshold_px=distance_threshold
    )
    # -----------------------------------------

    if not matching_ok:
        logging.error("Distance matching failed. Skipping subsequent steps.")
        # return or exit
        exit()

    # --- 6. Linking & 7. Pose Estimation (Loop through query images) ---
    logging.info("\n--- Linking Matches and Estimating Poses ---")
    refined_poses = {} # Dictionary to store: { image_name: pose_matrix (4x4, Map from Camera) }

    try:
        # Read query names using pathlib
        query_names = [line.strip() for line in query_image_list_file.read_text().splitlines() if line.strip()]
    except FileNotFoundError:
        logging.error(f"Query image list file not found at {query_image_list_file}. Cannot proceed.")
        # Handle error appropriately, e.g., return or exit
        exit() # Example
    except Exception as e:
        logging.error(f"Error reading query image list {query_image_list_file}: {e}", exc_info=True)
        exit() # Example

    if not query_names:
        logging.error("Query image list is empty.")
        exit() # Example
    
    processed_lidar_data_loaded, rendered_views_info_loaded = load_processed_data(
        output_dir=mid_data_dir,
        rebuild_kdtree=True # <--- MUST rebuild KDTree now!
    )
    if processed_lidar_data_loaded is None or rendered_views_info_loaded is None:
        logging.error("Failed to load processed data. Cannot continue."); exit()
    # Check if KDTree was successfully rebuilt
    if 'kdtree' not in processed_lidar_data_loaded or processed_lidar_data_loaded['kdtree'] is None:
        logging.error("KDTree is missing from loaded processed data. Cannot perform linking."); exit()

    # Define NN search distance threshold (e.g., based on voxel size)
    nn_distance_threshold = voxel_size * 2.0 # Example: Allow points within 2x voxel size

    for i, query_name in enumerate(query_names):
        logging.info(f"\n=== Processing Query Image: {query_name} ===")

        query_img_full_path = query_image_dir_undistorted / query_name # Define once

        # --- 6. Linking (Call NEW linking function) ---
        query_kps_np, processed_3d_pts_np = link_matches_via_depth( # <--- CALL NEW FUNCTION
            query_image_name=query_name,
            features_path=features_path,
            matches_path=matches_output_path, # Use the actual matches file
            rendered_views_info=rendered_views_info_loaded, # Has depth_map_path
            processed_lidar_data=processed_lidar_data_loaded, # Has kdtree
            camera_intrinsics=camera_intrinsics_matrix,
            nn_distance_threshold=nn_distance_threshold
        )

        # Check if linking returned non-empty numpy arrays
        # Using .shape[0] is safer than .size != 0 for empty arrays
        if query_kps_np.shape[0] == 0 or processed_3d_pts_np.shape[0] == 0:
            logging.warning(f"No valid 2D-3D links found for {query_name}. Skipping pose estimation.")
            continue
        
        # --- Visualization BEFORE PnP ---
        if query_img_full_path.exists():
            # 1. Visualize PnP Matches (Optional - keep if useful)
            vis_pnp_path_before = vis_pnp_output_dir / f"{Path(query_name).stem}_pnp_match_before.jpg"
            try:
                T_cam_map_prior = np.linalg.inv(render_poses_list[i]) if i < len(render_poses_list) else None
                if T_cam_map_prior is not None:
                    visualize_pnp_matches(
                        query_image_path=query_img_full_path, points2D=query_kps_np,
                        points3D=processed_3d_pts_np, camera_intrinsics=camera_intrinsics_matrix,
                        output_path=vis_pnp_path_before, estimated_pose_cam_from_map=T_cam_map_prior,
                        dist_coeffs=None
                    )
            except Exception as e_vis_pnp:
                logging.error(f"Error visualizing PnP matches (before): {e_vis_pnp}")

            # 2. Visualize Map Projection using PRIOR pose
            vis_proj_path_before = vis_pnp_output_dir / f"{Path(query_name).stem}_map_projection_prior.jpg"
            try:
                T_cam_map_prior = np.linalg.inv(render_poses_list[i]) if i < len(render_poses_list) else None
                if T_cam_map_prior is not None:
                    visualize_map_projection(
                        query_image_path=query_img_full_path,
                        processed_lidar_data=processed_lidar_data_loaded, # Pass full map data
                        camera_intrinsics=camera_intrinsics_matrix,
                        pose_cam_from_map=T_cam_map_prior, # Use prior pose
                        output_path=vis_proj_path_before,
                        dist_coeffs=None, # Assume undistorted query image
                        point_size=1 # Use small points for dense projection
                        # max_vis_points=500000 # Adjust if needed
                    )
                else:
                    logging.warning("Prior pose not available for projection visualization.")
            except np.linalg.LinAlgError:
                logging.error(f"Could not invert prior pose {i} for visualization.")
            except Exception as e_vis_proj:
                logging.error(f"Error visualizing prior map projection: {e_vis_proj}")

        else:
            logging.warning(f"Undistorted query image not found for visualization: {query_img_full_path}")

        # --- 7. Pose Estimation ---
        final_pose_mat, pnp_inliers_indices = estimate_final_pose_opencv(
            query_kps_np,
            processed_3d_pts_np,
            camera_intrinsics_matrix,
            image_width,
            image_height,
            dist_coeffs=None # Or pass your actual distortion coeffs array (e.g., camera_distortion_array)
        )

        if final_pose_mat is not None:
            refined_poses[query_name] = final_pose_mat # Store successful pose
        else:
            logging.warning(f"Failed to estimate pose for {query_name}. No refined visualization generated.")
        
        # --- Visualization AFTER PnP ---
        if query_img_full_path.exists() and final_pose_mat is not None:
            try:
                T_cam_map_estimated = np.linalg.inv(final_pose_mat)

                # 1. Visualize PnP Matches with refined pose (Optional - keep if useful)
                vis_pnp_path_after = vis_pnp_output_dir / f"{Path(query_name).stem}_pnp_viz_refined.jpg"
                visualize_pnp_matches(
                    query_image_path=query_img_full_path, points2D=query_kps_np,
                    points3D=processed_3d_pts_np, camera_intrinsics=camera_intrinsics_matrix,
                    output_path=vis_pnp_path_after, estimated_pose_cam_from_map=T_cam_map_estimated,
                    dist_coeffs=None, inlier_indices=pnp_inliers_indices
                )

                # 2. Visualize Map Projection using REFINED pose
                vis_proj_path_after = vis_pnp_output_dir / f"{Path(query_name).stem}_map_projection_refined.jpg"
                visualize_map_projection(
                    query_image_path=query_img_full_path,
                    processed_lidar_data=processed_lidar_data_loaded,
                    camera_intrinsics=camera_intrinsics_matrix,
                    pose_cam_from_map=T_cam_map_estimated, # Use refined pose
                    output_path=vis_proj_path_after,
                    dist_coeffs=None,
                    point_size=1
                    # max_vis_points=500000
                )

            except np.linalg.LinAlgError:
                logging.error("Failed to invert estimated pose matrix for visualization.")
            except Exception as e_vis:
                logging.error(f"Error during refined PnP/Projection visualization for {query_name}: {e_vis}")
        else:
            logging.warning(f"Undistorted query image not found for refined visualization: {query_img_full_path}")

    # --- Save Results ---
    logging.info(f"\n--- Saving {len(refined_poses)} refined poses to {results_file} ---")
    try:
        # Ensure parent directory exists
        results_file.parent.mkdir(parents=True, exist_ok=True)
        # Open results file using pathlib
        with results_file.open('w') as f_res:
            for img_name, pose_matrix_map_from_cam in refined_poses.items():
                if pose_matrix_map_from_cam is None: continue # Skip if pose estimation failed

                try:
                    # We need Camera from Map for the standard COLMAP/HLOC format (qw qx qy qz tx ty tz)
                    # Input pose_matrix is Map from Camera (T_map_cam)
                    # We need T_cam_map = T_map_cam.inverse()
                    T_cam_map_mat = np.linalg.inv(pose_matrix_map_from_cam)

                    # Extract quaternion (w, x, y, z) and translation (x, y, z) from T_cam_map
                    # Create Rotation3d object from the matrix
                    R_pnp = pycolmap.Rotation3d(T_cam_map_mat[:3, :3])
                    # Get the quaternion [w, x, y, z] from the Rotation3d object
                    q = R_pnp.quat
                    t = T_cam_map_mat[:3, 3] # Translation vector t_cam_map

                    # Format: image_name qw qx qy qz tx ty tz (Camera -> Map transformation)
                    f_res.write(f"{img_name} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]}\n")

                except np.linalg.LinAlgError as e_inv:
                    logging.error(f"Error inverting pose matrix for {img_name}: {e_inv}. Skipping save.")
                except Exception as e_save:
                    logging.error(f"Error formatting/saving pose for {img_name}: {e_save}", exc_info=True)

        logging.info(f"Successfully saved poses.")

    except IOError as e:
        logging.error(f"Error writing results file {results_file}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during results saving: {e}", exc_info=True)


    print("Pipeline finished.")

# ==============================================================================
# Function to Load and Prepare Pose Data
# ==============================================================================
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
            logging.error(f"No poses found for camera '{target_camera}' in the CSV.")
            return None

        # Convert timestamp to integer (assuming nanoseconds)
        df_cam['timestamp'] = df_cam['timestamp'].astype(np.int64)

        # Convert pose columns to numeric (float64 for precision) and handle errors
        pose_cols = [f'p{i}' for i in range(16)]
        for col in pose_cols:
            df_cam[col] = pd.to_numeric(df_cam[col], errors='coerce')

        # Drop rows with any NaN values that might have resulted from conversion errors
        df_cam.dropna(subset=['timestamp'] + pose_cols, inplace=True)
        if df_cam.empty:
             logging.error(f"No valid numeric pose data found for camera '{target_camera}' after cleaning.")
             return None

        # Sort by timestamp (essential for interpolation)
        df_cam.sort_values(by='timestamp', inplace=True)
        df_cam.reset_index(drop=True, inplace=True)

        # Convert flattened pose to 4x4 matrix
        poses = []
        timestamps = []
        for index, row in df_cam.iterrows():
            try:
                pose_flat = row[pose_cols].values.astype(np.float64)
                pose_matrix = pose_flat.reshape(4, 4)
                # Optional: Add validation check for pose matrix (e.g., check rotation part)
                # R_mat = pose_matrix[:3, :3]
                # if not np.allclose(R_mat.T @ R_mat, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R_mat), 1.0, atol=1e-5):
                #     logging.warning(f"Invalid rotation matrix found at timestamp {row['timestamp']}. Skipping row {index}.")
                #     continue
                poses.append(pose_matrix)
                timestamps.append(row['timestamp'])
            except ValueError:
                logging.warning(f"Could not reshape pose at timestamp {row['timestamp']}. Skipping row {index}.")
                continue
            except Exception as e:
                 logging.warning(f"Error processing pose at timestamp {row['timestamp']}: {e}. Skipping row {index}.")
                 continue

        if not timestamps:
            logging.error(f"No valid poses could be constructed for camera '{target_camera}'.")
            return None

        logging.info(f"Prepared {len(timestamps)} valid, sorted poses for '{target_camera}'.")
        # Return timestamps and poses as separate lists/arrays for bisect
        return np.array(timestamps), poses

    except FileNotFoundError:
        logging.error(f"Pose CSV file not found at: {csv_path}")
        return None
    except Exception as e:
        logging.error(f"Error reading or processing pose CSV: {e}", exc_info=True)
        return None

# ==============================================================================
# Function to Find or Interpolate Pose
# ==============================================================================
def get_pose_for_timestamp(query_ts, timestamps, poses, tolerance=1000):
    """Finds exact pose or interpolates between two poses using SLERP and LERP."""

    if timestamps is None or not poses:
        logging.warning(f"No pose data available to query for timestamp {query_ts}.")
        return None

    n_poses = len(timestamps)

    # Use bisect_left to find the insertion point for query_ts in the sorted timestamps
    # idx is the index of the first timestamp >= query_ts
    idx = bisect_left(timestamps, query_ts)

    # --- Handle different cases ---

    # Case 1: Exact or near-exact match found
    if idx < n_poses and abs(timestamps[idx] - query_ts) <= tolerance:
        logging.debug(f"Exact match found for timestamp {query_ts} at index {idx}.")
        return poses[idx]
    if idx > 0 and abs(timestamps[idx - 1] - query_ts) <= tolerance:
         logging.debug(f"Exact match found for timestamp {query_ts} at index {idx-1}.")
         return poses[idx-1]


    # Case 2: Query timestamp is before the first recorded pose
    if idx == 0:
        logging.warning(f"Query timestamp {query_ts} is before the first pose timestamp {timestamps[0]}. Cannot interpolate.")
        return None

    # Case 3: Query timestamp is after the last recorded pose
    if idx == n_poses:
        logging.warning(f"Query timestamp {query_ts} is after the last pose timestamp {timestamps[-1]}. Cannot interpolate.")
        return None

    # Case 4: Interpolation needed between index idx-1 and idx
    t0 = timestamps[idx - 1]
    t1 = timestamps[idx]
    pose0 = poses[idx - 1]
    pose1 = poses[idx]

    # Sanity check for time difference
    if t1 <= t0:
        logging.error(f"Invalid timestamp order for interpolation: t0={t0}, t1={t1}. Check sorting.")
        return None

    # Calculate interpolation factor
    alpha = (query_ts - t0) / (t1 - t0)
    logging.debug(f"Interpolating for {query_ts} between {t0} and {t1} with alpha={alpha:.4f}")

    # Extract rotation and translation
    try:
        R0 = R.from_matrix(pose0[:3, :3])
        R1 = R.from_matrix(pose1[:3, :3])
        T0 = pose0[:3, 3]
        T1 = pose1[:3, 3]
    except ValueError as e:
         logging.error(f"Failed to create Rotation object from matrices at {t0} or {t1}. Invalid rotation? Error: {e}")
         return None

    # Interpolate rotation using SLERP
    try:
        slerp_interpolator = Slerp([t0, t1], R.concatenate([R0, R1]))
        R_interp = slerp_interpolator([query_ts])[0] # Evaluate interpolator
    except Exception as e:
        logging.error(f"SLERP failed between timestamps {t0} and {t1}: {e}")
        return None

    # Interpolate translation using LERP
    T_interp = T0 + alpha * (T1 - T0)

    # Reconstruct the interpolated pose matrix
    pose_interp = np.eye(4)
    pose_interp[:3, :3] = R_interp.as_matrix()
    pose_interp[:3, 3] = T_interp

    return pose_interp

def get_init_poses(init_pose_path, camera_name, query_img_list):
    # --- Load Pose Data ---
    pose_data = load_and_prepare_poses(init_pose_path, camera_name)
    if pose_data is None:
        logging.error("Failed to load pose data. Exiting.")
        exit()
    sorted_timestamps, sorted_poses = pose_data

    # --- Process Query Images ---
    query_poses = []
    not_found_count = 0

    try:
        with open(query_img_list, 'r') as f:
            query_image_names = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logging.error(f"Query image list file not found: {query_img_list}")
        exit()

    logging.info(f"\n--- Looking up poses for {len(query_image_names)} query images ---")
    start_lookup_time = time.time()

    for image_name in query_image_names:
        try:
            # Extract timestamp from filename (assuming it's the part before '.jpg')
            base_name = os.path.splitext(image_name)[0]
            query_ts = int(base_name)

            # Get the pose
            interpolated_pose = get_pose_for_timestamp(
                query_ts,
                sorted_timestamps,
                sorted_poses
            )

            if interpolated_pose is not None:
                query_poses.append(interpolated_pose)
                logging.debug(f"Found/Interpolated pose for {image_name} (ts: {query_ts})")
            else:
                logging.warning(f"Could not determine pose for {image_name} (ts: {query_ts}).")
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
    
    return query_poses

# ==============================================
# Example Main Block to Run the Pipeline
# ==============================================

if __name__ == "__main__":
    # --- Configuration ---
    # Make sure the LiDAR map has intensity if you want to use highlighting
    LIDAR_MAP_PATH = "whole_map.pcd" # e.g., merged_point_cloud_with_norm_intensity.ply
    QUERY_IMG_DIR = "panoramic_2/" # e.g., "data/robotcar_images/"
    QUERY_IMG_LIST = "query_image_list.txt" # Text file, one image name per line (e.g., frame_00101.png)
    OUTPUT_DIR = "output/lidar_loc_pipeline/"
    CAMERA_NAME = "panoramic_2"
    INIT_POSE_PATH = "null_0_0_0_local2global_cam_pose.csv"

    # Define camera parameters (EXAMPLE VALUES - USE YOUR CALIBRATED VALUES)
    IMG_W, IMG_H, FX, FY, CX, CY, K1, K2, K3, K4 = 1920, 1440, 444.437439, 444.489471, 960.74469, 719.184692, 0.2189769, -0.0567729, 0.007488, -0.0008241
    K_MATRIX = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
    D_query = np.array([[K1], [K2], [K3], [K4]], dtype=np.float64) # Shape (4, 1)

    render_poses = get_init_poses(INIT_POSE_PATH, CAMERA_NAME, QUERY_IMG_LIST)

    # Add more realistic poses based on initial estimates or trajectory sampling
    # render_poses = [render_pose_1, render_pose_2, ...] # Add many more poses here!

    # --- Check files/folders ---
    if not os.path.exists(LIDAR_MAP_PATH): raise FileNotFoundError(f"LiDAR map not found: {LIDAR_MAP_PATH}")
    if not os.path.isdir(QUERY_IMG_DIR): raise NotADirectoryError(f"Query image directory not found: {QUERY_IMG_DIR}")
    if not os.path.exists(QUERY_IMG_LIST): raise FileNotFoundError(f"Query image list not found: {QUERY_IMG_LIST}")
    if not render_poses: raise ValueError("Render poses list cannot be empty.")
    print(f"Found {len(render_poses)} rendering poses.")

    # --- Run ---
    run_pipeline(
        lidar_map_file=LIDAR_MAP_PATH,
        query_image_dir=QUERY_IMG_DIR,
        query_image_list_file=QUERY_IMG_LIST,
        output_dir=OUTPUT_DIR,
        render_poses_list=render_poses,
        camera_intrinsics_matrix=K_MATRIX,
        camera_distortion_array=D_query,
        image_width=IMG_W, image_height=IMG_H,
        # Optional params:
        voxel_size=0.03, # Adjust voxel size as needed
        min_height=-2.0, # Filter ground points more aggressively if needed
        device="auto", # Use "CUDA:0" or "CPU:0" to force
        render_shading_mode='normal', # 'checkerboard' or 'normal'
        render_point_size=3, # Tune this based on point density and resolution! Smaller for dense clouds.
        intensity_highlight_threshold=0.1, # Example: Highlight points with intensity > 0.8
        feature_conf='superpoint_aachen', # Or 'superpoint_max', 'superpoint_inloc', etc.
        matcher_conf='superpoint+lightglue' # Or 'NN-superpoint' (if descriptors match), 'superglue-fast'
    )