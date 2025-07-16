#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_images_txt(filepath: Path) -> dict:
    """Reads an images.txt file into a dictionary keyed by image_id."""
    images = {}
    if not filepath.is_file():
        logging.error(f"images.txt not found at {filepath}"); return {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip().startswith('#'): continue
            parts = line.strip().split()
            if len(parts) < 10: continue
            
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            
            # Reconstruct the pose matrix to get camera center
            # T_cam_world = [R | t]
            # Camera center in world coords is C = -R.T @ t
            rotation = R.from_quat([qx, qy, qz, qw])
            R_mat = rotation.as_matrix()
            t_vec = np.array([tx, ty, tz])
            
            cam_center = -R_mat.T @ t_vec
            
            images[image_id] = {
                'name': parts[9],
                'cam_center': cam_center,
                'R': R_mat
            }
    logging.info(f"Read {len(images)} image poses from {filepath.name}")
    return images


def main(args):
    images_txt_path = Path(args.workspace_path) / "sparse" / "images.txt"
    output_cfg_path = Path(args.workspace_path) / "dense" / "fusion.cfg"
    
    # Parameters for selecting source images
    max_source_images_per_ref = args.max_source_images
    max_dist_between_cams = args.max_dist
    
    images_data = read_images_txt(images_txt_path)
    if not images_data:
        logging.critical("Could not read image data. Aborting config generation.")
        return

    # Convert to a list for easier iteration and indexing
    image_ids = sorted(images_data.keys())
    num_images = len(image_ids)
    
    # Pre-calculate all camera centers for efficient distance calculation
    cam_centers_array = np.array([images_data[img_id]['cam_center'] for img_id in image_ids])
    
    logging.info(f"Generating image pairs for {num_images} images...")
    
    with open(output_cfg_path, 'w') as f:
        for i, ref_id in enumerate(image_ids):
            ref_image = images_data[ref_id]
            ref_name = ref_image['name']
            ref_center = ref_image['cam_center']
            
            # Calculate distances from this reference camera to all other cameras
            distances = np.linalg.norm(cam_centers_array - ref_center, axis=1)
            
            # Get indices of all other cameras sorted by distance
            sorted_indices = np.argsort(distances)
            
            source_image_names = []
            for j in sorted_indices:
                # Skip the reference image itself
                if i == j: continue
                
                # Stop if we have enough source images
                if len(source_image_names) >= max_source_images_per_ref: break
                
                # Check if the candidate source image is within the max distance
                if distances[j] > max_dist_between_cams:
                    # Since they are sorted by distance, we can stop searching
                    break
                
                src_id = image_ids[j]
                source_image_names.append(images_data[src_id]['name'])
            
            if not source_image_names:
                logging.warning(f"Could not find any source images for {ref_name} within {max_dist_between_cams}m.")
                continue

            # Write the entry to fusion.cfg
            # Format: reference_image_name
            # source_image_name_1,source_image_name_2,...
            f.write(f"{ref_name}\n")
            f.write(','.join(source_image_names) + "\n")
            
    logging.info(f"Successfully wrote image pairs configuration to {output_cfg_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate a fusion.cfg file with image pairs for COLMAP's patch_match_stereo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--workspace_path', type=str, required=True,
        help="Path to the COLMAP project workspace (the directory containing 'sparse', 'dense', etc.)."
    )
    parser.add_argument(
        '--max_source_images', type=int, default=50,
        help="Maximum number of source images to select for each reference image."
    )
    parser.add_argument(
        '--max_dist', type=float, default=10.0,
        help="Maximum distance (in meters) between camera centers to be considered a valid pair."
    )
    main(parser.parse_args())