import re
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import csv
import sys # For sys.exit

def parse_value(value_str):
    """Attempts to parse a value string into bool, int, float, or string."""
    value_str = value_str.strip()
    # Remove trailing commas if any (sometimes present in text formats)
    if value_str.endswith(','):
        value_str = value_str[:-1]

    # Handle strings explicitly
    if value_str.startswith('"') and value_str.endswith('"'):
        return value_str[1:-1] # Remove quotes

    # Handle numbers (float first due to potential decimals/e-notation)
    try:
        return float(value_str)
    except ValueError:
        pass # Not a float

    try:
        return int(value_str)
    except ValueError:
        pass # Not an int

    # Add boolean checks if needed (e.g., true/false)
    # if value_str.lower() == 'true':
    #     return True
    # if value_str.lower() == 'false':
    #     return False

    # Fallback to string if none of the above
    return value_str # Return as is if no other type matches


def parse_camera_configs(config_text):
    """
    Parses multi-config text data similar to protobuf text format.

    Args:
        config_text: A string containing the configuration data.

    Returns:
        A list of dictionaries, where each dictionary represents a parsed 'config' block.
        Returns an empty list if parsing fails or no configs are found.
    """
    configs = []
    current_config_data = None
    # Use a stack to keep track of the current dictionary being populated
    # Stack elements are dictionaries.
    dict_stack = []

    lines = config_text.strip().split('\n')

    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#'): # Skip empty lines and comments
            continue

        # Start of a top-level config block
        if line == 'config {':
            if current_config_data is not None:
                print(f"Warning: Found 'config {{' before previous one ended at line {line_num + 1}. Starting new config.")
                # Optionally handle error or save previous incomplete config
            current_config_data = {}
            dict_stack = [current_config_data] # Start stack with the root dict
            continue

        # End of a block
        if line == '}':
            if not dict_stack:
                print(f"Warning: Found unexpected '}}' at line {line_num + 1}. Ignoring.")
                continue

            closed_dict = dict_stack.pop()

            # If the stack is now empty, it means we closed a top-level 'config' block
            if not dict_stack and current_config_data is not None:
                 # Check if the closed dict is the one we started with
                 if closed_dict is current_config_data:
                      configs.append(current_config_data)
                      current_config_data = None # Reset for the next config
                 else:
                      # This case should ideally not happen with balanced braces
                      print(f"Warning: Mismatched braces ending config block near line {line_num + 1}.")
                      # Decide how to handle: append anyway? discard?
                      configs.append(closed_dict) # Append the dict that was closed
                      current_config_data = None

            continue # Move to the next line

        # Inside a block, process assignments or nested blocks
        if not dict_stack:
            # We are not inside any '{...}' block, but the line is not a start/end brace
            # This could be data outside the main 'config' structure
            print(f"Warning: Found data outside expected block structure at line {line_num + 1}: '{line}'. Ignoring.")
            continue

        # Check for nested block start (e.g., "parameters {")
        match_block = re.match(r'^(\w+)\s*\{$', line)
        if match_block:
            key = match_block.group(1)
            new_dict = {}
            parent_dict = dict_stack[-1] # Get the current dictionary from stack top
            parent_dict[key] = new_dict   # Add the new dictionary to its parent
            dict_stack.append(new_dict)   # Push the new dictionary onto the stack
            continue

        # Check for key-value pair (e.g., "x: 1.23")
        match_kv = re.match(r'^(\w+)\s*:\s*(.+)$', line)
        if match_kv:
            key = match_kv.group(1)
            value_str = match_kv.group(2).strip()
            value = parse_value(value_str) # Parse the value string
            current_dict = dict_stack[-1] # Get the current dictionary
            current_dict[key] = value     # Assign the key-value pair
            continue

        # If the line doesn't match any known pattern inside a block
        print(f"Warning: Could not parse line {line_num + 1}: '{line}'. Ignoring.")


    # Check if a config was started but not properly closed
    if current_config_data is not None:
         print("Warning: Reached end of input but the last 'config' block was not closed.")
         # Optionally append the incomplete config if needed
         # configs.append(current_config_data)

    return configs

def load_camera_configs(parsed_configs):
    cameras = []
    for i, cfg in enumerate(parsed_configs):
        camera_dev = cfg.get('camera_dev')
        if camera_dev is None:
            print(f"Warning: 'camera_dev' not found in config #{i+1}. Skipping this config entry.")
            continue

        extrinsic_base = cfg.get('parameters', {}).get('extrinsic', {}).get('sensor_to_cam', {})
        
        position_data = extrinsic_base.get('position', {})
        x = position_data.get('x')
        y = position_data.get('y')
        z = position_data.get('z')

        orientation_data = extrinsic_base.get('orientation', {})
        qx = orientation_data.get('qx')
        qy = orientation_data.get('qy')
        qz = orientation_data.get('qz')
        qw = orientation_data.get('qw')

        if None in [x, y, z, qx, qy, qz, qw]:
            missing_fields = [name for name, val in [('x',x),('y',y),('z',z),('qx',qx),('qy',qy),('qz',qz),('qw',qw)] if val is None]
            print(f"Warning: Incomplete extrinsic data for camera_dev '{camera_dev}' in config #{i+1}. Missing fields: {missing_fields}. Skipping.")
            continue
        
        try:
            # Ensure they are floats, as parse_value might return int or string representation of float
            x, y, z = float(x), float(y), float(z)
            qx, qy, qz, qw = float(qx), float(qy), float(qz), float(qw)
        except (ValueError, TypeError) as e:
            print(f"Warning: Extrinsic data for camera_dev '{camera_dev}' in config #{i+1} contains non-numeric values. Error: {e}. Skipping.")
            continue

        try:
            # Scipy R.from_quat expects [x, y, z, w]
            # It normalizes quaternions internally, so explicit normalization is often not needed
            # unless very non-unit quaternions are expected.
            quat_cam = np.array([qx, qy, qz, qw])
            rot = R.from_quat(quat_cam).as_matrix()
        except Exception as e:
            print(f"Warning: Could not create rotation from quaternion for camera_dev '{camera_dev}' in config #{i+1}: {[qx, qy, qz, qw]}. Error: {e}. Skipping.")
            continue
            
        # 构建4x4变换矩阵: T_sensor_cam (transform from camera frame to sensor frame)
        T_sensor_cam = np.eye(4)
        T_sensor_cam[:3, :3] = rot
        T_sensor_cam[:3, 3] = [x, y, z]
        cameras.append((camera_dev, T_sensor_cam))
    return cameras

def load_sensor_poses(pose_file):
    """
    Loads sensor poses from a file.
    Each line in the file is expected to be in the format:
    timestamp x y z qx qy qz qw
    (space-separated values)
    Timestamp will be truncated to 16 digits if it's longer.
    """
    sensor_poses = []
    with open(pose_file, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'): # Skip empty lines and comments
                print(f"Info: Skipping empty or comment line {i+1} in {pose_file}")
                continue
            
            parts = line.split() # Splits by any whitespace
            
            if len(parts) != 8: # Expected: timestamp, x, y, z, qx, qy, qz, qw
                print(f"Warning: Line {i+1} in {pose_file} has {len(parts)} values, expected 8. Skipping: '{line}'")
                continue
            
            try:
                timestamp_str = parts[0]
                # Check and truncate timestamp if it's longer than 16 digits
                if len(timestamp_str) > 16:
                    original_timestamp = timestamp_str
                    timestamp_str = timestamp_str[:16]
                    print(f"Info: Timestamp on line {i+1} in {pose_file} was '{original_timestamp}', truncated to '{timestamp_str}'.")
                
                # Ensure the timestamp (even if shorter than 16 digits originally) consists only of digits
                if not timestamp_str.isdigit():
                    print(f"Warning: Timestamp '{timestamp_str}' on line {i+1} in {pose_file} is not purely numeric after potential truncation. Skipping: '{line}'")
                    continue
                
                timestamp = timestamp_str # Keep as string

                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                qx = float(parts[4])
                qy = float(parts[5])
                qz = float(parts[6])
                qw = float(parts[7])
            except ValueError as e:
                print(f"Warning: Could not parse numeric values in line {i+1} of {pose_file}: '{line}'. Error: {e}. Skipping.")
                continue
            except IndexError: 
                print(f"Warning: Line {i+1} in {pose_file} is malformed (not enough parts). Skipping: '{line}'")
                continue

            try:
                quat_np = np.array([qx, qy, qz, qw])
                rotation = R.from_quat(quat_np) 
                rotation_matrix = rotation.as_matrix() # 3x3
            except Exception as e: 
                print(f"Warning: Could not create rotation from quaternion in line {i+1} of {pose_file}: {[qx, qy, qz, qw]}. Error: {e}. Skipping.")
                continue

            T_sensor_global = np.eye(4)
            T_sensor_global[:3, :3] = rotation_matrix
            T_sensor_global[:3, 3] = [x, y, z]
            
            sensor_poses.append((timestamp, T_sensor_global))
            
    if not sensor_poses:
        print(f"Warning: No sensor poses were successfully loaded from {pose_file}.")
    return sensor_poses


def compute_global_camera_poses(sensor_poses, cameras):
    global_camera_poses = []
    for timestamp, T_sensor_global in sensor_poses:
        for camera_dev, T_sensor_cam in cameras:
            T_camera_global = T_sensor_global @ T_sensor_cam
            global_camera_poses.append((timestamp, camera_dev, T_camera_global))
    return global_camera_poses

def write_camera_poses_to_csv(global_poses, output_file):
    """Writes global camera poses to a CSV file."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for timestamp, camera_dev, matrix in global_poses:
            flattened = matrix.flatten().tolist()
            writer.writerow([timestamp, camera_dev] + flattened)
    print(f"Successfully wrote {len(global_poses)} global camera poses to {output_file}")

def write_sensor_poses_to_csv(sensor_poses_data, output_file):
    """Writes loaded sensor poses to a CSV file."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for timestamp, matrix in sensor_poses_data:
            flattened = matrix.flatten().tolist()
            writer.writerow([timestamp] + flattened)
    print(f"Successfully wrote {len(sensor_poses_data)} sensor poses to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Calculate global camera poses from sensor extrinsics and sensor trajectory. Also outputs processed sensor poses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-c", "--camera-cfg",
        required=True,
        help="Camera configuration file path (protobuf-like text format)."
    )
    parser.add_argument(
        "-p", "--input-ego-pose-file", # Renamed for clarity (was --ego-pose)
        required=True,
        help="Input ego sensor pose file path. Each line: timestamp x y z qx qy qz qw (space-separated)."
    )
    parser.add_argument(
        "-e", "--output-ego-csv", # New argument for sensor pose CSV output
        required=True,
        help="Output CSV file path for processed global ego sensor poses (timestamp, matrix)."
    )
    parser.add_argument(
        "-o", "--output-camera-csv", # Renamed for clarity (was --output-file)
        required=True,
        help="Output CSV file path for global camera poses (timestamp, camera_dev, matrix)."
    )

    args = parser.parse_args()

    try:
        with open(args.camera_cfg, "r") as f:
            config_text = f.read()
    except FileNotFoundError:
        print(f"Error: Camera config file ({args.camera_cfg}) not found!")
        sys.exit(1)

    parsed_configs = parse_camera_configs(config_text)
    if not parsed_configs:
        print(f"Error: No camera configurations parsed from {args.camera_cfg}. Exiting.")
        sys.exit(1)

    cameras = load_camera_configs(parsed_configs)
    if not cameras:
        print(f"Error: No valid camera extrinsics loaded from parsed configurations. Exiting.")
        sys.exit(1)
    print(f"Loaded {len(cameras)} camera extrinsic configurations.")

    try:
        sensor_poses = load_sensor_poses(args.input_ego_pose_file)
    except FileNotFoundError:
        print(f"Error: Input ego pose file ({args.input_ego_pose_file}) not found!")
        sys.exit(1)
        
    if not sensor_poses:
        print(f"Error: No sensor poses loaded from {args.input_ego_pose_file}. Exiting.")
        sys.exit(1)
    print(f"Loaded {len(sensor_poses)} sensor poses.")
    
    # Write loaded sensor poses to their own CSV file
    try:
        write_sensor_poses_to_csv(sensor_poses, args.output_ego_csv)
    except IOError as e:
        print(f"Error writing sensor poses to {args.output_ego_csv}: {e}")
        sys.exit(1)
        
    global_camera_poses = compute_global_camera_poses(sensor_poses, cameras)
    if not global_camera_poses:
        # This might be normal if, for example, sensor_poses was empty (though we check above)
        # or if cameras was empty (also checked above).
        # If sensor_poses and cameras are non-empty, this implies an issue in compute_global_camera_poses
        # or that it simply resulted in an empty list for valid reasons.
        print(f"Warning: No global camera poses were computed. The output camera pose file will be empty.")
    
    try:
        write_camera_poses_to_csv(global_camera_poses, args.output_camera_csv)
    except IOError as e:
        print(f"Error writing camera poses to {args.output_camera_csv}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()