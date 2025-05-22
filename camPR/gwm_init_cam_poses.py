import re
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import csv

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
    for cfg in parsed_configs:
        camera_dev = cfg.get('camera_dev')
        extrinsic = cfg.get('parameters', {}).get('extrinsic', {}).get('sensor_to_cam', {})
        x, y, z = extrinsic.get('position', {}).get('x'), extrinsic.get('position', {}).get('y'), extrinsic.get('position', {}).get('z')
        qx, qy, qz, qw = extrinsic.get('orientation', {}).get('qx'), extrinsic.get('orientation', {}).get('qy'), extrinsic.get('orientation', {}).get('qz'), extrinsic.get('orientation', {}).get('qw')
        rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
        # 构建4x4变换矩阵
        T_sensor_cam = np.eye(4)
        T_sensor_cam[:3, :3] = rot
        T_sensor_cam[:3, 3] = [x, y, z]
        cameras.append((camera_dev, T_sensor_cam))
    return cameras

def load_sensor_poses(pose_file):
    sensor_poses = []
    with open(pose_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            timestamp = row[0]
            matrix = np.array(list(map(float, row[1:]))).reshape(4, 4)
            sensor_poses.append((timestamp, matrix))
    return sensor_poses

def compute_global_camera_poses(sensor_poses, cameras):
    global_camera_poses = []
    for timestamp, T_sensor_global in sensor_poses:
        for camera_dev, T_sensor_cam in cameras:
            # T_camera_global = T_sensor_global @ np.linalg.inv(T_sensor_cam)
            # T_sensor_cam 实际上是camera到sensor的变换矩阵
            T_camera_global = T_sensor_global @ T_sensor_cam
            global_camera_poses.append((timestamp, camera_dev, T_camera_global))
    return global_camera_poses

def write_to_csv(global_poses, output_file):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入数据
        for timestamp, camera_dev, matrix in global_poses:
            flattened = matrix.flatten().tolist()
            writer.writerow([timestamp, camera_dev] + flattened)

def main():
    # 配置命令行参数
    parser = argparse.ArgumentParser(
        description="Camera 在建图全局坐标中的初始pose",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-c", "--camera-cfg",
        required=True,
        help="相机内外参文件地址"
    )
    parser.add_argument(
        "-p", "--ego-pose",
        required=True,
        help="自车 sensor local2global pose文件地址"
    )
    parser.add_argument(
        "-o", "--output-file",
        required=True,
        help="输出文件地址"
    )

    # 参数验证
    args = parser.parse_args()

    # 从文件读取配置内容
    try:
        with open(args.camera_cfg, "r") as f:
            config_text = f.read()
    except FileNotFoundError:
        print(f"Error: cameras.cfg ({args.camera_cfg})not found!")
        exit()

    parsed_configs = parse_camera_configs(config_text)

    cameras = load_camera_configs(parsed_configs)

    sensor_poses = load_sensor_poses(args.ego_pose)
    
    global_camera_poses = compute_global_camera_poses(sensor_poses, cameras)

    write_to_csv(global_camera_poses, args.output_file)

if __name__ == "__main__":
    main()