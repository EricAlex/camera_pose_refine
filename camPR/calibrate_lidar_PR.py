import open3d as o3d
import numpy as np
import math
import argparse
import sys

def parse_pcd_header(pcd_path):
    """
    解析PCD文件的头部信息。
    
    Args:
        pcd_path (str): PCD文件的路径。

    Returns:
        dict: 包含头部信息的字典，例如 'fields', 'points', 'data_start_line'。
              如果解析失败，则返回 None。
    """
    header = {}
    try:
        with open(pcd_path, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) > 1:
                    key = parts[0].upper()
                    if key == 'FIELDS':
                        header['fields'] = parts[1:]
                    elif key == 'SIZE':
                        header['size'] = [int(s) for s in parts[1:]]
                    elif key == 'TYPE':
                        header['type'] = parts[1:]
                    elif key == 'COUNT':
                        header['count'] = [int(c) for c in parts[1:]]
                    elif key == 'POINTS':
                        header['points'] = int(parts[1])
                    elif key == 'DATA':
                        header['data_type'] = parts[1].lower()
                        header['data_start_line'] = i + 1
                        # 确保所有关键信息都已读取
                        required_keys = ['fields', 'points', 'data_type']
                        if not all(k in header for k in required_keys):
                            print(f"错误: PCD文件头信息不完整，缺少 {set(required_keys) - set(header.keys())}")
                            return None
                        return header
    except FileNotFoundError:
        print(f"错误: 文件未找到 '{pcd_path}'")
        return None
    except Exception as e:
        print(f"错误: 解析PCD文件头时发生未知错误: {e}")
        return None
        
    print("错误: 未能在文件中找到 'DATA' 标志，PCD文件格式可能不正确。")
    return None

def calculate_pitch_roll_correction(normal_vector):
    """
    根据平面法向量计算lidar的pitch和roll校正值。
    
    Args:
        normal_vector (np.ndarray): 长度为3的单位法向量 (nx, ny, nz)。
    
    Returns:
        tuple: (pitch_correction_deg, roll_correction_deg) 校正角度（单位：度）。
    """
    nx, ny, nz = normal_vector

    # 确保法向量指向上方（Z分量为正），这对于atan2的计算很重要
    if nz < 0:
        nx, ny, nz = -nx, -ny, -nz
        print("提示: 原始法向量朝下，已翻转使其朝上。")
    
    # 计算Roll（绕X轴旋转）
    # Roll是法向量在YZ平面上的投影与Z轴的夹角。
    # 需要一个负的roll来抵消这个偏差。
    roll_rad = -math.atan2(ny, nz)
    
    # 计算Pitch（绕Y轴旋转）
    # Pitch是法向量与YZ平面的夹角，或者说是法向量在XZ平面上的投影与Z轴的夹角。
    # 如果nx为正，表示法向量朝+X方向倾斜，意味着雷达向上倾斜（pitch为负）。
    # 因此需要一个正的pitch来矫正（向下倾斜）。
    # atan2(-nx, nz) 直接给出了正确的校正角度。
    # 当nx>0 (向上倾斜), -nx<0, atan2(-,+) -> 负值 -> 向上pitch校正
    # 这似乎与直觉相反，让我们重新思考一下：
    # 标准定义：pitch down为正。如果法向量nx > 0，说明地面法线朝前倾斜，
    # 意味着雷达本身是朝上(up)倾斜的。要把它调平，需要向下(down)调整，即一个正的pitch。
    # math.atan2(y, x)
    # 我们的向量在XZ平面投影是 (nx, nz)。我们想把它转到Z轴。
    # 旋转角度是 atan2(nx, nz)。这个角度为正表示从Z轴向X轴的正向旋转。
    # 这对应一个向下的Pitch。所以校正值就是这个角度。
    pitch_rad = math.atan2(nx, nz)

    # 转换为度
    pitch_deg = math.degrees(pitch_rad)
    roll_deg = math.degrees(roll_rad)
    
    return pitch_deg, roll_deg

def process_pcd_and_calibrate(pcd_path, box_range=10.0, ransac_threshold=0.05):
    """
    主处理函数：读取、过滤、拟合平面并计算外参校正。
    
    Args:
        pcd_path (str): PCD文件路径。
        box_range (float): X和Y的截取范围 (-box_range, box_range)。
        ransac_threshold (float): RANSAC平面拟合的距离阈值。
    """
    print(f"--- 开始处理文件: {pcd_path} ---")

    # 1. 解析PCD文件头
    header = parse_pcd_header(pcd_path)
    if not header:
        return

    print("PCD文件头解析成功:")
    print(f"  - 字段: {header['fields']}")
    print(f"  - 点数: {header['points']}")
    print(f"  - 数据类型: {header['data_type']}")

    if header['data_type'] != 'ascii':
        print(f"错误: 此脚本仅支持 'ascii' 类型的PCD文件，但文件中是 '{header['data_type']}'。")
        return

    # 2. 读取点云数据
    try:
        # 使用numpy高效读取ASCII数据
        all_data = np.loadtxt(pcd_path, skiprows=header['data_start_line'])
        print(f"成功读取 {all_data.shape[0]} 个点。")
        
        # 创建一个更易于操作的字段字典
        field_map = {name: i for i, name in enumerate(header['fields'])}
        if not all(f in field_map for f in ['x', 'y', 'z', 'segLabel']):
            print(f"错误: PCD文件缺少必要的字段 'x', 'y', 'z', 'segLabel'。可用字段: {list(field_map.keys())}")
            return

    except Exception as e:
        print(f"错误: 使用numpy读取点云数据时失败: {e}")
        return

    # 3. 过滤点云
    # 3.1. 选取语义为“地面点”的点云（segLabel值为0）
    seg_label_idx = field_map['segLabel']
    ground_mask = (all_data[:, seg_label_idx] == 0)
    ground_points = all_data[ground_mask]
    
    if ground_points.shape[0] == 0:
        print("错误: 文件中没有找到 'segLabel' 为 0 的地面点。无法继续。")
        return
    print(f"根据 'segLabel == 0' 筛选后，剩余 {ground_points.shape[0]} 个地面点。")

    # 3.2. 截取x和y在（-10， 10）范围内的点
    x_idx, y_idx = field_map['x'], field_map['y']
    box_mask = (np.abs(ground_points[:, x_idx]) < box_range) & \
               (np.abs(ground_points[:, y_idx]) < box_range)
    filtered_points = ground_points[box_mask]

    if filtered_points.shape[0] < 3:
        print(f"错误: 在 x,y (-{box_range}, {box_range}) 范围内筛选后，剩余点数少于3个 ({filtered_points.shape[0]}个)。无法拟合平面。")
        return
    print(f"在 x,y (-{box_range}, {box_range}) 范围内截取后，剩余 {filtered_points.shape[0]} 个点用于平面拟合。")

    # 4. 使用Open3D进行平面拟合
    # 提取XYZ坐标
    xyz_points = filtered_points[:, [field_map['x'], field_map['y'], field_map['z']]]
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_points)
    
    # 使用RANSAC拟合平面
    # segment_plane返回 (a, b, c, d) 使得 ax + by + cz + d = 0
    # 法向量为 (a, b, c)
    try:
        plane_model, inliers = pcd.segment_plane(distance_threshold=ransac_threshold,
                                                 ransac_n=3,
                                                 num_iterations=1000)
    except Exception as e:
        print(f"错误: Open3D平面拟合失败: {e}")
        return

    if len(inliers) < 10: # 如果内点太少，认为拟合不可靠
        print(f"警告: RANSAC拟合的内点数量过少 ({len(inliers)}个)，结果可能不可靠。")

    [a, b, c, d] = plane_model
    print(f"\n拟合的平面方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    
    # 5. 计算法向量并进行校正
    normal = np.array([a, b, c])
    
    # 检查法向量是否有效
    norm_val = np.linalg.norm(normal)
    if np.isclose(norm_val, 0):
        print("错误: 拟合得到的法向量为零向量，无法进行校准。")
        return
        
    # 归一化法向量
    normal /= norm_val
    print(f"归一化后的平面法向量: [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]")

    # 理想的法向量（竖直朝上）
    target_normal = np.array([0, 0, 1.0])

    # 检查法向量是否已经足够竖直
    # 计算与理想法向量的点积，如果接近1，说明已经对齐
    dot_product = np.dot(normal, target_normal)
    angle_diff_deg = math.degrees(math.acos(np.clip(dot_product, -1.0, 1.0)))

    if angle_diff_deg < 0.1: # 如果夹角小于0.1度
        print("\n分析结果: 拟合平面的法向量已基本竖直朝上，无需调整。")
    else:
        print("\n分析结果: 拟合平面的法向量不完全竖直，需要调整外参。")
        pitch_corr_deg, roll_corr_deg = calculate_pitch_roll_correction(normal)
        
        print("\n--- 校准建议 ---")
        print(f"需要调整的 Pitch (绕Y轴): {pitch_corr_deg:+.4f} 度  (正值表示向下俯仰)")
        print(f"需要调整的 Roll  (绕X轴): {roll_corr_deg:+.4f} 度  (正值表示向右侧倾)")
        print("------------------")

    print(f"\n--- 处理完成 ---")


if __name__ == "__main__":
    # 使用 argparse 设置命令行参数
    parser = argparse.ArgumentParser(
        description="从PCD文件中读取地面点，拟合平面，并计算Lidar的Pitch和Roll外参校正值。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("pcd_file", help="输入的PCD文件路径。")
    parser.add_argument(
        "--range", 
        type=float, 
        default=20.0,
        help="用于拟合的地面点的X, Y坐标范围的绝对值。\n"
             "例如: --range 10.0 表示使用 x, y 在 (-10, 10) 范围内的点。\n"
             "(默认值: 10.0)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="RANSAC平面拟合的距离阈值（米）。\n"
             "点到平面的距离小于此值才被视为内点。\n"
             "(默认值: 0.05)"
    )
    
    args = parser.parse_args()
    
    # 运行主程序
    process_pcd_and_calibrate(args.pcd_file, args.range, args.threshold)