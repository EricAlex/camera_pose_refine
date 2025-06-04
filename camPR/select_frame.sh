      
#!/bin/bash

# 参数设置
cam_name="camera_4"
# cam_name="panoramic_1"
src_dir="${cam_name}"
dst_dir="input_tmp/${cam_name}"
percent=10  # 修改此处为任意百分比（例如 20 表示 20%）

# ---- New: Define the list file path ----
# Get the parent directory of dst_dir (e.g., /home/xin/Downloads/Hierarchical-Localization/camPR/)
list_file_dir=$(dirname "$dst_dir")
list_file="${list_file_dir}/query_image_list_${cam_name}.txt"

# 创建目标目录
mkdir -p "$dst_dir"
# Ensure the directory for the list file also exists (might be redundant but safe)
mkdir -p "$list_file_dir"

# ---- New: Clear or create the list file before starting ----
> "$list_file"

# 获取文件列表
# Using nullglob to handle cases where no files match the pattern
shopt -s nullglob
files_raw=("$src_dir"/*)
shopt -u nullglob

# Check if any files were found
if [ ${#files_raw[@]} -eq 0 ]; then
  echo "No files found in $src_dir"
  echo "Operation complete. Copied 0/0 files. List file '$list_file' is empty."
  exit 0
fi

# Sort the files
files=($(printf "%s\n" "${files_raw[@]}" | sort))

# 计算抽取数量
total=${#files[@]}
# Ensure total is not zero before division
if [ "$total" -eq 0 ]; then
  echo "No files found after sorting (this shouldn't normally happen if files were found initially)."
  echo "Operation complete. Copied 0/0 files. List file '$list_file' is empty."
  exit 0
fi

num_to_copy=$(( (total * percent + 50) / 100 ))  # 四舍五入

# Handle edge case: if percent is very low, num_to_copy might be 0.
if [ "$num_to_copy" -eq 0 ] && [ "$total" -gt 0 ] && [ "$percent" -gt 0 ]; then
    echo "Warning: Percentage $percent% is too low for $total files, rounding resulted in 0. Copying at least 1 file."
    num_to_copy=1
elif [ "$num_to_copy" -eq 0 ]; then
    echo "Target percentage ($percent%) results in 0 files to copy from $total total files."
    echo "Operation complete. Copied 0/$total files. List file '$list_file' is empty."
    exit 0
fi


# 计算步长
# Ensure step is at least 1 to avoid potential issues or infinite loops
step=$(( total / num_to_copy ))
if [ "$step" -lt 1 ]; then
    step=1
fi

# 拷贝文件并记录文件名
echo "Starting copy process..."
copied=0
for (( i=0; i<total && copied<num_to_copy; i+=step )); do
    src_file="${files[i]}"
    filename=$(basename "$src_file") # Extract just the filename

    # Copy the file
    if cp -v "$src_file" "$dst_dir/"; then
        # ---- New: If copy is successful, add filename to list file ----
        echo "$filename" >> "$list_file"
        ((copied++))
    else
        # Optional: Report error if copy failed
        echo "Warning: Failed to copy '$src_file'" >&2
    fi
done

echo "----------------------------------------"
echo "操作完成!"
echo "拷贝了 $copied/$total 个文件 (目标比例 $percent%)"
echo "源目录: $src_dir"
echo "目标目录: $dst_dir"
echo "已拷贝文件名列表: $list_file"
echo "----------------------------------------"
