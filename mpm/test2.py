import os
import numpy as np

# 指定文件夹路径
folder_path = r'D:\study\mpm\破裂炮号列表\J-TEXT'

# 获取文件夹中所有的.npy文件
npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

# 遍历并读取每个.npy文件
for file_name in npy_files:
    file_path = os.path.join(folder_path, file_name)
    data = np.load(file_path)

    # 打印文件名和数据形状（或直接打印数据）
    print(f"File: {file_name}")
    print(f"Data shape: {data.shape}")
    print("Data content:")
    print(data)
    print("-" * 20)

import os
import numpy as np

# 指定文件夹路径
folder_path = r'D:\study\mpm\破裂炮号列表'

# 获取文件夹中所有的.npy文件
npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

# 遍历并读取每个.npy文件
for file_name in npy_files:
    file_path = os.path.join(folder_path, file_name)
    data = np.load(file_path)

    # 打印文件名和数据形状（或直接打印数据）
    print(f"File: {file_name}")
    print(f"Data shape: {data.shape}")
    print("Data content:")
    print(data)
    print("-" * 40)