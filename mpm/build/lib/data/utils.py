import numpy as np
import h5py


def calculate_rms(data):
    # 计算rms
    rms = np.sqrt(np.mean(data ** 2))
    return rms


def remove_prefix(input_string: str, prefix: str) -> str:
    """
    去除输入字符串的指定前缀
    :param input_string: 需要去除前缀的字符串
    :param prefix: 需要去除的前缀
    :return: 去掉了指定前缀的字符串，如果原字符串没有该前缀，则返回原字符串
    """
    if input_string.startswith(prefix):
        return input_string[len(prefix):]
    else:
        return input_string


def read_hdf5(file_path) -> dict:
    """
    读取HDF5文件的内容，返回字典。输出结构如下
    root
    ├── group1
    │   ├── dataset1 (data: [1,2,3], attr: unit='m')
    │   └── dataset2 (data: 42)
    └── group2
        └── dataset3 (data: [[1,2],[3,4]], attr: desc='matrix')

    :param file_path: HDF5文件的路径。
    :return: 如果指定了dataset_name，则返回该数据集的内容。否则，返回包含所有数据集的一个字典。
    """
    def recursively_extract(h5_obj):
        result = {}
        for key in h5_obj:
            item = h5_obj[key]
            if isinstance(item, h5py.Dataset):
                result[key] = {}
                result[key]['attrs'] = {}
                for name, value in item.attrs.items():
                    result[key]['attrs'][name] = value
                # 转换为普通 Python 类型（如 numpy array 或 scalar）
                result[key]['dataset'] = item[()]  # item[()] 会自动转为合适的类型
            elif isinstance(item, h5py.Group):
                result[key] = recursively_extract(item)
        return result
    with h5py.File(file_path, 'r') as f:
        return recursively_extract(f)

