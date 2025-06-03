from data import utils
from general_functions import functions


def get_raw_data(data_source, shot: str) -> dict:
    """
    读取原始数据
    Args:
        data_source: hdf5文件地址
        shot: 炮号

    Returns:返回嵌套字典

    """
    data_source = functions.replace_shot_100(data_source, shot)
    raw_data = utils.read_hdf5(data_source)
    return raw_data


def get_is_disruption():
    pass
