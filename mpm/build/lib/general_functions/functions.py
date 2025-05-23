from h5py import File
import yaml
import os
import json
import argparse
import numpy as np
from datetime import datetime
from pymongo import MongoClient
from ruamel.yaml import YAML
from general_functions.database_data import get_is_running_shot
from alarmSystem.Data.db.collectionDB import CollectionDB
from typing import Optional


def get_input_params(description):
    """
    读取命令行输入参数
    :param description: 输入描述
    :return: 输入的配置文件地址和炮号
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('config_path', type=str, help='config path', default='')
    parser.add_argument('--name', '-n', type=str, help='ball_mill_name', default='')
    parser.add_argument('--shot', '-s', type=str, help='shot', default="")
    args = parser.parse_args()
    # print("参数输入完成")
    return args.config_path, args.name, args.shot


def read_config(config_path):
    """
    :param config_path: 配置文件地址
    :return: 配置文件信息
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config_data


def get_prefix_address(path, split='$bm$'):
    """
    将配置文件中的data_source的$split$前的前缀地址取出
    :param path: 地址
    :param split:切割处
    :return:切割处前的代码
    """
    prefix_address = ""
    while split in path:
        prefix_address, _ = os.path.split(path)
        path = prefix_address
    return prefix_address


def replace_string(string, replacement, key):
    """
    替换字符串中$key$对应的部分为replacement
    :param string: 被替换的字符串
    :param replacement: 替换的内容
    :param key: 被替换的关键字 如$bm$ bm为被替换的关键字
    :return:
    """
    replacement_key = f"${key}$"
    string = string.replace(replacement_key, replacement)
    return string


def replace_ball_mill_name(string, ball_mill_name):
    string = replace_string(string, ball_mill_name, 'bm')
    return string


def replace_sensor(string, sensor):
    string = replace_string(string, sensor, 'sensor')
    return string


def replace_shot_100(string, shot):
    """
    替换炮号，分为shot_2和shot两部分，shot_2表示每一百炮存一个文档
    :param string:
    :param shot:
    :return:
    """
    shot_2 = int(int(shot) / 100)
    string = replace_string(string, str(shot_2), 'shot_2')
    string = replace_string(string, str(shot), 'shot')
    return string


def replace_bm_shot_path(path, bm, shot):
    """
    :param path: 被替换的地址
    :param bm: 球磨机名称 如bm1
    :param shot: 炮号
    :return:
    """
    path = replace_shot_100(path, shot)
    path = replace_ball_mill_name(path, bm)
    return path


def replace_path(data_source, shot: str, bm, sensor):
    """
    :param data_source: config文件中的datasource,Inference等数据存放地
    :param shot: 炮号
    :param bm: 球磨机名称 如bm1
    :param sensor: 传感器名称 如sensor1
    :return: 替换了所有$str$的datasource
    """
    data_source = replace_bm_shot_path(data_source, bm, shot)
    data_source = replace_sensor(data_source, sensor)
    # print(data_source)
    return data_source


def get_folder_path(source_file):
    # 获取文件所在的文件夹地址
    folder_path = os.path.dirname(source_file)
    return folder_path


def rename_file(target_file_path, new_file_name):
    """
    将目标路径对应的文件名修改为new_file_name（包括文件后缀）
    """
    target_folder_path = os.path.dirname(target_file_path)
    new_file_path = os.path.join(target_folder_path, new_file_name)
    # print(new_file_path)
    if os.path.exists(target_file_path) and not os.path.exists(new_file_path):
        os.rename(target_file_path, new_file_path)
        if os.path.exists(target_file_path) and os.path.exists(new_file_path):
            os.remove(target_file_path)


def read_hdf5(data_path):
    """
    :param data_path: hdf5文件地址
    :return:
        sample_data: 采集设置的参数
        data: 数据
    """
    with File(data_path, 'r') as f:
        sample_data, data = {}, {}
        for key in f.keys():
            if key == 'Attribute':
                for name, value in f[key].attrs.items():
                    sample_data[name] = value
                    # print(f"{key}: {value}")
            else:
                data[key] = f[key][:]  # data[key]数据类型为numpy数组
    return sample_data, data


def get_single_sensor_data(data_source, shot, bm, sensor):
    """
    返回单个sensor采集的hdf5文件的信息
    :param data_source: config文件中的datasource即数据存放地
    :param shot: 炮号
    :param bm: 球磨机名称 如bm1
    :param sensor: 传感器名称 如sensor1
    :return:
        sample_data: 采集设置的参数
        data: 数据
    """
    data_source = replace_path(data_source, shot, bm, sensor)
    sample_data, data = read_hdf5(data_source)
    return sample_data, data


def get_sample_time(sample_data):
    """
    通过采样数据得到采样时间
    :param sample_data: 采样的数据
    :return: 采样时间
    """
    try:
        date_time = sample_data['CreateTime']
        return date_time
    except Exception as e:
        return None


def get_sensors_data(data_source, shot: str, name, sensors):
    """
    获取所有传感器的数据
    """
    sensors_data = {}
    sample_data = None
    for sensor in sensors:
        try:
            sample_data, data = get_single_sensor_data(data_source, shot, name, sensor)
            sensors_data[sensor] = data
        except Exception as e:
            sensors_data[sensor] = None
            continue
    return sample_data, sensors_data


def create_shots_raw_data_dict(sensors, channels):
    """
    创建存储多炮原始数据的字典格式
    """
    shots_data = {}
    for sensor in sensors:
        shots_data[sensor] = {}
        for channel in channels:
            shots_data[sensor][channel] = []
    return shots_data


def slice_data(data, drop_last, length: Optional[int] = None):
    """
    对数据进行切片
    :param data: 输入数据
    :param drop_last: 是否去除最后一个数据点
    :param length: 将数据需要切成的长度
    :return:
    """
    if drop_last:
        data = data[:-1]
    if length is not None:
        data = data[:length]
    return data


def get_shots_raw_data(data_source, shots, name, sensors, channels, drop_last=True, length=None):
    """
    取出shots列表炮号的原始数据
    :return:
    """
    shots_data = create_shots_raw_data_dict(sensors, channels)
    for shot in shots:
        shot = str(shot)
        _, sensors_data = get_sensors_data(data_source, shot, name, sensors)
        for sensor in sensors:
            for channel in channels:
                if sensors_data[sensor] is None:
                    # print(shot, sensor, channel)
                    continue
                else:
                    data = slice_data(sensors_data[sensor][channel], drop_last, length)
                    shots_data[sensor][channel].append(data)
    return shots_data


def get_is_running_shots_data(db, collection_name, shot_num, max_shot, data_source, name, sensors, channels,
                              drop_last=True, min_shot=-1):
    """
    从数据库获取is_running为True的炮号。根据该炮号得到正在运行的炮的原始数据
    """
    # print(db, collection_name, shot_num, max_shot, min_shot)
    is_running_shots = get_is_running_shot(db, collection_name, shot_num, max_shot, min_shot)
    # print(is_running_shots)
    is_running_shots_data = get_shots_raw_data(data_source, is_running_shots, name, sensors, channels, drop_last)
    return is_running_shots_data

def verify_list_all_None(lst: list):
    """
    确定输入的列表元素是否全为None，True则全为None
    :return:
    """
    all_none = all(x is None for x in lst)
    return all_none


def verify_dict_value_all_None(dct: dict):
    """
    确定输入的字典的values是否全是None
    """
    value_list = list(dct.values())
    result = verify_list_all_None(value_list)
    return result


def channel_to_axis(channels_to_axis, channel):
    """
    输入channel0，1，2与x,y,z的对应关系和channel，返回该channel对应的轴向
    :param channels_to_axis: channel0，1，2与x,y,z的对应关系 如config['channels']
    :param channel: 需要确定对应轴向的channel
    :return: 输入的channel对应的x,y,z轴
    """
    for channel_axis_dict in channels_to_axis:
        channel_, _ = channel_axis_dict.copy().popitem()
        if channel == channel_:
            return channel_axis_dict[channel]


def return_datatime(data_time_float):
    """
    将输入的时间浮点数转化为时间戳, 若hdf5中时间浮点数不存在，则返回None
    :param data_time_float: 时间浮点数 np.float64
    :return: 时间类型数据
    """
    try:
        # 将数字转换为字符串
        data_str = str(data_time_float)
        # 使用 strptime 方法解析日期时间，只到分钟
        date_time = datetime.strptime(data_str[:12], "%Y%m%d%H%M")
        formatted_date = date_time.strftime("%Y-%m-%d %H:%M")
    except Exception as e:
        formatted_date = None
    return formatted_date


def save_json(json_data, json_name, output_path):
    """
    :param json_data: 需要以json形式保存的数据
    :param json_name: 保存json文件的名字
    :param output_path: json文件的保存地址
    :return:
    """
    output_path = os.path.join(output_path, json_name)
    json_string = json.dumps(json_data)
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json_file.write(json_string)


def replace_placeholders(string, replacement_params):
    """
    :param string: 需要被替换的字符串，替换$key$
    :param replacement_params: 用于替换的键值对，key表示原字符串被替换的部分，value表示替换的部分
    :return:
    """
    for key, value in replacement_params.items():
        placeholder_key = f"${key}$"
        string = string.replace(placeholder_key, value)
    return string


def create_folder(folder_path):
    """
    创建文件夹
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def create_output_folder(output_path, shot, bm):
    """
    创建输出的文件夹
    :param output_path: 输出地址（含被替换部分）
    :param shot:
    :param bm:
    :return:
    """
    shot_2 = int(int(shot) / 100)
    output_path = output_path.replace('$shot$', shot)
    output_path = output_path.replace('$shot_2$', str(shot_2))
    output_path = output_path.replace('$bm$', bm)
    # output_path = replace_path(output_path, shot, bm, sensor)
    try:
        create_folder(output_path)
        return output_path
    except Exception as e:
        pass


def return_single_summary_data(output_path, shot, bm, model_name):
    """
    返回对应single_summary_modelname.json文件数据
    :param output_path: 输出地址（含被替换部分）config['Inference_path']部分
    :param shot: 返回的炮号
    :param bm: 球磨机名称
    :param model_name: 模型名称 如rms fft
    :return: json文件数据
    """
    output_path = create_output_folder(output_path, shot, bm)
    json_name = 'single_summary' + '_' + model_name + '.json'
    single_summary_path = os.path.join(output_path, json_name)
    with open(single_summary_path, 'r') as file:
        single_summary_data = json.load(file)
    return single_summary_data


def get_threshold_config(threshold_config_path, bm):
    """
    得到阈值的配置文件信息
    :param: bm 球磨机名称
    :return:
    """
    threshold_config_path = threshold_config_path.replace('$bm$', bm)
    threshold_config, ruamel_yaml = load_yaml(threshold_config_path)
    return threshold_config, ruamel_yaml, threshold_config_path


def load_yaml(file_path):
    """
    得到配置文件信息，便于更新
    :param file_path:
    :return:
    """
    ruamel_yaml = YAML()
    yaml.preserve_quotes = True  # 保留引号
    ruamel_yaml.default_flow_style = False  # 使用块样式而非流样式
    with open(file_path, 'r', encoding='utf-8') as file:
        data = ruamel_yaml.load(file)
    return data, ruamel_yaml


def save_yaml(data, file_path, ruamel_yaml):
    """
    保存修改好的配置文件数据
    :param data: 配置文件数据
    :param file_path: 保存地址
    :param ruamel_yaml: 之前读取创建的YAML对象
    :return:
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        ruamel_yaml.dump(data, file)


def alarm_config_axis(axis, data_list, h, hh, sensor_threshold, h_std_num: float=4, hh_std_num: float=5):
    """
    根据已有的数据列表与异常分位点，得到异常阈值列表
    :param h_std_num:
    :param hh_std_num:
    :param hh: 2级阈值分位点
    :param h: 1级阈值分位点
    :param axis: 轴向数据
    :param data_list:
    :param sensor_threshold:异常阈值字典
    :return:
    """
    # h_threshold = np.percentile(data_list, 100 * h)
    # hh_threshold = np.percentile(data_list, 100 * hh)
    mean = np.mean(data_list)
    std = np.std(data_list, ddof=1)
    h_threshold = mean + h_std_num * std
    hh_threshold = mean + hh_std_num * std
    sensor_threshold['of_h_' + axis.split("_")[0]] = h_threshold
    sensor_threshold['of_hh_' + axis.split("_")[0]] = hh_threshold
    return sensor_threshold


def order_alarm_dict(sensor_threshold_dict, round_num=5):
    """
    将异常阈值字典按照次序排列输出，并将字典内的数值类数据变为可录入yaml文件格式
    :param round_num: 保留的数据小数点位数
    :param sensor_threshold_dict:
    :return:
    """
    for axis, threshold in sensor_threshold_dict.items():
        sensor_threshold_dict[axis] = convert_to_serializable(threshold, round_num)
        sensor_threshold_dict[axis] = convert_floats_to_strings(sensor_threshold_dict[axis])
    return sensor_threshold_dict


def convert_to_serializable(data, round_num):
    """递归地将数据转换为可序列化的格式."""
    if isinstance(data, float):
        return round(data, round_num)  # 截断为可接受的精度
    elif isinstance(data, dict):
        return {key: convert_to_serializable(value, round_num) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item, round_num) for item in data]
    return data


def convert_floats_to_strings(data):
    """递归地将浮点数转换为字符串格式，以避免序列化问题."""
    if isinstance(data, float):
        return str(data)  # 将浮点数转换为字符串
    elif isinstance(data, dict):
        return {key: convert_floats_to_strings(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_floats_to_strings(item) for item in data]
    return data


def get_sensor_alarm_threshold(sensors_threshold, sensor):
    """
    取出alarm_config对应传感器的sensor_threshold
    :param sensors_threshold: alarm_config的sensors_threshold
    :param sensor: 需要进行判别是否异常的传感器
    :return:
    """
    for sensor_threshold in sensors_threshold:
        if sensor_threshold['sensor_prefix'] == sensor:
            sensor_threshold_ = {k: v for k, v in sensor_threshold.items() if k != 'sensor_prefix'}
            return sensor_threshold_
        else:
            continue


def single_shot_summary(bm, shot, sensors, datatime_float, is_running=None):
    """
    将单独炮号的信息模型总结，新设计
    :param sensors: summary sensors信息汇总字典
    :param bm: 球磨机名称 bm1 bm2
    :param shot: 炮号
    :param is_running: 该炮号采集时，球磨机是否正在运行
    :return: 单独炮号分析总结
    """
    single_summary = dict()
    single_summary['ball_mill_name'] = bm
    single_summary['shot'] = int(shot)
    # 获取当前的日期和时间
    # current_datetime = datetime.now()
    # # 使用 strftime 方法格式化输出，不包含毫秒部分
    # formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    single_summary['time'] = return_datatime(datatime_float)
    single_summary['is_running'] = is_running
    single_summary['sensors'] = sensors
    return single_summary


def ensure_unique_index(collection, index_name):
    # 检查索引是否已经存在
    if not any(index['name'] == f'{index_name}_1' for index in collection.list_indexes()):
        # 如果不存在，则创建唯一索引
        try:
            collection.create_index([(index_name, 1)], unique=True)
        except Exception as e:
            pass


def save_summary_mongodb(single_summary, db, collection_name, shot):
    collection = db[collection_name]
    shot = int(shot)
    # 确定是否已经创建唯一shot索引， 若无创建唯一索引
    ensure_unique_index(collection, 'shot')
    # 检查文档是否已经存在
    existing_document = collection.find_one({"shot": shot})
    if existing_document:
        # 替换现有文档
        result = collection.replace_one({"shot": shot}, single_summary)
    else:
        # 插入新文档
        result = collection.insert_one(single_summary)


def save_single_summary_mongodb(single_summary, address, collection_name, shot, database_name='bm'):
    """
    将数据保存到MongoDB数据库,如果当前炮号已被分析，则对已有的结果进行替换
    :param shot: 炮号
    :param single_summary: 分析得到的数据
    :param address: 数据库域名地址
    :param collection_name: 集合名称
    :param database_name: 数据库名称
    :return:
    """
    shot = int(shot)
    # 创建 MongoDB 客户端
    client = MongoClient(address)
    # 选择数据库
    db = client[database_name]
    # 选择集合
    collection = db[collection_name]
    # 确定是否已经创建唯一shot索引， 若无创建唯一索引
    ensure_unique_index(collection, 'shot')
    # 检查文档是否已经存在
    existing_document = collection.find_one({"shot": shot})
    if existing_document:
        # 替换现有文档
        result = collection.replace_one({"shot": shot}, single_summary)
    else:
        # 插入新文档
        result = collection.insert_one(single_summary)
    # 关闭客户端连接
    client.close()


def create_and_save_single_summary(bm, shot, sensors, address, collection_name, database_name, is_running=None):
    single_summary = single_shot_summary(bm, shot, sensors, is_running)
    save_single_summary_mongodb(single_summary, address, collection_name, database_name)


def get_rms_record(is_running_collection: str, name: str, shot: int, db, replace_key='bm'):
    is_running_collection = replace_string(is_running_collection, name, replace_key)
    query = {'shot': int(shot)}
    rms_record = CollectionDB(db, is_running_collection).find_record(query)
    return rms_record[0]


def save_alarm_config(file_path, alarm_config_data):
    pass
