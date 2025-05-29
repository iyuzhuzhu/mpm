from general_functions import functions, database_data
from data import loader, utils, processor
from general_functions.models import BasicModel

import matplotlib.pyplot as plt
import numpy as np


def generate_time_vector(Fs, L):
    """
    根据采样频率和信号长度生成时间向量。
    :param Fs: 采样频率 (samples per second)
    :param L: 信号长度 (number of samples)
    :return: 时间向量
    """
    T = 1 / Fs  # 采样周期
    return np.arange(0, L) * T  # 时间向量


def plot_time_series(Fs, L, data, title=None, xlabel='time (s)', grid=True):
    """
    绘制时序数据图，给定采样频率和信号长度。
    :param Fs: 采样频率 (samples per second)
    :param L: 信号长度 (number of samples)
    :param data: 数据序列（一维）
    :param title: 图像标题（可选）
    :param xlabel: x轴标签
    :param ylabel: y轴标签
    :param grid: 是否显示网格
    """
    time = generate_time_vector(Fs, L)
    plt.figure(figsize=(12, 5))
    plt.plot(time, data, color='blue', linewidth=1.2)
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    if grid:
        plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


class Rms(BasicModel):
    def __init__(self, config_path, shot, model_name='rms'):
        super().__init__(config_path, shot, model_name)
        raw_data = self.load_raw_data()
        # print(raw_data)
        tags_rms = self.calculate_tags_rms(raw_data)
        avg_rms = self.calculate_directional_average_rms(tags_rms)
        print(raw_data)
        print(tags_rms)
        print(avg_rms)
        # print(V_avg_rms)
        IP = raw_data['data']['\\IP']
        fs = IP['attrs']['SampleRate']
        dataset = IP['dataset']
        plot_time_series(fs, len(dataset), dataset)

    def calculate_tags_rms(self, raw_data, remove_pre=None):
        tags_rms = {}
        data = raw_data['data']
        for tag in self.tags:
            key = utils.remove_prefix(tag, remove_pre) + '_rms'
            if tag in data.keys():
                dataset = data[tag]['dataset']
                rms = processor.calculate_tag_rms(dataset)
                tags_rms[key] = rms
            else:
                tags_rms[key] = None
        return tags_rms

    def plot_model(self, sensors_data=None, sample_data=None):
        """
        绘制振动波形图像，并复制plot_desc文件
        """
        # sample_data, sensors_data = functions.get_sensors_data(self.data_source, self.shot, self.name, self.sensors)
        plot_config_data = self.get_plot_config()
        save_plot_folder = plot_config_data['save_plot_folder']
        output_path = functions.create_output_folder(self.config['Inference_path'], self.shot, self.name)
        output_folder = os.path.join(output_path, save_plot_folder)
        functions.create_folder(output_folder)
        plots.copy_and_rename_file(self.config['plot_desc_path'], output_folder,
                                   self.config['plot_desc_file'])  # 复制plot_desc
        for plot_data in plot_config_data['plots']:
            sensor = plot_data['name']
            channel = plot_data['plot_channel']
            drop_last = plot_data['drop_last']
            if sensors_data[sensor] is not None:
                x, y = self.get_plot_x_y(sample_data['SampleRate'], sensors_data[sensor][channel], drop_last)
                # print(x, y)
                plots.plot_config(x, y, plot_data, output_folder)

    @staticmethod
    def calculate_directional_average_rms(data):
        """
        计算纵场线圈与真空室上的传感器在R、phi、Z方向上的平均rms，并以特定格式返回结果。
        :param data: 包含rms值的字典
        :return: 返回两类位置（TF, V）在三个方向（R, phi, Z）上的平均rms值，按指定格式组织
                 若某方向无数据则设置为 None
        """
        F_sum = {'R': 0, 'phi': 0, 'Z': 0}
        V_sum = {'R': 0, 'phi': 0, 'Z': 0}
        F_count = {'R': 0, 'phi': 0, 'Z': 0}
        V_count = {'R': 0, 'phi': 0, 'Z': 0}

        for key, value in data.items():
            if value is None:
                continue
            direction = None
            if 'R' in key:
                direction = 'R'
            elif 'phi' in key or 'PHI' in key:
                direction = 'phi'
            elif 'Z' in key:
                direction = 'Z'

            if direction is not None:
                if 'TF' in key:
                    F_sum[direction] += value
                    F_count[direction] += 1
                elif 'V' in key:
                    V_sum[direction] += value
                    V_count[direction] += 1

        # 构建结果字典，若 count == 0 则设为 None
        result = {}
        for loc, sums, counts in [('TF', F_sum, F_count), ('V', V_sum, V_count)]:
            for direction in sums.keys():
                avg_key = f"{loc}_{direction}_rms"
                if counts[direction] > 0:
                    result[avg_key] = float(sums[direction] / counts[direction])
                else:
                    result[avg_key] = None  # 没有数据时设为 None

        return result


def main():
    # test_shots_calculate()
    # # 输入参数
    # config_path, name, shot = functions.get_input_params('rms')
    config_path = './config.yml'
    shot = '1103600'
    Rms(config_path, shot, model_name='rms')
    # test_shots_calculate()


if __name__ == '__main__':
    main()
