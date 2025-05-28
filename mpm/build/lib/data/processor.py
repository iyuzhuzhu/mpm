from data import utils
import numpy as np
from scipy.fftpack import fft


def calculate_tag_rms(tag_dataset):
    try:
        rms = utils.calculate_rms(tag_dataset)
        return rms
    except Exception as e:
        print(e)
        return None


def FFT(Fs, data):
    """
    对输入信号进行FFT
    :param Fs:  采样频率
    :param data:待FFT的序列
    :return:
    """
    N = len(data)  # 信号长度
    # N = np.power(2, np.ceil(np.log2(L)))  # 下一个最近二次幂，也即N个点的FFT
    result = np.abs(fft(x=data)) / N * 2  # N点FFT
    axisFreq = np.arange(int(N / 2)) * Fs / N  # 频率坐标
    result = result[range(int(N / 2))]  # 因为图形对称，所以取一半
    # print(result)
    return axisFreq, result