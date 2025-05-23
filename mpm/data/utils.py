import numpy as np


def calculate_rms(data):
    # 计算rms
    rms = np.sqrt(np.mean(data ** 2))
    return rms


