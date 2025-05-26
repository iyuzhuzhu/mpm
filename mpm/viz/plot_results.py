from matplotlib import pyplot as plt


def plot_fft_spectrum(freq_axis, fft_result, title, save_path):
    """
    绘制FFT频谱图
    :param freq_axis: 频率轴数据，例如由FFT函数返回
    :param fft_result: 幅值谱数据，例如由FFT函数返回
    """
    plt.figure(figsize=(10, 6))
    plt.plot(freq_axis, fft_result, color='red')
    plt.title(title)
    plt.xlabel('频率 (Hz)')
    plt.ylabel('幅值')
    plt.grid(True)
    plt.tight_layout()
    plt.show()