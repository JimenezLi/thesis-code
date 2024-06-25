import matplotlib.pyplot as plt
import statsmodels.tsa.statespace.sarimax
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np

from scipy.signal import argrelextrema
import scipy.interpolate as spi

from src.data_process import data_import


def get_arima(data, length, order=(3, 0, 3)):
    train_data = data[:length]
    test_data = data[length:]

    print(sm.tsa.stattools.adfuller(train_data))
    print(acorr_ljungbox(train_data, lags=[6, 12], boxpierce=True))

    # acf = plot_acf(train_data)
    # plt.title("Autocorrelation Function (ACF) plot of MUF")
    # plt.show()
    # pacf = plot_pacf(train_data)
    # plt.title("Partial Autocorrelation Function (PACF) plot of MUF")
    # plt.show()

    # aic = sm.tsa.arma_order_select_ic(train_data, ic='aic', max_ar=10, max_ma=5)['aic_min_order']
    # print('AIC: ', aic)

    # arima_model = statsmodels.tsa.statespace.sarimax.SARIMAX(train_data, order=(aic[0], 1, aic[1]), seasonal_order=(0, 1, 0, 36))
    # arima_model = statsmodels.tsa.statespace.sarimax.SARIMAX(train_data, order=(3, 1, 3), seasonal_order=(0, 1, 0, 36))
    # arima_model = sm.tsa.arima.ARIMA(train_data, order=(aic[0], 0, aic[1]))
    arima_model = sm.tsa.arima.ARIMA(train_data, order=order)
    arima_res = arima_model.fit()

    # return np.array(arima_res.predict(len(train_data), len(data) - 1))

    y_true = test_data
    y_pred = arima_res.predict(len(train_data), len(data) - 1)

    plt.plot(y_true)
    plt.plot(y_pred)
    plt.show()

    return np.array(y_pred)


def sifting(data):
    index = list(range(len(data)))

    max_peaks = list(argrelextrema(data, np.greater)[0])
    min_peaks = list(argrelextrema(data, np.less)[0])

    ipo3_max = spi.splrep(max_peaks, data[max_peaks], k=3)  # 样本点导入，生成参数
    iy3_max = spi.splev(index, ipo3_max)  # 根据观测点和样条参数，生成插值

    ipo3_min = spi.splrep(min_peaks, data[min_peaks], k=3)  # 样本点导入，生成参数
    iy3_min = spi.splev(index, ipo3_min)  # 根据观测点和样条参数，生成插值

    iy3_mean = (iy3_max + iy3_min) / 2
    return data - iy3_mean


def hasPeaks(data):
    max_peaks = list(argrelextrema(data, np.greater)[0])
    min_peaks = list(argrelextrema(data, np.less)[0])

    if len(max_peaks) > 3 and len(min_peaks) > 3:
        return True
    else:
        return False


# 判断IMFs
def isIMFs(data):
    max_peaks = list(argrelextrema(data, np.greater)[0])
    min_peaks = list(argrelextrema(data, np.less)[0])

    if min(data[max_peaks]) < 0 or max(data[min_peaks]) > 0:
        return False
    else:
        return True


def getIMFs(data):
    while not isIMFs(data):
        data = sifting(data)
    return data


# EMD函数
def EMD(data):
    IMFs = []
    while hasPeaks(data):
        data_imf = getIMFs(data)
        data = data - data_imf
        IMFs.append(data_imf)
    return IMFs


if __name__ == '__main__':
    # 绘制对比图
    data = data_import.data_import_2014_hour()
    data = np.array(data['mf'])
    train_length = 288

    IMFs = EMD(data)
    n = len(IMFs) + 1

    # # 原始信号
    # plt.figure(figsize=(18, 15))
    # plt.subplot(n, 1, 1)
    # plt.plot(data, label='Origin')
    # plt.title("Origin ")
    #
    # # 若干条IMFs曲线
    # for i in range(0, len(IMFs)):
    #     plt.subplot(n, 1, i + 2)
    #     plt.plot(IMFs[i])
    #     plt.ylabel('Amplitude')
    #     plt.title("IMFs " + str(i + 1))
    #
    # plt.legend()
    # plt.xlabel('time (s)')
    # plt.ylabel('Amplitude')
    # plt.savefig('IMFs.png')
    # plt.show()

    arrays = []

    # for imf in IMFs:
    #     imf = np.array(imf)
    #     arrays.append(get_arima(imf, train_length))
    arrays.append(get_arima(IMFs[0], train_length, (1, 0, 2)))
    arrays.append(get_arima(IMFs[1], train_length, (8, 0, 0)))
    arrays.append(get_arima(IMFs[2], train_length, (12, 0, 0)))
    arrays.append(get_arima(IMFs[3], train_length, (13, 0, 0)))

    # plt.plot(data[train_length:])
    plt.plot(np.sum(IMFs, axis=0)[train_length:])
    plt.plot(np.sum(arrays, axis=0))
    plt.show()

