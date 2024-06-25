import matplotlib.pyplot as plt
import statsmodels.tsa.statespace.sarimax
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from PyEMD import EMD, Visualisation
import numpy as np

from src.data_process import data_import


def get_arima(data, length):
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

    aic = sm.tsa.arma_order_select_ic(train_data, ic='aic', max_ar=10, max_ma=5)['aic_min_order']
    print('AIC: ', aic)

    arima_model = statsmodels.tsa.statespace.sarimax.SARIMAX(train_data, order=(aic[0], 1, aic[1]), seasonal_order=(0, 1, 0, 36))
    # arima_model = statsmodels.tsa.statespace.sarimax.SARIMAX(train_data, order=(3, 1, 3), seasonal_order=(0, 1, 0, 36))
    # arima_model = sm.tsa.arima.ARIMA(train_data, order=(bic[0], 0, bic[1]))
    arima_res = arima_model.fit()

    return np.array(arima_res.predict(len(train_data), len(data) - 1))

    # y_true = test_data
    # y_pred = arima_res.predict(len(train_data), len(data) - 1)
    #
    # plt.plot(y_true)
    # plt.plot(y_pred)
    #
    # plt.show()


def get_lstm(train_data_lstm):
    pass


if __name__ == '__main__':
    data = data_import.data_import_2014_hour()
    data = np.array(data['mf'])

    train_length = 288

    # res = STL(data, period=36).fit()
    # res.plot()
    # plt.show()
    #
    # data_arima = res.seasonal
    # data_lstm = res.resid
    #
    # train_data_arima = data_arima[:288]
    # test_data_arima = data_arima[288:]
    # train_data_lstm = data_lstm[:288]
    # test_data_lstm = data_lstm[288:]

    # result_arima = get_arima(train_data_arima, test_data_arima)
    emd = EMD()
    emd.emd(data)
    imfs, res = emd.get_imfs_and_residue()

    imfs1 = np.sum([imf for imf in imfs], axis=0)
    # imfs1 += res
    # plt.plot(imfs1)
    # plt.plot(data)
    # plt.show()
    # raise 1

    vis = Visualisation()
    vis.plot_imfs(imfs=imfs, residue=res, include_residue=True)
    vis.show()
    raise 1

    arrays = []

    for imf in imfs:
        arrays.append(get_arima(imf, train_length))

    # y_true = np.sum([imf[288:] for imf in imfs], axis=0)
    # y_pred = np.sum(arrays, axis=0)
    #
    # plt.plot(y_true)
    # plt.plot(y_pred)
    # plt.show()

    # print(np.sum(arrays, axis=0))

