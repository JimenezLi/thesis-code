import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.statespace.sarimax
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from src.data_process import data_import
import matplotlib
import seaborn as sns
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import norm

matplotlib.rc("font", family='DengXian')

if __name__ == '__main__':
    df = data_import.data_import_2014_hour()
    data = df.copy()

    # print(df.head(5))
    # print(df.info())
    # print(data.info())

    # plt.plot(data.index, data['mf'].values)
    # plt.show()

    # train = data.loc[:'2020/1/1', :]
    # test = data.loc['2020/1/1':, :]
    train = data.loc[:288, :]
    test = data.loc[288:, :]
    print(train.info())
    print(test.info())
    print(sm.tsa.stattools.adfuller(train))
    print(acorr_ljungbox(train, lags=[6, 12], boxpierce=True))

    # acf = plot_acf(train['mf'])
    # plt.title("Autocorrelation Function (ACF) plot of MUF")
    # plt.show()
    # pacf = plot_pacf(train['mf'])
    # plt.title("Partial Autocorrelation Function (PACF) plot of MUF")
    # plt.show()
    #
    # trend_evaluate_aic = sm.tsa.arma_order_select_ic(train['mf'], ic='aic', max_ar=10, max_ma=5)['aic_min_order']
    # print(trend_evaluate_aic)
    # trend_evaluate_bic = sm.tsa.arma_order_select_ic(train['mf'], ic='bic', max_ar=10, max_ma=5)['bic_min_order']
    # print(trend_evaluate_bic)
    # raise Exception()

    # model = sm.tsa.arima.ARIMA(train, order=(2, 0, 2))

    model = statsmodels.tsa.statespace.sarimax.SARIMAX(train, order=(3, 0, 2), seasonal_order=(0, 1, 0, 36))
    arima_res = model.fit()
    print(arima_res.summary())

    y_true = test['mf']
    y_pred = arima_res.predict(test.index.min(), test.index.max())

    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print('MAE:', mae)
    print('MSE:', mse)
    print('MAPE:', mape)

    plt.plot(test.index, y_true)
    plt.plot(test.index, y_pred)
    # plt.plot(train.index, train['mf'])
    # plt.plot(train.index, arima_res.fittedvalues)
    plt.title('Prediction result of ARIMA model')
    plt.xlabel('date')
    plt.ylabel('MUF (MHz)')
    plt.legend(['true', 'pred'])
    plt.show()

    residual = y_pred - y_true
    print('Residual mean: ', np.mean(residual))
    print('Durbin-Watson:', durbin_watson(residual))
    plt.plot(test.index, residual)
    plt.show()

    sns.distplot(residual, fit=stats.norm)
    plt.xlabel('residual')
    plt.show()

    stats.probplot(residual, plot=plt)
    plt.show()

    y_pred_normalized = (-10 + np.array(y_pred) - np.mean(residual)) / np.std(residual)
    probability = norm.cdf(y_pred_normalized)
    # plt.plot(probability)
    # plt.xlabel('date')
    # plt.ylabel('Probability of Communication for 10MHz')
    # plt.legend()
    # plt.show()

    time = range(1, 73)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(time, y_true, '-', label='true')
    ax.plot(time, y_pred, '-', label='pred')
    ax2 = ax.twinx()
    ax2.plot(time, probability, linestyle=':', color='green', label='10MHz Communication Probability')
    ax.legend(loc=0)
    ax.grid()
    ax.set_xlabel("date")
    ax.set_ylabel("MUF (MHz)")
    ax2.set_ylabel("Probability of Communication for 10MHz")
    ax2.set_ylim(0, 1)
    ax2.legend(loc=0)
    plt.show()







