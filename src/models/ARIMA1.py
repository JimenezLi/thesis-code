import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.statespace.sarimax
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from src.data_process import data_import
import matplotlib

matplotlib.rc("font", family='DengXian')

if __name__ == '__main__':
    df = pd.read_csv('../../res/test_periodical.csv')
    data = df.copy()
    data = data.set_index('time')
    # data = data.diff()
    # data['mf'][0] = data['mf'][1]
    # print(data.head(5))

    # print(df.head(5))
    # print(df.info())
    # print(data.info())

    # plt.plot(data.index, data['mf'].values)
    # plt.show()

    train = data.loc[:'80', :]
    test = data.loc['80':, :]
    print(train.info())
    print(test.info())
    print(sm.tsa.stattools.adfuller(train))
    print(acorr_ljungbox(train, lags=[6, 12], boxpierce=True))

    # acf = plot_acf(train['mf'])
    # plt.title("最高可用频率的自相关图")
    # plt.show()
    # pacf = plot_pacf(train['mf'])
    # plt.title("最高可用频率的偏自相关图")
    # plt.show()
    #
    # trend_evaluate_aic = sm.tsa.arma_order_select_ic(train['mf'], ic='aic', max_ar=10, max_ma=5)['aic_min_order']
    # print(trend_evaluate_aic)
    # trend_evaluate_bic = sm.tsa.arma_order_select_ic(train['mf'], ic='bic', max_ar=10, max_ma=5)['bic_min_order']
    # print(trend_evaluate_bic)
    # raise Exception()

    model = sm.tsa.arima.ARIMA(train, order=(1, 1, 0))

    # model = statsmodels.tsa.statespace.sarimax.SARIMAX(train, order=(3, 0, 3), seasonal_order=(0, 1, 1, 61))
    arima_res = model.fit()
    print(arima_res.summary())

    y_true = test['mf']
    y_pred = arima_res.predict(test.index.min(), test.index.max())

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print('MSE:', mse)
    print('RMSE:', rmse)
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
