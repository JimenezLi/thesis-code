import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
import numpy as np
import math
import statsmodels.api as sm
import statsmodels.tsa.statespace.sarimax
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from src.data_process import data_import
import matplotlib
import seaborn as sns
from scipy import stats
from statsmodels.stats.stattools import durbin_watson

from src.data_process import data_import


# 数据的差分转换
def difference(data_set, interval=1):
    diff = list()
    for i in range(interval, len(data_set)):
        value = data_set[i] - data_set[i - interval]
        diff.append(value)
    return pd.Series(diff)


# 对预测的数据进行逆差分转换
def invert_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# 将数据转换为监督学习集，移位后产生的NaN值补0
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# 将数据缩放到[-1,1]之间
def scale(train, test):
    # 创建一个缩放器，将数据集中的数据缩放到[-1,1]的取值范围中
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # 使用数据来训练缩放器
    scaler = scaler.fit(train)
    # 使用缩放器来将训练集和测试集进行缩放
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# 将预测值进行逆缩放，使用之前训练好的缩放器，x为一维数组，y为实数
def invert_scale(scaler, X, y):
    # 将X,y转换为一个list列表
    new_row = [x for x in X] + [y]
    # 将列表转换为数组
    array = np.array(new_row)
    # 将数组重构成一个形状为[1,2]的二维数组->[[10,12]]
    array = array.reshape(1, len(array))
    # 逆缩放输入的形状为[1,2]，输出形状也是如此
    invert = scaler.inverse_transform(array)
    # 只需要返回y值即可
    return invert[0, -1]


# 构建一个LSTM模型
def fit_lstm(train, batch_size, nb_epoch, neurons):
    # 将数据对中的x和y分开
    X, y = train[:, 0:-1], train[:, -1]
    # 将2D数据拼接成3D数据，形状为[N*1*1]
    X = X.reshape(X.shape[0], 1, X.shape[1])

    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        # shuffle是不混淆数据顺序
        his = model.fit(X, y, batch_size=batch_size, verbose=1, shuffle=False)
        # 每训练完一次就重置一次网络状态，网络状态与网络权重不同
        model.reset_states()
    return model


# 开始单步预测
def forecast_lstm(model, batch_size, X):
    # 将形状为[1:]的，包含一个元素的一维数组X，转换形状为[1,1,1]的3D张量
    X = X.reshape(1, 1, len(X))
    X = np.asarray(X).astype('float32')
    # 输出形状为1行一列的二维数组yhat
    yhat = model.predict(X, batch_size=batch_size)
    # 将yhat中的结果返回
    return yhat[0, 0]


if __name__ == '__main__':
    data = data_import.data_import_2014_hour()
    raw_value = np.array(data['mf'])

    # 将数据进行差分转换，例如[[4.6838],[4.6882],[4.7048]]转换为[[4.6882-4.6838],[4.7048-4.6882]]
    diff_value = np.array(difference(raw_value, 1))
    # print(diff_value)

    train = diff_value[:-72]
    test = diff_value[-72:]
    print(sm.tsa.stattools.adfuller(train))
    print(acorr_ljungbox(train, lags=[6, 12], boxpierce=True))

    # acf = plot_acf(train)
    # plt.title("Autocorrelation Function (ACF) plot of MUF")
    # plt.show()
    # pacf = plot_pacf(train)
    # plt.title("Partial Autocorrelation Function (PACF) plot of MUF")
    # plt.show()

    # trend_evaluate_aic = sm.tsa.arma_order_select_ic(train, ic='aic', max_ar=10, max_ma=5)['aic_min_order']
    # print(trend_evaluate_aic)
    # trend_evaluate_bic = sm.tsa.arma_order_select_ic(train, ic='bic', max_ar=10, max_ma=5)['bic_min_order']
    # print(trend_evaluate_bic)
    # raise 1

    arima_model = statsmodels.tsa.statespace.sarimax.SARIMAX(train, order=(3, 0, 3), seasonal_order=(0, 1, 0, 36))
    # arima_model = sm.tsa.arima.ARIMA(train, order=(3, 0, 3))
    arima_res = arima_model.fit()

    # print(len(train))
    # print(len(arima_res.fittedvalues))
    # raise 1

    plt.plot(np.array(train))
    plt.plot(np.array(arima_res.fittedvalues))
    plt.show()
    raise 1

    train_arima = arima_res.fittedvalues
    predict_arima = arima_res.predict(len(train), len(diff_value) - 1)
    plt.plot(test)
    plt.plot(predict_arima)
    plt.show()
    raise 1

    train_lstm = train - train_arima
    test_lstm = test - predict_arima
    # print(len(train_lstm), len(test_lstm))
    data_lstm = np.concatenate((train_lstm, test_lstm))
    # print(len(data_lstm))
    # raise 1

    # 将序列形式的数据转换为监督学习集形式，例如[[10],[11],[12],[13]]
    # 在此将其转换为监督学习集形式：[[0,10],[10,11],[11,12],[12,13]]，
    # 即前一个数作为输入，后一个数作为对应的输出
    supervised = timeseries_to_supervised(data_lstm, 1)
    supervised_value = supervised.values

    # 将数据集分割为训练集和测试集，设置后1000个数据为测试集
    testNum = 72
    train, test = supervised_value[:-testNum], supervised_value[-testNum:]
    print(train, test)

    # 将训练集和测试集都缩放到[-1,1]之间
    scaler, train_scaled, test_scaled = scale(train, test)

    # 构建一个LSTM模型并训练，样本数为1，训练次数为5，LSTM层神经元个数为4
    lstm_model = fit_lstm(train_scaled, 1, 1, 4)
    # 遍历测试集，对数据进行单步预测
    predictions = list()
    for i in range(len(test_scaled)):
        # 将测试集拆分为X和y
        X, y = test[i, 0:-1], test[i, -1]
        # 将训练好的模型、测试数据传入预测函数中
        yhat = forecast_lstm(lstm_model, 1, X)
        # 将预测值进行逆缩放
        yhat = invert_scale(scaler, X, yhat)
        # 对预测的y值进行逆差分
        # yhat = invert_difference(raw_value, yhat, len(test_scaled) + 1 - i)
        # 存储正在预测的y值
        predictions.append(yhat)

    predictions = np.array(predictions)
    predict666 = predict_arima + predictions

    plt.plot(diff_value[-72:])
    plt.plot(predict666)
    plt.show()
    raise 1

    # print(len(predictions))
    # raise 1

    # 计算方差
    rmse = mean_squared_error(raw_value[-testNum:], predictions)
    print("Test RMSE:", rmse)
    plt.plot(raw_value[-testNum:])
    plt.plot(predictions)
    plt.legend(['true', 'pred'])
    plt.show()