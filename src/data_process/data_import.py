import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def data_import_csv(filename: str):
    """
    :param filename: The file name imported from 'res' directory
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../../res/{filename}')
    df = pd.read_csv(path, parse_dates=['time'])
    return df


def data_import_tsv(filename: str):
    """
    :param filename: The file name imported from 'res' directory
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../../res/{filename}')
    df = pd.read_csv(path, sep='\t', parse_dates=['time'])
    return df


def data_import_2014():
    dfs = [data_import_tsv(f'2year/{y}.tsv') for y in range(2014, 2024, 2)]
    df = pd.concat(dfs, ignore_index=True)
    df = df.set_index('time')
    df = df.loc[:, ['mf']]

    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    # resample_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='10D')

    df = df.reindex(index=date_range)
    # df = df.reindex(index=resample_date_range, method='ffill')

    # df = df.interpolate(method='spline', order=5)
    df = df.interpolate(method='nearest')

    df['end_of_10days'] = df.index
    df['end_of_10days'] = df['end_of_10days'].dt.floor('10D') + pd.DateOffset(days=1)
    df = df.groupby('end_of_10days')['mf'].max().reset_index()
    df = df.set_index('end_of_10days')

    df['mf'] /= 10 ** 6

    return df


def data_import_year_hour(year):
    dfs = []
    for m in range(1, 12, 2):
        dfs.append(data_import_tsv('2month/{:d}-{:02d}.tsv'.format(year, m)))
    df = pd.concat(dfs, ignore_index=True)
    # df = df.set_index('time')
    # df = df.loc[:, ['mf']]

    df['mf'] /= 10 ** 6

    return df


def data_import_2014_hour():
    dfs = []
    for y in range(2012, 2022):
        # for m in range(1, 12, 2):
        #     dfs.append(data_import_tsv('2month/{:d}-{:02d}.tsv'.format(y, m)))
        dfs.append(data_import_year_hour(y))
    df = pd.concat(dfs, ignore_index=True)

    df = df.set_index('time')

    df['day'] = df.index
    df['day'] = df['day'].dt.floor('D')
    df = df.groupby('day')['mf'].mean().reset_index()

    # return df
    # df = df.set_index('day')

    # day_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    # df = df.reindex(index=day_range)
    # df = df.interpolate()

    def to_10days(date):
        year = date.year
        month = date.month
        day = min((date.day + 9) // 10, 3) * 10 - 9
        return '{:04d}-{:02d}-{:02d}'.format(year, month, day)

    df['10days'] = df['day'].apply(to_10days)

    # for i in range(2012, 2022):
    #     for j in range(1, 13):
    #         for k in range(1, 22, 10):
    #             print(pd)

    df = df.groupby('10days')['mf'].mean().reset_index()

    df['10days'] = pd.to_datetime(df['10days'], format='%Y-%m-%d')
    df = df.set_index('10days')

    all_dates = []
    current_date = datetime(2012, 1, 1)
    while current_date.year <= 2021:
        if current_date.day in [1, 11, 21]:
            all_dates.append(current_date)
        current_date += timedelta(days=1)

    all_dates = pd.DatetimeIndex(all_dates)
    df = df.reindex(all_dates).reset_index()
    df = df.set_index('index')

    # day_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    # df = df.reindex(index=day_range)
    # df = df[(df.index.day == 1) | (df.index.day == 11) | (df.index.day == 21)]

    # df['10days'] = df.index.year * 10000 + df.index.month * 100 + df.index.day
    # df.index.day = 1
    # df['10days'] = df.index
    # df['10days'] = df['10days'].dt.floor('10D')
    # df = df.reset_index()
    # df = df.set_index('10days')

    # df = df.set_index('index')
    # df = df.interpolate()
    # plt.plot(df.index, df['mf'])
    # plt.xlabel('date')
    # plt.ylabel('MUF (MHz)')
    # plt.legend()
    # plt.show()

    # df = df.loc[:, ['mf']]

    # date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    # resample_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='10D')

    # df = df.reindex(index=date_range)
    # df = df.reindex(index=resample_date_range, method='ffill')

    # df = df.interpolate(method='spline', order=5)
    # df = df.interpolate(method='nearest')

    # df['end_of_10days'] = df.index
    # df['end_of_10days'] = df['end_of_10days'].dt.floor('10D') + pd.DateOffset(days=1)
    # df = df.groupby('end_of_10days')['mf'].max().reset_index()
    # df = df.set_index('end_of_10days')

    # df['mf'] /= 10 ** 6

    df = df.interpolate()

    return df


if __name__ == '__main__':
    df = data_import_2014_hour()

    print(df.info())
    print(df.head(5))
    print(df.tail(5))

    plt.plot(df.index, df['mf'])
    plt.show()

    raise 1

    # df1 = data_import_year_hour(2014)
    # for i in range(24):
    #     dfi = df1[df1.index.hour == i]
    #     print(dfi['mf'].var())
    #     plt.plot(dfi.index, dfi['mf'])
    # for y in range(2014, 2024):
    #     df = data_import_year_hour(y)
    #     dfi_var = lambda hour: df[df.index.hour == hour]['mf'].var()
    #     df_var = [dfi_var(h) for h in range(24)]
    #     plt.plot(range(24), df_var)

    # plt.legend(range(2014, 2024))
    # plt.show()
    #
    # raise 1
    # import maidenhead_distance

    # dist = maidenhead_distance.distance_from_lat_lon_in_km
    # df_csv = data_import_csv('2.csv')
    # df_csv = data_import_tsv('test.tsv')
    # cond = dist(df_csv['rx_lat'], df_csv['rx_lon'], df_csv['tx_lat'], df_csv['tx_lon']) > 500
    # df_csv1 = df_csv.where(cond)
    df_2014 = data_import_2014()
    print(df_2014.head(5))
    print(df_2014.tail(5))

    # plt.figure(figsize=(10, 10), dpi=100)
    # plt.scatter(df_csv['rx_lon'], df_csv['rx_lat'])
    # # plt.scatter(df_csv1['rx_lon'], df_csv1['rx_lat'])
    # # plt.scatter([[1,2],[3,4],[5,6]])
    plt.plot(df_2014.index, df_2014['mf'])
    plt.show()
