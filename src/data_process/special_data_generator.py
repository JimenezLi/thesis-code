import os
import pandas as pd
import numpy as np

np.random.seed(0)


def generate_periodical():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../../res/test_periodical.csv')
    with open(path, mode='w') as f:
        print('time,mf', file=f)
        for i in range(100):
            print(f'{i},{100 + 60 * (i % 2) + np.random.randint(0, 10)}', file=f)
        f.close()


if __name__ == '__main__':
    generate_periodical()
    df = pd.read_csv('../../res/test_periodical.csv')
    print(df.head(5))
