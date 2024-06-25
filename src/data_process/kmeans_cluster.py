import numpy as np
import pandas as pd
import random
import sys
import time
import maidenhead
import maidenhead_distance
import matplotlib.pyplot as plt


def kmeans_cluster(samples, iteration: int, cluster: int):
    result = []
    for i in range(cluster):
        result.append([])
    for it in range(iteration):
        pass
        # for i in range(length):


class KMeansClusterer:
    def __init__(self, ndarray, cluster_num, threshold=0.001):
        self.ndarray = ndarray
        self.cluster_num = cluster_num
        self.points = self.__pick_start_point(ndarray, cluster_num)
        self.threshold = threshold

    def cluster(self):
        result = []
        for i in range(self.cluster_num):
            result.append([])
        for item in self.ndarray:
            distance_min = sys.maxsize
            index = -1
            for i in range(len(self.points)):
                distance = self.__distance(item, self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index] = result[index] + [item.tolist()]
        new_center = []
        for item in result:
            new_center.append(self.__center(item).tolist())
        if -self.threshold < (self.points - new_center).all() < self.threshold:
            return result, new_center

        self.points = np.array(new_center)
        return self.cluster()

    def __center(self, lst):
        return np.array(lst).mean(axis=0)

    def __distance(self, p1, p2):
        tmp = 0
        for i in range(len(p1)):
            tmp += pow(p1[i] - p2[i], 2)
        return pow(tmp, 0.5)

    def __pick_start_point(self, ndarray, cluster_num):
        if cluster_num < 0 or cluster_num > ndarray.shape[0]:
            raise Exception("Wrong cluster number")

        indexes = random.sample(list(np.arange(0, ndarray.shape[0], step=1)), cluster_num)
        points = []
        for index in indexes:
            points.append(ndarray[index].tolist())
        return np.array(points)


if __name__ == '__main__':
    import data_import

    # df = data_import.data_import_tsv('test.tsv')
    df = data_import.data_import_csv('2.csv')
    samples = np.array(list(zip(df['rx_lon'], df['rx_lat'])))

    cluster_num = 4
    c = KMeansClusterer(samples, cluster_num)
    result, centers = c.cluster()
    print(centers)
    for index, i in enumerate(centers):
        print(maidenhead.to_maiden(i[1], i[0]), len(result[index]))

    # plt.figure(figsize=(10, 10), dpi=100)
    # colors = ['red', 'orange', 'green', 'blue']

    for index, r in enumerate(result):
        x = [i[0] for i in r]
        y = [i[1] for i in r]
        # plt.scatter(x, y, color=colors[index])
        plt.scatter(x, y, s=2)

    plt.scatter([i[0] for i in centers], [i[1] for i in centers], color='black', s=4)

    plt.title('Center of most data decided by K-Means')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.legend([maidenhead.to_maiden(i[1], i[0]) for i in centers])

    plt.show()
