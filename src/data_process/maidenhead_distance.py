import maidenhead as mh
import numpy as np


def distance_from_lat_lon_in_km(rx_lat, rx_lon, tx_lat, tx_lon):
    R = 6371  # Radius of the Earth
    p = np.pi / 180

    a = 0.5 - np.cos((rx_lat - tx_lat) * p) / 2 + np.cos(rx_lat * p) * np.cos(tx_lat * p) * (
            1 - np.cos((rx_lon - tx_lon) * p)) / 2
    return 2 * R * np.arcsin(np.sqrt(a))


def distance_from_maidenhead_in_km(rx_loc, tx_loc):
    rx_lat, rx_lon = mh.to_location(rx_loc)
    tx_lat, tx_lon = mh.to_location(tx_loc)
    return distance_from_lat_lon_in_km(rx_lat, rx_lon, tx_lat, tx_lon)


if __name__ == '__main__':
    print(distance_from_lat_lon_in_km(36.12, -86.67, 33.94, -118.40))
    print(distance_from_maidenhead_in_km('JN48FW', 'JN47UL'))
    print(distance_from_maidenhead_in_km('EN52', 'IO82xl'))  # 6227
