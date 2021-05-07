
import re
import sys
import time
import math
import torch
# import kdtree
import pickle
import random
import numpy as np
from h52pic import *
from tqdm import tqdm
from tools.distance_compution import trajectory_distance_combain,trajecotry_distance_list


class ProgressBar (object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'
    
    def __init__(self, total, width=40, fmt=DEFAULT, symbol='=',
                 output=sys.stderr):
        assert len (symbol) == 1
        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub (r'(?P<name>%\(.+?\))d', r'\g<name>%dd' % len (str (total)), fmt)
        self.current = 0
    
    def __call__(self):
        percent = self.current / float (self.total)
        size = int (self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'
        
        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining}
        
        print ('\r' + self.fmt % args, file=self.output, end='')
    
    def done(self):
        self.current = self.total
        self ()
        print ('', file=self.output)

def preprocess_trajectory_dataset(Index_Lat_Lon_Dataset, latitude_unit, longitude_unit, latitude_min, longitude_min):
    preprocess_trajectories = []
    for trajectory in tqdm(Index_Lat_Lon_Dataset):
        if len(trajectory[0]) > 32:
            gps_trajectory = [[latitude_min + latitude_unit*p[2],
                               longitude_min + longitude_unit*p[1]] for p in trajectory[0]][:32] # lat lon
            preprocess_trajectories.append(gps_trajectory)
    return preprocess_trajectories


if __name__ == "__main__":
    porto_range = {
        "lon_min": -8.735152,
        "lat_min": 40.953673,
        "lon_max": -8.156309,
        "lat_max": 41.307945
    }
    
    engrider = EngriderMeters(porto_range, 100, 100)
    lat_unit = engrider.lat_unit
    lon_unit = engrider.lon_unit
    print(lat_unit)
    
    test_lat_lon_dataset = pickle.load(open("/home/xiaoziyang/Downloads/trajectory/bert_similarity/porto_test_grid.pkl", "rb"))
    Token_dict = pickle.load(open("/home/xiaoziyang/Downloads/trajectory/bert_similarity/porto_token_dict.pkl", "rb"))
    test_dataset = preprocess_trajectory_dataset(test_lat_lon_dataset, lat_unit, lon_unit, porto_range["lat_min"], porto_range["lon_min"])[:10000]

    np_traj_coord = []
    for trajectory in test_dataset:
        np_traj_coord.append(np.array([point for point in trajectory]))  # lat lon

    np_traj_coord_test = np_traj_coord[:1000]
    print(np_traj_coord[0])
    print(np_traj_coord[1])
    print(len(np_traj_coord))

    distance_type = 'discret_frechet'

    trajecotry_distance_list(np_traj_coord_test, np_traj_coord, batch_size=25, processors=120, distance_type=distance_type,
                             data_name='porto')

    # trajectory_distance_combain(1000000, batch_size=250, metric_type=distance_type, data_name='porto')
