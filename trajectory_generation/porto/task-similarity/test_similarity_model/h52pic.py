from __future__ import print_function
import numpy as np
import pandas as pd
import os
import time
import pickle
import json
from datetime import datetime
# from tensorflow.keras.utils import to_categorical
# import  tensorflow as tf
from pytz import timezone
import h5py
from sklearn.model_selection import train_test_split
# from MultiGridFusion import MultiGridFusion
from geopy.distance import distance

class Engrider:
    def __init__(self, geo_range, grid_lon_div, grid_lat_div):
        self.geo_range = geo_range
        self.grid_lon_div = grid_lon_div
        self.grid_lat_div = grid_lat_div
        self._calculate_unit()

    def _calculate_unit(self):
        self.lon_unit = (self.geo_range['lon_max'] - self.geo_range['lon_min']) / self.grid_lon_div
        self.lat_unit = (self.geo_range['lat_max'] - self.geo_range['lat_min']) / self.grid_lat_div

    def to_grid_id(self, df):
        region_lat_pos = int((df[1] - self.geo_range['lat_min']) / self.lat_unit)
        region_lon_pos = int((df[0] - self.geo_range['lon_min']) / self.lon_unit)
        return region_lon_pos, region_lat_pos
    
    def to_str_idx(self, df):
        lon_pos, lat_pos = self.to_grid_id(df)
        return str(lat_pos * self.grid_lon_div + lon_pos), lon_pos, lat_pos


class EngriderMeters(Engrider):
    def __init__(self, geo_range, lon_meters, lat_meters):
        self.geo_range = geo_range
        self.lon_meters = lon_meters
        self.lat_meters = lat_meters
        self._calculate_unit()
        
    def _calculate_unit(self):
        lon_dis = distance((self.geo_range['lat_max'], self.geo_range['lon_min']), (self.geo_range['lat_max'], self.geo_range['lon_max'])).meters
        lat_dis = distance((self.geo_range['lat_max'], self.geo_range['lon_min']), (self.geo_range['lat_min'], self.geo_range['lon_min'])).meters
        self.lon_unit = (self.geo_range['lon_max'] - self.geo_range['lon_min']) / lon_dis * self.lon_meters
        self.lat_unit = (self.geo_range['lat_max'] - self.geo_range['lat_min']) / lat_dis * self.lat_meters
        self.grid_lon_div = int(np.ceil(lon_dis / self.lon_meters))
        print('Range lontitude distance: %.2f, latitude distance: %.2f' %(lon_dis, lat_dis))

def if_in_range(point, map_range):
    return map_range['lat_min'] < point[1] < map_range['lat_max'] and \
           map_range['lon_min'] < point[0] < map_range['lon_max']

