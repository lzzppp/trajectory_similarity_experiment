
import torch
# import kdtree
import pickle
import random
import numpy as np
from h52pic import *
from tqdm import tqdm
# from src.dataloader_trainer import EncoderwithDecoderTrainer
# from src.model.model_similarity import TERT, trajectory_Distance_Loss
# from src.utils import bool_flag, check_data_params, network_to_half, initialize_exp, get_optimizer

def preprocess_trajectory_dataset(Index_Lat_Lon_Dataset, Index_Dict, latitude_unit, longitude_unit, latitude_min, longitude_min):
    preprocess_trajectories = []
    for trajectory in tqdm(Index_Lat_Lon_Dataset):
        grid = [Index_Dict[p[0]] for p in trajectory[0]]
        gps_trajectory = [[latitude_min + latitude_unit*p[2],
                           longitude_min + longitude_unit*p[1]] for p in trajectory[0]] # lat lon
        preprocess_trajectories.append([grid, gps_trajectory])
    lats, lons = [], []
    for trajectory in tqdm(preprocess_trajectories):
        for lat, lon in trajectory[1]:
            lats.append(lat)
            lons.append(lon)
    mean_lat, mean_lon, std_lat, std_lon = np.mean(lats), np.mean(lons), np.std(lats), np.std(lons)
    return preprocess_trajectories, [mean_lat, mean_lon, std_lat, std_lon]

if __name__ == "__main__":
    shanghai_range = {
        "lon_min": 120.0011,
        "lat_min": 30.003,
        "lon_max": 121.999,
        "lat_max": 31.9996
    }
    
    porto_range = {
        "lon_min": -8.735152,
        "lat_min": 40.953673,
        "lon_max": -8.156309,
        "lat_max": 41.307945
    }
    
    engrider = EngriderMeters(porto_range, 100, 100)
    lat_unit = engrider.lat_unit
    lon_unit = engrider.lon_unit
    
    lat_lon_dataset = pickle.load(open("/home/xiaoziyang/Downloads/trajectory/bert_similarity/porto_train_grid.pkl", "rb"))
    lat_lon_dataset += pickle.load(open("/home/xiaoziyang/Downloads/trajectory/bert_similarity/porto_test_grid.pkl", "rb"))
    Token_dict = pickle.load(open("/home/xiaoziyang/Downloads/trajectory/bert_similarity/porto_token_dict.pkl", "rb"))
    dAtAset, trajectory_features = preprocess_trajectory_dataset(lat_lon_dataset, Token_dict, lat_unit, lon_unit, porto_range["lat_min"], porto_range["lon_min"])
    
    print(trajectory_features)
    
    pickle.dump(trajectory_features, open("/home/xiaoziyang/Downloads/trajectory/bert_similarity/porto_trajectory_features.pkl", "wb"))