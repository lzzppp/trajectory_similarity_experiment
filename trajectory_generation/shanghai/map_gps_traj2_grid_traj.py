import pickle

import math
import pickle
from glob import glob
from tqdm import tqdm
from h52pic import *
from itertools import groupby

# porto_range = {
# 	"lon_min": -8.735152,
# 	"lat_min": 40.953673,
# 	"lon_max": -8.156309,
# 	"lat_max": 41.307945
# }

shanghai_range = {
    "lon_min": 120.0011,
    "lat_min": 30.003,
    "lon_max": 121.999,
    "lat_max": 31.9996
}

# porto_range = {
#     "lon_min":-8.835152,
#     "lat_min":40.853673,
#     "lon_max":-8.056309,
#     "lat_max":41.407945,
# }

def traj2cell_ids(traj_time, engrider):
    traj_grid_seqs = [[], []]
    cell_seqs = [engrider.to_str_idx(coord) for coord in traj_time[0]]  # lon lat
    target = cell_seqs[0][0]
    traj_grid_seqs[0].append(cell_seqs[0])
    traj_grid_seqs[1].append(traj_time[1][0])
    index = 1
    while index < len(cell_seqs):
        if cell_seqs[index][0] == target:
            index += 1
        else:
            traj_grid_seqs[0].append(cell_seqs[index])
            traj_grid_seqs[1].append(traj_time[1][index])
            target = cell_seqs[index][0]
            index += 1
    return traj_grid_seqs

def delete_sationery_point(trajectory_sationery):
    delete_trajectory = [[trajectory_sationery[0][0]], [trajectory_sationery[1][0]]]
    for i in range (1, len (trajectory_sationery[0])):
        if trajectory_sationery[0][i - 1][0] != trajectory_sationery[0][i][0] or trajectory_sationery[0][i - 1][1] != \
                trajectory_sationery[0][i][1]:
            delete_trajectory[0].append (trajectory_sationery[0][i])
            delete_trajectory[1].append (trajectory_sationery[1][i])
    return delete_trajectory


if __name__ == "__main__":
    ###################################################################################
    engrider = EngriderMeters(shanghai_range, 360, 360)
    in_range_trajectory_dataset = pickle.load(open("shanghai_taxi_data.pkl", "rb"))
    trajectory_grid_dataset=[]
    for trajectory in tqdm(in_range_trajectory_dataset):
        trajectory_and_time = [[], []]
        for idx, point in enumerate(trajectory):
            trajectory_and_time[0].append([float(point[2]), float(point[3])])
            if idx==0:
                tim = 0
                trajectory_and_time[1].append(tim)
            else:
                tim += time.strptime(point[1], "%Y-%m-%d  %H:%M:%S").tm_sec + \
                       time.strptime(point[1], "%Y-%m-%d  %H:%M:%S").tm_min * 60 + \
                       time.strptime(point[1], "%Y-%m-%d  %H:%M:%S").tm_hour * 3600 - \
                       time.strptime(trajectory[idx - 1][1], "%Y-%m-%d  %H:%M:%S").tm_sec - \
                       time.strptime(trajectory[idx - 1][1], "%Y-%m-%d  %H:%M:%S").tm_min * 60 - \
                       time.strptime(trajectory[idx - 1][1], "%Y-%m-%d  %H:%M:%S").tm_hour * 3600
                trajectory_and_time[1].append(tim)
        station_process_trajectory = delete_sationery_point(trajectory_and_time)
        trajectory_grid = traj2cell_ids(station_process_trajectory, engrider)
        if len(trajectory_grid[0]) > 20:
            trajectory_grid_dataset.append(trajectory_grid)
    pickle.dump(trajectory_grid_dataset, open("trajectory_grid_lat_lon_dataset.pkl", "wb"))
    ################################################################################