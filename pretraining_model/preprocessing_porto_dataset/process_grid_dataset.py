import math
import pickle
from glob import glob
from tqdm import tqdm
from h52pic import *
from itertools import groupby

porto_range = {
    "lon_min":-8.735152,
    "lat_min":40.953673,
    "lon_max":-8.156309,
    "lat_max":41.307945,
}
# porto_range = {
#     "lon_min":-8.835152,
#     "lat_min":40.853673,
#     "lon_max":-8.056309,
#     "lat_max":41.407945,
# }

def traj2cell_ids(traj_time, engrider):
    traj_grid_seqs=[[],[]]
    cell_seqs = [engrider.to_str_idx(coord) for coord in traj_time[0]]
    for cell in groupby(cell_seqs):
        x = cell[0]
        for j in range(len(cell_seqs)):
            if x==cell_seqs[j]:
                traj_grid_seqs[0].append(x)
                traj_grid_seqs[1].append(traj_time[1][j])
                break
            
    return traj_grid_seqs

def delete_sationery_point(trajectory_sationery):
    delete_trajectory = [[trajectory_sationery[0][0]], [trajectory_sationery[1][0]]]
    for i in range(1, len(trajectory_sationery[0])):
        if trajectory_sationery[0][i-1][0]!=trajectory_sationery[0][i][0] or trajectory_sationery[0][i-1][1]!=trajectory_sationery[0][i][1]:
            delete_trajectory[0].append(trajectory_sationery[0][i])
            delete_trajectory[1].append(trajectory_sationery[1][i])
    return delete_trajectory

if __name__ == "__main__":
    # trajectory_raw_dataset = pickle.load(open("in_range_porto_dataset.pkl", "rb"))
    # in_range_trajectory_dataset = []
    # print(len(trajectory_raw_dataset))
    # for trajectory in tqdm(trajectory_raw_dataset):
    #     # print(trajectory)
    #     in_range=True
    #     for point in trajectory:
    #         if point[0] < porto_range["lon_min"] or point[0] > porto_range["lon_max"] or point[1] < porto_range["lat_min"] or point[1] > porto_range["lat_max"]:
    #             in_range=False
    #             break
    #     if in_range:
    #         in_range_trajectory_dataset.append(trajectory)
    # print(len(in_range_trajectory_dataset))
    # pickle.dump(in_range_trajectory_dataset, open("in_range_porto_dataset.pkl", "wb"))
    ####################################################################################
    # engrider = EngriderMeters(porto_range, 100, 100)
    # in_range_trajectory_dataset = pickle.load(open("in_range_porto_dataset.pkl", "rb"))
    # trajectory_grid_dataset=[]
    # for trajectory in tqdm(in_range_trajectory_dataset):
    #     try:
    #         station_process_trajectory = delete_sationery_point([trajectory, [15*(i+1) for i in range(len(trajectory))]])
    #         trajectory_grid = traj2cell_ids(station_process_trajectory, engrider)
    #         if len(trajectory_grid[0]) > 20:
    #             trajectory_grid_dataset.append(trajectory_grid)
    #     except:
    #         a=1
    # pickle.dump(trajectory_grid_dataset, open("trajectory_grid_dataset.pkl", "wb"))
    #################################################################################
    # trajectory_grid_dict = {}
    # new_trajectory_dataset = []
    # trajectory_grid_dataset = pickle.load(open("trajectory_grid_dataset.pkl", "rb"))
    # for trajectory in tqdm(trajectory_grid_dataset):
    #     grid_dict_trajectory=[]
    #     for point in trajectory[0]:
    #         if point not in trajectory_grid_dict:
    #             trajectory_grid_dict[point]=len(trajectory_grid_dict) + 1
    #         grid_dict_trajectory.append(trajectory_grid_dict[point])
    #     new_trajectory_dataset.append([grid_dict_trajectory, trajectory[1]])
    # pickle.dump(new_trajectory_dataset, open("trajectory_new_grid_dataset.pkl", "wb"))
    # pickle.dump(trajectory_grid_dict, open("trajectory_grid_dict.pkl", "wb"))
    ###########################################################################
    trajectory_grid_dataset = pickle.load(open("trajectory_new_grid_dataset.pkl", "rb"))
    trajectory_grid_dict = pickle.load(open("trajectory_grid_dict.pkl", "rb"))
    print(len(trajectory_grid_dict))
    for line in trajectory_grid_dataset:
        print(line)