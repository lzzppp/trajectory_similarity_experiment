
import math
import pickle
from glob import glob
from tqdm import tqdm
from h52pic import *
from itertools import groupby

porto_range = {
	"lon_min": -8.735152,
	"lat_min": 40.953673,
	"lon_max": -8.156309,
	"lat_max": 41.307945
}

# /mnt/data4/lizepeng/porto/porto_grid_time_dataset.pkl

def traj2cell_ids(traj_time, engrider):
	traj_grid_seqs = [[], []]
	cell_seqs = [engrider.to_str_idx(coord) for coord in traj_time[0]]  # lon lat
	target = cell_seqs[0]
	traj_grid_seqs[0].append(target)
	traj_grid_seqs[1].append(traj_time[1][0])
	index = 1
	while index < len(cell_seqs):
		if cell_seqs[index][0] == target[0]:
			index += 1
		else:
			traj_grid_seqs[0].append(cell_seqs[index])
			traj_grid_seqs[1].append(traj_time[1][index])
			target = cell_seqs[index]
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
	engrider = EngriderMeters(porto_range, 100, 100)
	porto_process_dataset = pickle.load(open("porto_split_segement_without_outline_dataset.pkl", "rb"))
	porto_grid_time_dataset = []
	# 时间序列从1开始
	for trajectory in tqdm(porto_process_dataset): # lon lat
		if len(trajectory) < 20:
			continue
		gps_with_time = [trajectory, []]
		gps_with_time[1] = [i+1 for i in range(len(trajectory))]
		station_process_trajectory = delete_sationery_point(gps_with_time)
		trajectory_grid = traj2cell_ids(station_process_trajectory, engrider)
		if len(trajectory_grid[0]) > 20:
			porto_grid_time_dataset.append(trajectory_grid)
	pickle.dump(porto_grid_time_dataset, open("porto_grid_lat_lon_time_dataset.pkl", "wb"))