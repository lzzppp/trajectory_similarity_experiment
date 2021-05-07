
import pickle
from tqdm import tqdm

# 经度 longitude
# 纬度 latitude

if __name__ == "__main__":
	trajectory_gps_dataset = []
	trajectory_grid_gps_dataset = pickle.load(open("shanghai_trajectory.pkl", "rb")) # 含有grid的数据
	for trajectory_grid_gps in tqdm(trajectory_grid_gps_dataset): # 轨迹数据格式为lat lon
		trajectory_gps_dataset.append(trajectory_grid_gps[1])
		# print(trajectory_grid_gps[1])
	
	pickle.dump(trajectory_gps_dataset, open("shanghai_trajectory_gps.pkl", "wb"))