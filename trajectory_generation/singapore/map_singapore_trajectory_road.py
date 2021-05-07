import pickle
import numpy as np
from map_traj_network import Map_network

if __name__ == "__main__":
	lat_gap = 0.00001
	lon_gap = 0.00001
	city_map_network_dict = pickle.load(open("city_map_network_dict.pkl", "rb"))
	# grid_map_xy_dict = pickle.load(open("grid_map_xy_dict.pkl", "rb"))
	map_network = Map_network(min_lon=103.603226, max_lon=104.088684,
	                          min_lat=1.165715, max_lat=1.46639,
	                          gap=lon_gap)
	
	new_trajectory = []
	trajectory = pickle.load(open("trajectory.pkl", "rb"))
	for gps_point in trajectory:
		lat, lon = gps_point[0], gps_point[1] # 纬度 经度
		grid = map_network.Gps_point_to_grid(lat, lon)
		grid_query = city_map_network_dict[grid]
		stay_position_lat_lon = []
		for i in range(-5, 5):
			for j in range(-5, 5):
				x, y = grid_query[0] + i, grid_query[1] + j
				grid_near = x * map_network.y_size + y
				grid_near_query = city_map_network_dict[grid_near]
				stay_position_lat_lon.append([grid_near,
					                          grid_near_query[2],
				                              grid_near_query[3]])
		distance = np.array([(gps_point[0]-new_gps_point[1])**2
		                     + (gps_point[1]-new_gps_point[2])**2 for new_gps_point in stay_position_lat_lon])
		min_idx = np.argmin(distance)
		new_trajectory.append(stay_position_lat_lon[min_idx])
