
import pickle
import numpy as np
from tqdm import tqdm

# 经纬度 Latitude and longitude

class Map_network(object):
	def __init__(self,
	             min_lon, max_lon,
	             min_lat, max_lat,
	             gap):
		self.min_lon = min_lon
		self.max_lon = max_lon
		self.min_lat = min_lat
		self.max_lat = max_lat
		self.lon_gap = gap
		self.lat_gap = gap
		self.x_size = 10000
		self.y_size = 10000
		self.get_network_lat_lon_size()
		self.grid_map_dict = {}
	
	def Gps_point_to_grid(self, gps_lat, gps_lon):
		x = int((gps_lat - self.min_lat)/self.lat_gap)
		y = int((gps_lon - self.min_lon)/self.lon_gap)
		grid_code = x * self.y_size + y
		self.grid_map_dict[grid_code] = [x, y]
		return grid_code
	
	def get_network_lat_lon_size(self):
		self.x_size = int((self.max_lat - self.min_lat)/self.lat_gap)
		self.y_size = int((self.max_lon - self.min_lon)/self.lon_gap)
		print("x size: ", self.x_size, " y_size: ", self.y_size)

# 1,2015-04-01 00:00:00,103.87326,1.35535

if __name__ == "__main__":
	lat_gap = 0.00001
	lon_gap = 0.00001
	map_network = Map_network(min_lon=103.603226, max_lon=104.088684,
	                          min_lat=1.165715, max_lat=1.46639,
	                          gap=lon_gap)
	city_map_network_dict = {}
	for x_idx in tqdm(range(map_network.x_size)):
		for y_idx in range(map_network.y_size):
			lat, lon = x_idx*lat_gap + map_network.min_lat + 0.000000001, y_idx*lon_gap + map_network.min_lon + 0.000000001
			grid = map_network.Gps_point_to_grid(lat, lon)
			# [x_, y_] = map_network.grid_map_dict[grid]
			# assert x_ == x_idx and y_ == y_idx
			city_map_network_dict[grid] = [x_idx, y_idx, lat, lon]
			
	pickle.dump(city_map_network_dict, open("city_map_network_dict.pkl", "wb"))
	# pickle.dump(map_network.grid_map_dict, open("grid_map_xy_dict.pkl", "wb"))
	
	with open("singapore_network_map_road.txt") as file:
		for idx, grid_key in enumerate(city_map_network_dict):
			grid_query = city_map_network_dict[grid_key]
			file.write(str(idx + 1) + ",2015-04-01 00:00:00,"+str(grid_query[3])+","+str(grid_query[2])+"\n")