
import time
import pickle
from tqdm import tqdm
from glob import glob

if __name__ == "__main__":
	lat_range = [1000, -1000] # 经度 120.0011, 121.999
	lon_range = [1000, -1000] # 纬度 30.003, 31.9996
	taxi_file_list = glob("Taxi_070220/*")
	taxi_all_data = []
	length_all = []
	for file in tqdm(taxi_file_list):
		taxi_data_list = []
		with open(file) as read_file:
			for line in read_file:
				line = line.rstrip("\n")
				line = line.split(",")
				if float(line[2]) < lat_range[0]:
					lat_range[0] = float(line[2])
				if float(line[2]) > lat_range[1]:
					lat_range[1] = float(line[2])
				if float(line[3]) < lon_range[0]:
					lon_range[0] = float(line[3])
				if float(line[3]) > lon_range[1]:
					lon_range[1] = float(line[3])
				taxi_data_list.append(line[:4])
		if len(taxi_data_list) > 30:
			length_all.append(len(taxi_data_list))
			taxi_all_data.append(taxi_data_list)
	print('max(length_all)', max(length_all))
	print('min(length_all)', min(length_all))
	print('lon_range', lon_range)
	print('lat_range', lat_range)
	
	taxi_all_data_split = []
	for line in tqdm(taxi_all_data):
		new_line = []
		i, j = 0, 0
		while j < len(line):
			if j == len(line) - 1:
				new_line.append(line[j])
				taxi_all_data_split.append(new_line)
				j += 1
				continue
			time_j = time.strptime(line[j][1], "%Y-%m-%d  %H:%M:%S").tm_sec + \
			         time.strptime(line[j][1], "%Y-%m-%d  %H:%M:%S").tm_min * 60 + \
			         time.strptime(line[j][1], "%Y-%m-%d  %H:%M:%S").tm_hour * 3600
			time_j_1 = time.strptime(line[j+1][1], "%Y-%m-%d  %H:%M:%S").tm_sec + \
			           time.strptime(line[j+1][1], "%Y-%m-%d  %H:%M:%S").tm_min * 60 + \
			           time.strptime(line[j+1][1], "%Y-%m-%d  %H:%M:%S").tm_hour * 3600
			new_line.append(line[j])
			if (time_j_1 - time_j)/60 > 5:
				taxi_all_data_split.append(new_line)
				new_line = []
			j += 1
	print(len(taxi_all_data_split))
	pickle.dump(taxi_all_data_split, open("shanghai_split_data.pkl", "wb"))
	for line in taxi_all_data_split:
		print(line)