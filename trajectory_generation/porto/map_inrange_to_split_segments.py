
import pickle
from tqdm import tqdm
from geopy.distance import geodesic

if __name__ == "__main__":
	porto_inrange_dataset = pickle.load(open("porto_without_outline_dataset.pkl", "rb"))
	porto_distance_split_dataset = []
	porto_split_segement_dataset = []
	
	for trajectory in tqdm(porto_inrange_dataset):
		index = 0
		trajectory_segement = []
		while index < len(trajectory):
			if index == len(trajectory) - 1:
				trajectory_segement.append(trajectory[index])
				porto_distance_split_dataset.append(trajectory_segement)
				index += 1
				continue
			trajectory_segement.append(trajectory[index])
			point_index = trajectory[index]
			point_index_1 = trajectory[index + 1]
			dist = geodesic((point_index[1], point_index[0]), (point_index_1[1], point_index_1[0])).km
			if dist > 0.8: # 这里是分割轨迹的最大距离 0.8km
				porto_distance_split_dataset.append(trajectory_segement)
				trajectory_segement = []
			index += 1
	
	pickle.dump(porto_distance_split_dataset, open("porto_distance_split_without_outline_dataset.pkl", "wb")) # 从inrang数据中根据轨迹中两点距离进行分割，目前设定是0.8km
	
	# trajectory_segment = [] 下面是对于轨迹长度进行切分，250长度以内为一段
	for trajectory in tqdm(porto_distance_split_dataset):
		index = 0
		trajectory_segment = []
		while index < len(trajectory):
			if index == len(trajectory) - 1:
				trajectory_segment.append(trajectory[index])
				porto_split_segement_dataset.append(trajectory_segment)
				index += 1
				continue
			trajectory_segment.append(trajectory[index])
			if len(trajectory_segment) == 250:
				porto_split_segement_dataset.append(trajectory_segment)
				trajectory_segment = []
			index += 1
	
	pickle.dump(porto_split_segement_dataset, open("porto_split_segement_without_outline_dataset.pkl", "wb")) # 从distance_split数据集中根据数据长短进行切分 长度250以内的算作一条轨迹