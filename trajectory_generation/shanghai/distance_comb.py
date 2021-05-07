
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm

if __name__ == "__main__":
	distance_file_Prefix = "/mnt/data4/lizepeng/shanghai/features/shanghai_discret_frechet_distance_"
	distance_file_list = []
	for i in range(1, 61):
		distance_file_list.append(distance_file_Prefix + str(i * 250))
	
	for idx, distance_file in tqdm(enumerate(distance_file_list)):
		if idx == 0:
			distance = pickle.load(open(distance_file, "rb"))
		else:
			distance = np.vstack((distance, pickle.load(open(distance_file, "rb"))))
	
	pickle.dump(distance, open("shanghai_distance.pkl", "wb"))
	np.save('shanghai_distance.npy', distance)