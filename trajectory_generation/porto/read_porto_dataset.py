
import pickle
from tqdm import tqdm
from geopy.distance import geodesic

if __name__ == "__main__":
	read_dataset = pickle.load(open("/mnt/data4/lizepeng/porto/porto_grid_time_dataset.pkl", "rb"))
	for line in read_dataset:
		print(line)