import pickle
from tqdm import tqdm

porto_range = {
	"lon_min": -8.735152,
	"lat_min": 40.953673,
	"lon_max": -8.156309,
	"lat_max": 41.307945
}

if __name__ == "__main__":
	porto_inrange_dataset = []
	porto_raw_dataset = pickle.load(open("porto_raw_dataset.pkl", "rb")) # lon lat
	for line in tqdm(porto_raw_dataset, "map raw porto dataset to in range dataset"):
		out_range = False
		for point in line:
			if point[0] < porto_range['lon_min'] or point[0] > porto_range['lon_max'] or point[1] < porto_range['lat_min'] or point[1] > porto_range['lat_max']:
				out_range = True
				break
		if not out_range:
			porto_inrange_dataset.append(line)
	pickle.dump(porto_inrange_dataset, open("porto_in_range_dataset.pkl", "wb"))