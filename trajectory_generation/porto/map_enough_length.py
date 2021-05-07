import pickle
from tqdm import tqdm

grid_dict = pickle.load(open("porto_token_dict.pkl", "rb"))
grid_dataset = pickle.load(open("porto_grid_time_dataset.pkl", "rb"))
grid_lat_lon_dataset = pickle.load(open("porto_grid_lat_lon_time_dataset.pkl", "rb"))

word_frequency = {}
for grid in list(grid_dict.keys()):
	word_frequency[grid] = 0

for trajectory in tqdm(grid_dataset):
	for point_grid in trajectory[0]:
		word_frequency[point_grid] += 1

pickle.dump(word_frequency, open("porto_token_frequency.pkl", "wb"))

new_grid_dataset = []
new_grid_lat_lon_dataset = []

for idx, trajectory in enumerate(grid_dataset):
	op = True
	for point_grid in trajectory[0]:
		if word_frequency[point_grid] < 50:
			op = False
	if op:
		new_grid_dataset.append(trajectory)
		new_grid_lat_lon_dataset.append(grid_lat_lon_dataset[idx])

print(len(new_grid_dataset))
pickle.dump(new_grid_dataset, open("porto_grid_time_enough_dataset.pkl", "wb"))
pickle.dump(new_grid_lat_lon_dataset, open("porto_grid_lat_lon_time_enough_dataset.pkl", "wb"))