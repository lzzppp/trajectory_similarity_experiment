import pickle

max_length=510

data = pickle.load(open("trajectory_new_grid_dataset.pkl", "rb"))

new_data = []
for line in data:
    new_data.append([line[0][:510], line[1][:510]])

pickle.dump(new_data, open("trajectory_new_new_grid_dataset.pkl", "wb"))