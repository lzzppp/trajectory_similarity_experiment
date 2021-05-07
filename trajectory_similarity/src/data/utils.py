
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# max_length=510
#
# data = pickle.load(open("trajectory_new_new_grid_dataset.pkl", "rb"))
#
# lengths = []
# lengths_show = []
# token_dict = {}
# new_data = []
# lengths_dict = {}
# for i in range(1, 511):
#     lengths_dict[i] = 0
# for line in tqdm(data):
#     new_data.append([line[0][:250], line[1][:250]])
    # print(len(line[0]))
    # lengths.append(len(line[0]))
    # lengths_dict[len(line[0])] += 1
# for i in range(1, 511):
#     lengths_show.append(lengths_dict[i])

# print("mean", np.mean(np.array(lengths)))
# print("median", np.median(np.array(lengths)))
#
# print(sum(lengths_show[250:]) / 1172843)
#
# plt.plot(list(range(len(lengths_show))), lengths_show)
# plt.show()
# pickle.dump(new_data, open("trajectory_process_grid_dataset.pkl", "wb"))

token_dict = {}
time_dict = {}
data = pickle.load(open("trajectory_process_grid_dataset.pkl", "rb"))
for line in data:
	for token in line[0]:
		if token not in token_dict:
			token_dict[token] = len(token_dict) + 3

print(len(token_dict))