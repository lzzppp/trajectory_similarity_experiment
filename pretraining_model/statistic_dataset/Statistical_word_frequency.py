import pickle
import matplotlib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# matplotlib.use('agg')

if __name__ == "__main__":
	word_frequency = {}
	# process_trajectory_dataset = pickle.load(open("trajectory_process_grid_dataset.pkl", "rb"))
	# for line in tqdm(process_trajectory_dataset):
	# 	for point in line[0]:
	# 		if point not in word_frequency:
	# 			word_frequency[point] = 1
	# 		else:
	# 			word_frequency[point] += 1
	# pickle.dump(word_frequency, open("word_frequency_dict.pkl", "wb"))
	word_frequency = pickle.load(open("word_frequency_dict.pkl", "rb"))
	word_frequency_list = []
	for key in word_frequency:
		word_frequency_list.append(word_frequency[key])
	# sorted(word_frequency_list)
	# print(np.argmax(np.array(word_frequency_list)))
	# print(word_frequency_list[np.argmax(np.array(word_frequency_list))])
	# print(np.argmin(np.array(word_frequency_list)))
	# print(word_frequency_list[np.argmin(np.array(word_frequency_list))])
	print(word_frequency_list[1825])
	plt.plot(list(range(len(word_frequency_list))), word_frequency_list)
	# plt.savefig("./word_frequency.jpg")
	plt.show()