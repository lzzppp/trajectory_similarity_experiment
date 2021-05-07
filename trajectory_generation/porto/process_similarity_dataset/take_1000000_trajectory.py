import pickle
import random
import kdtree

def bulid_kd_tree(dataset, k=5):
	trajectory_sequences = []
	for trajectory in dataset:
		trajectory_sequence_k = []
		seq_k = int((len(trajectory) * 1.0) / k)
		for i in range(k-1):
			tree_point = [0.0, 0.0]
			for p in trajectory[i * seq_k:(i + 1) * seq_k]:
				tree_point[0] += p[0]
				tree_point[1] += p[1]
			trajectory_sequence_k.extend([tree_point[0] / (seq_k * 1.0), tree_point[1] / (seq_k * 1.0)])
		tree_point = [0, 0]
		for p in trajectory[(k-1) * seq_k:]:
			tree_point[0] += p[0]
			tree_point[1] += p[1]
		trajectory_sequence_k.extend(
			[tree_point[0] / (len (trajectory[(k-1) * seq_k:]) * 1.0), tree_point[1] / (len (trajectory[(k-1) * seq_k:]) * 1.0)])
		# print(len(traj_sequence_k))
		# print(trajectory_sequence_k)
		trajectory_sequences.append(trajectory_sequence_k)
	kd_tree_entity = kdtree.KDTree(trajectory_sequences, list(range(len(trajectory_sequences))))
	return kd_tree_entity, trajectory_sequences

raw_Dataset = pickle.load(open("porto_trajectory.pkl", "rb"))
raw_grid_Dataset = pickle.load(open("tert_dataset/porto_grid_lat_lon_time_enough_dataset.pkl", "rb"))

train_size = int(0.6*len(raw_Dataset))
valid_size = int(0.2*len(raw_Dataset))
test_size = len(raw_Dataset) - train_size - valid_size

index_list = list(range(len(raw_Dataset)))
random.shuffle(index_list)

shuffle_dataset = [raw_Dataset[i] for i in index_list]
shuffle_grid_dataset = [raw_grid_Dataset[k] for k in index_list]

train_dataset = shuffle_dataset[:train_size]
train_grid_dataset = shuffle_grid_dataset[:train_size]

valid_dataset = shuffle_dataset[train_size:train_size+valid_size]
valid_grid_dataset = shuffle_grid_dataset[train_size:train_size+valid_size]

test_dataset = shuffle_dataset[train_size+valid_size:]
test_grid_dataset = shuffle_grid_dataset[train_size+valid_size:]

pickle.dump(train_dataset, open("/mnt/data4/lizepeng/porto/bert_similarity/porto_train_trajectory.pkl", "wb"))
pickle.dump(train_grid_dataset, open("/mnt/data4/lizepeng/porto/bert_similarity/porto_train_grid.pkl", "wb"))
pickle.dump(valid_dataset, open("/mnt/data4/lizepeng/porto/bert_similarity/porto_valid_trajectory.pkl", "wb"))
pickle.dump(valid_grid_dataset, open("/mnt/data4/lizepeng/porto/bert_similarity/porto_valid_grid.pkl", "wb"))
pickle.dump(test_dataset, open("/mnt/data4/lizepeng/porto/bert_similarity/porto_test_trajectory.pkl", "wb"))
pickle.dump(test_grid_dataset, open("/mnt/data4/lizepeng/porto/bert_similarity/porto_test_grid.pkl", "wb"))

print("have done split task!")

kd, train_sequence_dataset = bulid_kd_tree(train_dataset)

pickle.dump(train_sequence_dataset, open("/mnt/data4/lizepeng/porto/bert_similarity/porto_train_sequence.pkl", "wb"))
pickle.dump(kd, open("/mnt/data4/lizepeng/porto/bert_similarity/porto_train_kd_tree.pkl", "wb"))