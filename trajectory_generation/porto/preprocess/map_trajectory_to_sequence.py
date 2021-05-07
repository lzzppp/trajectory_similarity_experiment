
import os
import kdtree
import pickle
from tqdm import tqdm
import multiprocessing

def bulid_kd_tree(dataset, k=5):
	trajectory_sequences = []
	for grid_trajectory in dataset:
		trajectory = grid_trajectory[1]
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
		trajectory_sequences.append(trajectory_sequence_k)
	kd_tree = kdtree.KDTree(trajectory_sequences, list(range(len(trajectory_sequences))))
	return kd_tree, trajectory_sequences

def run_proc(kd, sequences, s, k_trajs):
	print(s)
	idx = s
	index_lists = []
	for sequence in tqdm(sequences):
		k_d_tree_list = kd.search_knn(sequence, k_trajs + 1)[1:]
		index_list = [k_d_node[0].label for k_d_node in k_d_tree_list]
		assert idx not in index_list
		index_lists.append(index_list)
		idx += 1
	pickle.dump(index_lists, open("kdtree_index_list_" + str(s) + ".pkl", "wb"))

if __name__ == "__main__":
	batch_id = 0
	k_traj = 5
	trajectory_dataset = pickle.load(open("porto_trajectory.pkl", "rb"))
	KD, Sequences = bulid_kd_tree(trajectory_dataset)
	del trajectory_dataset
	batch_size = 10000
	
	run_proc(KD, Sequences[batch_id*batch_size:(batch_id+1)*batch_size], batch_id, k_trajs=5)