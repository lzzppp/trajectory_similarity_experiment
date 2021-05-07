import torch
import heapq
import pickle
import numpy as np
from tqdm import tqdm

embedding_matrix = np.load("/mnt/data4/lizepeng/porto/bert_similarity/porto_test_trajectory_embedding.npy")
# distance_matrix = []

for i in tqdm(range(1,81)):
	if i == 1:
		distance_matrix = pickle.load(open("/home/lizepeng/test_similarity_model/features/porto_discret_frechet_distance_" + str(i*25), "rb"))
	else:
		distance_matrix = np.vstack((distance_matrix,
		                             pickle.load(open("/home/lizepeng/test_similarity_model/features/porto_discret_frechet_distance_" + str(i*25), "rb"))))

# distance_matrix.shape[0]

with torch.no_grad():
	all_num=0
	for i in tqdm(range(distance_matrix.shape[0])):
		distance = list(distance_matrix[i,:])
		anchor_embedding = np.expand_dims(embedding_matrix[i,:],0).repeat(distance_matrix.shape[1],axis=0)
		# torch.exp (-torch.norm (traj_anchor_sub_embedding - anchor_sub_embedding, p=2, dim=-1))
		predict_distance = list(np.linalg.norm(anchor_embedding - embedding_matrix, ord=2, axis=1))
		pred_topk = heapq.nsmallest(50, range(len(predict_distance)), predict_distance.__getitem__)
		dist_topk = heapq.nsmallest(50, range(len(distance)), distance.__getitem__)
		right_list = [l for l in pred_topk if l in dist_topk]
		all_num += len(right_list)
print(all_num)