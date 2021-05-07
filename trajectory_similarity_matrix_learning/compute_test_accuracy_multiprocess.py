import torch
import heapq
import pickle
import numpy as np
from tqdm import tqdm
import multiprocessing

def run_proc(index, embed_matrix):
    print("process " + str(index) + " have started !")
    distance_matrix = pickle.load(open("features/porto_discret_frechet_distance_" + str(index * 25), "rb"))
    num_10 = 0
    num_50 = 0
    for j in range(distance_matrix.shape[0]):
        distance = list(distance_matrix[j, :])
        anchor_embedding = np.expand_dims(embed_matrix[(index - 1) * 25 + j, :], 0).repeat(distance_matrix.shape[1], axis=0)
        predict_distance = list(np.linalg.norm(anchor_embedding - embed_matrix, ord=2, axis=1))
        pred_topk_10 = heapq.nsmallest(11, range(len(predict_distance)), predict_distance.__getitem__)[1:]
        dist_topk_10 = heapq.nsmallest(11, range(len(distance)), distance.__getitem__)[1:]
        for predi in pred_topk_10:
            if predi in dist_topk_10:
                num_10 += 1
        pred_topk_50 = heapq.nsmallest(51, range (len (predict_distance)), predict_distance.__getitem__)[1:]
        dist_topk_50 = heapq.nsmallest(51, range (len (distance)), distance.__getitem__)[1:]
        for predi in pred_topk_50:
            if predi in dist_topk_50:
                num_50 += 1
    pickle.dump([num_10, num_50], open("accuracy_result/"+str(index)+".pkl", "wb"))
    print("process " + str (index) + " have done !")
    
embedding_matrix = np.load("porto_test_trajectory_embedding_traj2sim.npy")

pool = multiprocessing.Pool(processes=40)
with torch.no_grad():
    for iter_num in tqdm(range(1,401)):
        pool.apply_async(run_proc, (iter_num, embedding_matrix))
    pool.close()
    pool.join()

all_num_10 = 0
all_num_50 = 0

for i in tqdm(range(1, 401)):
    num_list = pickle.load(open("accuracy_result/"+str(i)+".pkl", "rb"))
    all_num_10 += num_list[0]
    all_num_50 += num_list[1]

print((all_num_10 * 1.0)/ 100000.0)
print((all_num_50 * 1.0)/500000.0)
