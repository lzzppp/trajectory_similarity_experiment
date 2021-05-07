import re
import sys
import time
import math
import torch
import heapq
import pickle
import random
import numpy as np
from h52pic import *
from tqdm import tqdm

if __name__ == "__main__":
    pred_similarity_matrix = pickle.load(open("similarity_matrix.pkl", "rb"))
    # print(pred_similarity_matrix)
    print(len(pred_similarity_matrix))
    print(len(pred_similarity_matrix[0]))
    num_10 = 0
    num_50 = 0
    for index in tqdm(range(1, 41)):
        similarity_matrix = pickle.load(open ("features/porto_discret_frechet_distance_" + str (index * 25), "rb"))
        for j in range(similarity_matrix.shape[0]):
            distance = list(similarity_matrix[j, :]*16.0)
            predict_distance = pred_similarity_matrix[(index - 1) * 25 + j]
            pred_topk_10 = heapq.nsmallest(11, range(len(predict_distance)), predict_distance.__getitem__)[1:]
            dist_topk_10 = heapq.nsmallest(11, range(len(distance)), distance.__getitem__)[1:]
            for predi in pred_topk_10:
                if predi in dist_topk_10:
                    num_10 += 1
            pred_topk_50 = heapq.nsmallest(51, range(len(predict_distance)), predict_distance.__getitem__)[1:]
            dist_topk_50 = heapq.nsmallest(51, range(len(distance)), distance.__getitem__)[1:]
            for predi in pred_topk_50:
                if predi in dist_topk_50:
                    num_50 += 1
    print(num_10/10000)
    print(num_50/50000)