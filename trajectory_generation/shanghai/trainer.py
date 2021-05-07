# -*- coding:utf-8 -*-
import math
import torch
import kdtree
import random
# import cPickle
import pickle
import numpy as np
# from dtw import dtw
# import torch.nn as nn
from tqdm import tqdm
from random import shuffle
import torch.nn.functional as F
# from traj_dist import distance as tdist
from SSM import FrechetDistanceLoop, FrechetDistanceRecursive, DynamicTimeWarpingLoop
from model_example2 import Traj2Sim_encoder
from traj2sim_loss import Traj2SimLoss

mail_pre_degree = 16.0
# alpha_input = 0.4417847609617142
alpha_input = 65.34532984876961

def get_topk_recall(pred, targ, k):
    # pred_similarity_k = heapq.nsmallest(k, range(len(a)), pred.take)
    top_k_preds = np.argsort(-pred)[:, ::-1][:, :k]
    top_k_idxs = np.argsort(-targ)[:, ::-1][:, :k]
    match_num = 0
    for top_k_idx, top_k_pred in zip(top_k_idxs, top_k_preds):
        match_num += len([x for x in top_k_pred if x in top_k_idx])
    return (match_num * 1.0) / (k * 1000.0)

def get_topkin_topm_recall(pred, targ, k, m):
    top_k_preds = np.argsort(-pred)[:, ::-1][:, :k]
    top_k_idxs = np.argsort(-targ)[:, ::-1][:, :m]
    match_num = 0
    for top_k_idx, top_k_pred in zip(top_k_idxs, top_k_preds):
        match_num += len([x for x in top_k_idx if x in top_k_pred])
    return (match_num * 1.0) / (k * 1000.0)

def eucl_dist(x, y):
    """
    Usage
    -----
    L2-norm between point x and y

    Parameters
    ----------
    param x : numpy_array
    param y : numpy_array

    Returns
    -------
    dist : float
           L2-norm between x and y
    """
    dist = np.linalg.norm(x - y)
    return dist

def pad_sequence(traj_grids, maxlen=100, pad_value=0.0):
    paddec_seqs = []
    for traj in traj_grids:
        # pad_r = np.zeros_like (traj[0]) * pad_value
        pad_traj = traj + [[0.0, 0.0]] * (maxlen - len(traj))
        paddec_seqs.append(pad_traj)
    return paddec_seqs

def pad_sequence_grid(traj_grids, maxlen=100, pad_value=0.0):
    paddec_seqs = []
    for traj in traj_grids:
        # pad_r = np.zeros_like (traj[0]) * pad_value
        pad_traj = traj + [0] * (maxlen - len(traj))
        paddec_seqs.append(pad_traj)
    return paddec_seqs

def D_sampling_index(n1, n2):
    n1_list = random.sample(list(range(n1)), 10)
    n2_list = random.sample(list(range(n2)), 10)
    return [(n1n1, n2n2) for n1n1, n2n2 in zip(n1_list, n2_list)]

def E_sampling_index(n1, n3):
    n1_list = random.sample(list(range(n1)), 10)
    n3_list = random.sample(list(range(n3)), 10)
    return [(n1n1, n3n3) for n1n1, n3n3 in zip(n1_list, n3_list)]

def batch_generator(train_trajectorys, train_raw_trajectorys, train_trajectory_sequence_k, train_trajectory_sequence_grid, k_dtree, batch_size=32, metric_type="discret_frechet"):
    j = 0
    euclidean_norm = lambda x, y: np.linalg.norm(x - y)
    while j < len(train_trajectorys):
        anchor_input_nearest, anchor_input_farest, nearest_input, farest_input, D_subtrajectory_pair, E_subtrajectory_pair, nearest_distance, farest_distance, nearest_distance_all, farest_distance_all = [], [], [], [], [], [], [], [], [], []
        anchor_grid_nearest, anchor_grid_farest, nearest_grid, farest_grid = [], [], [], []
        anchor_nearest_length, anchor_farest_length, nearest_length, farest_length = [], [], [], []
        nearest_tuple_, farest_tuple_ = [], []
        trajectory_anchor, trajectory_nearest, trajectory_farest, trajectory_anchor_length, trajectory_nearest_length, trajectory_farest_length = [], [], [], [], [], []
        trajectory_anchor_grid, trajectory_nearest_grid, trajectory_farest_grid = [], [], []
        if j + batch_size > len(train_trajectorys):
            batch_size = len(train_trajectorys) - j
        for i in range(batch_size):
            train_index = i + j
            trajectory = train_trajectorys[train_index]
            raw_trajectory = train_raw_trajectorys[train_index]
            trajectory_grid = train_trajectory_sequence_grid[train_index]
            
            trajectory_anchor.append(trajectory)
            trajectory_anchor_grid.append(trajectory_grid)
            trajectory_anchor_length.append(len(trajectory))
            
            k_d_tree_list = k_dtree.search_knn(train_trajectory_sequence_k[i + j], 5)
            nearest_index = int(random.choice(k_d_tree_list)[0].label)
            while nearest_index == i + j:
                nearest_index = int(random.choice(k_d_tree_list)[0].label)
            trajectory_nearest.append(train_trajectorys[nearest_index])
            trajectory_nearest_grid.append(train_trajectory_sequence_grid[nearest_index])
            trajectory_nearest_length.append(len(train_trajectorys[nearest_index]))
            
            farest_index = random.choice(list(range(len(train_trajectorys))))
            while farest_index == i + j or farest_index == nearest_index:
                farest_index = random.choice(list(range(len(train_trajectorys))))
            trajectory_farest.append(train_trajectorys[farest_index])
            trajectory_farest_grid.append(train_trajectory_sequence_grid[farest_index])
            trajectory_farest_length.append(len(train_trajectorys[farest_index]))
            
            # print(i + j, nearest_index, farest_index)
            # nearest_d, nearest_cost_matrix, nearest_acc_cost_matrix, nearest_path = dtw(np.array(trajectory), np.array(train_trajectorys[nearest_index]), dist=euclidean_norm)
            # farest_d, farest_cost_matrix, farest_acc_cost_matrix, farest_path = dtw(np.array(trajectory), np.array(train_trajectorys[farest_index]), dist=euclidean_norm)
            
            # near_distance_matrix, nearest_path = FrechetDistanceLoop.FrechetDistance(np.array(raw_trajectory), np.array(train_raw_trajectorys[nearest_index]))
            # far_distance_matrix, farest_path = FrechetDistanceLoop.FrechetDistance(np.array(raw_trajectory), np.array(train_raw_trajectorys[farest_index]))
            
            near_distance_matrix, nearest_path = DynamicTimeWarpingLoop.DynamicTimeWarping(np.array(raw_trajectory), np.array(train_raw_trajectorys[nearest_index]))
            far_distance_matrix, farest_path = DynamicTimeWarpingLoop.DynamicTimeWarping(np.array(raw_trajectory), np.array(train_raw_trajectorys[farest_index]))
            
            anchor_choose_point_near = random.sample(list(nearest_path[0]), 10)
            nearest_choose_match = [nearest_path[1][pp] for pp in anchor_choose_point_near]
            nearest_choose_unmatch = []
            
            for ind in anchor_choose_point_near:
                new_index_list = list(range(len(train_trajectorys[nearest_index])))
                new_index_list.pop(nearest_path[1][ind])
                nearest_choose_unmatch.append(random.choice(new_index_list))

            anchor_choose_point_far = random.sample(list(farest_path[0]), 10)
            farest_choose_match = [farest_path[1][ppp] for ppp in anchor_choose_point_far]
            farest_choose_unmatch = []

            for ind in anchor_choose_point_far:
                new_index_list = list(range(len(train_trajectorys[farest_index])))
                new_index_list.pop(farest_path[1][ind])
                farest_choose_unmatch.append(random.choice(new_index_list))
            
            nearest_tuple_.append([[pppp, qm, qum] for pppp, qm, qum in zip(anchor_choose_point_near, nearest_choose_match, nearest_choose_unmatch)])
            farest_tuple_.append([[pppp, qm, qum] for pppp, qm, qum in zip(anchor_choose_point_far, farest_choose_match, farest_choose_unmatch)])
            
            # nearest_distance_all.append(tdist.cdist([np.array(trajectory)], [np.array(train_trajectorys[nearest_index])], metric=metric_type)[0])
            # farest_distance_all.append(tdist.cdist([np.array(trajectory)], [np.array(train_trajectorys[farest_index])], metric=metric_type)[0])
            
            nearest_distance_all.append(math.exp(-float(near_distance_matrix[len(trajectory) - 1][len(train_trajectorys[nearest_index]) - 1] / alpha_input) * mail_pre_degree))
            farest_distance_all.append(math.exp(-float(far_distance_matrix[len(trajectory) - 1][len(train_trajectorys[farest_index]) - 1] / alpha_input) * mail_pre_degree))
            
            nearest_sampling = D_sampling_index(len(trajectory), len(train_trajectorys[nearest_index]))
            farest_sampling = E_sampling_index(len(trajectory), len(train_trajectorys[farest_index]))
            
            # print(nearest_sampling)
            # print(farest_sampling)
            
            for sub_trajectory_length, nearest_sub_trajectory_length in nearest_sampling:
                # anchor_input_nearest.append(trajectory[:sub_trajectory_length + 1])
                # anchor_grid_nearest.append(trajectory_grid[:sub_trajectory_length + 1])
                # nearest_input.append(train_trajectorys[nearest_index][:nearest_sub_trajectory_length + 1])
                # nearest_grid.append(train_trajectory_sequence_grid[nearest_index][:nearest_sub_trajectory_length + 1])
                anchor_nearest_length.append(sub_trajectory_length + 1)
                nearest_length.append(nearest_sub_trajectory_length + 1)
                nearest_distance.append(math.exp(-float(near_distance_matrix[sub_trajectory_length][nearest_sub_trajectory_length] / alpha_input) * mail_pre_degree))
                    
            for sub_trajectory_length, farest_sub_trajectory_length in farest_sampling:
                # anchor_input_farest.append(trajectory[:sub_trajectory_length + 1])
                # anchor_grid_farest.append(trajectory_grid[:sub_trajectory_length + 1])
                # farest_input.append(train_trajectorys[farest_index][:farest_sub_trajectory_length + 1])
                # farest_grid.append(train_trajectory_sequence_grid[farest_index][:farest_sub_trajectory_length + 1])
                anchor_farest_length.append(sub_trajectory_length + 1)
                farest_length.append(farest_sub_trajectory_length + 1)
                farest_distance.append(math.exp(-float(far_distance_matrix[sub_trajectory_length][farest_sub_trajectory_length] / alpha_input) * mail_pre_degree))
        
        # max_anchor_nearest_length = max(anchor_nearest_length)
        # max_anchor_farest_length = max (anchor_farest_length)
        # max_nearest_lenght = max(nearest_length)
        # max_farest_lenght = max(farest_length)
        max_trajectory_length = max(trajectory_anchor_length)
        max_trajectory_nearest_length = max(trajectory_nearest_length)
        max_trajectory_farest_length = max(trajectory_farest_length)
        # anchor_nearest_input = pad_sequence(anchor_input_nearest, maxlen=max_anchor_nearest_length)
        # anchor_farest_input = pad_sequence(anchor_input_farest, maxlen=max_anchor_farest_length)
        # nearest_input = pad_sequence(nearest_input, maxlen=max_nearest_lenght)
        # farest_input = pad_sequence(farest_input, maxlen=max_farest_lenght)
        trajectory_input = pad_sequence(trajectory_anchor, maxlen=max_trajectory_length)
        trajectory_nearest_input = pad_sequence(trajectory_nearest, maxlen=max_trajectory_nearest_length)
        trajectory_farest_input = pad_sequence(trajectory_farest, maxlen=max_trajectory_farest_length)
        # anchor_nearest_grid_input = pad_sequence_grid(anchor_grid_nearest, maxlen=max_anchor_nearest_length)
        # anchor_farest_grid_input = pad_sequence_grid(anchor_grid_farest, maxlen=max_anchor_farest_length)
        # nearest_grid_input = pad_sequence_grid(nearest_grid, maxlen=max_nearest_lenght)
        # farest_grid_input = pad_sequence_grid(farest_grid, maxlen=max_farest_lenght)
        trajectory_grid_input = pad_sequence_grid(trajectory_anchor_grid, maxlen=max_trajectory_length)
        trajectory_nearest_grid_input = pad_sequence_grid(trajectory_nearest_grid, maxlen=max_trajectory_nearest_length)
        trajectory_farest_grid_input = pad_sequence_grid(trajectory_farest_grid, maxlen=max_trajectory_farest_length)
        # alpha = max([max(nearest_distance), max(farest_distance), max(nearest_distance_all), max(farest_distance_all)])
        yield ([[np.array(trajectory_input), np.array(trajectory_grid_input)],
                [np.array(trajectory_nearest_input), np.array(trajectory_nearest_grid_input)],
                [np.array(trajectory_farest_input), np.array(trajectory_farest_grid_input)]],
               [anchor_nearest_length,
                anchor_farest_length,
                nearest_length,
                farest_length,
                trajectory_anchor_length,
                trajectory_nearest_length,
                trajectory_farest_length],
               [np.array(nearest_distance),
                np.array(farest_distance),
                np.array(nearest_distance_all),
                np.array(farest_distance_all),
                nearest_tuple_,
                farest_tuple_])
        j += batch_size

def get_test_batch(index_, DATASET, DATASET_GRID, test_batch_size, DATALENGTH):
    j = 0
    if j + test_batch_size >= DATALENGTH:
        test_batch_size = DATALENGTH - j
    while j < DATALENGTH:
        anchor_input, other_input, anchor_input_length, other_input_length = [], [], [], []
        anchor_grid_input, other_grid_input = [], []
        for ii in range(test_batch_size):
            anchor_input.append(DATASET[index_])
            anchor_grid_input.append(DATASET_GRID[index_])
            anchor_input_length.append(len(DATASET[index_]))
            other_input.append(DATASET[ii + j])
            other_grid_input.append(DATASET_GRID[ii + j])
            other_input_length.append(len(DATASET[ii + j]))
        anchor_length = max(anchor_input_length)
        other_length = max(other_input_length)
        anchor_input = pad_sequence(anchor_input, maxlen=anchor_length)
        other_input = pad_sequence(other_input, maxlen=other_length)
        anchor_grid_input = pad_sequence_grid(anchor_grid_input, maxlen=anchor_length)
        other_grid_input = pad_sequence_grid(other_grid_input, maxlen=other_length)
        # anchor_input_array = np.array(anchor_input)
        # other_input_array = np.array(other_input)
        yield ([anchor_input, anchor_grid_input],
               [other_input, other_grid_input],
               anchor_input_length,
               other_input_length)
        j += test_batch_size

def test_computing_method(self, all_dataset, all_grid_dataset, trainsize):
    test_pred_dist = []
    for idx in range(trainsize, trainsize + 1000):
        idx_tensor = []
        for test_batch in get_test_batch(idx, all_dataset, all_grid_dataset, 1000, trainsize + 1000):
            input1_ = torch.FloatTensor(test_batch[0][0]).cuda()
            input2_ = torch.FloatTensor(test_batch[1][0]).cuda()
            input1_grid = torch.LongTensor(test_batch[0][1]).cuda()
            input2_grid = torch.LongTensor(test_batch[1][1]).cuda()
            input1_length = test_batch[2]
            input2_length = test_batch[3]
            input1 = torch.cat((input1_, self.grid_embedding(input1_grid)), dim=-1)
            input2 = torch.cat((input2_, self.grid_embedding(input2_grid)), dim=-1)
            init_hidden_state_all = [torch.zeros([input1.shape[0], self.hidden_size]).cuda(),
                                     torch.zeros([input1.shape[0], self.hidden_size]).cuda()]
            # predict_distance_ = self.forward(input1, input2)
            vector1_raw = self.rnn_encoder([input1, input1_length], init_hidden_state_all)
            vector2_raw = self.rnn_encoder([input2, input2_length], init_hidden_state_all)
            vector1 = self.f_hid2sim(vector1_raw)
            vector2 = self.f_hid2sim(vector2_raw)
            idx_tensor.append(F.pairwise_distance(vector1, vector2, p=2))
        idx_tensor = torch.cat(idx_tensor)
        test_pred_dist.append(idx_tensor.unsqueeze(0))
    test_pred_dist = torch.cat(test_pred_dist)
    return test_pred_dist

# def computing_traj_embedding(self, ):

class Preprocesser(object):
    def __init__(self, delta = 0.005, lat_range = [1,2], lon_range = [1,2]):
        self.delta = delta
        self.lat_range = lat_range
        self.lon_range = lon_range
        self._init_grid_hash_function()

    def _init_grid_hash_function(self):
        dXMax, dXMin, dYMax, dYMin = self.lon_range[1], self.lon_range[0], self.lat_range[1], self.lat_range[0]
        x = self._frange(dXMin, dXMax, self.delta)
        y = self._frange(dYMin, dYMax, self.delta)
        self.x = x
        self.y = y

    def _frange(self, start, end=None, inc=None):
        "A range function, that does accept float increments..."
        if end == None:
            end = start + 0.0
            start = 0.0
        if inc == None:
            inc = 1.0
        L = []
        while 1:
            next = start + len(L) * inc
            if inc > 0 and next >= end:
                break
            elif inc < 0 and next <= end:
                break
            L.append(next)
        return L

    def get_grid_index(self, tuple):
        test_tuple = tuple
        test_x,test_y = test_tuple[0],test_tuple[1]
        x_grid = int ((test_x-self.lon_range[0])/self.delta)
        y_grid = int ((test_y-self.lat_range[0])/self.delta)
        index = (y_grid)*(len(self.x)) + x_grid
        return x_grid, y_grid, index

    def traj2grid_seq(self, trajs = [], isCoordinate = False):
        grid_traj = []
        for r in trajs:
            x_grid, y_grid, index = self.get_grid_index((r[2],r[1]))
            # print (r[1]+tx,r[0]+ty)
            # print x_grid, y_grid, index
            grid_traj.append(index)
        privious = None
        hash_traj = []
        for index, i in enumerate(grid_traj):
            if privious==None:
                privious = i
                if isCoordinate == False:
                    hash_traj.append(i)
                elif isCoordinate == True:
                    hash_traj.append(trajs[index][1:])
            else:
                if i==privious:
                    pass
                else:
                    if isCoordinate == False:
                        hash_traj.append(i)
                    elif isCoordinate == True:
                        hash_traj.append(trajs[index][1:])
                    privious = i
        return hash_traj

    def _traj2grid_preprocess(self, traj_feature_map, isCoordinate =False):
        trajs_hash = []
        trajs_keys = traj_feature_map.keys()
        for traj_key in trajs_keys:
            traj = traj_feature_map[traj_key]
            trajs_hash.append(self.traj2grid_seq(traj, isCoordinate))
        return trajs_hash

if __name__ == "__main__":
    sequence_k = 10
    train_size = 4000
    load_old_model_params = True
    # porto range
    # porto_lat_range = [40.953673, 41.307945]
    # porto_lon_range = [-8.735152, -8.156309]
    # geolife range
    
    distance_type = 'dtw'
    distances = np.load("distance_porto/porto_{}_distance_all_5000.npy".format(distance_type))
    traj_trajs = pickle.load(open("porto_traj.pkl", "rb"))
    traj_sequences = pickle.load(open("porto_norm.pkl", "rb"))
    traj_grid_sequences = pickle.load(open("porto_grid.pkl", "rb"))
    trajs = []
    for index in traj_sequences:
        # print(index)
        # trajs.append([point[1:] for point in traj_sequences[index]])
        trajs.append([[point[0], point[1]] for point in index])

    train_trajs = trajs[:train_size]
    train_raw_trajs = traj_trajs[:train_size]
    train_grids = traj_grid_sequences[:train_size]
    # trajs_sequence_k = cPickle.load(open("toy_traj_sequence_k", "rb"))[:train_size]
    trajs_sequence_k = []
    for traj in tqdm(train_raw_trajs):
        traj_sequence_k = []
        seq_k = int((len(traj) * 1.0) / 10.0)
        for i in range(9):
            tree_point = [0, 0]
            for p in traj[i*seq_k:(i+1)*seq_k]:
                tree_point[0] += p[0]
                tree_point[1] += p[1]
            traj_sequence_k.extend([tree_point[0]/(seq_k*1.0), tree_point[1]/(seq_k*1.0)])
        tree_point = [0, 0]
        for p in traj[9 * seq_k:]:
            tree_point[0] += p[0]
            tree_point[1] += p[1]
        traj_sequence_k.extend([tree_point[0]/(len(traj[9*seq_k:])*1.0), tree_point[1]/(len(traj[9*seq_k:])*1.0)])
        # print(len(traj_sequence_k))
        trajs_sequence_k.append(traj_sequence_k)
    kd = kdtree.KDTree (trajs_sequence_k, list (range (len (trajs_sequence_k))))
    model = Traj2Sim_encoder("LSTM", 130, 128).cuda()
    loss_func = Traj2SimLoss().cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    
    result = [-1000, -1000, -1000]
    # model.apply(weight_init)
    if load_old_model_params:
        model.load_state_dict(torch.load("traj2sim_model_params.pkl"))
        print("Load model succeed ! ! !")
    model.train()
    for epoch in tqdm(range(200)):
        tloss = 0.0
        # index_list = list(range(train_size))
        # shuffle(index_list)
        # train_trajs = [train_trajs[index_trajectory] for index_trajectory in index_list]
        # trajs_sequence_k = [trajs_sequence_k[index_trajectory_k] for index_trajectory_k in index_list]
        # train_raw_trajs = [train_raw_trajs[index_trajectory_raw_traj] for index_trajectory_raw_traj in index_list]
        # train_grids = [train_grids[index_trajectory_grid] for index_trajectory_grid in index_list]
        # kd = kdtree.KDTree(trajs_sequence_k, list(range(len(trajs_sequence_k))))
        for batch in batch_generator(train_trajs, train_raw_trajs, trajs_sequence_k, train_grids, kd):
            input_data_raw, input_length, target_data = batch[0], batch[1], batch[2]

            target_sub1, target_sub2, target_all1, target_all2 = torch.from_numpy(target_data[0]).float().cuda(), \
                                                                 torch.from_numpy(target_data[1]).float().cuda(), \
                                                                 torch.from_numpy(target_data[2]).float().cuda(), \
                                                                 torch.from_numpy(target_data[3]).float().cuda()
            nearest_tuple, farest_tuple = target_data[4], target_data[5]
            input_data = [torch.FloatTensor(nparray[0]).cuda() for nparray in input_data_raw]
            input_grid_data = [torch.LongTensor(nparray[1]).cuda() for nparray in input_data_raw]

            # [input_data[0], input_grid_data[0]],
            # [input_data[1], input_grid_data[1]],
            # [input_data[2], input_grid_data[2]],
            # [input_data[3], input_grid_data[3]],

            nt_pt_de, ft_pt_de, nt_pt_de_al, ft_pt_de_al, m_ar_vr, m_nt_vr, m_ft_vr = model([input_data[0], input_grid_data[0]],
                                                                                            [input_data[1], input_grid_data[1]],
                                                                                            [input_data[2], input_grid_data[2]],
                                                                                            input_length)

            loss = loss_func (nt_pt_de,
                              ft_pt_de,
                              nt_pt_de_al,
                              ft_pt_de_al,
                              m_ar_vr,
                              m_nt_vr,
                              m_ft_vr,
                              target_sub1,
                              target_sub2,
                              target_all1,
                              target_all2,
                              nearest_tuple,
                              farest_tuple,
                              alpha_input)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tloss += loss.item()
            # print(loss)
            del input_data, target_sub1, target_sub2, target_all1, target_all2, nt_pt_de, ft_pt_de, nt_pt_de_al, ft_pt_de_al, m_ar_vr, m_nt_vr, m_ft_vr, loss
            torch.cuda.empty_cache()
        #scheduler.step()
        if epoch != False:
            with torch.no_grad ():
                pred_dist = test_computing_method(model, trajs, traj_grid_sequences, train_size).cpu().numpy()
                target_dist = distances[train_size:]
                HR10_recall = get_topk_recall(pred_dist, target_dist, 10)
                HR50_recall = get_topk_recall(pred_dist, target_dist, 50)
                R10_50_recall = get_topkin_topm_recall (pred_dist, target_dist, 10, 50)
                if HR10_recall > result[0]:
                    result[0] = HR10_recall
                    result[1] = HR50_recall
                    result[2] = R10_50_recall
                print ('****************************')
                print ('HR10Recall : ', HR10_recall)
                print ('HR50Recall : ', HR50_recall)
                print ('10@50Recall : ', R10_50_recall)
                print ('****************************\n')
        # del input_data, target_sub1, target_sub2, target_all1, target_all2, nt_pt_de, ft_pt_de, nt_pt_de_al, ft_pt_de_al, m_ar_vr, m_nt_vr, m_ft_vr, loss
        # torch.cuda.empty_cache()
        print("Tloss: ", tloss)
        torch.save(model.state_dict(), 'traj2sim_model_params.pkl')
        print('lr: ', optimizer.state_dict()['param_groups'][0]['lr'])
    print('HR10Recall : ', result[0])
    print('HR50Recall : ', result[1])
    print('10@50Recall : ', result[2])
    pickle.dump(result, open(distance_type + "_t2vec_result.pkl", "wb"))
