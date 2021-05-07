
import re
import sys
import time
import math
import heapq
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from logging import getLogger
from collections import OrderedDict
from src.utils import get_optimizer, update_lambdas
from src.dataset import myDataset
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
# from apex.fp16_utils import FP16_Optimizer
from SSM import FrechetDistanceLoop, FrechetDistanceRecursive, DynamicTimeWarpingLoop
from torch.optim import Adam

logger = getLogger()
mail_pre_degree = 16.0
# alpha_input = 0.4417847609617142
# alpha_input = 65.34532984876961

class ProgressBar(object):
    DEFAULT='Progress: %(bar)s %(percent)3d%%'
    FULL='%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total,width = 40,fmt = DEFAULT,symbol = '=',
                 output=sys.stderr):
        assert len(symbol) == 1
        self.total=total
        self.width=width
        self.symbol=symbol
        self.output=output
        self.fmt=re.sub(r'(?P<name>%\(.+?\))d', r'\g<name>%dd' % len(str(total)), fmt)
        self.current=0

    def __call__(self):
        percent=self.current/float(self.total)
        size=int(self.width*percent)
        remaining=self.total-self.current
        bar='[' + self.symbol * size + ' ' * (self.width-size) + ']'

        args={
            'total':self.total,
            'bar':bar,
            'current':self.current,
            'percent':percent*100,
            'remaining':remaining}
        
        print ('\r'+self.fmt%args,file=self.output,end = '')

    def done(self):
        self.current=self.total
        self()
        print('', file=self.output)

class TranSimiTrainer(object):

    def __init__(self, model, loss_func, params, dataset, kd_tree_index, traj_sequences, traj_features, batch_size=16, test_batch_size=128, k_traj=5, test_trajectory_prob=0.1):
        self.index = 0
        self.loss_func = loss_func
        self.model = model
        self.data = dataset
        self.k_traj = k_traj
        self.params = params
        self.traj_features = traj_features
        self.kd_tree_index = kd_tree_index
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.data_length = len(dataset)
        self.traj_sequences = traj_sequences
        self.test_trajectory_num = test_trajectory_prob
        self.n_sentences = 0
        self.stats = {
            "top10@acc": 0.0,
            "top50@acc": 0.0,
            "top10@50acc": 0.0
        }

        self.optimizer = Adam(self.model.parameters(), 0.001)

    def get_nearest_index(self, anch_index):
        k_d_tree_list = self.kd_tree_index[anch_index]
        nearest_choice_index = int(random.choice(k_d_tree_list))
        return nearest_choice_index
    
    def get_farest_index(self):
        data_index_list = list(range(self.data_length))
        farest_choice_index = random.choice(data_index_list)
        return farest_choice_index

    def get_attention_mask(self, length1, length2, max_length_of_1_and_2):
        attn_mask = [[float(0.0)]*ai + [float('-inf')]*(length1 - ai) + [float(0.0)]*length2 + [float('-inf')]*(max_length_of_1_and_2 - length1 - length2) for ai in
                     range(1, length1 + 1)]
        attn_mask += [[float(0.0)]*length1 + [float(0.0)]*ni + [float('-inf')]*(length2 - ni) + [float('-inf')]*(max_length_of_1_and_2 - length1 - length2) for ni in
                      range(1, length2 + 1)]
        attn_mask += [[float('-inf')]*max_length_of_1_and_2 for _ in range(max_length_of_1_and_2 - length1 - length2)]
        return attn_mask
    
    def get_pair_indexs(self, leng1, leng2, leng1_max):
        pair_index = [[], []]
        for i in range(leng1):
            for j in range(leng2):
                pair_index[0].append(i)
                pair_index[1].append(j+leng1_max)
        return pair_index
    
    def data_generation(self):
        self.index = 0
        batch_size = self.batch_size
        while self.index < self.data_length:
            anchor_length, far_length, near_length = [], [], []
            near_mask_padding, far_mask_padding = [], []
            anchor_trajectorys, far_trajectorys, near_trajectorys = [], [], []
            anchor_near_distance, anchor_far_distance = [], []
            if self.index + batch_size > self.data_length:
                batch_size = self.data_length - self.index
            for anchor_index in range(self.index, self.index+batch_size):
                anchor_trajectory = self.data[anchor_index][1][:125]
                anchor_length.append(len(anchor_trajectory))
                near_index = self.get_nearest_index(anchor_index)
                far_index = self.get_farest_index()
                near_trajectory = self.data[near_index][1][:125]
                near_length.append(len(near_trajectory))
                far_trajectory = self.data[far_index][1][:125]
                far_length.append(len(far_trajectory))
                anchor_trajectorys.append(anchor_trajectory)
                near_trajectorys.append(near_trajectory)
                far_trajectorys.append(far_trajectory)
                near_distance_matrix, nearest_path = FrechetDistanceLoop.FrechetDistance(np.array(anchor_trajectory),
                                                                                         np.array(near_trajectory))
                far_distance_matrix, farest_path = FrechetDistanceLoop.FrechetDistance(np.array(anchor_trajectory),
                                                                                       np.array(far_trajectory))
                near_distance_matrix = np.exp(-near_distance_matrix)*mail_pre_degree
                far_distance_matrix = np.exp(-far_distance_matrix) * mail_pre_degree
                if anchor_index == self.index:
                    anchor_near_distance = near_distance_matrix.reshape(len(anchor_trajectory) * len(near_trajectory))
                    anchor_far_distance = far_distance_matrix.reshape(len(anchor_trajectory) * len(far_trajectory))
                else:
                    anchor_near_distance = np.hstack((anchor_near_distance, near_distance_matrix.reshape(len(anchor_trajectory) * len(near_trajectory))))
                    anchor_far_distance = np.hstack((anchor_far_distance, far_distance_matrix.reshape(len(anchor_trajectory) * len(far_trajectory))))
            
            anchor_length_max = max(anchor_length)
            near_length_max = max(near_length)
            far_length_max = max(far_length)

            anchor_near_trajectory_pair, anchor_far_trajectory_pair = [], []
            anchor_near_indexs, anchor_far_indexs = [], []
            
            ind = 0
            for alength, nlength, flength in zip(anchor_length, near_length, far_length):
                anchor_near_indexs.append(self.get_pair_indexs(alength, nlength, anchor_length_max))
                anchor_far_indexs.append(self.get_pair_indexs(alength, flength, anchor_length_max))
                near_mask_padding.append([False]*alength + [True]*(anchor_length_max - alength) + [False]*nlength + [True]*(near_length_max - nlength))
                far_mask_padding.append([False]*alength + [True]*(anchor_length_max - alength) + [False]*flength + [True]*(far_length_max - flength))
                anchor_near_trajectory_pair.append([[(atp[0] - self.traj_features[0]) / self.traj_features[2],
                                                     (atp[1] - self.traj_features[1]) / self.traj_features[3]] for atp in
                                                      anchor_trajectorys[ind]] + [[0.0, 0.0] for anii in range(anchor_length_max - alength)] +
                                                   [[(ntp[0] - self.traj_features[0]) / self.traj_features[2],
                                                     (ntp[1] - self.traj_features[1]) / self.traj_features[3]] for ntp in
                                                      near_trajectorys[ind]] + [[0.0, 0.0] for nnii in range(near_length_max - nlength)])
                anchor_far_trajectory_pair.append ([[(atp[0] - self.traj_features[0]) / self.traj_features[2],
                                                      (atp[1] - self.traj_features[1]) / self.traj_features[3]] for atp
                                                     in anchor_trajectorys[ind]] + [[0.0, 0.0] for afii in range(anchor_length_max - alength)] +
                                                    [[(ftp[0] - self.traj_features[0]) / self.traj_features[2],
                                                      (ftp[1] - self.traj_features[1]) / self.traj_features[3]] for ftp in
                                                     far_trajectorys[ind]] + [[0.0, 0.0] for ffii in range(far_length_max - flength)])
                ind += 1
            near_mask_attention = [[float(0.0)]*len1 + [float('-inf')]*(anchor_length_max - len1) + [float(0.0)]*near_length_max for len1 in range(1, anchor_length_max+1)] +\
                                  [[float(0.0)]*anchor_length_max + [float(0.0)]*len2 + [float('-inf')]*(near_length_max - len2) for len2 in range(1, near_length_max+1)]
            far_mask_attention = [[float(0.0)]*len3 + [float('-inf')]*(anchor_length_max - len3) + [float(0.0)]*far_length_max for len3 in range(1, anchor_length_max+1)] +\
                                 [[float(0.0)]*anchor_length_max + [float(0.0)]*len4 + [float('-inf')]*(far_length_max - len4) for len4 in range(1, far_length_max+1)]
            self.index += batch_size
            yield anchor_near_trajectory_pair, anchor_far_trajectory_pair, \
                  near_mask_attention, far_mask_attention, \
                  near_mask_padding, far_mask_padding, \
                  anchor_near_distance, anchor_far_distance, \
                  anchor_near_indexs, anchor_far_indexs
    
    def train(self, epoch):
        self.index = 0
        self.model.train()

        progress = ProgressBar(len(self.data) // self.batch_size, fmt=ProgressBar.FULL)
        tloss = 0.0
        loss_list = []
        
        for batch in self.data_generation():
            anchor_near_pair = torch.FloatTensor(batch[0]).cuda()
            anchor_far_pair = torch.FloatTensor(batch[1]).cuda()
            anchor_near_attention_mask = torch.FloatTensor(batch[2]).cuda()
            anchor_far_attention_mask = torch.FloatTensor(batch[3]).cuda()
            anchor_near_padding_mask = torch.BoolTensor(batch[4]).cuda()
            anchor_far_padding_mask = torch.BoolTensor(batch[5]).cuda()
            anchor_near_distance = torch.FloatTensor(batch[6]).cuda()
            anchor_far_distance = torch.FloatTensor(batch[7]).cuda()
            indexs1, indexs2, indexs3, indexs4 = [p[0] for p in batch[8]], [p[0] for p in batch[9]], [p[1] for p in batch[8]], [p[1] for p in batch[9]]
            anchor_near_out, anchor_far_out = self.model(anchor_near_pair, anchor_far_pair,
                                                         anchor_near_attention_mask, anchor_far_attention_mask,
                                                         anchor_near_padding_mask, anchor_far_padding_mask)
            loss = self.loss_func(anchor_near_out, anchor_far_out,
                                  indexs1, indexs2, indexs3, indexs4,
                                  anchor_near_distance, anchor_far_distance)
            progress.current += 1
            progress()
            self.optimizer.zero_grad ()
            loss.backward ()
            self.optimizer.step ()
            tloss += loss.cpu ().detach ().numpy ()

            if int (self.n_sentences / self.batch_size) % 10 == 0 and self.n_sentences > 0:
                logger.info (tloss)
                loss_list.append (tloss)
                tloss = 0

            self.n_sentences += self.batch_size