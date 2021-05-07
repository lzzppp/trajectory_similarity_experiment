
import math
import torch
import random
import pickle
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from utils import ProgressBar
from logging import getLogger
from frechetdist import frdist
import torch.nn.functional as F
from SSM import FrechetDistanceLoop

logger = getLogger()
mail_pre_degree = 8.0
num_heads = 8

class RegressSimiTrainer(object):
    def __init__(self, model, loss_func, data, kd_index, features, batch_size=32):
        self.model = model
        self.loss_function = loss_func
        self.data = data
        self.data_index = kd_index
        self.features = features
        self.batch_size = batch_size
        
        self.optimizer = Adam(self.model.parameters (), 0.0001)
        
        self.index = 0
        self.n_sentences = 0
    
    def standard_trajectory_point(self, gps):
        gps_stand = [(gps[0]-self.features[0])/self.features[2],
                     (gps[1]-self.features[1])/self.features[3]]
        return gps_stand
    
    def get_index(self, anch_index, mode='near'):
        assert mode in ['near', 'far']
        if mode == 'near':
            k_d_tree_list = self.data_index[anch_index]
            nearest_choice_index = int(random.choice (k_d_tree_list))
            return nearest_choice_index
        elif mode == 'far':
            data_index_list = list(range(len(self.data)))
            farest_choice_index = random.choice(data_index_list)
            return farest_choice_index
    
    def get_triple_index(self, length1, length2, k=10):
        double_index = [[], []]
        length1_randoms = random.sample (list (range (0, length1 - 1)), k)
        length2_randoms = random.sample (list (range (0, length2 - 1)), k)
        for length1_random, length2_random in zip (length1_randoms, length2_randoms):
            double_index[0].append (length1_random)
            double_index[1].append (length2_random)
        double_index[0].append (length1_random - 1)
        double_index[1].append (length2_random - 1)
        return double_index
    
    def get_memory_mask(self, q_length_list, k_length_list, q_length_max, k_length_max):
        memory_mask_init = [[float (0.0) for _ in range (k_length_max)] for _ in range (q_length_max)]
        for q_index, v_index in zip (q_length_list, k_length_list):
            for k_ in range (k_length_max):
                if k_ > v_index:
                    memory_mask_init[q_index][k_] = float('-inf')
            # memory_mask_init[q_index][v_index] = float(0.0)
        memory_mask = [memory_mask_init for _ in range(num_heads)]
        return memory_mask
    
    def batch_genarator(self):
        self.index = 0
        batch_size = self.batch_size
        while self.index < len(self.data):
            if batch_size + self.index > len(self.data):
                batch_size = len(self.data) - self.index
            anchor_gps, near_gps, far_gps = [], [], []
            anchor_length, near_length, far_length = [], [], []
            anchor_near_length, anchor_far_length, near_anchor_length, far_anchor_length = [], [], [], []
            anchor_mask, near_mask, far_mask = [], [], []
            anchor_near_memory_mask, near_anchor_memory_mask, anchor_far_memory_mask, far_anchor_memory_mask = [], [], [], []
            anchor_position, near_position, far_position = [], [], []
            near_distance, far_distance = [], []
            for i in range(self.index, self.index + batch_size):
                anchor_gps.append(self.data[i])
                anchor_length.append(len(self.data[i]))
                near_index = self.get_index(i)
                near_gps.append(self.data[near_index])
                near_length.append(len(self.data[near_index]))
                far_index = self.get_index(i, mode='far')
                far_gps.append(self.data[far_index])
                far_length.append(len(self.data[far_index]))
                near_distance.append(math.exp(-FrechetDistanceLoop.FrechetDistance(np.array(anchor_gps[-1]), np.array(near_gps[-1]))[0][-1, -1] * mail_pre_degree))
                far_distance.append(math.exp(-FrechetDistanceLoop.FrechetDistance(np.array(anchor_gps[-1]), np.array(far_gps[-1]))[0][-1, -1] * mail_pre_degree))
                
            anchor_length_max = max(anchor_length)
            near_length_max = max(near_length)
            far_length_max = max(far_length)
            ind = 0
            for a_length, n_length, f_length in zip(anchor_length, near_length, far_length):
                anchor_mask.append([False]*a_length+[True]*(anchor_length_max-a_length))
                near_mask.append([False]*n_length+[True]*(near_length_max-n_length))
                far_mask.append([False]*f_length+[True]*(far_length_max-f_length))
                anchor_position.append(list(range(1, anchor_length_max+1)))
                near_position.append(list(range(1, near_length_max+1)))
                far_position.append(list(range(1, far_length_max+1)))
                anchor_gps[ind] = anchor_gps[ind] + [[0.0, 0.0] for ai in range(anchor_length_max - a_length)]
                near_gps[ind] = near_gps[ind] + [[0.0, 0.0] for ni in range(near_length_max - n_length)]
                far_gps[ind] = far_gps[ind] + [[0.0, 0.0] for fi in range(far_length_max - f_length)]
                anchor_near_double = self.get_triple_index(a_length, n_length)
                anchor_far_double = self.get_triple_index(a_length, f_length)
                anchor_near_memory_mask.extend(self.get_memory_mask(anchor_near_double[0], anchor_near_double[1], anchor_length_max, near_length_max))
                anchor_far_memory_mask.extend(self.get_memory_mask(anchor_far_double[0], anchor_far_double[1], anchor_length_max, far_length_max))
                near_anchor_memory_mask.extend(self.get_memory_mask(anchor_near_double[1], anchor_near_double[0], near_length_max, anchor_length_max))
                far_anchor_memory_mask.extend(self.get_memory_mask(anchor_far_double[1], anchor_far_double[0], far_length_max, anchor_length_max))
                anchor_near_length.extend(anchor_near_double[0])
                near_anchor_length.extend(anchor_near_double[1])
                anchor_far_length.extend(anchor_far_double[0])
                far_anchor_length.extend(anchor_far_double[1])
                # anchor_near_length.append(a_length - 1)
                # near_anchor_length.append(n_length - 1)
                # anchor_far_length.append(a_length - 1)
                # far_anchor_length.append(f_length - 1)
                ind += 1
            anchor_attention_mask = [[float(0.0)] * anchor_len + [float('-inf')]*(anchor_length_max - anchor_len) for anchor_len in range(1, anchor_length_max + 1)]
            near_attention_mask = [[float(0.0)] * near_len + [float('-inf')]*(near_length_max - near_len) for near_len in range(1, near_length_max + 1)]
            far_attention_mask = [[float(0.0)] * far_len + [float('-inf')]*(far_length_max - far_len) for far_len in range(1, far_length_max + 1)]
            
            self.index += batch_size
            yield anchor_gps, near_gps, far_gps, \
                  anchor_mask, near_mask, far_mask, \
                  anchor_position, near_position, far_position, \
                  anchor_attention_mask, near_attention_mask, far_attention_mask, \
                  anchor_length, near_length, far_length, \
                  near_distance, far_distance, \
                  anchor_near_memory_mask, near_anchor_memory_mask, anchor_far_memory_mask, far_anchor_memory_mask, \
                  anchor_near_length, near_anchor_length, anchor_far_length, far_anchor_length
        
    def train_step(self, epoch):
        self.model.train()
        
        progress = ProgressBar (len (self.data) // self.batch_size, fmt=ProgressBar.FULL)
        tloss = 0.0
        loss_list = []
        
        for batch in self.batch_genarator():
            anchor_gps_data, near_gps_data, far_gps_data = torch.FloatTensor(batch[0]).cuda().transpose(0, 1), \
                                                           torch.FloatTensor(batch[1]).cuda().transpose(0, 1), \
                                                           torch.FloatTensor(batch[2]).cuda().transpose(0, 1)
            anchor_mask_data, near_mask_data, far_mask_data = torch.BoolTensor(batch[3]).cuda(), \
                                                              torch.BoolTensor(batch[4]).cuda(), \
                                                              torch.BoolTensor(batch[5]).cuda()
            anchor_position_data, near_position_data, far_position_data = torch.LongTensor(batch[6]).cuda(), \
                                                                          torch.LongTensor(batch[7]).cuda(), \
                                                                          torch.LongTensor(batch[8]).cuda()
            anchor_attention_mask_data, near_attention_mask_data, far_attention_mask_data = torch.FloatTensor(batch[9]).cuda(), \
                                                                                            torch.FloatTensor(batch[10]).cuda(), \
                                                                                            torch.FloatTensor(batch[11]).cuda()
            # anchor_length_data, near_length_data, far_length_data = batch[12], batch[13], batch[14]
            
            near_distance_data, far_distance_data = torch.FloatTensor(batch[15]).cuda(), \
                                                    torch.FloatTensor(batch[16]).cuda()
            anchor_near_memory_mask_data, near_anchor_memory_mask_data, anchor_far_memory_mask_data, far_anchor_memory_mask_data = torch.FloatTensor(batch[17]).cuda(), \
                                                                                                                                   torch.FloatTensor(batch[18]).cuda(), \
                                                                                                                                   torch.FloatTensor(batch[19]).cuda(), \
                                                                                                                                   torch.FloatTensor(batch[20]).cuda()

            anchor_near_length_data, near_anchor_length_data, anchor_far_length_data, far_anchor_length_data = batch[21], batch[22], batch[23], batch[24]
            
            pred_near_distance, pred_far_distance = self.model(anchor_gps_data, near_gps_data, far_gps_data,
                                                               anchor_mask_data, near_mask_data, far_mask_data,
                                                               anchor_position_data, near_position_data, far_position_data,
                                                               anchor_attention_mask_data, near_attention_mask_data, far_attention_mask_data,
                                                               anchor_near_memory_mask_data, near_anchor_memory_mask_data, anchor_far_memory_mask_data, far_anchor_memory_mask_data,
                                                               anchor_near_length_data, near_anchor_length_data, anchor_far_length_data, far_anchor_length_data)
            loss = self.loss_function(near_distance_data, far_distance_data,
                                      pred_near_distance, pred_far_distance)
            progress.current += 1
            progress ()
            self.optimizer.zero_grad ()
            loss.backward ()
            self.optimizer.step ()
            tloss += loss.cpu ().detach ().numpy ()
            
            if int (self.n_sentences / self.batch_size) % 10 == 0 and self.n_sentences > 0:
                logger.info (tloss)
                loss_list.append (tloss)
                tloss = 0
            
            self.n_sentences += self.batch_size
            torch.cuda.empty_cache()
        pickle.dump (loss_list, open ("loss_store/loss_list_" + str (epoch) + ".pkl", "wb"))
        progress.done ()
