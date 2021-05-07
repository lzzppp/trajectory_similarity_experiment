
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

class EncoderwithDecoderTrainer(object):

    def __init__(self, model, loss_func, params, dataset, kd_tree, traj_sequences, batch_size=16, test_batch_size=128, k_traj=5, test_trajectory_prob=0.1):
        self.index = 0
        self.loss_func = loss_func
        self.model = model
        self.data = dataset
        self.k_traj = k_traj
        self.params = params
        self.kd_tree = kd_tree
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
        
        self.optimizer = Adam(self.model.parameters(), lr=1e-5)
    
    def evaluate(self, pred_matrix, target_matrix):
        top_10_match_10, top_50_match_50, top_50_match_10 = 0, 0, 0
        for test_i in tqdm(range(self.test_trajectory_num)):
            pred_test_i = np.array(pred_matrix[test_i])
            target_test_i = target_matrix[test_i]
            # pred_test_i_top50 = heapq.nsmallest(pred_test_i, range(len(pred_test_i)), pred_test_i.take)
            pred_test_i_top50 = heapq.nsmallest(50, range(len(pred_test_i)), pred_test_i.take)
            pred_test_i_top10 = pred_test_i_top50[:10]
            target_test_i_top50 = heapq.nsmallest(50, range(len(target_test_i)), target_test_i.take)
            target_test_i_top10 = target_test_i_top50[:10]
            for ten_p in pred_test_i_top10:
                if ten_p in target_test_i_top10:
                    top_10_match_10 += 1
                if ten_p in target_test_i_top10 and ten_p in pred_test_i_top50:
                    top_50_match_10 += 1
            for fif_p in pred_test_i_top50:
                if fif_p in target_test_i_top50:
                    top_50_match_50 += 1
            
        self.stats["top10@acc"] = top_10_match_10 / (10.0*self.test_trajectory_num)
        self.stats["top50@acc"] = top_50_match_50 / (50.0*self.test_trajectory_num)
        self.stats["top10@50acc"] = top_50_match_10 / (10.0*self.test_trajectory_num)
        
    def get_batch_quick(self):
        batch_size = self.batch_size
        while self.index < self.data_length:
            if self.index + batch_size > self.data_length:
                batch_size = self.data_length - self.index
            trajectory_gps, trajectory_grid, anchor_gps, anchor_grid, farest_gps, farest_grid, anchor_distance, farest_distance = [], [], [], [], [], [], [], []
            trajectory_length, anchor_length, farest_length = [], [], []
            for i in range(self.index, self.index + batch_size):
                trajectory_gps.append(self.data[i][1])
                trajectory_grid.append(self.data[i][0])
                trajectory_length.append(len(self.data[i][0]))
                k_d_tree_list = self.kd_tree.search_knn(self.traj_sequences[i], self.k_traj)
                anchor_index = int(random.choice(k_d_tree_list)[0].label)
                while anchor_index == i:
                    anchor_index = int(random.choice(k_d_tree_list)[0].label)
                anchor_gps.append(self.data[anchor_index][1])
                anchor_grid.append(self.data[anchor_index][0])
                anchor_length.append(len(self.data[anchor_index][0]))
                farest_index = random.choice(list(range(self.data_length)))
                while farest_index == i or farest_index == anchor_index:
                    farest_index = random.choice(list(range(self.data_length)))
                farest_gps.append(self.data[farest_index][1])
                farest_grid.append(self.data[farest_index][0])
                farest_length.append(len(self.data[farest_index][0]))

                near_distance_matrix, nearest_path = FrechetDistanceLoop.FrechetDistance(np.array(trajectory_gps[-1]), np.array(anchor_gps[-1]))
                far_distance_matrix, farest_path = FrechetDistanceLoop.FrechetDistance(np.array(trajectory_gps[-1]), np.array(farest_gps[-1]))
                
                anchor_distance.append(math.exp(-float(near_distance_matrix[trajectory_length[-1] - 1][anchor_length[-1] - 1]) * mail_pre_degree))
                farest_distance.append(math.exp(-float(far_distance_matrix[trajectory_length[-1] - 1][farest_length[-1] - 1]) * mail_pre_degree))
                
                # anchor_distance.append(math.exp(-float(self.distance[self.train_index_dict[i]][self.train_index_dict[anchor_index]]) * mail_pre_degree))
                # farest_distance.append(math.exp(-float(self.distance[self.train_index_dict[i]][self.train_index_dict[farest_index]]) * mail_pre_degree))
                
            trajectory_length_max = max(trajectory_length)
            anchor_length_max = max(anchor_length)
            farest_length_max = max(farest_length)
            # print(trajectory_length_max, " ", anchor_length_max, " ", farest_length_max)
            # padding all data
            trajectory_mask, anchor_mask, farest_mask, trajectory_grid_pos, anchor_grid_pos, farest_grid_pos = [], [], [], [], [], []
            for ind, ti, ai, fi in zip(range(batch_size), trajectory_length, anchor_length, farest_length):
                trajectory_mask.append([False] * ti + [True] * (trajectory_length_max - ti))
                anchor_mask.append([False] * ai + [True] * (anchor_length_max - ai))
                farest_mask.append([False] * fi + [True] * (farest_length_max - fi))
                # trajectory_grid_pos.append([j+1 for j in range(trajectory_length_max)])
                # anchor_grid_pos.append([j+1 for j in range(anchor_length_max)])
                # farest_grid_pos.append([j+1 for j in range(farest_length_max)])
                
                trajectory_gps[ind] = trajectory_gps[ind] + [[0.0, 0.0] for ttii in range(trajectory_length_max - ti)]
                trajectory_grid[ind] = trajectory_grid[ind] + [0]*(trajectory_length_max - ti)
                anchor_gps[ind] = anchor_gps[ind] + [[0.0, 0.0] for aaii in range(anchor_length_max - ai)]
                anchor_grid[ind] = anchor_grid[ind] + [0]*(anchor_length_max - ai)
                farest_gps[ind] = farest_gps[ind] + [[0.0, 0.0] for ffii in range(farest_length_max - fi)]
                farest_grid[ind] = farest_grid[ind] + [0]*(farest_length_max - fi)
                
                # trajectory_gps[ind] += [[0.0, 0.0] for ttii in range(trajectory_length_max - ti)]
                # trajectory_grid[ind] += [0]*(trajectory_length_max - ti)
                # anchor_gps[ind] += [[0.0, 0.0] for aaii in range(anchor_length_max - ai)]
                # anchor_grid[ind] += [0]*(anchor_length_max - ai)
                # farest_gps[ind] += [[0.0, 0.0] for ffii in range(farest_length_max - fi)]
                # farest_grid[ind] += [0]*(farest_length_max - fi)

            self.index += batch_size
            self.batch_size = batch_size
            yield trajectory_gps, trajectory_grid, trajectory_mask, \
                  anchor_gps, anchor_grid, anchor_mask, \
                  farest_gps, farest_grid, farest_mask, \
                  trajectory_length, anchor_length, farest_length,\
                  anchor_distance, farest_distance
    
    def get_batch(self):
        batch_size = self.batch_size
        while self.index < self.data_length:
            if self.index + batch_size > self.data_length:
                batch_size = self.data_length - self.index
            trajectory_gps, trajectory_grid, anchor_gps, anchor_grid, farest_gps, farest_grid, anchor_distance, farest_distance = [], [], [], [], [], [], [], []
            trajectory_length, anchor_length, farest_length = [], [], []
            for i in range(self.index, self.index + batch_size):
                trajectory_gps.append(self.data[i][1])
                trajectory_grid.append(self.data[i][0])
                trajectory_length.append(len(self.data[i][0]))
                k_d_tree_list = self.kd_tree.search_knn(self.traj_sequences[i], self.k_traj)
                anchor_index = int(random.choice(k_d_tree_list)[0].label)
                while anchor_index == i:
                    anchor_index = int(random.choice(k_d_tree_list)[0].label)
                anchor_gps.append(self.data[anchor_index][1])
                anchor_grid.append(self.data[anchor_index][0])
                anchor_length.append(len(self.data[anchor_index][0]))
                farest_index = random.choice(list(range(self.data_length)))
                while farest_index == i or farest_index == anchor_index:
                    farest_index = random.choice(list(range(self.data_length)))
                farest_gps.append(self.data[farest_index][1])
                farest_grid.append(self.data[farest_index][0])
                farest_length.append(len(self.data[farest_index][0]))

                near_distance_matrix, nearest_path = FrechetDistanceLoop.FrechetDistance(np.array(trajectory_gps[-1]), np.array(anchor_gps[-1]))
                far_distance_matrix, farest_path = FrechetDistanceLoop.FrechetDistance(np.array(trajectory_gps[-1]), np.array(farest_gps[-1]))

                anchor_distance.append(math.exp(-float(near_distance_matrix[trajectory_length[-1] - 1][anchor_length[-1] - 1]) * mail_pre_degree))
                farest_distance.append(math.exp(-float(far_distance_matrix[trajectory_length[-1] - 1][farest_length[-1] - 1]) * mail_pre_degree))
            trajectory_length_max = max(trajectory_length)
            anchor_length_max = max(anchor_length)
            farest_length_max = max(farest_length)
            trajectory_mask = [[False] * ti + [True] * (trajectory_length_max - ti) for ti in trajectory_length]
            anchor_mask = [[False] * ai + [True] * (anchor_length_max - ai) for ai in anchor_length]
            farest_mask = [[False] * fi + [True] * (farest_length_max - fi) for fi in farest_length]
            trajectory_grid_pos = [[j+1 for j in range(trajectory_length_max)] for _ in range(batch_size)]
            anchor_grid_pos = [[j+1 for j in range(anchor_length_max)] for _ in range(batch_size)]
            farest_grid_pos = [[j+1 for j in range(farest_length_max)] for _ in range(batch_size)]
            trajectory_gps = [trajectory_gps[ti] + [[0.0, 0.0] for _ in range(trajectory_length_max - trajectory_length[ti])] for ti in range(batch_size)]
            trajectory_grid = [trajectory_grid[ti] + [0]*(trajectory_length_max- trajectory_length[ti]) for ti in range(batch_size)]
            anchor_gps = [anchor_gps[ai] + [[0.0, 0.0] for _ in range(anchor_length_max - anchor_length[ai])] for ai in range(batch_size)]
            anchor_grid = [anchor_grid[ai] + [0]*(anchor_length_max - anchor_length[ai]) for ai in range(batch_size)]
            farest_gps = [farest_gps[fi] + [[0.0, 0.0] for _ in range(farest_length_max - farest_length[fi])] for fi in range(batch_size)]
            farest_grid = [farest_grid[fi] + [0]*(farest_length_max - farest_length[fi]) for fi in range(batch_size)]
            self.index += batch_size
            yield trajectory_gps, trajectory_grid, trajectory_mask, trajectory_grid_pos, \
                  anchor_gps, anchor_grid, anchor_mask, anchor_grid_pos,\
                  farest_gps, farest_grid, farest_mask, farest_grid_pos,\
                  trajectory_length, anchor_length, farest_length,\
                  anchor_distance, farest_distance
    
    def produce_trajectory_embedding_vector(self):
        test_batch_size = self.test_batch_size
        while self.index < self.test_data_length:
            if self.index + self.test_batch_size > self.test_data_length:
                test_batch_size = self.test_data_length - self.index
            trajectory_gps, trajectory_grid, trajectory_length = [], [], []
            for i in range(self.index, self.index + test_batch_size):
                trajectory_gps.append(self.test_data[i][1])
                trajectory_grid.append(self.test_data[i][0])
                trajectory_length.append(len(self.test_data[i][0]))
            trajectory_max_length = max(trajectory_length)
            trajectory_mask = []
            for j, ti in enumerate(trajectory_length):
                trajectory_mask.append([False] * ti + [True] * (trajectory_max_length - ti))
                trajectory_gps[j] = trajectory_gps[j] + [[0.0, 0.0] for _ in range(trajectory_max_length - ti)]
                trajectory_grid[j] = trajectory_grid[j] + [0]*(trajectory_max_length - ti)
            self.index += test_batch_size
            self.test_batch_size = test_batch_size
            yield trajectory_gps, trajectory_grid, trajectory_mask, trajectory_max_length
        
    def train(self, epoch):
        self.index = 0
        self.model.train()

        progress = ProgressBar(len(self.data) // self.batch_size, fmt=ProgressBar.FULL)
        tloss = 0.0
        loss_list = []
        
        for batch in self.get_batch_quick():
            #############################################
            traj_gps = torch.FloatTensor(batch[0]).cuda()
            traj_grid = torch.LongTensor(batch[1]).cuda()
            traj_mask = torch.BoolTensor(batch[2]).cuda()
            # traj_pos = torch.LongTensor(batch[3]).cuda()
            traj_pos = torch.arange(1, max(batch[9]) + 1, device=torch.device("cuda:0")).unsqueeze(0).expand(self.batch_size, max(batch[9]))
            #############################################
            anch_gps = torch.FloatTensor(batch[3]).cuda()
            anch_grid = torch.LongTensor(batch[4]).cuda()
            anch_mask = torch.BoolTensor(batch[5]).cuda()
            # anch_pos = torch.LongTensor(batch[7]).cuda()
            anch_pos = torch.arange(1, max(batch[10]) + 1, device=torch.device("cuda:0")).unsqueeze(0).expand(self.batch_size, max(batch[10]))
            #############################################
            fare_gps = torch.FloatTensor(batch[6]).cuda()
            fare_grid = torch.LongTensor(batch[7]).cuda()
            fare_mask = torch.BoolTensor(batch[8]).cuda()
            # fare_pos = torch.LongTensor(batch[11]).cuda()
            fare_pos = torch.arange(1, max(batch[11]) + 1, device=torch.device("cuda:0")).unsqueeze(0).expand(self.batch_size, max(batch[11]))
            #############################################
            anchor_dist, farest_dist = torch.FloatTensor(batch[12]).cuda(), torch.FloatTensor(batch[13]).cuda()
            #############################################
            anchor_pred, farest_pred = self.model(traj_gps, traj_mask, traj_grid, traj_pos,
                                                  anch_gps, anch_mask, anch_grid, anch_pos,
                                                  fare_gps, fare_mask, fare_grid, fare_pos)
            loss = self.loss_func(anchor_pred, farest_pred,
                                  anchor_dist, farest_dist)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # tloss += loss.item()
            tloss += loss.cpu().detach().numpy()
            progress.current += 1
            progress()
            if int(self.n_sentences / self.batch_size) % 10 == 0 and self.n_sentences > 0:
                # print("*"*int(self.n_sentences / self.batch_size), tloss)
                logger.info(tloss)
                loss_list.append (tloss)
                tloss = 0

            # number of processed sentences / words
            self.n_sentences += self.batch_size
            
            # del traj_gps, traj_mask, traj_grid, traj_pos, \
            #     anch_gps, anch_mask, anch_grid, anch_pos, \
            #     fare_gps, fare_mask, fare_grid, fare_pos, \
            #     anchor_dist, farest_dist, loss
            # torch.cuda.empty_cache()
        tloss = 0.0
        pickle.dump(loss_list, open ("loss_store/loss_list_" + str (epoch) + ".pkl", "wb"))
        progress.done()
    
    def test(self, epoch):
        self.model.eval()
        self.index = 0
        
        produce_progress = ProgressBar(len(self.test_data) // self.test_batch_size, fmt=ProgressBar.FULL)
        with torch.no_grad():
            for idx, produce_batch in enumerate(self.produce_trajectory_embedding_vector()):
                #######################################################
                traj_gps = torch.FloatTensor(produce_batch[0]).cuda()
                traj_grid = torch.LongTensor(produce_batch[1]).cuda()
                traj_mask = torch.BoolTensor(produce_batch[2]).cuda()
                traj_pos = torch.arange(1, produce_batch[3] + 1, device=torch.device("cuda:0")).unsqueeze(0).expand(self.test_batch_size, produce_batch[3])
                #######################################################
                
                trajectory_zone_embedding = self.model.traj_embedding(traj_grid, traj_pos, traj_gps, traj_mask)
                _, trajectory_embedding = self.model.trajectory_encoder(trajectory_zone_embedding)
                
                if idx == 0:
                    trajectory_embedding_matrix = trajectory_embedding.transpose(0, 1)
                else:
                    trajectory_embedding_matrix = torch.cat((trajectory_embedding_matrix, trajectory_embedding.transpose(0, 1)))
                produce_progress.current += 1
                produce_progress()
            produce_progress.done()
        
        print("Produced ", self.test_data_length, " trajectories' embedding vector done ! ")
        
        pred_distance_matrix = []
        for test_index in tqdm(range(self.test_trajectory_num)):
            traj_embed = trajectory_embedding_matrix[test_index, :]
            pred_distances = []
            for anchor_index in range(self.test_data_length):
                pred_distance = torch.norm(traj_embed - trajectory_embedding_matrix[anchor_index, :], p=2, dim=-1).cpu().numpy()[0]
                pred_distances.append(pred_distance)
            pred_distance_matrix.append(pred_distances)

        print("Produced ", self.test_trajectory_num, " trajectories' predict distance ! ")
        
        inde = 0
        for test_index in tqdm(self.test_trajectory_index):
            if inde == 0:
                target_distance_matrix = self.distance[test_index, :]
            else:
                target_distance_matrix = np.vstack((target_distance_matrix, self.distance[test_index, :]))
            inde += 1
        
        self.evaluate(pred_distance_matrix, target_distance_matrix)
        
        print(self.stats)