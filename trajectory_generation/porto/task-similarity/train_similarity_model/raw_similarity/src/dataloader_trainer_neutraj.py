
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

    def __init__(self, model, loss_func, params, dataset, kd_tree_index, traj_sequences, traj_feature,
                 batch_size=16, test_batch_size=128, k_traj=5, test_trajectory_prob=0.1):
        self.index = 0
        self.loss_func = loss_func
        self.model = model
        self.data = dataset
        self.k_traj = k_traj
        self.params = params
        self.traj_feature = traj_feature # [mean_lat, mean_lon, std_lat, std_lon]
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
        
        self.optimizer = Adam(self.model.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma = 0.8)
    
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
    
    def get_sub_length_tuple(self, t_length, a_length, f_length, k_sub=10):
        
        return (random.sample(list(range(1, t_length + 1)), k_sub),
                random.sample(list(range(1, t_length + 1)), k_sub),
                random.sample(list(range(1, a_length + 1)), k_sub),
                random.sample(list(range(1, f_length + 1)), k_sub))
    
    def get_batch_quick(self):
        batch_size = self.batch_size
        while self.index < self.data_length:
            if self.index + batch_size > self.data_length:
                batch_size = self.data_length - self.index
            trajectory_input, anchor_input, farest_input, anchor_distance, farest_distance = [], [], [], [], []
            trajectory_gps, anchor_gps, farest_gps = [], [], []
            trajectory_length, anchor_length, farest_length = [], [], []
            trajectory_anchor_sub_length, trajectory_farest_sub_length, anchor_sub_length, farest_sub_length = [], [], [], []
            anchor_sub_distance, farest_sub_distance = [], [] # trajectory_feature [mean_lat, mean_lon, std_lat, std_lon]
            for batchi, i in enumerate(range(self.index, self.index + batch_size)):
                trajectory_input.append([[(token[0]-self.traj_feature[0])/self.traj_feature[2],
                                          (token[1]-self.traj_feature[1])/self.traj_feature[3],
                                           token[2], token[3]] for token in self.data[i]])
                trajectory_gps.append([p[:2] for p in self.data[i]])
                trajectory_length.append(len(self.data[i]))
                k_d_tree_list = self.kd_tree_index[i]
                anchor_index = int(random.choice(k_d_tree_list))
                anchor_input.append([[(token[0]-self.traj_feature[0])/self.traj_feature[2],
                                      (token[1]-self.traj_feature[1])/self.traj_feature[3],
                                       token[2], token[3]] for token in self.data[anchor_index]])
                anchor_gps.append([p[:2] for p in self.data[anchor_index]])
                anchor_length.append(len(self.data[anchor_index]))
                farest_index = random.choice(list(range(self.data_length)))
                while farest_index == i or farest_index == anchor_index:
                    farest_index = random.choice(list(range(self.data_length)))
                farest_input.append([[(token[0]-self.traj_feature[0])/self.traj_feature[2],
                                      (token[1]-self.traj_feature[1])/self.traj_feature[3],
                                       token[2], token[3]] for token in self.data[farest_index]])
                farest_gps.append([p[:2] for p in self.data[farest_index]])
                farest_length.append(len(self.data[farest_index]))

                near_distance_matrix, nearest_path = FrechetDistanceLoop.FrechetDistance(np.array(trajectory_gps[-1]), np.array(anchor_gps[-1]))
                far_distance_matrix, farest_path = FrechetDistanceLoop.FrechetDistance(np.array(trajectory_gps[-1]), np.array(farest_gps[-1]))
                
                anchor_distance.append(math.exp(-float(near_distance_matrix[trajectory_length[-1] - 1][anchor_length[-1] - 1]) * mail_pre_degree))
                farest_distance.append(math.exp(-float(far_distance_matrix[trajectory_length[-1] - 1][farest_length[-1] - 1]) * mail_pre_degree))
                
                traj_anch_fare_sub_length_tuple = self.get_sub_length_tuple(trajectory_length[-1], anchor_length[-1], farest_length[-1])
                trajectory_anchor_sub_length.extend(traj_anch_fare_sub_length_tuple[0])
                trajectory_farest_sub_length.extend(traj_anch_fare_sub_length_tuple[1])
                anchor_sub_length.extend(traj_anch_fare_sub_length_tuple[2])
                farest_sub_length.extend(traj_anch_fare_sub_length_tuple[3])
                
                anchor_sub_distance.extend([math.exp(-float(near_distance_matrix[trajectory_anchor_sub_length[sub_i + batchi*10] - 1][anchor_sub_length[sub_i + batchi*10] - 1]) * mail_pre_degree)
                                            for sub_i in range(10)])
                farest_sub_distance.extend([math.exp(-float(far_distance_matrix[trajectory_farest_sub_length[sub_j + batchi*10] - 1][farest_sub_length[sub_j + batchi*10] - 1]) * mail_pre_degree)
                                            for sub_j in range(10)])
                
            trajectory_length_max = max(trajectory_length)
            anchor_length_max = max(anchor_length)
            farest_length_max = max(farest_length)
            
            for ind, ti, ai, fi in zip(range(batch_size), trajectory_length, anchor_length, farest_length):

                trajectory_input[ind] = trajectory_input[ind] + [[0.0, 0.0, 0.0, 0.0] for _ in range(trajectory_length_max - ti)]
                anchor_input[ind] = anchor_input[ind] + [[0.0, 0.0, 0.0, 0.0] for _ in range(anchor_length_max - ai)]
                farest_input[ind] = farest_input[ind] + [[0.0, 0.0, 0.0, 0.0] for _ in range(farest_length_max - fi)]

            self.index += batch_size
            self.batch_size = batch_size
            yield trajectory_input, anchor_input, farest_input, \
                  trajectory_length, anchor_length, farest_length,\
                  anchor_distance, farest_distance, \
                  trajectory_anchor_sub_length, trajectory_farest_sub_length, anchor_sub_length, farest_sub_length, \
                  anchor_sub_distance, farest_sub_distance
    
        
    def train(self, epoch):
        self.index = 0
        self.model.train()

        progress = ProgressBar(len(self.data) // self.batch_size, fmt=ProgressBar.FULL)
        tloss = 0.0
        loss_list = []
        
        for batch in self.get_batch_quick():
            #############################################
            traj_input = torch.Tensor(batch[0]).cuda()
            traj_len = batch[3]
            traj_anch_sub_len = batch[8]
            traj_fare_sub_len = batch[9]
            #############################################
            anch_input = torch.Tensor(batch[1]).cuda()
            anch_len = batch[4]
            anch_sub_len = batch[10]
            #############################################
            fare_input = torch.Tensor(batch[2]).cuda()
            fare_len = batch[5]
            fare_sub_len = batch[11]
            #############################################
            anchor_dist, farest_dist = torch.FloatTensor(batch[6]).cuda(), torch.FloatTensor(batch[7]).cuda()
            anchor_sub_dist, farest_sub_dist = torch.FloatTensor(batch[12]).cuda(), torch.FloatTensor(batch[13]).cuda()
            #############################################
            anchor_pred, farest_pred, anchor_sub_pred, farest_sub_pred = self.model(traj_input, traj_len, traj_anch_sub_len, traj_fare_sub_len,
                                                                                    anch_input, anch_len, anch_sub_len,
                                                                                    fare_input, fare_len, fare_sub_len)
            loss = self.loss_func(anchor_pred, farest_pred,
                                  anchor_dist, farest_dist,
                                  anchor_sub_pred, farest_sub_pred,
                                  anchor_sub_dist, farest_sub_dist)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            tloss += loss.cpu().detach().numpy()
            progress.current += 1
            progress()
            if int(self.n_sentences / self.batch_size) % 10 == 0 and self.n_sentences > 0:
                logger.info(tloss)
                loss_list.append (tloss)
                tloss = 0

            self.n_sentences += self.batch_size
            torch.cuda.empty_cache()
        
        # self.scheduler.step()
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