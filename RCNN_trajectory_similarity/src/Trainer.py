import math
import torch
import random
import pickle
import datetime
import numpy as np
from logging import getLogger
import torch.nn.functional as F
from SSM import FrechetDistanceLoop
from utils import adjusting_rate, ProgressBar

logger = getLogger()
scale = 100.0
tscale = 1.0
mail_pre_degree = 16.0

class MatrixTrainer(object):
    def __init__(self, MODEL, OPT, train_batch_size, dataset, trajectory_feature, trajectory_match_indexs):
        self.model = MODEL
        self.loss_fn = torch.nn.MSELoss()
        # self.loss_fn2 = torch.nn.KLDivLoss
        self.opt = OPT
        self.train_batch_size = train_batch_size
        self.trajectory_match_index = trajectory_match_indexs
        self.dataset = dataset
        self.data_name = "porto"
        self.dataset_length = len(dataset)
        self.trajectory_feature = trajectory_feature
        self.dataset_index_list = list(range(self.dataset_length))
        self.progressbar = ProgressBar(self.dataset_length // self.train_batch_size, fmt=ProgressBar.FULL)
        # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001, weight_decay=0.001)
        # self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.opt.learning_rate,
        #                                         momentum=self.opt.momentum, weight_decay=self.opt.weight_decay)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.0001)
    
    
    def get_unmatch_index(self, anchor_index):
        unmatch_index = random.choice(self.dataset_index_list)
        while unmatch_index == anchor_index and (unmatch_index not in self.trajectory_match_index[anchor_index]):
            unmatch_index = random.choice(self.dataset_index_list)
        return unmatch_index
    
    def get_match_index(self, anchor_index):
        k_d_tree_list = self.trajectory_match_index[anchor_index]
        match_index = int(random.choice(k_d_tree_list))
        while match_index == anchor_index:
            match_index = int(random.choice(k_d_tree_list))
        return match_index
    
    def batch_generator(self):
        index = 0
        batch_size = self.train_batch_size
        while index < self.dataset_length:
            if index + batch_size > self.dataset_length:
                batch_size = self.dataset_length - index
            anchor_trajectory, match_trajectory, unmatch_trajectory = [], [], []
            anchor_grid, match_grid, unmatch_grid = [], [], []
            anchor_length, match_length = [], []
            distance_matrix_list_m, distance_matrix_xy_list_m = [], []
            distance_matrix_list_um, distance_matrix_xy_list_um = [], []
            for anchor_i in range(index, index + batch_size):
                # anchor_trajectory.append([[(anchor_p[0]-self.trajectory_feature[0])/self.trajectory_feature[2],
                #                            (anchor_p[1]-self.trajectory_feature[1])/self.trajectory_feature[3]] for anchor_p in self.dataset[anchor_i]])
                anchor_trajectory.append(self.dataset[anchor_i][0][:32])
                anchor_grid.append(self.dataset[anchor_i][1][:32])
                match_i = self.get_match_index(anchor_i)
                # match_trajectory.append([[(match_p[0]-self.trajectory_feature[0])/self.trajectory_feature[2],
                #                           (match_p[1]-self.trajectory_feature[1])/self.trajectory_feature[3]] for match_p in self.dataset[match_i]])
                match_trajectory.append(self.dataset[match_i][0][:32])
                match_grid.append(self.dataset[match_i][1][:32])
                anchor_length.append(len(anchor_trajectory[-1]))
                match_length.append(len(match_trajectory[-1]))
                unmatch_i = self.get_unmatch_index(anchor_i)
                unmatch_trajectory.append(self.dataset[unmatch_i][0][:32])
                unmatch_grid.append(self.dataset[unmatch_i][1][:32])
                distance_match = FrechetDistanceLoop.FrechetDistance(np.array(self.dataset[anchor_i][0]),
                                                                      np.array(self.dataset[match_i][0]))[0] * mail_pre_degree
                distance_unmatch = FrechetDistanceLoop.FrechetDistance(np.array(self.dataset[anchor_i][0]),
                                                                      np.array(self.dataset[unmatch_i][0]))[0] * mail_pre_degree
                distance_matrix_list_m.append(np.exp(-distance_match))
                distance_matrix_list_um.append(np.exp(-distance_unmatch))
            anchor_length_min, match_length_min = min(anchor_length), min(match_length)
            matrix_x, matrix_y = (anchor_length_min // 32)*32, (match_length_min // 32)*32
        
            input_matrix_m, input_matrix_um = [], []
            for i in range(batch_size):
                line_xy_m, line_xy_um = [], []
                for x in range(matrix_x):
                    line_x_m, line_x_um = [], []
                    for y in range(matrix_y):
                        line_x_m.append(((anchor_trajectory[i][x][0] - match_trajectory[i][y][0])**2 +
                                         (anchor_trajectory[i][x][1] - match_trajectory[i][y][1])**2)**0.5 * mail_pre_degree * scale)
                        line_x_um.append(((anchor_trajectory[i][x][0] - unmatch_trajectory[i][y][0]) ** 2 +
                                          (anchor_trajectory[i][x][1] - unmatch_trajectory[i][y][1]) ** 2)**0.5 * mail_pre_degree * scale)
                    line_xy_m.append(line_x_m)
                    line_xy_um.append(line_x_um)
                input_matrix_m.append([line_xy_m for _ in range(3)])
                input_matrix_um.append([line_xy_um for _ in range(3)])
                distance_matrix_xy_list_m.append(distance_matrix_list_m[i][:matrix_x, :matrix_y])
                distance_matrix_xy_list_um.append(distance_matrix_list_um[i][:matrix_x, :matrix_y])
                anchor_trajectory[i] = [[(p[0] - self.trajectory_feature[0]) / self.trajectory_feature[2] * tscale,
                                         (p[1] - self.trajectory_feature[1]) / self.trajectory_feature[3] * tscale] for p in anchor_trajectory[i]]

                match_trajectory[i] = [[(p[0] - self.trajectory_feature[0]) / self.trajectory_feature[2] * tscale,
                                        (p[1] - self.trajectory_feature[1]) / self.trajectory_feature[3] * tscale] for p in match_trajectory[i]]

                unmatch_trajectory[i] = [[(p[0] - self.trajectory_feature[0]) / self.trajectory_feature[2] * tscale,
                                          (p[1] - self.trajectory_feature[1]) / self.trajectory_feature[3] * tscale] for p in unmatch_trajectory[i]]
            
            index += batch_size
            yield input_matrix_m, input_matrix_um, distance_matrix_xy_list_m, distance_matrix_xy_list_um, \
                  anchor_trajectory, match_trajectory, unmatch_trajectory, \
                  anchor_grid, match_grid, unmatch_grid
            
    def train_step(self, epoch):
        # loss_sum = 0
        num_batch = 0
        loss_list = []
        realepoch = epoch + 1
        for batch in self.batch_generator():
            input_data_m = torch.FloatTensor(batch[0]).cuda()
            input_data_um = torch.FloatTensor(batch[1]).cuda()
            target_data_m = torch.FloatTensor(batch[2]).cuda()
            target_data_um = torch.FloatTensor(batch[3]).cuda()
            
            anch_gps, matc_gps, unma_gps = torch.FloatTensor(batch[4]).cuda(), torch.FloatTensor(batch[5]).cuda(), torch.FloatTensor(batch[6]).cuda()
            anch_grid, matc_grid, unma_grid = torch.LongTensor(batch[7]).cuda(), torch.LongTensor(batch[8]).cuda(), torch.LongTensor(batch[9]).cuda()

            target_data_m_total = target_data_m[:, -1, -1].unsqueeze(1)
            target_data_um_total = target_data_um[:, -1, -1].unsqueeze(1)
            
            pred_data_m, feature_target_data_m, feature_predict_data_m = self.model(input_data_m, anch_gps, anch_grid, matc_gps, matc_grid)
            pred_data_um, feature_target_data_um, feature_predict_data_um = self.model(input_data_um, anch_gps, anch_grid, unma_gps, unma_grid)
            
            pred_data_m_total = pred_data_m[:, -1, -1].unsqueeze(1)
            pred_data_um_total = pred_data_um[:, -1, -1].unsqueeze(1)
            example_m_pred = pred_data_m[0, -1, -1].item()
            example_um_pred = pred_data_um[0, -1, -1].item()
            example_m_targ = target_data_m[0, -1, -1].item()
            example_um_targ = target_data_um[0, -1, -1].item()
            target_loss = torch.sum(torch.mul(pred_data_m[:, -1, -1] - target_data_m[:, -1, -1],
                                              pred_data_m[:, -1, -1] - target_data_m[:, -1, -1])).item()/(pred_data_m.shape[0] * 1.0)
            loss_m = self.loss_fn(pred_data_m, target_data_m)
            loss_um = self.loss_fn(pred_data_um, target_data_um)
            loss_m_total = self.loss_fn(pred_data_m_total, target_data_m_total)
            loss_um_total = self.loss_fn(pred_data_um_total, target_data_um_total)

            feature_loss_m = F.kl_div(feature_predict_data_m.softmax(dim=-1).log(), feature_target_data_m.softmax(dim=-1), reduction="batchmean")
            feature_loss_um = F.kl_div(feature_predict_data_um.softmax(dim=-1).log(), feature_target_data_um.softmax(dim=-1), reduction="batchmean")
            
            loss = sum([loss_m, loss_um, loss_m_total, loss_um_total, feature_loss_m, feature_loss_um])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if realepoch % 1 == 0:
                info = 'Epoch:{} Loss_avg:{} target_loss:{} m_pred:{} m_targ:{} um_pred:{} um_targ:{}'.format(epoch + 1, loss.item (), target_loss, example_m_pred, example_m_targ, example_um_pred, example_um_targ)
                loss_list.append(loss.item())
                logger.info(info)
            num_batch += 1
        adjusting_rate(self.optimizer, self.opt.learning_rate, epoch + 1)
        torch.save(self.model.state_dict(), "ocd" + '_' + self.data_name + str(epoch + 23) + '.pt')
        pickle.dump(loss_list, open("loss_store/loss_" + str(epoch) + ".pkl", "wb"))
