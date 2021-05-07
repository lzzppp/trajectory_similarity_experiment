# coding=utf-8
import math
import torch
import kdtree
import random
import numpy as np
import torch.nn as nn
# from traj_dist import distance as tdist

# alpha = 1.0/10.0

def weight_init(m):
    try:
        nn.init.uniform_(m.weight, a=-128**(-0.5), b=128**(-0.5))
        nn.init.uniform_(m.bias, a=-128**(-0.5), b=128**(-0.5))
    except:
        nn.init.uniform_(m.weight, a=-128**(-0.5), b=128**(-0.5))

class f_hid_sim(nn.Module):
    def __init__(self, hidden_size):
        super (f_hid_sim, self).__init__()
        self.linear_i = nn.Linear(hidden_size, hidden_size)
        self.linear_c = nn.Linear(hidden_size, hidden_size)
        self.linear_o = nn.Linear(hidden_size, hidden_size)
        self.init_weight()
        
    def init_weight(self):
        nn.init.uniform_(self.linear_i.weight)
        nn.init.uniform_(self.linear_o.weight)
        nn.init.uniform_(self.linear_c.weight)
        nn.init.zeros_(self.linear_i.bias)
        nn.init.zeros_(self.linear_o.bias)
        nn.init.zeros_(self.linear_c.bias)
    
    def forward(self, hi):
        C = torch.mul (torch.sigmoid (self.linear_i (hi)),
                       torch.tanh (self.linear_c (hi)))
        hi_ = torch.mul (torch.sigmoid (self.linear_o (hi)),
                         torch.tanh (C))
        output = hi_ + hi
        return output

class f_hid_mve(nn.Module):
    def __init__(self, hidden_size, batch_size=32):
        super(f_hid_mve, self).__init__()
        self.rnncell = nn.LSTMCell(hidden_size, hidden_size // 2)
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.init_weight()
    
    def init_weight(self):
        nn.init.uniform_(self.rnncell.weight_hh)
        nn.init.uniform_(self.rnncell.weight_ih)
        # nn.init.constant_(self.rnncell.bias, 0)
        # nn.init.constant_(self.rnncell.bias_hh, 0)
        # nn.init.constant_(self.rnncell.bias_ih, 0)
    
    def forward(self, hidden_l, hidden_list):
        h = torch.zeros(hidden_l.shape[0], self.hidden_size // 2).cuda()
        C = torch.zeros(hidden_l.shape[0], self.hidden_size // 2).cuda()
        output = []
        for index in range(hidden_list.shape[1]):
            h, C = self.rnncell(torch.cat((hidden_list[:, index, :], hidden_l), dim=-1), (h, C))
            output.append(C.unsqueeze(1))
        output = torch.cat(output, dim=1)
        return output
            
class Traj2Sim_encoder(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, batch_size=32, Bidirectional=True):
        super(Traj2Sim_encoder, self).__init__()
        if rnn_type == 'GRU':
            if Bidirectional:
                self.rnn = nn.GRU(input_size, hidden_size//2, batch_first=True, bidirectional=Bidirectional)
            else:
                self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        elif rnn_type == 'LSTM':
            if Bidirectional:
                self.rnn = nn.LSTM(input_size, hidden_size//2, batch_first=True, bidirectional=Bidirectional)
                # self.rnncell = nn.LSTMCell(hidden_size, hidden_size)
            else:
                self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
                # self.rnncell = nn.LSTMCell(2 * hidden_size, 2 * hidden_size)
        self.bidirectional = Bidirectional
        self.f_hid2sim = f_hid_sim(hidden_size)
        self.f_hid2mve = f_hid_mve(hidden_size * 2)
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.init_weight()
        
    def init_weight(self):
        nn.init.uniform_(self.rnn.weight_hh_l0)
        nn.init.uniform_(self.rnn.weight_ih_l0)
        # nn.init.constant_(self.rnncell.bias, 0)
    
    def forward(self,
                anchor_nearest_input,
                anchor_farest_input,
                nearest_input,
                farest_input,
                trajectory_anchor_input,
                trajectory_nearest_input,
                trajectory_farest_input):
        anchor_nearest_output, (anchor_nearest_hidden_, anc) = self.rnn(anchor_nearest_input)
        anchor_nearest_output, (anchor_farest_hidden_, afc) = self.rnn(anchor_farest_input)
        nearest_output, (nearest_hidden_, nc) = self.rnn(nearest_input)
        farest_output, (farest_hidden_, fc) = self.rnn(farest_input)
        
        trajectory_anchor_output, (t_anchor_hidden_, tac) = self.rnn(trajectory_anchor_input)
        trajectory_nearest_output, (t_nearest_hidden_, tnc) = self.rnn(trajectory_nearest_input)
        trajectory_farest_output, (t_farest_hidden_, tfc) = self.rnn(trajectory_farest_input)
        
        if self.bidirectional:
            anchor_nearest_hidden = torch.cat((anchor_nearest_hidden_[0, :, :], anchor_nearest_hidden_[1, :, :]), dim=-1)
            anchor_farest_hidden = torch.cat((anchor_farest_hidden_[0, :, :], anchor_farest_hidden_[1, :, :]), dim=-1)
            nearest_hidden = torch.cat((nearest_hidden_[0, :, :], nearest_hidden_[1, :, :]), dim=-1)
            farest_hidden = torch.cat((farest_hidden_[0, :, :], farest_hidden_[1, :, :]), dim=-1)
            trajectory_hidden = torch.cat((t_anchor_hidden_[0, :, :], t_anchor_hidden_[1, :, :]), dim=-1)
            trajectory_nearest_hidden = torch.cat((t_nearest_hidden_[0, :, :], t_nearest_hidden_[1, :, :]), dim=-1)
            trajectory_farest_hidden = torch.cat((t_farest_hidden_[0, :, :], t_farest_hidden_[1, :, :]), dim=-1)
            m_anchor_vector = self.f_hid2mve(trajectory_hidden, trajectory_anchor_output)
            m_nearest_vector = self.f_hid2mve(trajectory_nearest_hidden, trajectory_nearest_output)
            m_farest_vector = self.f_hid2mve(trajectory_farest_hidden, trajectory_farest_output)
            
        else:
            anchor_nearest_hidden = anchor_nearest_hidden_.squeeze()
            anchor_farest_hidden = anchor_farest_hidden_.squeeze()
            nearest_hidden = nearest_hidden_.squeeze()
            farest_hidden = farest_hidden_.squeeze()
            trajectory_hidden = t_anchor_hidden_.squeeze()
            trajectory_nearest_hidden = t_nearest_hidden_.squeeze()
            trajectory_farest_hidden = t_farest_hidden_.squeeze()
            m_anchor_vector = self.f_hid2mve(trajectory_hidden, trajectory_anchor_output)
            m_nearest_vector = self.f_hid2mve(trajectory_nearest_hidden, trajectory_nearest_output)
            m_farest_vector = self.f_hid2mve(trajectory_farest_hidden, trajectory_farest_output)
            
        ##################### trajectory distance #############################
        anchor_nearest_vector = self.f_hid2sim(anchor_nearest_hidden)
        anchor_farest_vector = self.f_hid2sim(anchor_farest_hidden)
        nearest_vector = self.f_hid2sim(nearest_hidden)
        farest_vector = self.f_hid2sim(farest_hidden)
        trajectory_vector = self.f_hid2sim(trajectory_hidden)
        trajectory_nearest_vector = self.f_hid2sim(trajectory_nearest_hidden)
        trajectory_farest_vector = self.f_hid2sim(trajectory_farest_hidden)
        
        nearest_predict_distance = torch.exp(-torch.norm(anchor_nearest_vector - nearest_vector, p=2, dim=-1))
        farest_predict_distance = torch.exp(-torch.norm(anchor_farest_vector - farest_vector, p=2, dim=-1))
        
        nearest_predict_distance_all = torch.exp(-torch.norm(trajectory_vector - trajectory_nearest_vector, p=2, dim=-1))
        farest_predict_distance_all = torch.exp(-torch.norm(trajectory_vector - trajectory_farest_vector, p=2, dim=-1))

        return nearest_predict_distance, farest_predict_distance, nearest_predict_distance_all, farest_predict_distance_all, m_anchor_vector, m_nearest_vector, m_farest_vector

# model = Traj2Sim_encoder("LSTM", 2, 128)
#
# a_n_i = torch.rand([160, 10, 2])
# a_f_i = torch.rand([160, 10, 2])
# n_i = torch.rand([160, 10, 2])
# f_i = torch.rand([160, 10, 2])
# t_all = torch.rand([16, 10, 2])
# t_n_all = torch.rand([16, 10, 2])
# t_f_all = torch.rand([16, 10, 2])
# near_tuple = [[[1, 2, 3], [2, 4, 5], [1, 2, 3], [2, 4, 5]],
#               [[1, 2, 3], [2, 4, 5], [1, 2, 3], [2, 4, 5]],
#               [[1, 2, 3], [2, 4, 5], [1, 2, 3], [2, 4, 5]],
#               [[1, 2, 3], [2, 4, 5], [1, 2, 3], [2, 4, 5]],
#               [[1, 2, 3], [2, 4, 5], [1, 2, 3], [2, 4, 5]],
#               [[1, 2, 3], [2, 4, 5], [1, 2, 3], [2, 4, 5]],
#               [[1, 2, 3], [2, 4, 5], [1, 2, 3], [2, 4, 5]],
#               [[1, 2, 3], [2, 4, 5], [1, 2, 3], [2, 4, 5]],
#               [[1, 2, 3], [2, 4, 5], [1, 2, 3], [2, 4, 5]],
#               [[1, 2, 3], [2, 4, 5], [1, 2, 3], [2, 4, 5]],
#               [[1, 2, 3], [2, 4, 5], [1, 2, 3], [2, 4, 5]],
#               [[1, 2, 3], [2, 4, 5], [1, 2, 3], [2, 4, 5]],
#               [[1, 2, 3], [2, 4, 5], [1, 2, 3], [2, 4, 5]],
#               [[1, 2, 3], [2, 4, 5], [1, 2, 3], [2, 4, 5]],
#               [[1, 2, 3], [2, 4, 5], [1, 2, 3], [2, 4, 5]],
#               [[1, 2, 3], [2, 4, 5], [1, 2, 3], [2, 4, 5]]]
#
# targetsub1, targetsub2 = torch.rand ([160]), torch.rand ([160])
# targetall1, targetall2 = torch.rand ([16]), torch.rand ([16])
#
# out1, out2, out3, out4, out5, out6, out7 = model(a_n_i, a_f_i, n_i, f_i, t_all, t_n_all, t_f_all)
# # print(out1.shape)
# # print(out2.shape)
# # print(out3.shape)
# # print(out4.shape)
# # print(out5.shape)
#
#
# loss1_fun = Traj2SimLoss()
#
# # loss = loss1_fun(out1, out2, out3, out4, out5, out6, out7, targetsub1, targetsub2, targetall1, targetall2, near_tuple, near_tuple)
# #
# model.train ()
# loss1_fun.train ()
#
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
#
# for i in range(10000):
#     out1, out2, out3, out4, out5, out6, out7 = model(a_n_i, a_f_i, n_i, f_i, t_all, t_n_all, t_f_all)
#
#     loss = loss1_fun(out1, out2, out3, out4, out5, out6, out7, targetsub1, targetsub2, targetall1, targetall2, near_tuple, near_tuple)
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     print(loss)

lstm = nn.LSTM(2, 5, bias=True, batch_first=True)
input1 = torch.Tensor([[[1.2, 2.3]]])
input2 = torch.Tensor([[[1.2, 2.3], [0.0, 0.0]]])
input3 = torch.Tensor([[[1.2, 2.3], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
# print(input2.shape)
output1, (c1, hidden1) = lstm(input1)
output2, (c2, hidden2) = lstm(input2)
output3, (c3, hidden3) = lstm(input3)