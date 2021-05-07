from src.model.sam_cells import SAM_LSTMCell,SAM_GRUCell
from torch.nn import Module
from src.model import config

import torch.autograd as autograd
import torch.nn.functional as F
import torch
import torch.nn as nn

class trajectory_Distance_Loss (nn.Module):
    def __init__(self, sub=False, r=10.0, batch_size=32):
        super (trajectory_Distance_Loss, self).__init__ ()
        self.r = r
        self.sub = sub
        self.batch_size = batch_size
    
    # self.weight_of_sub_trajectory = self.get_weight_vector ()
    
    def get_weight_vector(self):
        weight_vector = []
        for i in range (self.batch_size):
            weight_vector.append (1)
            weight_vector.extend ([1.0 / 5.0] * self.r)
        return torch.FloatTensor (weight_vector)
    
    def forward(self, nearest_distance_predict, farest_distance_predict,
                nearest_distance_target, farest_distance_target,
                nearest_sub_distance_predict, farest_sub_distance_predict,
                nearest_sub_distance_target, farest_sub_distance_target):
        
        div_nearest = nearest_distance_target.view (-1, 1) - nearest_distance_predict.view (-1, 1)
        div_farest = farest_distance_target.view (-1, 1) - farest_distance_predict.view (-1, 1)
        div_nearest_sub = nearest_sub_distance_target.view (-1, 1) - nearest_sub_distance_predict.view (-1, 1)
        div_farest_sub = farest_sub_distance_target.view (-1, 1) - farest_sub_distance_predict.view (-1, 1)
        
        square_nearest = torch.mul (div_nearest.view (-1, 1), div_nearest.view (-1, 1))
        square_farest = torch.mul (div_farest.view (-1, 1), div_farest.view (-1, 1))
        square_nearest_sub = torch.mul (div_nearest_sub.view (-1, 1), div_nearest_sub.view (-1, 1))
        square_farest_sub = torch.mul (div_farest_sub.view (-1, 1), div_farest_sub.view (-1, 1))
        
        log_nearest = torch.mul (nearest_distance_target.view (-1, 1), square_nearest.view (-1, 1))
        log_farest = torch.mul (farest_distance_target.view (-1, 1), square_farest.view (-1, 1))
        log_nearest_sub = torch.mul (nearest_sub_distance_target.view (-1, 1), square_nearest_sub.view (-1, 1))
        log_farest_sub = torch.mul (farest_sub_distance_target.view (-1, 1), square_farest_sub.view (-1, 1))
        
        loss_nearest = torch.sum (log_nearest)
        loss_farest = torch.sum (log_farest)
        loss_nearest_sub = torch.sum (log_nearest_sub) / self.r
        loss_farest_sub = torch.sum (log_farest_sub) / self.r
        
        sub_trajectory_loss = sum ([loss_nearest, loss_farest, loss_nearest_sub, loss_farest_sub])
        
        if self.sub:
            return sub_trajectory_loss / self.r
        else:
            return sub_trajectory_loss

class RNNEncoder(Module):
    def __init__(self, input_size, hidden_size, grid_size, stard_LSTM= False, incell = True):
        super(RNNEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.stard_LSTM = stard_LSTM
        if self.stard_LSTM:
            if config.recurrent_unit=='GRU':
                self.cell = torch.nn.GRUCell(input_size - 2, hidden_size).cuda()
            elif config.recurrent_unit=='SimpleRNN':
                self.cell = torch.nn.RNNCell(input_size - 2, hidden_size).cuda()
            else:
                self.cell = torch.nn.LSTMCell(input_size - 2, hidden_size).cuda()
        else:
            if config.recurrent_unit=='GRU':
                self.cell = SAM_GRUCell(input_size, hidden_size, grid_size, incell=incell).cuda()
            elif config.recurrent_unit=='SimpleRNN':
                self.cell = SpatialRNNCell(input_size, hidden_size, grid_size, incell=incell).cuda()
            else:
                self.cell = SAM_LSTMCell(input_size, hidden_size, grid_size, incell=incell).cuda()

        print(self.cell)
        print('in cell update: {}'.format(incell))
        # self.cell = torch.nn.LSTMCell(input_size-2, hidden_size).cuda()
    def forward(self, inputs_a, initial_state = None):
        inputs, inputs_len = inputs_a
        time_steps = inputs.size(1)
        out = None
        if config.recurrent_unit == 'GRU' or config.recurrent_unit == 'SimpleRNN':
            out = initial_state
        else:
            out, state = initial_state

        outputs = []
        for t in range(time_steps):
            if self.stard_LSTM:
                cell_input = inputs[:, t, :][:,:-2]
            else:
                cell_input = inputs[:, t, :]
            if config.recurrent_unit == 'GRU' or config.recurrent_unit == 'SimpleRNN':
                out = self.cell(cell_input, out)
            else:
                out, state = self.cell(cell_input, (out, state))
            outputs.append(out.unsqueeze(1))
        mask_out = []
        for b, v in enumerate(inputs_len):
            mask_out.append(outputs[v-1][b,:].view(1,-1))
        return torch.cat(mask_out, dim=0), torch.cat(outputs, dim=1)

    def batch_grid_state_gates(self, inputs_a, initial_state = None):
        inputs, inputs_len = inputs_a
        time_steps = inputs.size(1)
        out, state = initial_state
        outputs = []
        gates_out_all = []
        batch_weight_ih = autograd.Variable(self.cell.weight_ih.data, requires_grad=False).cuda()
        batch_weight_hh = autograd.Variable(self.cell.weight_hh.data, requires_grad=False).cuda()
        batch_bias_ih = autograd.Variable(self.cell.bias_ih.data, requires_grad=False).cuda()
        batch_bias_hh = autograd.Variable(self.cell.bias_hh.data, requires_grad=False).cuda()
        for t in range(time_steps):
            # cell_input = inputs[:, t, :][:,:-2]
            cell_input = inputs[:, t, :]
            self.cell.update_memory(cell_input, (out, state),
                                    batch_weight_ih, batch_weight_hh,
                                    batch_bias_ih, batch_bias_hh)

class NeuTraj_Network(Module):
    def __init__(self,input_size, target_size, grid_size, batch_size, sampling_num, stard_LSTM = False, incell = True):
        super(NeuTraj_Network, self).__init__()
        self.input_size = input_size
        self.target_size = target_size
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.sampling_num = sampling_num
        if config.recurrent_unit=='GRU' or config.recurrent_unit=='SimpleRNN':
            self.hidden = autograd.Variable(torch.zeros(self.batch_size * (1 + self.sampling_num), self.target_size),
                                            requires_grad=False).cuda()
        else:
            self.hidden = (autograd.Variable(torch.zeros(self.batch_size * (1 + self.sampling_num), self.target_size),requires_grad=False).cuda(),
                           autograd.Variable(torch.zeros(self.batch_size * (1 + self.sampling_num), self.target_size),requires_grad=False).cuda())
        self.rnn = RNNEncoder(self.input_size, self.target_size, self.grid_size, stard_LSTM= stard_LSTM,
                              incell = incell).cuda()

    def forward(self, traj_input, traj_length, traj_anchor_sub_length, traj_farest_sub_length,
                traj_anchor_input, traj_anchor_length, anchor_sub_length,
                traj_farest_input, traj_farest_length, farest_sub_length):

        traj_embedding, traj_tensor = self.rnn([autograd.Variable(traj_input, requires_grad=False).cuda(), traj_length], self.hidden)
        traj_anchor_embedding, anchor_tensor = self.rnn([autograd.Variable(traj_anchor_input,requires_grad=False).cuda(), traj_anchor_length], self.hidden)
        traj_farest_embedding, farest_tensor = self.rnn([autograd.Variable(traj_farest_input,requires_grad=False).cuda(), traj_farest_length], self.hidden)

        traj_anchor_sub_embedding, traj_farest_sub_embedding, anchor_sub_embedding, farest_sub_embedding = [], [], [], []
        
        for i in range (traj_tensor.shape[0]):
            for sample_an_j, \
                sample_af_j, \
                sample_n_j, \
                sample_f_j in \
                    zip (traj_anchor_sub_length[i * 10:i * 10 + 10],
                         traj_farest_sub_length[i * 10:i * 10 + 10],
                         anchor_sub_length[i * 10:i * 10 + 10],
                         farest_sub_length[i * 10:i * 10 + 10]):
                traj_anchor_sub_embedding.append (traj_tensor[i, sample_an_j - 1, :].unsqueeze(0))
                traj_farest_sub_embedding.append (traj_tensor[i, sample_af_j - 1, :].unsqueeze(0))
                anchor_sub_embedding.append (anchor_tensor[i, sample_n_j - 1, :].unsqueeze(0))
                farest_sub_embedding.append (farest_tensor[i, sample_f_j - 1, :].unsqueeze(0))

        traj_anchor_sub_embedding = torch.cat (traj_anchor_sub_embedding)
        traj_farest_sub_embedding = torch.cat (traj_farest_sub_embedding)
        anchor_sub_embedding = torch.cat (anchor_sub_embedding)
        farest_sub_embedding = torch.cat (farest_sub_embedding)

        anchor_distance = torch.exp(-torch.norm(traj_embedding - traj_anchor_embedding, p=2, dim=-1))
        farest_distance = torch.exp(-torch.norm(traj_embedding - traj_farest_embedding, p=2, dim=-1))

        anchor_sub_distance = torch.exp(-torch.norm(traj_anchor_sub_embedding - anchor_sub_embedding, p=2, dim=-1))
        farest_sub_distance = torch.exp(-torch.norm(traj_farest_sub_embedding - farest_sub_embedding, p=2, dim=-1))

        return anchor_distance, farest_distance, \
               anchor_sub_distance, farest_sub_distance


    def spatial_memory_update(self, inputs_arrays, inputs_len_arrays):
        batch_traj_input = torch.Tensor(inputs_arrays[3])
        batch_traj_len = inputs_len_arrays[3]
        batch_hidden = (autograd.Variable(torch.zeros(len(batch_traj_len), self.target_size),requires_grad=False).cuda(),
                        autograd.Variable(torch.zeros(len(batch_traj_len), self.target_size),requires_grad=False).cuda())
        self.rnn.batch_grid_state_gates([autograd.Variable(batch_traj_input).cuda(), batch_traj_len],batch_hidden)