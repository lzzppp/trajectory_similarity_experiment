import torch
import torch.nn as nn
from torch.nn import Module, LSTM
from torch.nn.functional import softmax
from torch.nn.init import xavier_uniform_
from src.model.transformer_raw_model import S2sTransformer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MaskedLSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0., bidirectional=False):
        super(MaskedLSTM, self).__init__()
        self.batch_first = batch_first
        self.f_sim = f_hid_sim(hidden_size)
        self.lstm = LSTM(input_size, hidden_size, num_layers=num_layers, bias=bias,
             batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, input_tensor, seq_lens):
        # input_tensor shape: batch_size*time_step*dim , seq_lens: (batch_size,)  when batch_first = True
        total_length = input_tensor.size(1) if self.batch_first else input_tensor.size(0)
        x_packed = pack_padded_sequence(input_tensor, seq_lens, batch_first=self.batch_first, enforce_sorted=False)
        y_lstm, hidden = self.lstm(x_packed)
        y_padded, length = pad_packed_sequence(y_lstm, batch_first=self.batch_first, total_length=total_length)
        y_padded_sim = self.f_sim(y_padded)
        hidden_sim = self.f_sim(hidden[0].squeeze())
        return y_padded_sim, hidden_sim

class trajectory_Distance_Loss(nn.Module):
    def __init__(self, sub=False, r=10.0, batch_size=32):
        super(trajectory_Distance_Loss, self).__init__ ()
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

        div_nearest = nearest_distance_target.view(-1, 1) - nearest_distance_predict.view(-1, 1)
        div_farest = farest_distance_target.view(-1, 1) - farest_distance_predict.view(-1, 1)
        div_nearest_sub = nearest_sub_distance_target.view(-1, 1) - nearest_sub_distance_predict.view(-1, 1)
        div_farest_sub = farest_sub_distance_target.view(-1, 1) - farest_sub_distance_predict.view(-1, 1)
        
        square_nearest = torch.mul(div_nearest.view(-1, 1), div_nearest.view(-1, 1))
        square_farest = torch.mul(div_farest.view(-1, 1), div_farest.view(-1, 1))
        square_nearest_sub = torch.mul(div_nearest_sub.view(-1, 1), div_nearest_sub.view(-1, 1))
        square_farest_sub = torch.mul(div_farest_sub.view(-1, 1), div_farest_sub.view(-1, 1))
        
        log_nearest = torch.mul(nearest_distance_target.view(-1, 1), square_nearest.view(-1, 1))
        log_farest = torch.mul(farest_distance_target.view(-1, 1), square_farest.view(-1, 1))
        log_nearest_sub = torch.mul(nearest_sub_distance_target.view(-1, 1), square_nearest_sub.view(-1, 1))
        log_farest_sub = torch.mul(farest_sub_distance_target.view(-1, 1), square_farest_sub.view(-1, 1))
        
        loss_nearest = torch.sum(log_nearest)
        loss_farest = torch.sum(log_farest)
        loss_nearest_sub = torch.sum(log_nearest_sub) / self.r
        loss_farest_sub = torch.sum(log_farest_sub) / self.r
        
        sub_trajectory_loss = sum([loss_nearest, loss_farest, loss_nearest_sub, loss_farest_sub])
        
        if self.sub:
            return sub_trajectory_loss / self.r
        else:
            return sub_trajectory_loss


class f_hid_sim (nn.Module):
    def __init__(self, hidden_size):
        super (f_hid_sim, self).__init__ ()
        self.linear_i = nn.Linear (hidden_size, hidden_size)
        self.linear_c = nn.Linear (hidden_size, hidden_size)
        self.linear_o = nn.Linear (hidden_size, hidden_size)
        self.init_weight ()
    
    def init_weight(self):
        nn.init.uniform_ (self.linear_i.weight, a=-1 / (128 ** 0.5), b=1 / (128 ** 0.5))
        nn.init.uniform_ (self.linear_o.weight, a=-1 / (128 ** 0.5), b=1 / (128 ** 0.5))
        nn.init.uniform_ (self.linear_c.weight, a=-1 / (128 ** 0.5), b=1 / (128 ** 0.5))
        nn.init.zeros_ (self.linear_i.bias)
        nn.init.zeros_ (self.linear_o.bias)
        nn.init.zeros_ (self.linear_c.bias)
    
    def forward(self, hi):
        C = torch.mul (torch.sigmoid (self.linear_i (hi)),
                       torch.tanh (self.linear_c (hi)))
        hi_ = torch.mul (torch.sigmoid (self.linear_o (hi)),
                         torch.tanh (C))
        output = hi_ + hi
        return output


class TRAJ_ZONE_EMBEDDING(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.embedding = S2sTransformer(params.n_words, num_encoder_layers=params.n_layers)
        self.reload_tert_encoder_params(params.path)
        
        # for p in self.parameters():
        # 	p.requires_grad = False
    
    def reload_tert_encoder_params(self, path):
        self.embedding.load_state_dict(torch.load(path))
    
    def forward(self, grid, pos, gps, mask):
        grid_embedding = self.embedding.embedding(grid)
        grid_pos_embedding = self.embedding.pos_encoder(grid_embedding.transpose(0, 1), pos)
        traj_zone_embedding = self.embedding.encoder(grid_pos_embedding, mask=None, src_key_padding_mask=mask)
        
        traj_input = torch.cat((traj_zone_embedding.transpose(0, 1), gps), dim=-1)
        return traj_input
        
class TERT(nn.Module):
    def __init__(self, tert_params):
        super().__init__()
        # self.traj_embedding = TRAJ_ZONE_EMBEDDING(tert_params)
        self.traj_embedding = nn.Embedding(tert_params.n_words, embedding_dim=128, padding_idx=0)
        # self.trajectory_encoder = nn.GRU(tert_params.embed_size + 2, tert_params.embed_size, batch_first=True)
        self.trajectory_encoder = MaskedLSTM(tert_params.embed_size + 2, tert_params.embed_size, batch_first=True)
    
    def forward(self, traj_gps, traj_mask, traj_grid, traj_grid_pos, traj_length, traj_anchor_sub_length, traj_farest_sub_length,
                traj_anchor_gps, traj_anchor_mask, traj_anchor_grid, traj_anchor_grid_pos, traj_anchor_length, anchor_sub_length,
                traj_farest_gps, traj_farest_mask, traj_farest_grid, traj_farest_grid_pos, traj_farest_length, farest_sub_length):
        
        # traj_zone_embedding = self.traj_embedding(traj_grid, traj_grid_pos, traj_gps, traj_mask)
        # traj_anchor_zone_embedding = self.traj_embedding(traj_anchor_grid, traj_anchor_grid_pos, traj_anchor_gps, traj_anchor_mask)
        # traj_farest_zone_embedding = self.traj_embedding(traj_farest_grid, traj_farest_grid_pos, traj_farest_gps, traj_farest_mask)
        
        traj_zone_embedding = self.traj_embedding(traj_grid)
        traj_anchor_zone_embedding = self.traj_embedding(traj_anchor_grid)
        traj_farest_zone_embedding = self.traj_embedding(traj_farest_grid)

        traj_input = torch.cat((traj_zone_embedding, traj_gps), dim=-1)
        traj_anchor_input = torch.cat((traj_anchor_zone_embedding, traj_anchor_gps), dim=-1)
        traj_farest_input = torch.cat((traj_farest_zone_embedding, traj_farest_gps), dim=-1)
        
        traj_tensor, traj_embedding = self.trajectory_encoder(traj_input, traj_length)
        anchor_tensor, traj_anchor_embedding = self.trajectory_encoder(traj_anchor_input, traj_anchor_length)
        farest_tensor, traj_farest_embedding = self.trajectory_encoder(traj_farest_input, traj_farest_length)
        
        traj_anchor_sub_embedding, traj_farest_sub_embedding, anchor_sub_embedding, farest_sub_embedding = [], [], [], []

        for i in range(traj_tensor.shape[0]):
            for sample_an_j, \
                sample_af_j, \
                sample_n_j, \
                sample_f_j in \
                    zip(traj_anchor_sub_length[i * 10:i * 10 + 10],
                        traj_farest_sub_length[i * 10:i * 10 + 10],
                        anchor_sub_length[i * 10:i * 10 + 10],
                        farest_sub_length[i * 10:i * 10 + 10]):
                
                traj_anchor_sub_embedding.append(traj_tensor[i, sample_an_j - 1, :].unsqueeze(0))
                traj_farest_sub_embedding.append(traj_tensor[i, sample_af_j - 1, :].unsqueeze(0))
                anchor_sub_embedding.append(anchor_tensor[i, sample_n_j - 1, :].unsqueeze(0))
                farest_sub_embedding.append(farest_tensor[i, sample_f_j - 1, :].unsqueeze(0))

        traj_anchor_sub_embedding = torch.cat(traj_anchor_sub_embedding)
        traj_farest_sub_embedding = torch.cat(traj_farest_sub_embedding)
        anchor_sub_embedding = torch.cat(anchor_sub_embedding)
        farest_sub_embedding = torch.cat(farest_sub_embedding)
        
        anchor_distance = torch.exp(-torch.norm(traj_embedding - traj_anchor_embedding, p=2, dim=-1))
        farest_distance = torch.exp(-torch.norm(traj_embedding - traj_farest_embedding, p=2, dim=-1))
        
        anchor_sub_distance = torch.exp(-torch.norm(traj_anchor_sub_embedding - anchor_sub_embedding, p=2, dim=-1))
        farest_sub_distance = torch.exp(-torch.norm(traj_farest_sub_embedding - farest_sub_embedding, p=2, dim=-1))
        
        return anchor_distance, farest_distance, \
               anchor_sub_distance, farest_sub_distance
