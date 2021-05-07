import torch
import torch.nn as nn
from torch.nn import Module, LSTM
from torch.nn.functional import softmax
from torch.nn.init import xavier_uniform_
# from src.model.transformer_raw_model import S2sTransformer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

max_pos = 502

class LearnedPositionEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        # super().__init__(max_len, d_model)
        super ().__init__ ()
        self.dropout = nn.Dropout(p = dropout)
        self.embedding = nn.Embedding(max_pos, d_model, padding_idx=0)

    def forward(self, x, pos):
        weight = self.embedding(pos).transpose(0, 1)
        x = x + weight
        return self.dropout(x)

class MaskedLSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0., bidirectional=False):
        super(MaskedLSTM, self).__init__()
        self.batch_first = batch_first
        self.lstm = LSTM(input_size, hidden_size, num_layers=num_layers, bias=bias,
             batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, input_tensor, seq_lens):
        # input_tensor shape: batch_size*time_step*dim , seq_lens: (batch_size,)  when batch_first = True
        total_length = input_tensor.size(1) if self.batch_first else input_tensor.size(0)
        x_packed = pack_padded_sequence(input_tensor, seq_lens, batch_first=self.batch_first, enforce_sorted=False)
        y_lstm, hidden = self.lstm(x_packed)
        y_padded, length = pad_packed_sequence(y_lstm, batch_first=self.batch_first, total_length=total_length)
        return y_padded, hidden[0].squeeze()

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

class TRAJ_ZONE_EMBEDDING(nn.Module):
    def __init__(self, params, d_model=128, nhead=8, dim_feedforward=512, dropout=0.1, num_encoder_layers=4):
        super().__init__()
        # self.embedding = S2sTransformer(params.n_words, num_encoder_layers=params.n_layers)

        self.embedding = nn.Embedding(params.n_words, d_model)
        self.pos_encoder = LearnedPositionEncoding(d_model)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.reload_tert_encoder_params(params.path)
    
    def reload_tert_encoder_params(self, path):
        pretrained_dict = torch.load(path)
        model_dict = self.state_dict()
        keys = []
        for k, v in pretrained_dict.items ():
            if 'decoder' not in k and 'output' not in k:
                keys.append(k)

        i = 0
        for k, v in model_dict.items():
            if v.size() == pretrained_dict[keys[i]].size():
                model_dict[k] = pretrained_dict[keys[i]]
                i += 1
        self.load_state_dict(model_dict)
    
    def forward(self, grid1, pos1, gps1, mask1,
                      grid2, pos2, gps2, mask2):
        len1 = grid1.shape[1]
        grid = torch.cat((grid1, grid2), dim=1)
        mask = torch.cat((mask1, mask2), dim=1)
        gps = torch.cat((gps1, gps2), dim=1)
        # pos = torch.arange(1, grid.shape[-1] + 1, device=torch.device("cuda:0")).unsqueeze (0).expand(grid.shape[0], grid.shape[1])
        grid_embedding = self.embedding(grid)
        grid_pos_embedding = self.pos_encoder(grid_embedding.transpose(0, 1), pos2)
        traj_zone_embedding = self.encoder(grid_pos_embedding, mask=None, src_key_padding_mask=mask)
        
        traj_input = torch.cat((traj_zone_embedding.transpose(0, 1), gps), dim=-1)
        traj_input1 = traj_input[:, :len1, :]
        traj_input2 = traj_input[:, len1:, :]
        return traj_input1, traj_input2
        
class TERT(nn.Module):
    def __init__(self, tert_params):
        super().__init__()
        self.traj_embedding = TRAJ_ZONE_EMBEDDING(tert_params)
        self.trajectory_encoder = MaskedLSTM(tert_params.embed_size + 2, tert_params.embed_size, batch_first=True)
    
    def forward(self, traj_gps, traj_mask, traj_grid, traj_grid_pos, traj_length, traj_anchor_sub_length, traj_farest_sub_length,
                traj_anchor_gps, traj_anchor_mask, traj_anchor_grid, traj_anchor_grid_pos, traj_anchor_length, anchor_sub_length,
                traj_farest_gps, traj_farest_mask, traj_farest_grid, traj_farest_grid_pos, traj_farest_length, farest_sub_length):
        
        traj_zone_embedding_a, traj_anchor_zone_embedding = self.traj_embedding(traj_grid, traj_grid_pos, traj_gps, traj_mask,
                                                                                traj_anchor_grid, traj_anchor_grid_pos, traj_anchor_gps, traj_anchor_mask)
        # traj_anchor_zone_embedding = self.traj_embedding(traj_anchor_grid, traj_anchor_grid_pos, traj_anchor_gps, traj_anchor_mask)
        traj_zone_embedding_f, traj_farest_zone_embedding = self.traj_embedding(traj_grid, traj_grid_pos, traj_gps, traj_mask,
                                                                                traj_farest_grid, traj_farest_grid_pos, traj_farest_gps, traj_farest_mask)
        
        traj_tensor_a, traj_embedding_a = self.trajectory_encoder(traj_zone_embedding_a, traj_length)
        traj_tensor_f, traj_embedding_f = self.trajectory_encoder(traj_zone_embedding_f, traj_length)
        anchor_tensor, traj_anchor_embedding = self.trajectory_encoder(traj_anchor_zone_embedding, traj_anchor_length)
        farest_tensor, traj_farest_embedding = self.trajectory_encoder(traj_farest_zone_embedding, traj_farest_length)

        traj_anchor_sub_embedding, traj_farest_sub_embedding, anchor_sub_embedding, farest_sub_embedding = [], [], [], []

        for i in range(traj_tensor_a.shape[0]):
            for sample_an_j, \
                sample_af_j, \
                sample_n_j, \
                sample_f_j in \
                    zip(traj_anchor_sub_length[i * 10:i * 10 + 10],
                        traj_farest_sub_length[i * 10:i * 10 + 10],
                        anchor_sub_length[i * 10:i * 10 + 10],
                        farest_sub_length[i * 10:i * 10 + 10]):
                
                traj_anchor_sub_embedding.append(traj_tensor_a[i, sample_an_j - 1, :].unsqueeze(0))
                traj_farest_sub_embedding.append(traj_tensor_f[i, sample_af_j - 1, :].unsqueeze(0))
                anchor_sub_embedding.append(anchor_tensor[i, sample_n_j - 1, :].unsqueeze(0))
                farest_sub_embedding.append(farest_tensor[i, sample_f_j - 1, :].unsqueeze(0))

        traj_anchor_sub_embedding = torch.cat(traj_anchor_sub_embedding)
        traj_farest_sub_embedding = torch.cat(traj_farest_sub_embedding)
        anchor_sub_embedding = torch.cat(anchor_sub_embedding)
        farest_sub_embedding = torch.cat(farest_sub_embedding)
        
        anchor_distance = torch.exp(-torch.norm(traj_embedding_a - traj_anchor_embedding, p=2, dim=-1))
        farest_distance = torch.exp(-torch.norm(traj_embedding_f - traj_farest_embedding, p=2, dim=-1))
        
        anchor_sub_distance = torch.exp(-torch.norm(traj_anchor_sub_embedding - anchor_sub_embedding, p=2, dim=-1))
        farest_sub_distance = torch.exp(-torch.norm(traj_farest_sub_embedding - farest_sub_embedding, p=2, dim=-1))
        
        return anchor_distance, farest_distance, \
               anchor_sub_distance, farest_sub_distance
