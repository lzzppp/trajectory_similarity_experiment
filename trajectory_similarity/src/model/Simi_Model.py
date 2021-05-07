import torch
import torch.nn as nn
import torch.nn.functional as F


class trajectory_Distance_Loss(nn.Module):
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
    
    def forward(self, anchor_near_tensor, anchor_far_tensor,
                      near_anchor_indexs, far_anchor_indexs, near_indexs, far_indexs,
                      near_target, far_target):
        near_anchor_embedding, far_anchor_embedding, near_embedding, far_embedding = [], [], [], []
        for i in range(anchor_near_tensor.shape[1]):
            near_anchor_embedding.extend([anchor_near_tensor[nai, i, :].unsqueeze(0) for nai in near_anchor_indexs[i]])
            far_anchor_embedding.extend([anchor_far_tensor[fai, i, :].unsqueeze(0) for fai in far_anchor_indexs[i]])
            near_embedding.extend([anchor_near_tensor[ni, i, :].unsqueeze(0) for ni in near_indexs[i]])
            far_embedding.extend([anchor_far_tensor[fi, i, :].unsqueeze(0) for fi in far_indexs[i]])
        near_anchor_embedding = torch.cat((near_anchor_embedding))
        far_anchor_embedding = torch.cat((far_anchor_embedding))
        near_embedding = torch.cat((near_embedding))
        far_embedding = torch.cat((far_embedding))
        
        near_predict = torch.exp(-F.pairwise_distance(near_anchor_embedding, near_embedding, p=2))
        far_predict = torch.exp(-F.pairwise_distance(far_anchor_embedding, far_embedding, p=2))
        
        near_loss = near_predict - near_target
        far_loss = far_predict - far_target

        near_loss_square = torch.mul(near_loss, near_loss)
        far_loss_square = torch.mul(far_loss, far_loss)
        
        output = torch.sum(near_loss_square) + torch.sum(far_loss_square)
        
        return output
        

class TranSimiModel(nn.Module):
    def __init__(self, d_model=128, n_head=8, dim_feedforward=512, dropout=0.1, num_encoder_layers=4):
        super(TranSimiModel, self).__init__ ()
        self.embedding = nn.Linear(2, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, activation="gelu")
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
    def forward(self, anchor_near_trajectory_pair, anchor_far_trajectory_pair,
                anchor_near_attention_mask, anchor_far_attention_mask,
                anchor_near_padding_mask, anchor_far_padding_mask):
        
        anchor_near_embedding_pair = self.embedding(anchor_near_trajectory_pair).transpose(0, 1)
        anchor_far_embedding_pair = self.embedding(anchor_far_trajectory_pair).transpose(0, 1)
        
        anchor_near_trajectory_output = self.encoder(anchor_near_embedding_pair, mask=anchor_near_attention_mask, src_key_padding_mask=anchor_near_padding_mask)
        anchor_far_trajectory_output = self.encoder(anchor_far_embedding_pair, mask=anchor_far_attention_mask, src_key_padding_mask=anchor_far_padding_mask)
        
        return anchor_near_trajectory_output, anchor_far_trajectory_output