
import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
import functools
import pdb

class MultiheadAttention(nn.Module):
    def __init__(self,
                 flag=0,
                 hidden_size=128,
                 num_attention_heads=8,
                 attention_probs_dropout_prob=0.1,
                 share_att_key=False):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        _attention_head_size = int(hidden_size / num_attention_heads)
        self.attention_head_size = _attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.flag = flag
        if flag == 0:
            self.query_proj = nn.Linear(hidden_size, self.all_head_size)
            self.out_proj = nn.Linear(hidden_size, self.all_head_size)
        elif flag == 1:
            self.query_proj = nn.Linear(hidden_size, self.all_head_size)
            self.key_proj = nn.Linear(hidden_size, self.all_head_size)
            self.out_proj = nn.Linear(hidden_size, self.all_head_size)
        elif flag == 2:
            self.query_proj = nn.Linear(hidden_size, self.all_head_size)
            self.key_proj = nn.Linear(hidden_size, self.all_head_size)
            self.value_proj = nn.Linear(hidden_size, self.all_head_size)
        
        self.share_att_key = share_att_key
        self.pos_att_type = ['t2p', 'p2t'] # c2p|p2c

        # self.dropout = StableDropout(attention_probs_dropout_prob)
        # self._register_load_state_dict_pre_hook(self._pre_load_hook)

    def transpose_for_scores(self, x, attention_heads):
        new_x_shape = x.size()[:-1] + (attention_heads, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))

    def forward(self, query_states, key_states, value_states, attention_mask=None, key_padding_mask=None, need_weights=True):
        tgt_len, bsz, embed_dim = query_states.size()
        
        # if self.flag == 0:
        #     query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        #     key_layer = self.transpose_for_scores(self.query_proj(key_states), self.num_attention_heads)
        #     value_layer = self.transpose_for_scores(self.query_proj(value_states), self.num_attention_heads)
        # else:
        #     query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        #     key_layer = self.transpose_for_scores(self.key_proj(key_states), self.num_attention_heads)
        #     value_layer = self.transpose_for_scores(self.key_proj(value_states), self.num_attention_heads)
        
        if self.flag == 0:
            q = self.query_proj(query_states).contiguous().view(tgt_len, bsz * self.num_attention_heads, self.attention_head_size).transpose(0, 1)
            k = self.query_proj(key_states).contiguous().view(-1, bsz * self.num_attention_heads, self.attention_head_size).transpose(0, 1)
            v = self.query_proj(value_states).contiguous().view(-1, bsz * self.num_attention_heads, self.attention_head_size).transpose(0, 1)
        else:
            q = self.query_proj(query_states).contiguous().view(tgt_len, bsz * self.num_attention_heads, self.attention_head_size).transpose(0, 1)
            k = self.key_proj(key_states).contiguous().view(-1, bsz * self.num_attention_heads, self.attention_head_size).transpose(0, 1)
            v = self.key_proj(value_states).contiguous().view(-1, bsz * self.num_attention_heads, self.attention_head_size).transpose(0, 1)

        src_len = k.size(1)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if 't2p' in self.pos_att_type:
            scale_factor += 1
        if 'p2t' in self.pos_att_type:
            scale_factor += 1

        scale = math.sqrt(q.size(-1)*scale_factor)
        attention_scores = torch.bmm(q, k.transpose(1, 2))
        # attention_scores = torch.bmm(q, k.transpose(-1, -2))/scale

        attention_scores = attention_scores.view(-1, self.num_attention_heads, attention_scores.size(-2), attention_scores.size(-1))
        
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                attention_scores.masked_fill_(attention_mask, float('-inf'))
            else:
                attention_scores += attention_mask
        
        if key_padding_mask is not None:
            attention_scores = attention_scores.view(bsz, self.num_attention_heads, tgt_len, src_len)
            attention_scores = attention_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attention_scores = attention_scores.view(bsz * self.num_attention_heads, tgt_len, src_len)
        
        # bxhxlxd
        # _attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
        _attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = F.dropout(_attention_probs)
        attn_output = torch.bmm(attention_probs, v)
        assert list(attn_output.size ()) == [bsz * self.num_attention_heads, tgt_len, self.attention_head_size]
        attn_output = attn_output.transpose (0, 1).contiguous ().view (tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        if need_weights:
            # average attention weights over heads
            attention_probs = attention_probs.view (bsz, self.num_attention_heads, tgt_len, src_len)
            return attn_output, attention_probs.sum (dim=1) / self.num_attention_heads
        else:
            return attn_output, None

    def disentangled_attention_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(q, key_layer.size(-2), bucket_size = self.position_buckets, max_position = self.max_relative_positions)
        if relative_pos.dim()==2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim()==3:
            relative_pos = relative_pos.unsqueeze(1)
        # bxhxqxk
        elif relative_pos.dim()!=4:
            raise ValueError(f'Relative postion ids must be of dim 2 or 3 or 4. {relative_pos.dim()}')

        att_span = self.pos_ebd_size
        relative_pos = relative_pos.long().to(query_layer.device)

        rel_embeddings = rel_embeddings[self.pos_ebd_size - att_span:self.pos_ebd_size + att_span, :].unsqueeze(0) #.repeat(query_layer.size(0)//self.num_attention_heads, 1, 1)
        if self.share_att_key:
            pos_query_layer = self.transpose_for_scores(self.query_proj(rel_embeddings), self.num_attention_heads)\
                .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
            pos_key_layer = self.transpose_for_scores(self.key_proj(rel_embeddings), self.num_attention_heads)\
                .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
        else:
            if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                pos_key_layer = self.transpose_for_scores(self.pos_key_proj(rel_embeddings), self.num_attention_heads)\
                    .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
            if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                pos_query_layer = self.transpose_for_scores(self.pos_query_proj(rel_embeddings), self.num_attention_heads)\
                    .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)

        return score

    def _pre_load_hook(self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs):
        self_state = self.state_dict()
        if ((prefix + 'query_proj.weight') not in state_dict) and ((prefix + 'in_proj.weight') in state_dict):
          v1_proj = state_dict[prefix+'in_proj.weight']
          v1_proj = v1_proj.unsqueeze(0).reshape(self.num_attention_heads, -1, v1_proj.size(-1))
          q,k,v=v1_proj.chunk(3, dim=1)
          state_dict[prefix + 'query_proj.weight'] = q.reshape(-1, v1_proj.size(-1))
          state_dict[prefix + 'key_proj.weight'] = k.reshape(-1, v1_proj.size(-1))
          state_dict[prefix + 'key_proj.bias'] = self_state['key_proj.bias']
          state_dict[prefix + 'value_proj.weight'] = v.reshape(-1, v1_proj.size(-1))
          v1_query_bias = state_dict[prefix + 'q_bias']
          state_dict[prefix + 'query_proj.bias'] = v1_query_bias
          v1_value_bias = state_dict[prefix +'v_bias']
          state_dict[prefix + 'value_proj.bias'] = v1_value_bias

          v1_pos_key_proj = state_dict[prefix + 'pos_proj.weight']
          state_dict[prefix + 'pos_key_proj.weight'] = v1_pos_key_proj
          v1_pos_query_proj = state_dict[prefix + 'pos_q_proj.weight']
          state_dict[prefix + 'pos_query_proj.weight'] = v1_pos_query_proj
          v1_pos_query_proj_bias = state_dict[prefix + 'pos_q_proj.bias']
          state_dict[prefix + 'pos_query_proj.bias'] = v1_pos_query_proj_bias
          state_dict[prefix + 'pos_key_proj.bias'] = self_state['pos_key_proj.bias']

          del state_dict[prefix + 'in_proj.weight']
          del state_dict[prefix + 'q_bias']
          del state_dict[prefix + 'v_bias']
          del state_dict[prefix + 'pos_proj.weight']
          del state_dict[prefix + 'pos_q_proj.weight']
          del state_dict[prefix + 'pos_q_proj.bias']