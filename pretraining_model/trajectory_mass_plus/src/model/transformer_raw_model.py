
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.nn.init import xavier_uniform_

device=torch.device("cuda:0")
src_pad_idx=0
trg_pad_idx=0
max_pos=252

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

class S2sTransformer(nn.Module):

    def __init__(self,vocab_size,d_model=128,nhead=8,num_encoder_layers=6,
                 num_decoder_layers=6,dim_feedforward=512,dropout=0.1):
        super(S2sTransformer,self).__init__()

        # Preprocess
        self.embedding = nn.Embedding(vocab_size,d_model)
        
        self.pos_encoder = LearnedPositionEncoding(d_model)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward,dropout,activation="gelu")
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer,num_encoder_layers,encoder_norm)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model,nhead,dim_feedforward,dropout,activation="gelu")
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer,num_decoder_layers,decoder_norm)
        self.output_layer = nn.Linear(d_model,vocab_size)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead


    def forward(self, src, tgt, src_pos, tgt_pos, tgt_mask_raw, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        # word embedding
        src = self.embedding(src)
        tgt = self.embedding(tgt)

        # shape check
        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        # position encoding
        src = self.pos_encoder(src, src_pos)
        tgt = self.pos_encoder(tgt, tgt_pos)

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask).transpose(0, 1)
        predict_output = output[(~tgt_mask_raw).unsqueeze(-1).expand_as(output)].view(-1, self.d_model)
        output_ = self.output_layer(predict_output)
        # print(output_.shape)
        return output_
        # return softmax(output_, dim=-1)

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
                
# model = S2sTransformer(10)
# src = torch.LongTensor([[1, 3, 4, 0], [1, 5, 6, 7]]).transpose(0, 1)
# tgt = torch.LongTensor([[2, 3, 0], [2, 5, 6]]).transpose(0, 1)
# src_mask = torch.BoolTensor([[False, False, False, True], [False, False, False, False]])
# tgt_mask = torch.BoolTensor([[False, False, True], [False, False, False]])
# memory_mask = torch.Tensor([[float(0.0), float('-inf'), float('-inf')],
#                             [float(0.0), float(0.0), float('-inf')],
#                             [float(0.0), float(0.0), float(0.0)]])
#
# print(model.forward(src, tgt, src_mask=None, tgt_mask=memory_mask, memory_mask=None, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask, memory_key_padding_mask=src_mask).shape)
