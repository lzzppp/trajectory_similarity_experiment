import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.Component import S2sTransformer, f_hid_sim

class TCAN(nn.Module):
	def __init__(self, params):
		super(TCAN, self).__init__()
		self.encoder_decoder = S2sTransformer(params.n_words)
		self.transform = f_hid_sim(128)
		# self.load_params()
	
	def forward(self, a_gps_data, n_gps_data, f_gps_data,
	            a_mask_data, n_mask_data, f_mask_data,
	            a_position_data, n_position_data, f_position_data,
	            a_attention_mask_data, n_attention_mask_data, f_attention_mask_data,
	            anchor_len, near_len, far_len):
		near_anchor_output = self.encoder_decoder(a_gps_data, n_gps_data,
		                                          a_position_data, n_position_data,
		                                          tgt_mask=n_attention_mask_data, src_key_padding_mask=a_mask_data,
		                                          tgt_key_padding_mask=n_mask_data, memory_key_padding_mask=a_mask_data)
		anchor_near_output = self.encoder_decoder(n_gps_data, a_gps_data,
		                                          n_position_data, a_position_data,
		                                          tgt_mask=a_attention_mask_data, src_key_padding_mask=n_mask_data,
		                                          tgt_key_padding_mask=a_mask_data, memory_key_padding_mask=n_mask_data)
		far_anchor_output = self.encoder_decoder(a_gps_data, f_gps_data,
		                                         a_position_data, f_position_data,
		                                         tgt_mask=f_attention_mask_data, src_key_padding_mask=a_mask_data,
		                                         tgt_key_padding_mask=f_mask_data, memory_key_padding_mask=a_mask_data)
		anchor_far_output = self.encoder_decoder(f_gps_data, a_gps_data,
		                                         f_position_data, a_position_data,
		                                         tgt_mask=a_attention_mask_data, src_key_padding_mask=f_mask_data,
		                                         tgt_key_padding_mask=a_mask_data, memory_key_padding_mask=f_mask_data)
		
		# near_anchor_output = self.transform(near_anchor_output)
		# anchor_near_output = self.transform(anchor_near_output)
		# far_anchor_output = self.transform(far_anchor_output)
		# anchor_far_output = self.transform(anchor_far_output)
		
		near_anchor_embedding, anchor_near_embedding, far_anchor_embedding, anchor_far_embedding = [], [], [], []
		idx = 0
		for alen, nlen, flen in zip(anchor_len, near_len, far_len):
			near_anchor_embedding.append(near_anchor_output[idx, nlen - 1, :].unsqueeze(0))
			anchor_near_embedding.append(anchor_near_output[idx, alen - 1, :].unsqueeze(0))
			far_anchor_embedding.append(far_anchor_output[idx, flen - 1, :].unsqueeze(0))
			anchor_far_embedding.append(anchor_far_output[idx, alen - 1, :].unsqueeze(0))
			idx += 1
		
		near_anchor_embedding = torch.cat((near_anchor_embedding))
		anchor_near_embedding = torch.cat((anchor_near_embedding))
		far_anchor_embedding = torch.cat((far_anchor_embedding))
		anchor_far_embedding = torch.cat((anchor_far_embedding))
		
		near_anchor_embedding = self.transform(near_anchor_embedding)
		anchor_near_embedding = self.transform(anchor_near_embedding)
		far_anchor_embedding = self.transform(far_anchor_embedding)
		anchor_far_embedding = self.transform(anchor_far_embedding)
		
		pred_near_distance = torch.exp(-F.pairwise_distance(anchor_near_embedding, near_anchor_embedding, p=2))
		pred_far_distance = torch.exp(-F.pairwise_distance(anchor_far_embedding, far_anchor_embedding, p=2))
		
		# pred_near_distance = torch.cosine_similarity(anchor_near_embedding, near_anchor_embedding, dim=-1)
		# pred_far_distance = torch.cosine_similarity(anchor_far_embedding, far_anchor_embedding, dim=-1)
		
		return pred_near_distance, pred_far_distance