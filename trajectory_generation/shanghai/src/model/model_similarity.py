import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.nn.init import xavier_uniform_
from src.model.transformer_raw_model import S2sTransformer


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
	                  nearest_distance_target, farest_distance_target):

		div_nearest = nearest_distance_target.view (-1, 1) - nearest_distance_predict.view (-1, 1)
		div_farest = farest_distance_target.view (-1, 1) - farest_distance_predict.view (-1, 1)
		
		square_nearest = torch.mul (div_nearest.view (-1, 1), div_nearest.view (-1, 1))
		square_farest = torch.mul (div_farest.view (-1, 1), div_farest.view (-1, 1))
		
		log_nearest = torch.mul (nearest_distance_target.view (-1, 1), square_nearest.view (-1, 1))
		log_farest = torch.mul (farest_distance_target.view (-1, 1), square_farest.view (-1, 1))
		
		loss_nearest = torch.sum (log_nearest)
		loss_farest = torch.sum (log_farest)
		
		sub_trajectory_loss = sum ([loss_nearest, loss_farest])
		
		if self.sub:
			return sub_trajectory_loss / self.r
		else:
			return sub_trajectory_loss

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
		self.traj_embedding = TRAJ_ZONE_EMBEDDING(tert_params)
		self.trajectory_encoder = nn.GRU(tert_params.embed_size + 2, tert_params.embed_size, batch_first=True)
		
	
	def forward(self, traj_gps, traj_mask, traj_grid, traj_grid_pos,
	            traj_anchor_gps, traj_anchor_mask, traj_anchor_grid, traj_anchor_grid_pos,
	            traj_farest_gps, traj_farest_mask, traj_farest_grid, traj_farest_grid_pos):
		
		traj_zone_embedding = self.traj_embedding(traj_grid, traj_grid_pos, traj_gps, traj_mask)
		traj_anchor_zone_embedding = self.traj_embedding(traj_anchor_grid, traj_anchor_grid_pos, traj_anchor_gps, traj_anchor_mask)
		traj_farest_zone_embedding = self.traj_embedding(traj_farest_grid, traj_farest_grid_pos, traj_farest_gps, traj_farest_mask)
		
		_, traj_embedding = self.trajectory_encoder(traj_zone_embedding)
		_, traj_anchor_embedding = self.trajectory_encoder(traj_anchor_zone_embedding)
		_, traj_farest_embedding = self.trajectory_encoder(traj_farest_zone_embedding)
		 
		anchor_distance = torch.exp(-torch.norm(traj_embedding - traj_anchor_embedding, p=2, dim=-1)).transpose(0,1)
		farest_distance = torch.exp(-torch.norm(traj_embedding - traj_farest_embedding, p=2, dim=-1)).transpose(0,1)

		return anchor_distance, farest_distance
