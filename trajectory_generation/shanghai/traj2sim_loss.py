# coding=utf-8
import math
import torch
import kdtree
import random
import numpy as np
import torch.nn as nn


# def forward(self, tuples, m1, m2):
# 	point_predict = []
# 	for batch_idx in range (len (tuples)):
# 		nearest_tuple = tuples[batch_idx]
# 		point_line = []
# 		for index, index_list in enumerate (nearest_tuple):
# 			a_i, n_m_i, n_um_i = index_list[0], index_list[1], index_list[2]
# 			m_a_v = m1[batch_idx, a_i, :]
# 			m_n_m_v = m2[batch_idx, n_m_i, :]
# 			m_n_um_v = m2[batch_idx, n_um_i, :]
#
# 			predict_m = torch.exp (-torch.norm (m_a_v - m_n_m_v, p=2, dim=-1))
# 			predict_um = torch.exp (-torch.norm (m_a_v - m_n_um_v, p=2, dim=-1))
#
# 			point_line.append (torch.FloatTensor ([min (0, 0.01 + predict_um - predict_m) / self.r_]))
# 		point_line = torch.cat (point_line)
# 		point_predict.append (point_line.unsqueeze (0))
# 	point_predict = torch.cat (point_predict)
# 	return point_predict / self.r_

class Sub_trajectory_Distance_Loss (nn.Module):
	def __init__(self, sub=False, r=10.0, batch_size=32):
		super (Sub_trajectory_Distance_Loss, self).__init__ ()
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
	
	def forward(self, nearest_distance_predict, farest_distance_predict, nearest_distance_target,
	            farest_distance_target, alpha):
		# div_nearest = torch.exp(-nearest_distance_target.view(-1, 1)) - nearest_distance_predict.view(-1, 1)
		# div_farest = torch.exp(-farest_distance_target.view(-1, 1)) - farest_distance_predict.view(-1, 1)
		
		div_nearest = nearest_distance_target.view(-1, 1) - nearest_distance_predict.view(-1, 1)
		div_farest = farest_distance_target.view(-1, 1) - farest_distance_predict.view(-1, 1)
		
		square_nearest = torch.mul(div_nearest.view(-1, 1), div_nearest.view(-1, 1))
		square_farest = torch.mul(div_farest.view(-1, 1), div_farest.view(-1, 1))
		# print(alpha * nearest_distance_target)
		# weight_div_nearest = torch.mul (self.weight_of_sub_trajectory, div_nearest)
		# weight_div_farest = torch.mul (self.weight_of_sub_trajectory, div_farest)
		
		log_nearest = torch.mul(nearest_distance_target.view(-1, 1), square_nearest.view(-1, 1))
		log_farest = torch.mul(farest_distance_target.view(-1, 1), square_farest.view(-1, 1))
		
		loss_nearest = torch.sum(log_nearest)
		loss_farest = torch.sum(log_farest)
		
		sub_trajectory_loss = sum([loss_nearest, loss_farest])
		
		if self.sub:
			return sub_trajectory_loss / self.r
		else:
			return sub_trajectory_loss

class point_matching_loss (nn.Module):
	def __init__(self, r_=10.0, batch_size=32):
		super (point_matching_loss, self).__init__ ()
		self.batch_size = batch_size
		self.r_ = r_
	
	def forward(self, tuples, m1, m2):
		point_predict = []
		a, n_m, n_um = [], [], []
		batch_ = [[i] for i in range(len(tuples))]
		for batch_idx in range(len(tuples)):
			nearest_tuple = tuples[batch_idx]
			a_i, n_m_i, n_um_i = [], [], []
			for index, index_list in enumerate(nearest_tuple):
				a_i.append(index_list[0])
				n_m_i.append(index_list[1])
				n_um_i.append(index_list[2])
			a.append(a_i)
			n_m.append(n_m_i)
			n_um.append(n_um_i)
		m_a_v = m1[batch_, a, :]
		m_n_m_v = m2[batch_, n_m, :]
		m_n_um_v = m2[batch_, n_um, :]
		m_loss = torch.exp(-torch.pairwise_distance(m_a_v, m_n_m_v))
		um_loss = torch.exp(-torch.pairwise_distance(m_a_v, m_n_um_v))
		# loss = torch.relu(0.01 - m_loss + um_loss)
		loss = torch.clamp(0.01 - m_loss + um_loss, min=0.0)
		# loss_plus = torch.sum(m_loss, dim=1) + torch.sum(um_loss, dim=1)
		# point_predict = torch.cat(point_predict)
		loss_plus = torch.sum(loss, dim=1)
		return loss_plus / self.r_

class Trajectory_Point_Matching_Loss (nn.Module):
	def __init__(self, r_=10, batch_size=32):
		super (Trajectory_Point_Matching_Loss, self).__init__ ()
		self.r_ = r_
		self.batch_size = batch_size
		self.point_matching_nearest_loss = point_matching_loss ()
		self.point_matching_farest_loss = point_matching_loss ()
	
	def forward(self,
	            anchor_mvector,
	            nearest_mvector,
	            farest_mvector,
	            nearest_tuples,
	            farest_tuples):
		loss1 = self.point_matching_nearest_loss(nearest_tuples, anchor_mvector, nearest_mvector)
		loss2 = self.point_matching_farest_loss(farest_tuples, anchor_mvector, farest_mvector)
		point_matching_Loss = sum([torch.sum(loss1), torch.sum(loss2)])
		
		return point_matching_Loss

class Traj2SimLoss (nn.Module):
	def __init__(self, batch_size=32):
		super (Traj2SimLoss, self).__init__ ()
		self.batch_size = batch_size
		self.sub_trajectory_distance_loss = Sub_trajectory_Distance_Loss (sub=True)
		self.trajectory_distance_loss_total = Sub_trajectory_Distance_Loss ()
		self.trajectory_point_matching_loss = Trajectory_Point_Matching_Loss ()
	
	def forward(self, n_p_d, f_p_d, n_p_d_a, f_p_d_a, m_a_v, m_n_v, m_f_v,
	            target_sub1, target_sub2, target_all1, target_all2, n_tuple, f_tuple, alpha):
		
		loss1 = self.sub_trajectory_distance_loss(n_p_d, f_p_d, target_sub1, target_sub2, alpha)
		loss2 = self.trajectory_distance_loss_total(n_p_d_a, f_p_d_a, target_all1, target_all2, alpha)
		# loss3 = self.trajectory_point_matching_loss(m_a_v, m_n_v, m_f_v, n_tuple, f_tuple)
		# print(loss1)
		# print(loss2)
		# print(loss3)
		traj2sim_loss = sum([loss1, loss2])
		return traj2sim_loss
