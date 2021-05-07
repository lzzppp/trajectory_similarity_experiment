
import torch
from torch.optim import Adam
from src.tools import ProgressBar
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

class NPPTrainer(object):
	def __init__(self, dataset, model, train_batch_size):
		self.index = 0
		self.model = model
		self.n_sentences = 0
		self.dataset = dataset
		self.data_length = len(dataset)
		self.train_batch_size = train_batch_size
		self.optimizer = Adam(self.model.parameters(), lr=1e-5)
		self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.8)
	
	def batch_genarator(self):
		self.index = 0
		batch_size = self.train_batch_size
		while self.index < self.data_length:
			if self.index + batch_size > self.data_length:
				batch_size = self.data_length - self.index
			trajectory_batch_encoder_data, trajectory_batch_target_data, trajectory_batch_decoder_data, \
			trajectory_batch_encoder_length, trajectory_batch_decoder_length = [], [], [], [], []
			for data_index in range(self.index, self.index + batch_size):
				data_length = len(self.dataset[data_index])
				base_knowledge_length = int(data_length * 0.9)
				trajectory_batch_encoder_data.append(self.dataset[data_index][:base_knowledge_length])
				trajectory_batch_encoder_length.append(base_knowledge_length)
				trajectory_batch_decoder_data.append(self.dataset[data_index][base_knowledge_length-1:-1])
				trajectory_batch_target_data.extend(self.dataset[data_index][base_knowledge_length:])
				trajectory_batch_decoder_length.append(data_length - base_knowledge_length)
			
			trajectory_batch_encoder_data_padding, trajectory_batch_decoder_data_padding, trajectory_batch_target_data_padding = [], [], []
			trajectory_batch_encoder_position, trajectory_batch_decoder_position = [], []
			trajectory_batch_encoder_mask, trajectory_batch_decoder_mask = [], []
			max_encoder_length = max(trajectory_batch_encoder_length)
			max_decoder_length = max(trajectory_batch_decoder_length)
			for i in range(batch_size):
				trajectory_batch_encoder_data_padding.append(trajectory_batch_encoder_data[i] + (max_encoder_length - trajectory_batch_encoder_length[i])*[0])
				trajectory_batch_decoder_data_padding.append(trajectory_batch_decoder_data[i] + (max_decoder_length - trajectory_batch_decoder_length[i])*[0])
				# trajectory_batch_target_data_padding.append(trajectory_batch_target_data[i] + (max_decoder_length - trajectory_batch_decoder_length[i])*[0])
				trajectory_batch_encoder_mask.append([False]*trajectory_batch_encoder_length[i] + [True]*(max_encoder_length - trajectory_batch_encoder_length[i]))
				trajectory_batch_decoder_mask.append([False]*trajectory_batch_decoder_length[i] + [True]*(max_decoder_length - trajectory_batch_decoder_length[i]))
				trajectory_batch_encoder_position.append([epos + 1 for epos in range(trajectory_batch_encoder_length[i])] + (max_encoder_length - trajectory_batch_encoder_length[i]) * [0])
				trajectory_batch_decoder_position.append([dpos + trajectory_batch_encoder_length[i] + 1 for dpos in range(trajectory_batch_decoder_length[i])] + (max_decoder_length - trajectory_batch_decoder_length[i]) * [0])
			trajectory_batch_decoder_forward_mask = [[float (0.0)] * i + [float ('-inf')] * (max_decoder_length - i) for i in range(1, max_decoder_length + 1)]
			self.index += batch_size
			yield trajectory_batch_encoder_data_padding, trajectory_batch_decoder_data_padding,\
			      trajectory_batch_encoder_mask, trajectory_batch_decoder_mask,\
			      trajectory_batch_encoder_position, trajectory_batch_decoder_position, \
			      trajectory_batch_decoder_forward_mask, trajectory_batch_target_data
	
	def train_step(self):
		self.index = 0
		self.model.train()
		tloss = 0.0
		loss_list = []
		
		progress = ProgressBar(self.data_length//self.train_batch_size, fmt=ProgressBar.FULL)
		
		for enc_data, dec_data, enc_mask, dec_mask, enc_pos, dec_pos, dec_forward_mask, target_data in self.batch_genarator():
			enc_data = torch.LongTensor(enc_data).transpose(0, 1).cuda()
			dec_data = torch.LongTensor(dec_data).transpose(0, 1).cuda()
			enc_pos = torch.LongTensor(enc_pos).cuda()
			dec_pos = torch.LongTensor(dec_pos).cuda()
			target_data = torch.LongTensor(target_data).cuda()
			enc_mask = torch.BoolTensor(enc_mask).cuda()
			dec_mask = torch.BoolTensor(dec_mask).cuda()
			dec_forward_mask = torch.FloatTensor(dec_forward_mask).cuda()
			pred_out = self.model(enc_data, dec_data, enc_pos, dec_pos,
			                  tgt_mask=dec_forward_mask, src_key_padding_mask=enc_mask,
			                  tgt_key_padding_mask=dec_mask, memory_key_padding_mask=enc_mask)
			
			loss = F.cross_entropy(pred_out, target_data)
			
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			
			progress.current += 1
			progress()
			
			tloss += loss.cpu().detach().numpy()
			
			if int(self.n_sentences / self.train_batch_size) % 10 == 0 and self.n_sentences > 0:
				print(tloss)
				loss_list.append(tloss)
				tloss = 0

			self.n_sentences += self.train_batch_size
		progress.done()