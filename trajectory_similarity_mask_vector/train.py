import torch
import pickle
from h52pic import *
import torch.nn as nn
import torch.nn.functional as F
from src.model.models import TCAN
from src.Trainer import RegressSimiTrainer
from src.model.loss_function import LossFunction
from utils import preprocess_trajectory_dataset, PARAMS, initialize_exp, porto_range

if __name__ == "__main__":
	################################
	engrider = EngriderMeters(porto_range, 100, 100)
	lat_unit = engrider.lat_unit
	lon_unit = engrider.lon_unit
	model_params = PARAMS()
	################################
	logger = initialize_exp(engrider)
	
	train_lat_lon_dataset = pickle.load(open("/home/xiaoziyang/Downloads/trajectory/bert_similarity/porto_train_grid.pkl", "rb"))
	Token_dict = pickle.load(open("/home/xiaoziyang/Downloads/trajectory/bert_similarity/porto_token_dict.pkl", "rb"))
	train_dataset = preprocess_trajectory_dataset (train_lat_lon_dataset, Token_dict, lat_unit, lon_unit,
	                                               porto_range["lat_min"], porto_range["lon_min"])
	traj_features = pickle.load(open("/home/xiaoziyang/Downloads/trajectory/bert_similarity/porto_trajectory_features.pkl", "rb"))
	kd_index = pickle.load(open("/home/xiaoziyang/Downloads/trajectory/bert_similarity/porto_kdtree_index_list", "rb"))
	
	model = TCAN().cuda()
	# model.load_state_dict(torch.load("/home/xiaoziyang/Github/trajectory_similarity_mask/porto_store/simi_model_0.pth"))
	loss_f = LossFunction().cuda()
	trainer = RegressSimiTrainer(model, loss_f, train_dataset, kd_index, traj_features,
	                             batch_size=model_params.batch_size)
	
	for epoch in range (model_params.epochs):
		logger.info ("============ Starting epoch %i ... ============" % epoch)
		trainer.n_sentences = 0
		######################
		trainer.train_step(epoch)
		trainer.batch_size = model_params.batch_size
		######################
		torch.save (trainer.model.state_dict(), "porto_store/simi_model_" + str (epoch) + ".pth")
		# trainer.test(epoch)
		# trainer.test_batch_size = model_params.test_batch_size
		logger.info ("============ End of epoch %i ============" % epoch)