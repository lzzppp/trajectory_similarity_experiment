import torch
import random
from utils import *
from h52pic import *
from src.models import OCD
import torchvision.models as models
from src.Trainer import MatrixTrainer

def train():
	################################
	epochs = 100
	engrider = EngriderMeters(porto_range, 100, 100)
	lat_unit = engrider.lat_unit
	lon_unit = engrider.lon_unit
	model_params = PARAMS()
	################################
	logger = initialize_exp(engrider)
	
	train_lat_lon_dataset = pickle.load(open("/home/xiaoziyang/Downloads/trajectory/bert_similarity/porto_train_grid.pkl", "rb"))
	Token_dict = pickle.load(open("/home/xiaoziyang/Downloads/trajectory/bert_similarity/porto_token_dict.pkl", "rb"))
	traj_features = pickle.load(open("/home/xiaoziyang/Downloads/trajectory/bert_similarity/porto_trajectory_features.pkl", "rb"))
	traj_indexs = pickle.load(open("/home/xiaoziyang/Downloads/trajectory/bert_similarity/porto_kdtree_index_list2", "rb"))
	train_dataset = preprocess_trajectory_dataset(train_lat_lon_dataset, traj_indexs, lat_unit, lon_unit, porto_range["lat_min"], porto_range["lon_min"])
	
	model = OCD(input_channel=1, cls_num=1).cuda()
	model.train()
	# model_dict = model.state_dict()
	# vgg16 = models.vgg16(pretrained=True)
	# pretrained_dict = vgg16.state_dict()
	# pretrained_dict = {k: v for k, v in pretrained_dict.items () if k in model_dict}
	# model_dict.update(pretrained_dict)
	# model.load_state_dict(model_dict)
	# model.load_state_dict(torch.load("/home/xiaoziyang/Github/trajectory_similarity_matrix_learning/ocd_porto2.pt"))
	
	trainer = MatrixTrainer(model, opt, model_params.batch_size, train_dataset, traj_features, traj_indexs)
	
	for epoch in range(epochs):
		trainer.train_step(epoch)

if __name__ == "__main__":
	train()