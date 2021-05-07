
import torch
import pickle
import argparse
from tqdm import tqdm
from src.Trainer import NPPTrainer
from src.model.model_forecasting import S2sTransformer

def preprocess_raw_dataset(dataset, token_dict):
	new_dataset = []
	for grid_lon_lat_time_trajectory in tqdm(dataset):
		new_dataset.append([int(token_dict[goa[0]]) for goa in grid_lon_lat_time_trajectory[0]])
	return new_dataset
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.description = 'please enter -train_batch_size int -'
	parser.add_argument("--train_batch_size", "--train_batch_size", help="train batch size", dest="argA", type=int, default=64)
	parser.add_argument("-b", "--inputB", help="this is parameter b", type=int, default="1")
	args = parser.parse_args()
	args.n_words = 13416
	args.n_layers = 4
	args.train_batch_size = 64
	# print(dir(args))
	
	train_grid_lon_lat_dataset = pickle.load(open("/home/xiaoziyang/Downloads/trajectory/bert_similarity/porto_train_grid.pkl", "rb"))
	Token_dict = pickle.load(open("/home/xiaoziyang/Downloads/trajectory/bert_similarity/porto_token_dict.pkl", "rb"))
	
	train_dataset = preprocess_raw_dataset(train_grid_lon_lat_dataset, Token_dict)
	model = S2sTransformer(args.n_words, num_encoder_layers=args.n_layers).cuda()
	model.load_state_dict(torch.load("/home/xiaoziyang/Github/tert_model_similarity/model_store/porto_encoder_decoder_27.pth"))
	
	trainer = NPPTrainer(train_dataset, model, args.train_batch_size)
	
	trainer.train_step()