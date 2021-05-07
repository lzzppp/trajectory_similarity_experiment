
import torch
import kdtree
import pickle
import random
from h52pic import *
from tqdm import tqdm
from src.dataloader_trainer import EncoderwithDecoderTrainer
from src.model.model_similarity import TERT, trajectory_Distance_Loss
from src.utils import bool_flag, check_data_params, network_to_half, initialize_exp, get_optimizer

def preprocess_trajectory_dataset(Index_Lat_Lon_Dataset, Index_Dict, latitude_unit, longitude_unit, latitude_min, longitude_min):
	preprocess_trajectories = []
	for trajectory in tqdm(Index_Lat_Lon_Dataset):
		grid = [Index_Dict[p[0]] for p in trajectory[0]]
		gps_trajectory = [[latitude_min + latitude_unit*p[1],
						   longitude_min + longitude_unit*p[2]] for p in trajectory[0]] # lat lon
		preprocess_trajectories.append([grid, gps_trajectory])
	return preprocess_trajectories

def bulid_kd_tree(dataset, k=5):
	trajectory_sequences = []
	for grid_trajectory in dataset:
		trajectory = grid_trajectory[1]
		trajectory_sequence_k = []
		seq_k = int((len(trajectory) * 1.0) / k)
		for i in range(k-1):
			tree_point = [0.0, 0.0]
			for p in trajectory[i * seq_k:(i + 1) * seq_k]:
				tree_point[0] += p[0]
				tree_point[1] += p[1]
			trajectory_sequence_k.extend([tree_point[0] / (seq_k * 1.0), tree_point[1] / (seq_k * 1.0)])
		tree_point = [0, 0]
		for p in trajectory[(k-1) * seq_k:]:
			tree_point[0] += p[0]
			tree_point[1] += p[1]
		trajectory_sequence_k.extend(
			[tree_point[0] / (len (trajectory[(k-1) * seq_k:]) * 1.0), tree_point[1] / (len (trajectory[(k-1) * seq_k:]) * 1.0)])
		# print(len(traj_sequence_k))
		trajectory_sequences.append(trajectory_sequence_k)
	kd_tree = kdtree.KDTree(trajectory_sequences, list(range(len(trajectory_sequences))))
	return kd_tree, trajectory_sequences

class PARAMS(object):
	def __init__(self):
		
		self.n_words = 22339
		self.pad_index = 0
		self.eos_index = 1
		self.mask_index = 2
		self.path = "model_store/decoder90.pth"
		self.batch_size = 64
		self.epochs = 200
		self.embed_size = 128
		self.test_batch_size = 128
		self.test_trajectory_prob = 0.1
		self.distance_path = "/mnt/data4/lizepeng/shanghai/shanghai_distance.npy"
		self.train_size = 10000
		self.test_trajectory_nums = 1000
		self.have_computed_distance_num = 15000

if __name__ == "__main__":
	shanghai_range = {
		"lon_min": 120.0011,
		"lat_min": 30.003,
		"lon_max": 121.999,
		"lat_max": 31.9996
	}
	
	engrider = Engrider(shanghai_range, 360, 360)
	lat_unit = engrider.lat_unit
	lon_unit = engrider.lon_unit
	model_params = PARAMS()
	
	logger = initialize_exp(engrider)
	
	index_lat_lon_dataset = pickle.load(open("trajectory_grid_lat_lon_dataset.pkl", "rb"))
	index_dict = pickle.load(open("tert_dataset/shanghai_token_dict.pkl", "rb"))
	Dataset = preprocess_trajectory_dataset(index_lat_lon_dataset, index_dict, lat_unit, lon_unit, shanghai_range["lat_min"], shanghai_range["lon_min"])[:model_params.have_computed_distance_num]
	
	###################### Bulid train test dataset from raw dataset ######################
	trajectory_index = list(range(len(Dataset)))
	random.shuffle(trajectory_index)
	train_dataset_index = trajectory_index[:model_params.train_size]
	train_dataset = [Dataset[train_index] for train_index in train_dataset_index]
	
	test_dataset_index = random.sample(trajectory_index[model_params.train_size:], model_params.test_trajectory_nums)
	distance = np.load(model_params.distance_path)[:,:model_params.have_computed_distance_num]
	#######################################################################################
	
	# pickle.dump(Dataset, open("shanghai_trajectory.pkl", "wb"))
	kd, sequences = bulid_kd_tree(train_dataset)

	model = TERT(model_params).cuda()
	model.load_state_dict(torch.load("simi_model_store/simi_model_" + str(60) + ".pth"))
	loss_f = trajectory_Distance_Loss(batch_size=model_params.batch_size).cuda()
	trainer = EncoderwithDecoderTrainer(model, loss_f, model_params, train_dataset, Dataset, distance, train_dataset_index, test_dataset_index, kd, sequences, batch_size=model_params.batch_size, test_batch_size=model_params.test_batch_size, test_trajectory_prob=model_params.test_trajectory_nums)
	for epoch in range(model_params.epochs):
		logger.info("============ Starting epoch %i ... ============" % epoch)
		trainer.n_sentences = 0
		######################
		# trainer.train(epoch)
		trainer.batch_size = model_params.batch_size
		######################
		if epoch % 10 == 0 and epoch != 0:
			# torch.save(trainer.model.state_dict(), "simi_model_store/simi_model_" + str(epo) + ".pth")
			trainer.test(epoch)
			trainer.test_batch_size = model_params.test_batch_size
		logger.info("============ End of epoch %i ============" % epoch)