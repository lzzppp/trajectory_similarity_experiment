
import torch
import kdtree
import pickle
import random
from h52pic import *
from tqdm import tqdm
from src.dataloader_trainer_neutraj import EncoderwithDecoderTrainer
from src.model.NeuTraj_models import NeuTraj_Network, trajectory_Distance_Loss
from src.utils import bool_flag, check_data_params, network_to_half, initialize_exp, get_optimizer

spatial_width = 2

def preprocess_trajectory_dataset(Index_Lat_Lon_Dataset, Index_Dict, latitude_unit, longitude_unit, latitude_min, longitude_min):
    preprocess_trajectories = []
    for trajectory in tqdm(Index_Lat_Lon_Dataset):
        grid = [[latitude_min + latitude_unit*p[2], longitude_min + longitude_unit*p[1], p[2] + spatial_width, p[1] + spatial_width] for p in trajectory[0]] # lat lon
        # gps_trajectory = [[latitude_min + latitude_unit*p[2],
        #                    longitude_min + longitude_unit*p[1]] for p in trajectory[0]] # lat lon
        preprocess_trajectories.append(grid)
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
        
        # self.n_words = 22339 # shanghai
        self.n_words = 13416 # porto
        self.pad_index = 0
        self.eos_index = 1
        self.mask_index = 2
        self.path = "/home/xiaoziyang/Github/tert_model_similarity/model_store/porto_encoder_decoder_27.pth"
        self.n_layers = 4
        self.batch_size = 256
        self.epochs = 200
        self.embed_size = 128
        self.test_batch_size = 128
        self.test_trajectory_prob = 0.1
        # self.distance_path = "/mnt/data4/lizepeng/shanghai/shanghai_distance.npy"
        self.train_size = 900000
        self.test_trajectory_nums = 1000
        self.have_computed_distance_num = 1000000

if __name__ == "__main__":
    shanghai_range = {
        "lon_min": 120.0011,
        "lat_min": 30.003,
        "lon_max": 121.999,
        "lat_max": 31.9996
    }
    
    porto_range = {
        "lon_min": -8.735152,
        "lat_min": 40.953673,
        "lon_max": -8.156309,
        "lat_max": 41.307945
    }
    
    engrider = EngriderMeters(porto_range, 100, 100)
    
    print(engrider.to_str_idx([porto_range["lon_max"], porto_range["lat_max"]]))
    
    lat_unit = engrider.lat_unit
    lon_unit = engrider.lon_unit
    model_params = PARAMS()
    
    logger = initialize_exp(engrider)
    
    train_lat_lon_dataset = pickle.load(open("/home/xiaoziyang/Downloads/trajectory/bert_similarity/porto_train_grid.pkl", "rb"))
    Token_dict = pickle.load(open("/home/xiaoziyang/Downloads/trajectory/bert_similarity/porto_token_dict.pkl", "rb"))
    train_dataset = preprocess_trajectory_dataset(train_lat_lon_dataset, Token_dict, lat_unit, lon_unit, porto_range["lat_min"], porto_range["lon_min"])

    sequences = pickle.load(open("/home/xiaoziyang/Downloads/trajectory/bert_similarity/porto_train_sequence.pkl", "rb"))
    kd_index = pickle.load(open("/home/xiaoziyang/Downloads/trajectory/bert_similarity/porto_kdtree_index_list", "rb"))
    trajectory_feature = pickle.load(open("/home/xiaoziyang/Downloads/trajectory/bert_similarity/porto_trajectory_features.pkl", "rb"))
    
    model = NeuTraj_Network(4, model_params.embed_size, [400, 490], model_params.batch_size, 0, stard_LSTM=False, incell=True).cuda()
    
    loss_f = trajectory_Distance_Loss(batch_size=model_params.batch_size).cuda()
    trainer = EncoderwithDecoderTrainer(model, loss_f, model_params, train_dataset, kd_index, sequences, trajectory_feature,
                                        batch_size=model_params.batch_size,
                                        test_batch_size=model_params.test_batch_size,
                                        test_trajectory_prob=model_params.test_trajectory_nums)
    
    for epoch in range(model_params.epochs):
        logger.info("============ Starting epoch %i ... ============" % epoch)
        trainer.n_sentences = 0
        ######################
        trainer.train(epoch)
        trainer.batch_size = model_params.batch_size
        ######################
        torch.save(trainer.model.state_dict(), "simi_model_store/simi_model_" + str(epoch) + ".pth")
        # trainer.test(epoch)
        # trainer.test_batch_size = model_params.test_batch_size
        logger.info("============ End of epoch %i ============" % epoch)
