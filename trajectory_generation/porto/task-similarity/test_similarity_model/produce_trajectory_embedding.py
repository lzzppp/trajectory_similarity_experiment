
import re
import sys
import time
import math
import torch
# import kdtree
import pickle
import random
import numpy as np
from h52pic import *
from tqdm import tqdm
from src.dataloader_trainer import EncoderwithDecoderTrainer
from src.model.model_similarity import TERT, trajectory_Distance_Loss
from src.utils import bool_flag, check_data_params, network_to_half, initialize_exp, get_optimizer


class ProgressBar (object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'
    
    def __init__(self, total, width=40, fmt=DEFAULT, symbol='=',
                 output=sys.stderr):
        assert len (symbol) == 1
        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub (r'(?P<name>%\(.+?\))d', r'\g<name>%dd' % len (str (total)), fmt)
        self.current = 0
    
    def __call__(self):
        percent = self.current / float (self.total)
        size = int (self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'
        
        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining}
        
        print ('\r' + self.fmt % args, file=self.output, end='')
    
    def done(self):
        self.current = self.total
        self ()
        print ('', file=self.output)

def preprocess_trajectory_dataset(Index_Lat_Lon_Dataset, Index_Dict, latitude_unit, longitude_unit, latitude_min, longitude_min):
    preprocess_trajectories = []
    for trajectory in tqdm(Index_Lat_Lon_Dataset):
        grid = [Index_Dict[p[0]] for p in trajectory[0]]
        gps_trajectory = [[latitude_min + latitude_unit*p[2],
                           longitude_min + longitude_unit*p[1]] for p in trajectory[0]] # lat lon
        preprocess_trajectories.append([grid, gps_trajectory])
    return preprocess_trajectories

class PARAMS(object):
    def __init__(self):
        
        # self.n_words = 22339 # shanghai
        self.n_words = 13416 # porto
        self.pad_index = 0
        self.eos_index = 1
        self.mask_index = 2
        self.path = "/home/lizepeng/tert_model_similarity/model_store/porto_encoder_decoder_27.pth"
        self.simi_path = "/home/lizepeng/tert_model_similarity/simi_model_store/simi_model_0.pth"
        self.n_layers = 4
        self.batch_size = 32
        self.generator_batch_size = 128
        self.epochs = 200
        self.embed_size = 128
        self.test_batch_size = 128
        self.test_trajectory_prob = 0.1
        # self.distance_path = "/mnt/data4/lizepeng/shanghai/shanghai_distance.npy"
        self.train_size = 900000
        self.test_trajectory_nums = 1000
        self.have_computed_distance_num = 1000000

class TEST_TRAINER(object):
    def __init__(self, dataset, generator_batch_size):
        self.index = 0
        self.dataset = dataset
        self.dataset_length = len(dataset)
        self.generator_batch_size = generator_batch_size
    
    def trajectory_batch_generator(self):
        generator_batch_size = self.generator_batch_size
        while self.index < self.dataset_length:
            if self.index + generator_batch_size > self.dataset_length:
                generator_batch_size = self.dataset_length - self.index
            trajectory_gps, trajectory_grid = [], []
            trajectory_length = []
            for i in range(self.index, self.index + generator_batch_size):
                trajectory_gps.append(self.dataset[i][1])
                trajectory_grid.append(self.dataset[i][0])
                trajectory_length.append(len(self.dataset[i][0]))
            trajectory_length_max = max(trajectory_length)
            trajectory_mask, trajectory_grid_pos = [], []
            for ind, ti in enumerate(trajectory_length):
                trajectory_mask.append ([False] * ti + [True] * (trajectory_length_max - ti))
                trajectory_gps[ind] = trajectory_gps[ind] + [[0.0, 0.0] for ttii in range (trajectory_length_max - ti)]
                trajectory_grid[ind] = trajectory_grid[ind] + [0] * (trajectory_length_max - ti)

            self.index += generator_batch_size
            self.generator_batch_size = generator_batch_size
            yield trajectory_gps, trajectory_grid, trajectory_mask, trajectory_length

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
    lat_unit = engrider.lat_unit
    lon_unit = engrider.lon_unit
    model_params = PARAMS()
    
    logger = initialize_exp(engrider)
    
    test_lat_lon_dataset = pickle.load(open("/mnt/data4/lizepeng/porto/bert_similarity/porto_test_grid.pkl", "rb"))
    Token_dict = pickle.load(open("/mnt/data4/lizepeng/porto/bert_similarity/porto_token_dict.pkl", "rb"))
    test_dataset = preprocess_trajectory_dataset(test_lat_lon_dataset, Token_dict, lat_unit, lon_unit, porto_range["lat_min"], porto_range["lon_min"])
    test_trainer = TEST_TRAINER(test_dataset, generator_batch_size=model_params.generator_batch_size)
    
    model = TERT(model_params).cuda()
    model.load_state_dict(torch.load(model_params.simi_path))
    
    flag = 0
    model.eval()
    progress = ProgressBar(test_trainer.dataset_length // test_trainer.generator_batch_size, fmt=ProgressBar.FULL)
    with torch.no_grad():
        for batch in test_trainer.trajectory_batch_generator():
            traj_gps = torch.FloatTensor(batch[0]).cuda()
            traj_grid = torch.LongTensor(batch[1]).cuda()
            traj_mask = torch.BoolTensor(batch[2]).cuda()
            traj_length = batch[3]
            traj_pos = torch.arange(1, max(traj_length) + 1, device=torch.device("cuda")).unsqueeze(0).expand(len(traj_length), max(traj_length))
            traj_zone_embedding = model.traj_embedding(traj_grid, traj_pos, traj_gps, traj_mask)
            _, traj_embedding = model.trajectory_encoder(traj_zone_embedding, traj_length)
            progress.current += 1
            progress()
            if flag == 0:
                traj_embeddings = traj_embedding
            else:
                traj_embeddings = torch.cat((traj_embeddings, traj_embedding))
            flag += 1
    progress.done()
    print("all test trajectory embedding done, nums: ", traj_embeddings.shape[0])
    np.save("/mnt/data4/lizepeng/porto/bert_similarity/porto_test_trajectory_embedding.npy", traj_embeddings.cpu().numpy())