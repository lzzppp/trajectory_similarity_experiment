
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
import torch.nn as nn
from tqdm import tqdm
import torch.autograd as autograd
import torch.nn.functional as F
# from src.dataloader_trainer import EncoderwithDecoderTrainer
from src.models_test import OCD

mail_pre_degree = 16.0

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
        if len(trajectory[0]) > 32:
            gps_trajectory = [[latitude_min + latitude_unit*p[2],
                               longitude_min + longitude_unit*p[1]] for p in trajectory[0]] # lat lon
            preprocess_trajectories.append(gps_trajectory)
    return preprocess_trajectories

class PARAMS(object):
    def __init__(self):
        
        # self.n_words = 22339 # shanghai
        self.n_words = 13416 # porto
        self.pad_index = 0
        self.eos_index = 1
        self.mask_index = 2
        self.path = "/home/xiaoziyang/Github/tert_model_similarity/model_store/porto_encoder_decoder_27.pth"
        self.simi_path = "/home/xiaoziyang/Github/Traj2SimVector/simi_model_store/simi_model_7.pth"
        self.n_layers = 4
        self.batch_size = 32
        self.generator_batch_size = 1024
        self.epochs = 200
        self.embed_size = 128
        self.test_batch_size = 128
        self.test_trajectory_prob = 0.1
        # self.distance_path = "/mnt/data4/lizepeng/shanghai/shanghai_distance.npy"
        self.train_size = 900000
        self.test_trajectory_nums = 1000
        self.have_computed_distance_num = 1000000

def get_anchor_batch_pair(trajectory_data, query_index, batch_size=32):
    index = 0
    query_trajectory = trajectory_data[query_index]
    while index < len(trajectory_data):
        trajectory_matrixs = []
        if batch_size + index > len(trajectory_data):
            batch_size = len(trajectory_data) - index
        for pair_index in range(index, index+batch_size):
            # if pair_index != query_index:
            match_trajectory = trajectory_data[pair_index]
            matrix_x, matrix_y = 32, 32
            line_xy = []
            for x in range (matrix_x):
                line_x = []
                for y in range (matrix_y):
                    line_x.append (((query_trajectory[x][0] - match_trajectory[y][0]) ** 2 +
                                    (query_trajectory[x][1] - match_trajectory[y][1]) ** 2) ** 0.5 * mail_pre_degree)
                line_xy.append (line_x)
            trajectory_matrixs.append([line_xy for _ in range(3)])
                
        index += batch_size
        yield trajectory_matrixs

if __name__ == "__main__":
    
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
    
    # logger = initialize_exp(engrider)
    
    test_lat_lon_dataset = pickle.load(open("/home/xiaoziyang/Downloads/trajectory/bert_similarity/porto_test_grid.pkl", "rb"))
    Token_dict = pickle.load(open("/home/xiaoziyang/Downloads/trajectory/bert_similarity/porto_token_dict.pkl", "rb"))
    traj_features = pickle.load(open("/home/xiaoziyang/Downloads/trajectory/bert_similarity/porto_trajectory_features.pkl", "rb"))
    test_dataset = preprocess_trajectory_dataset(test_lat_lon_dataset, Token_dict, lat_unit, lon_unit, porto_range["lat_min"], porto_range["lon_min"])[:10000]
    # test_trainer = TEST_TRAINER(test_dataset, traj_features, generator_batch_size=model_params.generator_batch_size)
    
    model = OCD(input_channel=3, cls_num=1).cuda()
    model.load_state_dict(torch.load("/home/xiaoziyang/Github/trajectory_similarity_matrix_learning/ocd_porto33.pt"))
    
    flag = 0
    model.eval()
    with torch.no_grad():
        similarity_matrix = []
        for anchor_index in tqdm(range(1000)):
            anchor_similarity = []
            progress = ProgressBar(len(test_dataset)//1024, fmt=ProgressBar.FULL)
            for batch in get_anchor_batch_pair(test_dataset, anchor_index, batch_size=1024):
                input_data = torch.FloatTensor(batch).cuda()
                target_data = list(model(input_data).cpu().numpy()[:, -1, -1])
                anchor_similarity.extend(target_data)
                progress.current += 1
                progress()
            progress.done()
            similarity_matrix.append(anchor_similarity)
            flag += 1
    pickle.dump(similarity_matrix, open("similarity_matrix.pkl", "wb"))