
import random
import torch
import pickle
import numpy as np
from tqdm import tqdm
from src.dataset import myDataset

max_len=520

class DataLoader(object):
    
    def __init__(self, path):
        self.path = path
        self.raw_dataset = []
        self.token_dict = {}
        self.token_frequency_dict = {}
        self.time_dict = {}
        self.bacth_size = 32
        self.i = 0
        self.load()
        
    def load(self):
        raw_dataset = pickle.load(open(self.path, "rb"))
        dataset = []
        for i in tqdm(range(len(raw_dataset)), desc="Loading data"):
            for token, time in zip(raw_dataset[i][0], raw_dataset[i][1]):
                if token not in self.token_dict:
                    self.token_dict[token] = len(self.token_dict) + 3
                    self.token_frequency_dict[self.token_dict[token]] = 1
                else:
                    self.token_frequency_dict[self.token_dict[token]] += 1
                if time not in self.time_dict:
                    self.time_dict[time] = len(self.time_dict) + 1
            dataset.append([[self.token_dict[token] for token in raw_dataset[i][0]],
                            [self.time_dict[time] for time in raw_dataset[i][1]]])
        print("length of tokens :", len(self.token_dict))
        print("length of time tokens :", len(self.time_dict))
        self.raw_dataset = dataset

    def get_start_mask(self, length):
        mask_length = round (length / 2)
        start = 1
        end = length - mask_length
        start_index = random.randint (start, end)
        return list (range (start_index, start_index + mask_length + 1))

    def mask_word(self, w):
        _w_real = w
        _w_rand = np.random.randint (3, 42608, size=w.shape)
        _w_mask = np.full (w.shape, 2)
    
        probs = torch.multinomial (torch.Tensor ([0.8, 0.1, 0.1]), len (_w_real), replacement=True)
    
        _w = _w_mask * (probs == 0).numpy () + _w_real * (probs == 1).numpy () + _w_rand * (probs == 2).numpy ()
        return _w

    def process_dataset(self):
        batch_data_Raw, batch_data, batch_data2, lengths, length1, length2, position, pred_target, pred_mask = [], [], [], [], [], [], [], [], []
        for j in tqdm (range (len (self.raw_dataset))):
            batch_data_Raw.append ([[1] + self.raw_dataset[j][0] + [1], [0] + self.raw_dataset[j][1] + [self.raw_dataset[j][1][-1] + 15]])
            batch_data.append ([[1] + self.raw_dataset[j][0] + [1], [0] + self.raw_dataset[j][1] + [self.raw_dataset[j][1][-1] + 15]])
            lengths.append (len (self.raw_dataset[j][0]))
            length1.append (len (self.raw_dataset[j][0]) + 2)
    
        for length_idx in tqdm (range (len (lengths))):
            length = lengths[length_idx]
            shuffle_token_list = self.get_start_mask (length)
            position.append (shuffle_token_list)
            batch_data_mask = self.mask_word (
                np.array ([batch_data_Raw[length_idx][0][shuffle_index] for shuffle_index in shuffle_token_list]))
            pred_target.extend ([batch_data_Raw[length_idx][0][shuffle_index] for shuffle_index in shuffle_token_list])
            for idx, shuffle_index in enumerate (shuffle_token_list):
                batch_data[length_idx][0][shuffle_index] = batch_data_mask[idx]
            batch_data2.append (
                [[2] + [batch_data_Raw[length_idx][0][shuffle_index] for shuffle_index in shuffle_token_list[:-1]],
                 [batch_data_Raw[length_idx][1][shuffle_index] for shuffle_index in shuffle_token_list]])
            length2.append (len (shuffle_token_list))
        length1_max = max (length1)
        length2_max = max (length2)
        pred_target = torch.LongTensor (pred_target)
        pred_mask = torch.ByteTensor ([l2 * [True] + (length2_max - l2) * [False] for l2 in length2])
        batch_data_token = torch.LongTensor ([line[0] + (length1_max - len (line[0])) * [0] for line in batch_data])
        batch_data_time = [line[1] + (length1_max - len (line[1])) * [0] for line in batch_data]
        batch_data2_token = [line[0] + (length2_max - len (line[0])) * [0] for line in batch_data2]
        batch_data2_time = [line[1] + (length2_max - len (line[1])) * [0] for line in batch_data2]
        batch_data_position = [shuffle_position + [0] * (length2_max - len (shuffle_position)) for shuffle_position in
                               position]
        my_dataset = myDataset (batch_data_token, batch_data_time, batch_data2_token, batch_data2_time,
                                batch_data_position, pred_target, length1, length2, pred_mask)
        train_sampler = torch.utils.data.distributed.DistributedSampler (my_dataset)
        train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=16, sampler=train_sampler)
    
        return train_loader
    
    def get_batch(self):
        
        while self.i < len(self.raw_dataset):
            batch_data_Raw, batch_data, batch_data2, lengths, length1, length2, position, pred_target, pred_mask = [], [], [], [], [], [], [], [], []
            if self.i + self.bacth_size > len(self.raw_dataset):
                self.bacth_size = len(self.raw_dataset) - self.i
            for j in range(self.i, self.i + self.bacth_size):
                batch_data_Raw.append([[1] + self.raw_dataset[j][0] + [1], [0] + self.raw_dataset[j][1] + [self.raw_dataset[j][1][-1] + 15]])
                batch_data.append([[1] + self.raw_dataset[j][0] + [1], [0] + self.raw_dataset[j][1] + [self.raw_dataset[j][1][-1] + 15]])
                lengths.append(len(self.raw_dataset[j][0]))
                length1.append(len(self.raw_dataset[j][0]) + 2)
            for length_idx in range(len(lengths)):
                length = lengths[length_idx]
                shuffle_token_list = self.get_start_mask(length)
                position.append(shuffle_token_list)
                batch_data_mask = self.mask_word(np.array([batch_data_Raw[length_idx][0][shuffle_index] for shuffle_index in shuffle_token_list]))
                pred_target.extend([batch_data_Raw[length_idx][0][shuffle_index] for shuffle_index in shuffle_token_list])
                for idx, shuffle_index in enumerate(shuffle_token_list):
                    batch_data[length_idx][0][shuffle_index] = batch_data_mask[idx]
                batch_data2.append([[2] + [batch_data_Raw[length_idx][0][shuffle_index] for shuffle_index in shuffle_token_list[:-1]],
                                    [batch_data_Raw[length_idx][1][shuffle_index] for shuffle_index in shuffle_token_list]])
                length2.append(len(shuffle_token_list))
            length1_max = max(length1)
            length2_max = max(length2)
            pred_mask = [l2*[True] + (length2_max-l2)*[False] for l2 in length2]
            batch_data_token = [line[0] + (length1_max-len(line[0]))*[0] for line in batch_data]
            batch_data_time = [line[1] + (length1_max-len(line[1]))*[0] for line in batch_data]
            batch_data2_token = [line[0] + (length2_max-len(line[0]))*[0] for line in batch_data2]
            batch_data2_time = [line[1] + (length2_max-len(line[1]))*[0] for line in batch_data2]
            batch_data_position = [shuffle_position + [0]*(length2_max - len(shuffle_position)) for shuffle_position in position]
            self.i += self.bacth_size
            yield batch_data_token, batch_data_time, batch_data2_token, batch_data2_time, batch_data_position, pred_target, length1, length2, pred_mask
    
    def shuffle_dataset(self):
        random.shuffle(self.raw_dataset)