
import pickle
import numpy as np
from tqdm import tqdm

max_len=512

class DataLoader(object):
    
    def __init__(self, path):
        self.path = path
        self.raw_dataset = []
        self.token_dict = {}
        self.time_dict = {}
        self.load()
        
    def load(self):
        raw_dataset = pickle.load(open(self.path, "rb"))
        dataset = []
        for i in tqdm(range(len(raw_dataset)), desc="Loading data"):
            for token, time in zip(raw_dataset[i][0], raw_dataset[i][1]):
                if token not in self.token_dict:
                    self.token_dict[token] = len(self.token_dict) + 3
                if time not in self.time_dict:
                    self.time_dict[time] = len(self.time_dict) + 1
            dataset.append([[self.token_dict[token] for token in raw_dataset[i][0]],
                            [self.time_dict[time] for time in raw_dataset[i][1]]])
        print("length of tokens :", len(self.token_dict))
        print("length of time tokens :", len(self.time_dict))
        self.raw_dataset = dataset