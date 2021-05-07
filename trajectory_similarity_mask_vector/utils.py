import re
import sys
import logging
from tqdm import tqdm
import os
import re
import sys
import time
import pickle
import random
import inspect
import getpass
import argparse
import subprocess
import numpy as np
import torch
import logging
from torch import optim
from datetime import timedelta

porto_range = {
        "lon_min": -8.735152,
        "lat_min": 40.953673,
        "lon_max": -8.156309,
        "lat_max": 41.307945
    }

class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''

def create_logger(filepath, rank):
    """
    Create a logger.
    Use a different log file for each process.
    """
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        if rank > 0:
            filepath = '%s-%i' % (filepath, rank)
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger

def initialize_exp(params):
    """
    Initialize the experience:
    - dump parameters
    - create a logger
    """

    # create a logger
    logger = create_logger(os.path.join('train.log'), rank=0)
    # logger.info("============ Initialized logger ============")
    # logger.info("\n".join("%s: %s" % (k, str(v))
                          # for k, v in sorted(dict(vars(params)).items())))
    # logger.info("The experiment will be stored in %s\n" % params.dump_path)
    # logger.info("Running command: %s" % command)
    # logger.info("")
    return logger

def preprocess_trajectory_dataset(Index_Lat_Lon_Dataset, Index_Dict, latitude_unit, longitude_unit, latitude_min, longitude_min):
    preprocess_trajectories = []
    for trajectory in tqdm(Index_Lat_Lon_Dataset):
        grid = [Index_Dict[p[0]] for p in trajectory[0]]
        gps_trajectory = [[latitude_min + latitude_unit*p[2],
                           longitude_min + longitude_unit*p[1]] for p in trajectory[0]] # lat lon
        preprocess_trajectories.append([gps_trajectory, grid])
    return preprocess_trajectories


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

class PARAMS (object):
    def __init__(self):
        # self.n_words = 22339 # shanghai
        self.n_words = 13416  # porto
        self.pad_index = 0
        self.eos_index = 1
        self.mask_index = 2
        self.path = "/home/xiaoziyang/Github/tert_model_similarity/model_store/porto_encoder_decoder_27.pth"
        self.n_layers = 4
        self.batch_size = 128
        self.epochs = 200
        self.embed_size = 128
        self.test_batch_size = 128
        self.test_trajectory_prob = 0.1