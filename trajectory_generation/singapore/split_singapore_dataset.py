import os
import pickle
from tqdm import tqdm
# import matplotlib.pyplot as plt

driver_dict = pickle.load(open("driver_date/taxi_driver_dict.pkl", "rb"))
date_dict = pickle.load(open("driver_date/taxi_date_dict.pkl", "rb"))

with open("/mnt/data4/lizepeng/Singapore/Output.txt") as file:
    for line in tqdm(file):
        line = line.rstrip("\n")
        line = line.split(",")
        data = pickle.load(open ("data_split/" + line[0] + "/" + line[4] + ".pkl", "rb"))
        data.append(line)
        pickle.dump(data, open ("data_split/" + line[0] + "/" + line[4] + ".pkl", "wb"))