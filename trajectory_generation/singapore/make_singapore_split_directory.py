import os
import pickle
from tqdm import tqdm
# import matplotlib.pyplot as plt

driver_dict = pickle.load(open("driver_date/taxi_driver_dict.pkl", "rb"))
date_dict = pickle.load(open("driver_date/taxi_date_dict.pkl", "rb"))

for taxi in tqdm(driver_dict[0]):
	os.system("mkdir data_split/"+taxi)
	for date in date_dict[0]:
		# os.system("mkdir data_split/"+taxi+"/"+date+".pkl")
		pickle.dump([], open("./data_split/"+taxi+"/"+date+".pkl", "wb"))