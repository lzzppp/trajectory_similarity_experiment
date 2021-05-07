import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
# SCHEMA:
# TAXI_ID,LATITUDE,LONGITUDE,TAXI_STATUS,TAXI_DATE,TAXI_TIME,TRANSMIT_DATE,TRANSMIT_TIME,SPEED,DIRECTION,LOCATION,COMPNANY
#
# NOTES:
# 1. TAXI_STATUS MAPPING:
# AVAILABLE(1),BUSY(2),HIRED(3),ONCALL(4),CHANGE SHIFT(5),OTHERS(6)
#
# 2.SPEED IS THE INSTANEOUS SPEED
#
# 3. DIRECTION IS BELIEVED TO BE INACCURATE

token_dict = {}
taxi_shanghai_dataset = pickle.load(open("src/data/trajectory_grid_dataset_shanghai.pkl", "rb"))
for line in taxi_shanghai_dataset:
	for point in line[0]:
		if point not in token_dict:
			token_dict[point] = 0
		else:
			token_dict[point] += 1

token_list = []
for token in token_dict:
	token_list.append(token_dict[token])
	
plt.plot(list(range(1, len(token_list) + 1)), token_list)
plt.show()