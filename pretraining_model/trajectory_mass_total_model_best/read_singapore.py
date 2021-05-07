import pickle
from tqdm import tqdm
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

taxi_driver = {}
taxi_date = {}
text_file = open("/mnt/data4/lizepeng/Singapore/Output.txt")
for line in tqdm(text_file):
	line = line.rstrip("\n")
	line = line.split(",")
	if line[0] not in taxi_driver:
		taxi_driver[line[0]] = len(taxi_driver)
	if line[4] not in taxi_date:
		taxi_date[line[4]] = len(taxi_date)
text_file.close()
pickle.dump([taxi_driver, len(taxi_driver)], open("taxi_driver_dict.pkl", "wb"))
pickle.dump([taxi_date, len(taxi_date)], open("taxi_date_dict.pkl", "wb"))
print("read taxi driver done!")