import csv
import pickle
from tqdm import tqdm

porto_dataset = []
csv_reader = csv.reader(open("data/train.csv"))
next (csv_reader)
for line in tqdm (csv_reader):
    trajectory = []
    line = line[-1][1:-1].split('],[')
    # print(line[0].lstrip('[').split(","))
    if len(line) == 1:
        if line != ['']:
            trajectory.append([float(item) for item in line[0].lstrip('[').rstrip(']').split(',')])
    elif len(line) == 2:
        trajectory.append ((float (line[0].lstrip ('[').split (",")[0]),
                            float (line[0].lstrip ('[').split (",")[1])))
        # for checkin_point in line[1:-1]:
            # trajectory.append ([float (item) for item in checkin_point.split (',')])
        trajectory.append ((float (line[-1].rstrip (']').split (",")[0]),
                            float (line[-1].rstrip (']').split (",")[1])))
    else:
        trajectory.append ((float (line[0].lstrip ('[').split (",")[0]),
                            float (line[0].lstrip ('[').split (",")[1])))
        for checkin_point in line[1:-1]:
            trajectory.append ([float (item) for item in checkin_point.split (',')])
        trajectory.append ((float (line[-1].rstrip (']').split (",")[0]),
                            float (line[-1].rstrip (']').split (",")[1])))
    porto_dataset.append (trajectory)

csv_reader = csv.reader(open("data/test.csv"))
next (csv_reader)
for line in tqdm (csv_reader):
    trajectory = []
    line = line[-1][1:-1].split ('],[')
    # print(line[0].lstrip('[').split(","))
    if len (line) == 1:
        if line != ['']:
            trajectory.append ([float (item) for item in line[0].lstrip ('[').rstrip (']').split (',')])
    elif len (line) == 2:
        trajectory.append ((float (line[0].lstrip ('[').split (",")[0]),
                            float (line[0].lstrip ('[').split (",")[1])))
        # for checkin_point in line[1:-1]:
        # trajectory.append ([float (item) for item in checkin_point.split (',')])
        trajectory.append ((float (line[-1].rstrip (']').split (",")[0]),
                            float (line[-1].rstrip (']').split (",")[1])))
    else:
        trajectory.append ((float (line[0].lstrip ('[').split (",")[0]),
                            float (line[0].lstrip ('[').split (",")[1])))
        for checkin_point in line[1:-1]:
            trajectory.append ([float (item) for item in checkin_point.split (',')])
        trajectory.append ((float (line[-1].rstrip (']').split (",")[0]),
                            float (line[-1].rstrip (']').split (",")[1])))
    porto_dataset.append (trajectory)

pickle.dump (porto_dataset, open ("porto_raw_dataset.pkl", "wb"))
