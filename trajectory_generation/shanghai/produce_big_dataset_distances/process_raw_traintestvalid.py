# coding=utf-8
import pickle
import random
import cPickle

# "minlon":-8.735152,
# "minlat":40.953673,
# "maxlon":-8.156309,
# "maxlat":41.307945,

lon_range = [-8.735152, -8.156309]
lat_range = [40.953673, 41.307945]

with open("porto_bigger50_smaller600_dataset.pkl", "rb") as f:
    w = cPickle.load(f)
#
dataset_50 = []
# # for line in w:
# #     if len(line) < 600:
# #         dataset_50.append(line)
# #
# # pickle.dump(dataset_50, open("porto_bigger50_smaller600_dataset.pkl","wb"), protocol=2)

# for trajectory in w:
#     plus = True
#     for lon, lat in trajectory:
#         if lon < lon_range[0] or lon > lon_range[1] or lat < lat_range[0] or lat > lat_range[1]:
#             plus = False
#     if plus:
#         dataset_50.append(trajectory)
#
# pickle.dump(dataset_50, open("porto_in_range_dataset.pkl","wb"), protocol=2)

# with open("porto_in_range_dataset.pkl", "rb") as f:
#     w = cPickle.load(f)
#
# print "Loads raw dataset !!!"
#
# data_length = len(w)
# random.shuffle(w)
# train_size = int(len(w) * 0.3)
# valid_size = int(len(w) * 0.1)
# train_data = w[:train_size]
# valid_data = w[train_size:train_size + valid_size]
# test_data = w[train_size + valid_size:]
# # for trajectiry in w:
# #     print(trajectiry)
# # choosen_data = random.sample(w, 5000)
# # print(len(w))
# pickle.dump(train_data, open("porto_train.pkl","wb"), protocol=2)
# pickle.dump(valid_data, open("porto_valid","wb"), protocol=2)
# pickle.dump(test_data, open("porto_test.pkl","wb"), protocol=2)
data = pickle.load(open("porto_train.pkl"))
random_choosen = random.sample(data, 6000)
pickle.dump(random_choosen, open("porto_choosen_dataset.pkl", "wb"), protocol=2)
