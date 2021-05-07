# from tools import preprocess
# coor_path, data_name = preprocess.trajectory_feature_generation(path= './data/toy_trajs')
from tools.distance_compution import trajectory_distance_combain,trajecotry_distance_list
# import cPickle
import pickle
import numpy as np

def distance_comp(coor_path):
    traj_coord = pickle.load(open(coor_path, 'rb'))
    np_traj_coord = []
    for t in traj_coord:
        np_traj_coord.append(np.array([p for p in t])) # lat lon

    print(np_traj_coord[0])
    print(np_traj_coord[1])
    print(len(np_traj_coord))

    distance_type = 'discret_frechet'

    trajecotry_distance_list(np_traj_coord, batch_size=250, processors=20, distance_type=distance_type,
                             data_name='porto')

    trajectory_distance_combain(1000000, batch_size=250, metric_type=distance_type, data_name='porto')

if __name__ == '__main__':
    distance_comp('porto_trajectory.pkl') # shanghai lat lon
    # distance_comp('example.pkl')
