# import cPickle
import pickle
import traj_dist.distance as  tdist
import numpy as  np
import multiprocessing
import sys
from .SSM import DynamicTimeWarpingLoop

def trajectory_distance(traj_feature_map, traj_keys,  distance_type = "hausdorff", batch_size = 50, processors = 30):
    # traj_keys= traj_feature_map.keys()
    trajs = []
    for k in traj_keys:
        traj = []
        for record in traj_feature_map[k]:
            traj.append([record[1],record[2]])
        trajs.append(np.array(traj))

    pool = multiprocessing.Pool(processes=processors)
    # print np.shape(distance)
    batch_number = 0
    for i in range(len(trajs)):
        if (i!=0) & (i%batch_size == 0):
            print (batch_size*batch_number, i)
            pool.apply_async(trajectory_distance_batch, (i, trajs[batch_size*batch_number:i], trajs, distance_type,
                                                         'porto'))
            batch_number+=1
    pool.close()
    pool.join()

def trajecotry_distance_list(trajs, distance_type = "hausdorff", batch_size = 50, processors = 30, data_name = 'porto'):
    pool = multiprocessing.Pool(processes=processors)
    # print np.shape(distance)
    batch_number = 0
    for i in range(len(trajs) + 1):
        if (i!=0) & (i % batch_size == 0):
            print (batch_size*batch_number, i)
            pool.apply_async(trajectory_distance_batch, (i, trajs[batch_size*batch_number:i], trajs, distance_type,
                                                         data_name))
            batch_number+=1
    pool.close()
    pool.join()

def trajectory_distance_batch(i, batch_trjs, trjs, metric_type = "dtw", data_name = 'porto'):
    if metric_type == 'lcss' or  metric_type == 'edr' :
        trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type, eps= 0.003)
    # elif metric_type=='erp':
    #     trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type, eps=0.003)
    else:
        trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type)
        # trs_matrix, path_matrix = DynamicTimeWarpingLoop.calculate_dtw_dist_path(batch_trjs, trjs)
    pickle.dump(np.array(trs_matrix), open('./features/'+data_name+'_'+metric_type+'_distance_' + str(i), 'wb'))
    # pickle.dump(path_matrix, open('./features/'+data_name+'_'+'dtw'+'_path_'+str(i), 'wb'))
    print('complete: '+str(i))

def trajectory_distance_combain(trajs_len, batch_size = 100, metric_type = "dtw", data_name = 'porto'):
    distance_list = []
    a = 0
    for i in range(1,trajs_len+1):
        if (i!=0) & (i%batch_size == 0):
            distance_list.append(pickle.load(open('./features/'+data_name+'_'+metric_type+'_distance_' + str(i), 'rb')))
            print(distance_list[-1].shape)
    a = distance_list[-1].shape[1]
    distances = np.array(distance_list)
    # print(distances.shape)
    all_dis = distances.reshape((trajs_len,a))
    # print(all_dis.shape)
    pickle.dump(all_dis,open('./features/'+data_name+'_'+metric_type+'_distance_all_'+str(trajs_len),'wb'))
    np.save('./features/'+data_name+'_'+metric_type+'_distance_all_'+str(trajs_len)+'.npy', all_dis)
    return all_dis
