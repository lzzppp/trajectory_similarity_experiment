
import numpy as np

def pad_sequence(traj_grids, maxlen=100, pad_value = 0.0):
    paddec_seqs = []
    for traj in traj_grids:
        pad_r = np.zeros_like(traj[0])*pad_value
        while (len(traj) < maxlen):
            traj.append(pad_r)
        paddec_seqs.append(traj)
    return paddec_seqs

trajs = [[[1, 2, 0.5, 0.6], [3, 4, 0.7, 0.8]],
         [[5, 6, 0.3, 0.4], [7, 8, 1.1, 1.2], [1, 2, 0.5, 0.6], [3, 4, 0.7, 0.8]],
         [[15, 16, 1.3, 1.4], [71, 81, 2.1, 3.2], [21, 52, 1.5, 1.6], [83, 74, 6.7, 1.8]]]
print(pad_sequence(trajs, maxlen=5)[1])