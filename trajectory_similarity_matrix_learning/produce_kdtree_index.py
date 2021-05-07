import pickle
import random
import kdtree
from tqdm import tqdm
from multiprocessing import Pool

def run_index(batchi, kd, k_trajs=5):
    index_sequence = pickle.load(open("split_cache/train_sequence_split_"+str(batchi)+".pkl", "rb"))
    index_lists = []
    for sequence in index_sequence:
        k_d_tree_list = kd.search_knn(sequence, k_trajs + 1)[1:]
        # index_list = [k_d_node[0].label for k_d_node in k_d_tree_list]
        # assert idx not in index_list
        index_lists.append([k_d_node[0].label for k_d_node in k_d_tree_list])
    pickle.dump(index_lists, open ("features/kdtree_index_lists_" + str(batchi) + ".pkl", "wb"))
    print(str(batchi) + ' processor done !')
    
idx = 0
batch_size = 2000

train_sequence_raw = pickle.load(open("/home/xiaoziyang/Downloads/trajectory/bert_similarity/porto_train_sequence.pkl", "rb"))
train_grid_raw = pickle.load(open("/home/xiaoziyang/Downloads/trajectory/bert_similarity/porto_train_grid.pkl", "rb"))
# train_kd_tree = pickle.load(open("/home/xiaoziyang/Downloads/trajectory/porto_train_kd_tree.pkl", "rb"))
train_sequence = []
for train_sequenc, train_grid in zip(train_sequence_raw, train_grid_raw):
    if len(train_grid[0]) > 32:
        train_sequence.append(train_sequenc)

train_kd_tree = kdtree.KDTree(train_sequence, list(range(len(train_sequence))))

batch_num = int(len(train_sequence) / batch_size) + 1
print("batch_num: ", batch_num)

# split train sequences
for batch_i in range(batch_num):
    if batch_i == batch_num - 1:
        split_dataset = train_sequence[batch_i * batch_size:]
    else:
        split_dataset = train_sequence[batch_i * batch_size:(batch_i + 1) * batch_size]
    pickle.dump(split_dataset, open("split_cache/train_sequence_split_"+str(batch_i)+".pkl", "wb"))

del train_sequence, split_dataset

processor=47
p=Pool(processor)
for batch_j in range(batch_num):
    p.apply_async(run_index, args=(batch_j, train_kd_tree,))
    print(str(batch_j) + ' processor started !')
p.close()
p.join()
