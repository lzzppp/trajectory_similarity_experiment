
import pickle
from glob import glob

if __name__ == "__main__":
	index_list_file = ["features/kdtree_index_lists_", ".pkl"]
	for i in range(670):
		if i == 0:
			data = [[int(token) for token in token_list] for token_list in pickle.load(open(index_list_file[0] + str(i) + index_list_file[1], "rb"))]
		else:
			data.extend([[int(token) for token in token_list] for token_list in pickle.load(open(index_list_file[0] + str(i) + index_list_file[1], "rb"))])
	print(data)
	print(len(data))
	pickle.dump(data, open("/mnt/data4/lizepeng/porto/bert_similarity/porto_kdtree_index_list", "wb"))