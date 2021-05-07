
import pickle
from glob import glob
import matplotlib.pyplot as plt

if __name__ == "__main__":
	
	losses = pickle.load(open("model_store/loss_list_1.pkl", "rb")) + pickle.load(open("model_store/loss_list_2.pkl", "rb")) + pickle.load(open("model_store/loss_list_3.pkl", "rb")) + pickle.load(open("model_store/loss_list_4.pkl", "rb")) + pickle.load(open("model_store/loss_list_5.pkl", "rb")) + \
	         pickle.load(open("model_store/loss_list_6.pkl", "rb")) + pickle.load(open("model_store/loss_list_7.pkl", "rb")) + pickle.load(open("model_store/loss_list_8.pkl", "rb")) + pickle.load(open("model_store/loss_list_9.pkl", "rb"))
	loss_files = glob("model_store/loss_list_1*.pkl")
	for idx, loss_file in enumerate(loss_files[1:]):
		losses += pickle.load(open(loss_file, "rb"))
	
	loss_files = glob ("model_store/loss_list_2*.pkl")
	for idx, loss_file in enumerate (loss_files[1:]):
		losses += pickle.load (open (loss_file, "rb"))
	
	loss_files = glob ("model_store/loss_list_3*.pkl")
	for idx, loss_file in enumerate (loss_files[1:]):
		losses += pickle.load (open (loss_file, "rb"))
	
	loss_files = glob ("model_store/loss_list_4*.pkl")
	for idx, loss_file in enumerate (loss_files[1:]):
		losses += pickle.load (open (loss_file, "rb"))
	
	loss_files = glob ("model_store/loss_list_5*.pkl")
	for idx, loss_file in enumerate (loss_files[1:]):
		losses += pickle.load (open (loss_file, "rb"))
	
	loss_files = glob ("model_store/loss_list_6*.pkl")
	for idx, loss_file in enumerate (loss_files[1:]):
		losses += pickle.load (open (loss_file, "rb"))
	
	loss_files = glob ("model_store/loss_list_7*.pkl")
	for idx, loss_file in enumerate (loss_files[1:]):
		losses += pickle.load (open (loss_file, "rb"))
	
	loss_files = glob ("model_store/loss_list_8*.pkl")
	for idx, loss_file in enumerate (loss_files[1:]):
		losses += pickle.load (open (loss_file, "rb"))
	
	loss_files = glob ("model_store/loss_list_9*.pkl")
	for idx, loss_file in enumerate (loss_files[1:]):
		losses += pickle.load (open (loss_file, "rb"))
	
	plt.plot(list(range(len(losses))), losses)
	plt.show()