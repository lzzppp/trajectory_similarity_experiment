import torch
import pickle
from glob import glob
import matplotlib.pyplot as plt

def predict(tensor, pred_mask, y=None, get_scores=None):
	"""
	Given the last hidden state, compute word scores and/or the loss.
		`pred_mask` is a ByteTensor of shape (slen, bs), filled with 1 when
			we need to predict a word
		`y` is a LongTensor of shape (pred_mask.sum(),)
		`get_scores` is a boolean specifying whether we need to return scores
	"""
	masked_tensor = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, 5)
	print(tensor[pred_mask.unsqueeze(-1).expand_as(tensor)])
	# scores, loss = self.pred_layer (masked_tensor, y, get_scores)
	# return scores, loss

# tensor = torch.rand([2, 5, 5])
# tensor = torch.Tensor([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]],
#                        [[16, 17, 18, 19, 20], [11, 12, 13, 14, 15], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]])
# print(tensor)
# pred_mask = torch.BoolTensor([[True]*2+3*[False], [True]*3+2*[False]])
#
# predict(tensor, pred_mask)

# data = pickle.load(open("model_store/loss_list_1.pkl", "rb"))
# print(data)
# data2 = pickle.load(open("model_store/loss_list_2.pkl", "rb"))
# print(data2)
# data3 = pickle.load(open("model_store/loss_list_3.pkl", "rb"))
# print(data3)
# data4 = pickle.load(open("model_store/loss_list_4.pkl", "rb"))

data_all = []

data_all += pickle.load(open("model_store6/loss_0.pkl", "rb"))
data_all += pickle.load(open("model_store6/loss_1.pkl", "rb"))
data_all += pickle.load(open("model_store6/loss_2.pkl", "rb"))
# data_all += pickle.load(open("model_store5/loss_list_4.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_5.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_6.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_7.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_8.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_9.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_10.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_11.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_12.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_13.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_14.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_15.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_16.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_17.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_18.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_19.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_20.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_21.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_22.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_23.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_24.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_25.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_26.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_27.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_28.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_29.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_30.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_31.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_32.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_33.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_34.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_35.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_36.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_37.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_38.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_39.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_40.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_41.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_42.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_43.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_44.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_45.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_46.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_47.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_48.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_49.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_50.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_51.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_52.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_53.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_54.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_55.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_56.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_57.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_58.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_59.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_60.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_61.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_62.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_63.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_64.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_65.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_66.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_67.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_68.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_69.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_70.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_71.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_72.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_73.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_74.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_75.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_76.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_77.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_78.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_79.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store5/loss_list_80.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store3/loss_list_81.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store3/loss_list_82.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store3/loss_list_83.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store3/loss_list_84.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store3/loss_list_85.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store3/loss_list_86.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store3/loss_list_87.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store3/loss_list_88.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store3/loss_list_89.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store3/loss_list_90.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store3/loss_list_91.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store3/loss_list_92.pkl", "rb"))[1:]
# data_all += pickle.load(open("model_store3/loss_list_93.pkl", "rb"))[1:]
#
# data_all_old = []
# data_all_old += pickle.load(open("model_store/loss_list_1.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_2.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_3.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_4.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_5.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_6.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_7.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_8.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_9.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_10.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_11.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_12.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_13.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_14.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_15.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_16.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_17.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_18.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_19.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_20.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_21.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_22.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_23.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_24.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_25.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_26.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_27.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_28.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_29.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_30.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_31.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_32.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_33.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_34.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_35.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_36.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_37.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_38.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_39.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_40.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_41.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_42.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_43.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_44.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_45.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_46.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_47.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_48.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_49.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_50.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_51.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_52.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_53.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_54.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_55.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_56.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_57.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_58.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_59.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_60.pkl", "rb"))[1:]
# data_all_old += pickle.load(open("model_store/loss_list_61.pkl", "rb"))[1:]

print(min(data_all))
plt.plot(list(range(len(data_all))), data_all, label="pre-train-plus")
# plt.plot(list(range(len(data_all_old[:len(data_all)]))), data_all_old[:len(data_all)], label="pre-train")
plt.legend()
plt.show()