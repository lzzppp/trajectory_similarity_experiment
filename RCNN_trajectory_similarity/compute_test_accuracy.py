import torch
import heapq
import pickle
import numpy as np
from tqdm import tqdm
from src.models import OCD
import torch.nn.functional as F

model = OCD(input_channel=1, cls_num=1).cuda()
model.load_state_dict(torch.load("/home/xiaoziyang/Github/trajectory_similarity_rnn_cnn_learning/ocd_porto25.pt"))
model.eval()
embedding_matrix = np.load("porto_test_trajectory_embedding_matrix.npy")
batch_size = 10000

with torch.no_grad():
    all_num_10 = 0
    all_num_50 = 0
    for i in tqdm(range(1,41)):
        distance_matrix = pickle.load(open("/home/xiaoziyang/Github/trajectory_similarity_matrix_learning/features/porto_discret_frechet_distance_" + str(i*25), "rb"))
        for j in range(distance_matrix.shape[0]):
            distance = list(distance_matrix[j, :])
            pred_distance = []
            anchor_embedding = torch.FloatTensor(np.expand_dims(embedding_matrix[(i - 1) * 25 + j, :], 0).repeat(batch_size, axis=0)).cuda()
            for index in range(1):
                pair_embedding = torch.FloatTensor(embedding_matrix[index*batch_size:(index+1)*batch_size, :]).cuda()
                feature_predict = model.rnn_encoder.transform(torch.cat((anchor_embedding, pair_embedding), dim=-1)).unsqueeze(2)
                x = F.relu(model.conv14(feature_predict.unsqueeze(3)))
                # x = F.dropout(x, drop_rate)
                x = F.relu(model.conv15(x))
                # x = F.dropout(x, drop_rate)
                x = F.relu(model.upsampconv1(x))

                x = F.relu(model.conv16(x))
                # x = F.dropout(x, drop_rate)
                x = F.relu(model.upsampconv2 (x))

                x = F.relu(model.conv17(x))
                # x = F.dropout(x, drop_rate)
                x = F.relu(model.upsampconv3(x))

                x = F.relu(model.conv18(x))
                # x = F.dropout(x, drop_rate)
                x = F.relu(model.upsampconv4(x))

                x = F.relu(model.conv19(x))
                # x = F.dropout(x, drop_rate)
                x = F.relu(model.upsampconv5(x))

                x = F.relu(model.conv20(x))
                # x = F.dropout(x, drop_rate)
                x = model.conv21(x).squeeze()
                
                pred_distance.extend(list(x[:, -1, -1].cpu().numpy()))
            pred_topk_10 = heapq.nlargest(11, range(len(pred_distance)), pred_distance.__getitem__)[1:]
            dist_topk_10 = heapq.nsmallest(11, range(len(distance)), distance.__getitem__)[1:]
            for predi in pred_topk_10:
                if predi in dist_topk_10:
                    all_num_10 += 1
            pred_topk_50 = heapq.nlargest(51, range(len(pred_distance)), pred_distance.__getitem__)[1:]
            dist_topk_50 = heapq.nsmallest(51, range(len(distance)), distance.__getitem__)[1:]
            for predi in pred_topk_50:
                if predi in dist_topk_50:
                    all_num_50 += 1
                    
print((all_num_10*1.0)/10000.0)
print((all_num_50*1.0)/50000.0)
