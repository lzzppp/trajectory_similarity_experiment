# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

drop_rate = 0.5

# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

class f_hid_sim (nn.Module):
    def __init__(self, hidden_size):
        super (f_hid_sim, self).__init__ ()
        self.linear_i = nn.Linear (hidden_size, hidden_size)
        self.linear_c = nn.Linear (hidden_size, hidden_size)
        self.linear_o = nn.Linear (hidden_size, hidden_size)
        self.init_weight ()
    
    def init_weight(self):
        nn.init.uniform_ (self.linear_i.weight)
        nn.init.uniform_ (self.linear_o.weight)
        nn.init.uniform_ (self.linear_c.weight)
        nn.init.zeros_ (self.linear_i.bias)
        nn.init.zeros_ (self.linear_o.bias)
        nn.init.zeros_ (self.linear_c.bias)
    
    def forward(self, hi):
        C = torch.mul (torch.sigmoid (self.linear_i (hi)),
                       torch.tanh (self.linear_c (hi)))
        hi_ = torch.mul (torch.sigmoid (self.linear_o (hi)),
                         torch.tanh (C))
        output = hi_ + hi
        return output

class HighwayMLP(nn.Module):
    def __init__(self,
                 input_size,
                 gate_bias=-2,
                 activation_function=nn.functional.relu,
                 gate_activation=nn.functional.softmax):

        super(HighwayMLP, self).__init__()

        self.activation_function = activation_function
        self.gate_activation = gate_activation

        self.normal_layer = nn.Linear(input_size, input_size)

        self.gate_layer = nn.Linear(input_size, input_size)
        self.gate_layer.bias.data.fill_(gate_bias)

    def forward(self, x):

        normal_layer_result = self.activation_function(self.normal_layer(x))
        gate_layer_result = self.gate_activation(self.gate_layer(x))

        multiplyed_gate_and_normal = torch.mul(normal_layer_result, gate_layer_result)
        multiplyed_gate_and_input = torch.mul((1 - gate_layer_result), x)

        return torch.add(multiplyed_gate_and_normal,
                         multiplyed_gate_and_input)

class RNNEncoder(nn.Module):
    def __init__(self):
        super(RNNEncoder, self).__init__()
        self.rnn_encoder = nn.LSTM(input_size=130, hidden_size=256, batch_first=True)
        self.memory = nn.Embedding(14000, 128, padding_idx=0)
        self.transform =

    def forward(self, gps_data1, grid1, gps_data2, grid2):
        grid_data1 = self.memory(grid1)
        grid_data2 = self.memory(grid2)

        input_data1 = torch.cat((gps_data1, grid_data1), dim=-1)
        input_data2 = torch.cat((gps_data2, grid_data2), dim=-1)
        
        _, (hidden1, _) = self.rnn_encoder(input_data1)
        _, (hidden2, _) = self.rnn_encoder(input_data2)
        
        hidden = torch.cat((hidden1, hidden2, torch.mul(hidden1, hidden2), torch.abs(hidden1 - hidden2)), x.dim() - 1)
        hidden = self.transform(hidden).unsqueeze(2)
        
        return hidden.unsqueeze(3)

class OCD(nn.Module):
    def __init__(self, input_channel=3, cls_num=1):
        super(OCD, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2, 2),
                                      nn.Conv2d(64, 128, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 128, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2, 2),
                                      nn.Conv2d(128, 256, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 256, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 256, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2, 2),
                                      nn.Conv2d(256, 512, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2, 2),
                                      nn.Conv2d(512, 512, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2, 2))
        
        self.rnn_encoder = RNNEncoder()
        
        self.conv14 = nn.Conv2d(512, 4096, 7, stride=1, padding=3)
        nn.init.xavier_uniform_(self.conv14.weight)
        nn.init.constant_(self.conv14.bias, 0.1)
        # self.conv14_bn = nn.BatchNorm2d(4096)
        
        self.conv15 = nn.Conv2d(4096, 512, 1, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv15.weight)
        nn.init.constant_ (self.conv15.bias, 0.1)
        # self.conv15_bn = nn.BatchNorm2d(512)
        
        self.upsampconv1 = nn.ConvTranspose2d(512, 512, 2, stride=2, padding=0)
        
        self.conv16 = nn.Conv2d(512, 512, 5, stride=1, padding=2)
        nn.init.xavier_uniform_(self.conv16.weight)
        nn.init.constant_(self.conv16.bias, 0.1)
        # self.conv16_bn = nn.BatchNorm2d(512)
        
        self.upsampconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2, padding=0)
        self.conv17 = nn.Conv2d(256, 256, 5, stride=1, padding=2)
        nn.init.xavier_uniform_(self.conv17.weight)
        nn.init.constant_(self.conv17.bias, 0.1)
        # self.conv17_bn = nn.BatchNorm2d(256)
        
        self.upsampconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0)
        self.conv18 = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        nn.init.xavier_uniform_(self.conv18.weight)
        nn.init.constant_(self.conv18.bias, 0.1)
        # self.conv18_bn = nn.BatchNorm2d(128)
        
        self.upsampconv4 = nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0)
        self.conv19 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        nn.init.xavier_uniform_(self.conv19.weight)
        nn.init.constant_(self.conv19.bias, 0.1)
        # self.conv19_bn = nn.BatchNorm2d(64)
        
        self.upsampconv5 = nn.ConvTranspose2d(64, 32, 2, stride=2, padding=0)
        self.conv20 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        nn.init.xavier_uniform_(self.conv20.weight)
        nn.init.constant_(self.conv20.bias, 0.1)
        # self.conv20_bn = nn.BatchNorm2d(32)
        
        self.conv21 = nn.Conv2d(32, cls_num, 5, stride=1, padding=2)
        nn.init.xavier_uniform_(self.conv21.weight)
        nn.init.constant_(self.conv21.bias, 0.1)
    
    def forward(self, x, gps1, grid1, gps2, grid2):
        feature_target = self.features(x)
        
        feature_predict = self.rnn_encoder(gps1, grid1, gps2, grid2)
        
        x = F.relu(self.conv14(feature_predict))
        # x = F.dropout(x, drop_rate)
        x = F.relu(self.conv15(x))
        # x = F.dropout(x, drop_rate)
        x = F.relu(self.upsampconv1(x))
        
        x = F.relu(self.conv16(x))
        # x = F.dropout(x, drop_rate)
        x = F.relu(self.upsampconv2(x))
        
        x = F.relu(self.conv17(x))
        # x = F.dropout(x, drop_rate)
        x = F.relu(self.upsampconv3(x))
        
        x = F.relu(self.conv18 (x))
        # x = F.dropout(x, drop_rate)
        x = F.relu(self.upsampconv4(x))
        
        x = F.relu(self.conv19 (x))
        # x = F.dropout(x, drop_rate)
        x = F.relu(self.upsampconv5(x))
        
        x = F.relu(self.conv20 (x))
        # x = F.dropout(x, drop_rate)
        x = self.conv21(x).squeeze()
        
        # x = torch.exp(-x)
        return x, feature_target.squeeze(), feature_predict.squeeze()