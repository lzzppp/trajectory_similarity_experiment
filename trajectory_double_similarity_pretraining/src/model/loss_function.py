
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightMSELoss(nn.Module):
    def __init__(self):
        super(WeightMSELoss, self).__init__()

    def forward(self, input, target, isReLU=False):
        div = target - input.view(-1, 1)
        if isReLU:
            div = F.relu(div.view(-1, 1))
        square = torch.mul(div.view(-1, 1), div.view(-1, 1))

        loss = torch.sum(square)
        return loss


class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.positive_loss = WeightMSELoss()
        self.negative_loss = WeightMSELoss()
    
    def forward(self, near_target, far_target,
                      near_predict, far_predict):
        trajs_mse_loss = self.positive_loss(near_predict, near_target)
        negative_mse_loss = self.negative_loss(far_predict, far_target, isReLU=True)

        loss = sum([trajs_mse_loss, negative_mse_loss])
        return loss