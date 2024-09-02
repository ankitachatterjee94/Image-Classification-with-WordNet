import numpy as np
import torch
import torch.nn.functional as F
from typing import List


def loss_function(pred, target, soft_labels):
    y = np.zeros(pred.size())
    for i in range(len(target)):
        label = target[i]
        y[i] = soft_labels[label]
    y = torch.FloatTensor(y).to(pred.device)
    pred = F.log_softmax(pred, dim=1)
    loss = y * pred
    loss = -1. * loss.sum(-1).mean()
    return loss
