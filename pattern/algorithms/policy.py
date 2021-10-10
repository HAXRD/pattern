# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.nets.ed import EncoderDecoder
from utils.loss import mse

class NaivePolicy:
    """
    Naive policy class. Used for predicting ABS_patterns.
    """
    def __init__(self, args, device=torch.device("cpu")):
        self.args = args
        self.least_policy_buffer_size = args.least_policy_buffer_size
        self.num_train_policy = args.num_train_policy
        self.policy_batch_size = args.policy_batch_size
        

        self.device = device
        self.model = EncoderDecoder(1, 1).to(device)
        self.optim = torch.optim.Adam(self.model.parameters(),
                                      lr=self.args.policy_lr)

    def train(self, buffer):
        train_loss = 0.
        if buffer.size > self.least_policy_buffer_size:
            for _ in range(self.num_train_policy):
                data = buffer.sample(self.policy_batch_size)
                loss = self.update(data)
                train_loss += loss
            total_size = self.num_train_policy * self.policy_batch_size
            train_loss /= total_size
        return train_loss
    
    def update(self, data):
        return_loss = 0.
        GU_patterns, ABS_patterns = data

        bz = GU_patterns.size()[0]
        GU_patterns = torch.FloatTensor(GU_patterns).to(self.device)
        ABS_patterns = torch.FloatTensor(ABS_patterns).to(self.device).view(bz, -1)

        self.optim.zero_grad()
        pred_ABS_patterns = self.model(GU_patterns).view(bz, -1)

        loss = mse(pred_ABS_patterns, ABS_patterns)
        loss.backward()

        return_loss += loss.item() * bz
        return return_loss

class DistributionalPolicy:
    def __init__(self, args, device=torch.device("cpu")):
        raise NotImplementedError