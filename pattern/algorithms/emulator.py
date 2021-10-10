# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.nets.ed import EncoderDecoder
from utils.loss import hybrid_mse

def _t2n(x):
    return x.detach().cpu().numpy()

class Emulator:
    """
    Env emulator class. Used for planning.
    """
    def __init__(self, args, device=torch.device("cpu")):
        self.lr = args.emulator_lr
        self.K = int(args.world_len // args.granularity)
        self.least_emulator_buffer_size = args.least_emulator_buffer_size
        self.num_train_emulator = args.num_train_emulator
        self.emulator_batch_size = args.emulator_batch_size

        self.device = device
        self.model = EncoderDecoder(2, 1).to(device)
        self.optim = torch.optim.Adam(self.model.parameters(),
                                      lr=self.lr)

    def train(self, buffer):
        train_loss = 0.
        if buffer.size > self.least_emulator_buffer_size:
            for _ in range(self.num_train_emulator):
                data = buffer.sample(self.emulator_batch_size)
                loss = self.update(data)
                train_loss += loss
            total_size = self.num_train_emulator * self.emulator_batch_size
            train_loss /= total_size
        return train_loss

    def update(self, data):
        return_loss = 0.
        if self.emulator_replay_per == True:
            raise NotImplementedError
        else:
            GU_patterns, ABS_patterns, CGU_patterns = data

            bz = GU_patterns.size()[0]
            GU_patterns  = torch.FloatTensor(GU_patterns).to(self.device)
            ABS_patterns = torch.FloatTensor(ABS_patterns).to(self.device)
            CGU_patterns = torch.FloatTensor(CGU_patterns).to(self.device).view(bz, -1)

            self.optim.zero_grad()
            pred_CGU_patterns = self.model(torch.cat((GU_patterns, ABS_patterns), dim=1)).view(bz, -1)

            loss = hybrid_mse(pred_CGU_patterns)
            loss.backward()
            
            return_loss += loss.item() * bz
        return return_loss

