# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms2.nets.cvae import CVAE

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
        self.emulator_replay_per = args.emulator_replay_per

        self.device = device
        self.model = CVAE('emulator', 2, 1, 64, 64).to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr) # TODO: might need to add scheduler to adjust initial data-non-sufficient problem.

    def train(self, buffer):
        kld_weight = self.emulator_batch_size / buffer.max_size
        train_loss = 0.
        if buffer.size > self.least_emulator_buffer_size:
            for _ in range(self.num_train_emulator):
                data = buffer.sample(self.emulator_batch_size)
                loss = self.update(data, kld_weight)
                train_loss += loss
            total_size = self.num_train_emulator * self.emulator_batch_size
            train_loss /= total_size
        else:
            train_loss = None
            print(f'[emulator train] not enough buffer {buffer.size} < {self.least_emulator_buffer_size}, skipping...')
        return train_loss

    def update(self, data, kld_weight):
        return_loss = 0.
        if self.emulator_replay_per == True:
            raise NotImplementedError
        else:
            batch_P_GU, batch_P_ABS, batch_P_CGU = data
            batch_P_GU  = batch_P_GU.to(self.device)
            batch_P_ABS = batch_P_ABS.to(self.device)
            batch_P_CGU = batch_P_CGU.to(self.device)
            batch_size = batch_P_GU.size()[0]
            
            rtns = self.model(batch_P_GU, batch_P_ABS, batch_P_CGU)
            loss_dict = self.model.loss_function(*rtns, kld_weight=kld_weight)
            loss = loss_dict['loss']
            self.optim.zero_grad()
            loss.backward()

            return_loss += loss.item() * batch_size
        return return_loss