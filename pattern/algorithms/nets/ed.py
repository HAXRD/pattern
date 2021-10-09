# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import torch.nn as nn
import torch.nn.functional as F

class EncoderDecoder(nn.Module):
    def __init__(self, n_in_chs, n_out_chs, ):
        super(EncoderDecoder, self).__init__()
        self.n_in_chs  = n_in_chs
        self.n_out_chs = n_out_chs

        # Encoder
        self.conv1 = nn.Conv2d(n_in_chs, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(32, 64, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(64, n_out_chs, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))
        return x