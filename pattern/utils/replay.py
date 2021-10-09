# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import torch
import numpy as np

class EmulatorNaiveReplay(object):
    """
    Buffer to store emulator training data.
    ::
    """
    def __init__(self, K, max_size):
        self.K = K
        self.max_size = max_size
        self.size = 0
        self.ptr = 0

        self.GU_patterns = np.zeros((max_size, 1, K, K), dtype=np.float32)
        self.ABS_patterns = np.zeros((max_size, 1, K, K), dtype=np.float32)
        self.CGU_patterns = np.zeros((max_size, 1, K, K), dtype=np.float32)

    def add(self, data):
        GU_pattern, ABS_pattern, _, CGU_pattern = data
        self.GU_patterns[self.ptr] = GU_pattern.reshape(1, self.K, self.K)
        self.ABS_patterns[self.ptr] = ABS_pattern.reshape(1, self.K, self.K)
        self.CGU_patterns[self.ptr] = CGU_pattern.reshape(1, self.K, self.K)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def shuffle(self):
        perm = np.arange(self.size)
        np.random.shuffle(perm)

        self.GU_patterns  = self.GU_patterns[perm]
        self.ABS_patterns = self.ABS_patterns[perm]
        self.CGU_patterns = self.CGU_patterns[perm]
        print('shuffle...')

    def sample(self, batch_size):
        idc = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.GU_patterns[idc]),
            torch.FloatTensor(self.ABS_patterns[idc]),
            torch.FloatTensor(self.CGU_patterns[idc])
        )

    def data_loader(self, batch_size):
        cur = 0
        while cur + batch_size <= self.size:
            idc = np.arange(cur, cur + batch_size)
            cur += batch_size
            yield (
                torch.FloatTensor(self.GU_patterns[idc]),
                torch.FloatTensor(self.ABS_patterns[idc]),
                torch.FloatTensor(self.CGU_patterns[idc])
            )

class EmulatorPrioritizedReplay(object):
    def __init__(self):
        pass

class PolicyReplay(object):
    def __init__(self, K, max_size):
        self.K = K
        self.max_size = max_size
        self.size = 0
        self.ptr = 0

        self.GU_patterns = np.zeros((max_size, 1, K, K), dtype=np.float32)
        self.ABS_patterns = np.zeros((max_size, 1, K, K), dtype=np.float32)

    def add(self, data):
        GU_patterns, ABS_patterns, _ , _ = data
        self.GU_patterns[self.ptr] = GU_patterns.reshape(1, self.K, self.K)
        self.ABS_patterns[self.ptr] = ABS_patterns.reshape(1, self.K, self.K)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idc = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.GU_patterns[idc]),
            torch.FloatTensor(self.ABS_patterns[idc])
        )
