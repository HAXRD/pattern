# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import imp
import os.path as osp

def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    return imp.load_source('', pathname)