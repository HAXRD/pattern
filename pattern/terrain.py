# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

# Generate terrain (site specific) data

import os
import argparse
import random
import numpy as np
import scipy.io as sio
from pprint import pprint

def gen_BMs(world_len: int, mesh_len: int, N: int, save_dir: str, seed=123):
    '''
    Generate a world with building meshes.
    '''
    random.seed(seed)
    np.random.seed(seed)

    fname = f'terrain-{N}.mat'
    assert world_len % mesh_len == 0, f'world_len={world_len}, mesh_len={mesh_len}'
    M = world_len//mesh_len
    assert M*M - N >= 0

    raw_grids = np.ones(M*M, dtype=np.float32) * 90
    zero_idcs = sorted(random.sample(range(0, M*M), M*M-N))
    raw_grids[zero_idcs] = 0.
    grids = raw_grids.reshape((M, M)).astype(np.float32)

    mat = {
        'world_len': world_len,
        'mesh_len': mesh_len,
        'N': N,
        'grids': grids,
    }

    save_path = os.path.join(save_dir, fname)
    sio.savemat(save_path, mat)
    return mat, save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-len', type=int, default=1000)
    parser.add_argument('--mesh-len', type=int, default=50)
    parser.add_argument('--N', type=int, default=60)
    parser.add_argument('--save-dir', type=str, default='./')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    mat, save_path = gen_BMs(args.world_len, args.mesh_len, args.N, args.save_dir, args.seed)
    pprint(mat)
    pprint(save_path)