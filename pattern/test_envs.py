# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import sys
import os
import numpy as np
import random

from envs.sse.SSE_env import SSEEnv
from envs.env_wrappers import DummyVecEnv, SubprocVecEnv
from config import get_config

from pprint import pprint


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = SSEEnv(all_args)
            env.seed(all_args.seed + rank*1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        print(f'{all_args.n_rollout_threads} DummyVecEnv')
        return DummyVecEnv([get_env_fn(0)])
    else:
        print(f'{all_args.n_rollout_threads} SubprocVecEnv')
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def test_env(env, vis=False):
    GU_pattern = env.reset()
    if vis: env.render()
    ABS_pattern = np.zeros((env.world.K, env.world.K), dtype=np.float32)
    print(ABS_pattern.shape)
    for i in range(env.world.n_ABS):
        ABS_pattern[0, i] = 1.
    CGU_pattern = env.step(ABS_pattern)
    if vis: env.render()
    pprint([GU_pattern, ABS_pattern, CGU_pattern])
    env.info()

def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str, default='gridized',
                        help="Which scenario to run on")
    parser.add_argument('--num_agents', type=int,
                        default=1, help="number of agent")

    all_args = parser.parse_known_args(args)[0]

    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # env = SSEEnv(all_args)
    # test_env(env, vis=True)

    envs = make_train_env(all_args)
    GU_patterns = envs.reset()
    envs.render()
    pprint(GU_patterns)
    pprint(GU_patterns.shape)
    pprint(np.sum(GU_patterns))
    while True:
        pass

if __name__ == '__main__':
    main(sys.argv[1:])