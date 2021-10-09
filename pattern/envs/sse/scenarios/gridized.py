# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import numpy as np
from envs.sse.core import World, ABS, GU, BM
from envs.sse.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self, args, seed=0):
        # instantiate world object with given BMs info
        world = World(args.BMs_fname, seed)
        # set any world properties first
        world.n_ABS = args.n_ABS
        world.n_GU  = args.n_GU
        # add ABSs
        world.ABSs = [ABS(id=i, name=f'ABS-{i}', H=args.h_ABS) for i in range(world.n_ABS)]
        # add GUs
        world.GUs  = [GU(id=i, name=f'GU-{i}', H=args.h_GU) for i in range(world.n_GU)]
        # store velocities
        world.v_ABS = args.v_ABS
        world.v_GU  = args.v_GU
        # store altitudes
        world.h_ABS = args.h_ABS
        world.h_GU  = args.h_GU
        # store episode length (s)
        world.episode_length = args.episode_length
        # radii of NLoS & LoS
        world.R_2D_NLoS = args.R_2D_NLoS
        world.R_2D_LoS  = args.R_2D_LoS
        # store granularity for action-pattern
        world.granularity = args.granularity
        world.K = int(world.world_len // world.granularity)
        assert int(args.world_len) == int(world.world_len), f"{args.world_len}, {world.world_len}"
        assert int(args.granularity) == int(world.granularity), f"{args.granularity}, {world.granularity}"
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):

        # set random initial states
        for _abs in world.ABSs:
            _abs.state = world.gen_1_position(AVOID_COLLISION=False)
        for _gu in world.GUs:
            _gu.state = world.gen_1_position(AVOID_COLLISION=True)

    ############### gridized entity getter methods ###############
    def get_GU_pattern(self, world, NORMALIZED=True):
        granularity = world.granularity
        K = world.K

        GU_pattern = np.zeros((K, K), dtype=np.float32)
        for _gu in world.GUs:
            x_idx, y_idx = np.clip(_gu.state // granularity, 0, K - 1).astype(np.int32)
            GU_pattern[x_idx, y_idx] += 1.

        GU_pattern /= world.n_GU if NORMALIZED else 1.
        return GU_pattern

    def get_ABS_pattern(self, world):
        granularity = world.granularity
        K = world.K

        ABS_pattern = np.zeros((K, K), dtype=np.float32)
        for _abs in world.ABSs:
            x_idx, y_idx = np.clip(_abs.state // granularity, 0, K - 1).astype(np.int32)
            ABS_pattern[x_idx, y_idx] += 1.

        # ABS_pattern /= world.n_ABS if NORMALIZED else 1.
        return ABS_pattern

    def get_CGU_pattern(self, world, NORMALIZED=True):
        granularity = world.granularity
        K = world.K

        CGU_pattern = np.zeros((K, K), dtype=np.float32)
        for _gu in world.GUs:
            if len(_gu.covered_by) > 0:
                x_idx, y_idx = np.clip(_gu.state // granularity, 0, K - 1).astype(np.int32)
                CGU_pattern[x_idx, y_idx] += 1.

        CGU_pattern /= world.n_GU if NORMALIZED else 1.
        return CGU_pattern


