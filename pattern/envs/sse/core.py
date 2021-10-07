# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import os
import scipy.io as sio
import numpy as np
from collections import namedtuple
import seaborn
from numba import jit


COLORs = {
    'grey'  : np.array([0.5, 0.5, 0.5]), # BM color
    'red'   : np.array([0.9, 0.0, 0.0]), # uncovered GU
    'green' : np.array([0.0, 0.9, 0.0]), # covered GU
    'blue'  : np.array([0.0, 0.0, 0.9]), # ABS
    'yellow': np.array([0.9, 0.9, 0.0]), # LoS
    'orange': np.array([1.0, 0.5, 0.0])  # NLoS
}

@jit(nopython=True)
def compute_2D_distance(pos1, pos2):
    assert pos1.shape == (2,)
    assert pos2.shape == (2,)
    return np.sqrt(np.sum((pos1 - pos2)**2))

# basic class for all entities in World class
class Entity(object):
    def __init__(self,
                 id,
                 name,
                 H,
                 state,
                 size=8.):
        self.id    = id   # unique identifier for different subclass instances
        self.name  = name  # f'{subclass_type}-{id}'
        self.H     = H     # height of the entity
        self.state = state # shape=(2,), 2D position
        self.color = None
        self.size  = size

# building mesh class
class BM(Entity):
    def __init__(self,
                 id,
                 name,
                 H,
                 state,
                 size):
        super(BM, self).__init__(id=id, name=name, H=H, state=state, size=size)
        self.color = COLORs['grey']

# aerial base station / unmanned aerial vehicles class
class ABS(Entity):
    def __init__(self,
                 id,
                 name,
                 H,
                 state=None):
        super(ABS, self).__init__(id=id, name=name, H=H, state=state)
        self.action = None # np.ndarray (2,) a specific action within action space
        self.color = COLORs['blue']

# namedtuple for storing GU covered by which ABS(s) info
CoverTuple = namedtuple('CoverTuple', ('id', 'distance'))
# ground user class
class GU(Entity):
    def __init__(self,
                 id,
                 name,
                 H,
                 state=None):
        super(GU, self).__init__(id=id, name=name, H=H, state=state)
        self.action     = None # np.ndarray (2,) a specific action within action space
        self.covered_by = []   # (list[CoverTuple]) a list of tuples that include which ABSs are covering this GU along with the distance between them
        self.color = COLORs['red']

    # sort the `covered_by` list by `distance property`
    def sort_covered_by(self):
        self.covered_by = sorted(self.covered_by, key=lambda x: x.distance)

# site specific world
# entities include:
# - building meshes (BMs)
# - ground users (GUs)
# - aerial base stations / unmanned aerial vehicles (ABSs/UAVs)
class World(object):
    '''
    only the BMs' info is initialized during instantiation.
    :param path_to_load_BMs: path to BMs mat file.
    :param seed: seed for random number generator.
    '''
    def __init__(self, path_to_load_BMs):
        # TODO: maybe remove this
        # # seed setting
        # self.np_rng = np.random.RandomState(0)
        # self.seed = seed if seed is not None else 0
        # self.np_rng.seed(self.seed)

        # load site-specific info from terrain.mat
        world_len, mesh_len, N, grids = self._load_BMs(path_to_load_BMs)
        self.world_len = world_len
        self.mesh_len  = mesh_len         # side len of each BM
        self.n_BM  = N                    # num of BMs
        self.M = int(world_len//mesh_len) # num of maximum meshes along the world side
        self.grids = grids                # shape=(M, M), the value of each elem is the corresponding height

        # list of entities (can change at execution-time!)
        self.BMs = self._process_BMs()
        self.ABSs = []
        self.GUs = []
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # nums of entities
        self.n_ABS = 0
        self.n_GU  = 0
        # velocities of entities
        self.v_ABS = 0.
        self.v_GU  = 0.
        # altitudes of entities
        self.h_ABS = 0.
        self.h_GU  = 0.
        # episode length
        self.episode_length = 0.
        # radii of NLoS & LoS
        self.R_2D_NLoS = 0.
        self.R_2D_LoS  = 0.
        # granularity
        self.granularity = world_len
        self.K = self.world_len // self.granularity

    ############### public methods ###############
    # generate 1 2D position
    def gen_1_position(self, AVOID_COLLISION=True):
        while True:
            if self.dim_p == 2:
                x, y = self.world_len * np.random.rand(self.dim_p)
                if AVOID_COLLISION: # consider collision with BMs
                    x_idx, y_idx = np.clip(np.array([x, y]) // self.mesh_len, 0, self.M - 1).astype(np.int32)
                    if self.grids[x_idx, y_idx] == 0.:
                        return np.array([x, y], dtype=np.float32)
                else: # not consider collision with BMs
                    return np.array([x, y], dtype=np.float32)

    # update state of the world
    def step(self, ABS_UPDATE=False, GU_UPDATE=False):

        # applying the actions of GUs
        if GU_UPDATE:
            for _gu in self.GUs:
                next_state = np.clip(_gu.state + _gu.action * self.v_GU * self.episode_length, 0, self.world_len)
                next_x_idx, next_y_idx = np.clip((next_state // self.mesh_len).astype, 0, self.M - 1)
                if self.grids[next_x_idx, next_y_idx] == 0.:
                    _gu.state = next_state

        # update GUs' covered_by property
        if ABS_UPDATE or GU_UPDATE:
            for _gu in self.GUs:
                # empty out covered_by list
                _gu.covered_by = []
                # recompute covered_by list
                for _abs in self.ABSs:
                    distance = compute_2D_distance(_gu.state, _abs.state)
                    # the GU is covered by ABS only under 2 conditions:
                    # 1. within the NLoS range
                    # 2. within the ring between NLoS and LoS range and is LoS
                    if distance <= self.R_2D_NLoS or \
                       (distance > self.R_2D_NLoS and distance <= self.R_2D_LoS and self._judge_is_LoS(_abs, _gu)):
                        # update `covered_by` property
                        _gu.covered_by.append(CoverTuple(id=_abs.id, distance=distance))
                # change covered GU color
                if len(_gu.covered_by) == 0:
                    _gu.color = COLORs['red']
                else:
                    _gu.color = COLORs['green']

    ############### private methods ###############
    # load BMs' info from mat file
    def _load_BMs(self, path_to_load_BMs):
        if not os.path.exists(path_to_load_BMs) or not os.path.isfile(path_to_load_BMs):
            raise FileNotFoundError()
        else:
            mat = sio.loadmat(path_to_load_BMs)
            return mat['world_len'].item(), mat['mesh_len'].item(), mat['N'].item(), mat['grids']

    # process loaded BMs' info into precise building meshes (easy for rendering)
    def _process_BMs(self):
        BMs = []
        k = 0
        for i in range(self.M):
            for j in range(self.M):
                h = self.grids[i, j]
                if h > 0.:
                    state = self.mesh_len * (np.array([i, j], dtype=np.float32) + 0.5)
                    BMs.append(BM(id=k, name=f'BM-{k}', H=h, state=state, size=self.mesh_len))
        return BMs

    # judge if a given pair of ABS and GU is LoS
    def _judge_is_LoS(self, abs, gu):
        assert isinstance(abs, ABS)
        assert isinstance(gu, GU)
        assert abs.H > gu.H

        dx, dy = gu.state - abs.state

        # calculate `x` value of the intersection between
        # abs-gu line and the given BM edge line (by
        # providing its `y` value)
        def _cal_x(y):
            return abs.state[0] + (y - abs.state[1]) * dx / dy
        def _cal_y(x):
            return abs.state[1] + (x - abs.state[0]) * dy / dx

        abs_x_i, abs_y_i = np.clip(abs.state // self.mesh_len, 0, self.M - 1).astype(np.int32)
        gu_x_i, gu_y_i   = np.clip(gu.state // self.mesh_len, 0, self.M - 1).astype(np.int32)

        for _idx_i in range(min(abs_x_i, gu_x_i), max(abs_x_i, gu_x_i)+1):
            for _idx_j in range(min(abs_y_i, gu_y_i), max(abs_y_i, gu_y_i)+1):
                # taken by a BM
                if self.grids[_idx_i, _idx_j] > 0.:
                    w_x, e_x = _idx_i * self.mesh_len, (_idx_i+1) * self.mesh_len
                    s_y, n_y = _idx_j * self.mesh_len, (_idx_j+1) * self.mesh_len
                    # compute 2D projection intersection point as `point`
                    point = None
                    # different conditions for intersections (take `ABS` as origin)
                    # axises
                    if dx == 0 and dy > 0:      # GU: y-axis +
                        point = np.array([abs.state[0], n_y])
                    elif dx == 0 and dy < 0:    # GU: y-axis -
                        point = np.array([abs.state[0], s_y])
                    elif dx > 0 and dy == 0:    # GU: x-axis +
                        point = np.array([e_x, abs.state[1]])
                    elif dx < 0 and dy == 0:    # GU: x-axis -
                        point = np.array([w_x, abs.state[1]])
                    elif dx > 0 and dy > 0:     # GU: 1st quad
                        point_1 = np.array([_cal_x(n_y), n_y])
                        point_2 = np.array([e_x, _cal_y(e_x)])
                        if point_1[0] >=w_x and point_1[0] <= e_x:
                            point = point_1
                        elif point_2[1] >= s_y and point_2[1] <= n_y:
                            point = point_2
                        else:
                            pass # LoS
                    elif dx < 0 and dy > 0:     # GU: 2nd quad
                        point_1 = np.array([_cal_x(n_y), n_y])
                        point_4 = np.array([w_x, _cal_y(w_x)])
                        if point_1[0] >= w_x and point_1[0] <= e_x:
                            point = point_1
                        elif point_4[1] >= s_y and point_4[1] <= n_y:
                            point = point_4
                        else:
                            pass # LoS
                    elif dx < 0 and dy < 0:     # GU: 3rd quad
                        point_3 = np.array([_cal_x(s_y), s_y])
                        point_4 = np.array([w_x, _cal_y(w_x)])
                        if point_3[0] >= w_x and point_3[0] <= e_x:
                            point = point_3
                        elif point_4[1] >= s_y and point_4[1] <= n_y:
                            point = point_4
                        else:
                            pass # LoS
                    elif dx > 0 and dy < 0:     # GU: 4th quad
                        point_2 = np.array([e_x, _cal_y(e_x)])
                        point_3 = np.array([_cal_x(s_y), s_y])
                        if point_2[1] >= s_y and point_2[1] <= n_y:
                            point = point_2
                        elif point_3[0] >= w_x and point_3[0] <= e_x:
                            point = point_3
                        else:
                            pass # LoS
                    # check vertical intersection
                    if point is not None:
                        d = compute_2D_distance(abs.state, point)
                        D = compute_2D_distance(abs.state, gu.state)
                        dH = abs.H - gu.H
                        dh = dH - dH*d/D + gu.H
                        if dh < self.grids[_idx_i, _idx_j]: # 3D wise intersection
                            return False # NLoS

        return True # LoS

    @property
    def entities(self):
        return self.BMs + self.ABSs + self.GUs