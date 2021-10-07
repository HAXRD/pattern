# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np

from envs.sse import rendering
from envs.sse.core import BM, ABS, GU, COLORs

# site specific environment
class SiteSpecificEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"]
    }

    def __init__(self,
                 args,
                 world,
                 reset_callback=None,
                 get_GU_pattern_callback=None,
                 get_ABS_pattern_callback=None,
                 get_CGU_pattern_callback=None,
                 ):
        self.args  = args
        self.world = world
        self.reset_callback = reset_callback
        self.get_GU_pattern_callback  = get_GU_pattern_callback
        self.get_ABS_pattern_callback = get_ABS_pattern_callback
        self.get_CGU_pattern_callback = get_CGU_pattern_callback

        # render related
        self.cam_range = 1.2 * self.world.world_len
        self.viewer = None
        self._reset_render()

    ############### public methods ###############
    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def step(self, ABS_pattern):
        """
        :param  ABS_pattern: (2D array), shape==(K, K), gridized target ABSs' location
        :return CGU_pattern: (2D array), shape==(K, K), gridized covered GUs' location
        """
        # set locations (states) for each ABS
        K = self.world.K
        l = 0
        for i in range(K):
            for j in range(K):
                if ABS_pattern[i, j] == 1.:
                    self.world.ABSs[l].state = (np.array([i, j], dtype=np.float32) + 0.5) * self.world.granularity
                    l += 1
        # update world state
        self.world.step(ABS_UPDATE=True, GU_UPDATE=False) # core.step()
        # get CGU_pattern
        CGU_pattern = self.get_CGU_pattern_callback(self.world, self.args.normalize_pattern)
        return CGU_pattern

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # get GU_pattern
        GU_pattern = self.get_GU_pattern_callback(self.world, self.args.normalize_pattern)
        return GU_pattern

    def render(self, mode='human'):
        print('p1')
        # create viewer object to diplay entities
        if self.viewer == None:
            print('p1.1')
            self.viewer = rendering.Viewer(1000, 1000)
            print('p1.2')
        print('p2')
        # create rendering geometries
        self.render_geoms = []
        self.render_geoms_xform = []
        for entity in self.world.entities:
            if isinstance(entity, BM):
                geom = rendering.make_square(entity.size)
                xform = rendering.Transform()
                geom.set_color(*entity.color, alpha=0.5)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)
            elif isinstance(entity, ABS):
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                geom.set_color(*entity.color, alpha=0.5)
                geom.add_attr(xform)
                # NLoS radius
                geom_NLoS = rendering.make_circle(self.world.R_2D_NLoS, filled=False)
                xform_NLoS = rendering.Transform()
                geom_NLoS.set_color(*COLORs['orange'], alpha=0.5)
                geom_NLoS.add_attr(xform_NLoS)
                # LoS radius
                geom_LoS = rendering.make_circle(self.world.R_2D_LoS, filled=False)
                xform_LoS = rendering.Transform()
                geom_LoS.set_color(*COLORs['yellow'], alpha=0.5)
                geom_LoS.add_attr(xform_LoS)

                self.render_geoms.append((geom, geom_NLoS, geom_LoS))
                self.render_geoms_xform.append((xform, xform_NLoS, xform_LoS))
            elif isinstance(entity, GU):
                geom = rendering.make_triangle(entity.size)
                xform = rendering.Transform()
                geom.set_color(*entity.color, alpha=0.5)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)
        print('p3')
        world_geom = rendering.make_square(self.world.world_len)
        world_xform = rendering.Transform()
        world_geom.set_color(*COLORs['grey'], 0.1)
        world_geom.add_attr(world_xform)
        print('p4')

        # add geoms into viewer
        self.viewer.geoms = []
        for item in self.render_geoms:
            if isinstance(item, tuple):
                assert len(item) == 3
                for geom in item:
                    self.viewer.add_geom(geom)
            else:
                self.viewer.add_geom(item)
        self.viewer.add_geom(world_geom)
        print('p5')
        # set initial position
        pos = np.ones(2) * self.world.world_len / 2
        self.viewer.set_bounds(pos[0] - self.cam_range/2, pos[0] + self.cam_range/2, pos[1] - self.cam_range/2, pos[1] + self.cam_range/2)
        for xform_item, entity in zip(self.render_geoms_xform, self.world.entities):
            if isinstance(entity, BM):
                xform_item.set_translation(*entity.state)
            elif isinstance(entity, ABS):
                for xform in xform_item:
                    xform.set_translation(*entity.state)
            elif isinstance(entity, GU):
                xform_item.set_translation(*entity.state)
        print('p6')
        world_xform.set_translation(*pos)
        print(f'# of entities {len(self.world.entities)}')
        self.viewer.render()

    def info(self):
        def pprint_entity(entity):
            print(f'{entity.name}, {entity.state}, {entity.H}')
        # print BMs' info
        for _bm in self.world.BMs:
            pprint_entity(_bm)
        # print GUs' info
        for _gu in self.world.GUs:
            pprint_entity(_gu)
        # print ABSs' info
        for _abs in self.world.ABSs:
            pprint_entity(_abs)

    ############### private methods ###############
    # reset rendering related
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None
