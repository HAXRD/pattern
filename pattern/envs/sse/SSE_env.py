# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

from envs.sse.scenarios import load
from envs.sse.environment import SiteSpecificEnv

def SSEEnv(args):
    '''
    TODO:
    '''
    # load scenario from script
    scenario = load(args.scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(args)
    # create environment
    env = SiteSpecificEnv(args, world, scenario.reset_world, scenario.get_GU_pattern, scenario.get_ABS_pattern, scenario.get_CGU_pattern)

    return env