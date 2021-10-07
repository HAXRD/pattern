# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import argparse

def get_config():
    '''
    The configuration parser for common hyperparameters of environment.
    '''
    parser = argparse.ArgumentParser(
        description='pattern', formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare params
    parser.add_argument("--experiment_name", type=str, default="check",
                        help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed for numpy/torch.")
    parser.add_argument("--cuda", action="store_false", default=True,
                        help="by default True, will use GPU to train; otherwise will use CPU.")
    parser.add_argument("--n_training_threads", type=int, default=1,
                        help="number of torch threads for training.")
    parser.add_argument("--n_rollout_threads", type=int, default=1,
                        help="number of parallel envs for training.")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                        help="number of parallel envs for evaluation.")
    parser.add_argument("--n_render_rollout_threads", type=int, default=1,
                        help="number of parallel envs for rendering.")
    parser.add_argument("--num_env_episodes", type=int, default=10e6,
                        help="number of environment episode to train.")
    parser.add_argument("--user_name", type=str, default="haxrd",
                        help="[for wandb usage], to specify user's name for simply collecting training data.")
    parser.add_argument("--use_wandb", action="store_false", default=True,
                        help="[for wandb usage], by default True, will log date to wandb server.")

    # env params
    parser.add_argument("--env_name", type=str, default="sitespecific",
                        help="specify the name of environment.")
    parser.add_argument("--BMs_fname", type=str, default="terrain-0.mat",
                        help="file name of BMs mat file.")
    parser.add_argument("--world_len", type=float, default=1000., 
                        help="side length of the square shape world.")
    parser.add_argument("--granularity", type=float, default=250.,
                        help="pattern size.")
    parser.add_argument("--episode_length", type=float, default=40.,
                        help="duration of 1 episode.")
    parser.add_argument("--n_ABS", type=int, default=3,
                        help="number of ABSs.")
    parser.add_argument("--n_GU", type=int, default=2,
                        help="number of GUs.")
    parser.add_argument("--v_ABS", type=float, default=25.,
                        help="maximum velocity of ABSs (currently not quite useful).")
    parser.add_argument("--v_GU", type=float, default=2.,
                        help="maximum velocity of GUs.")
    parser.add_argument("--h_ABS", type=float, default=90.,
                        help="height of ABSs.")
    parser.add_argument("--h_GU", type=float, default=1.0,
                        help="height of GUs.")
    parser.add_argument("--R_2D_NLoS", type=float, default=100.,
                        help="2D radius for NLoS.")
    parser.add_argument("--R_2D_LoS", type=float, default=200.,
                        help="2D radius for LoS.")
    parser.add_argument("--normalize_pattern", action="store_false", default=True,
                        help="by default True, normalize GUs & CGUs patterns by the number of GUs; else normalized by 1.")

    # env-model φ params
    parser.add_argument("--use_emulator_ckpt", action="store_false", default=True,
                        help="by default True, use stored ckpt to as φ_0; else interact with environment (that has no site-specific data) to collect experience and train a baseline env emulator as φ_0")
    parser.add_argument("--emulator_ckpt", type=str, default="env emulator.ckpt",
                        help="file name to env emulator φ0 ckpt")
    parser.add_argument("--num_emulator_warmup", type=int, default=10e5,
                        help="number of episodes to emulate to get φ_0")
    parser.add_argument("--num_planning_random_warmup", type=int, default=1e4,
                        help="number of episodes to use random permutations as ABS patterns.")
    parser.add_argument("--planning_random_size", type=int, default=2<<13,
                        help="size of different planning ABS patterns.")
    parser.add_argument("--planning_top_k", type=int,default=2<<7,
                        help="choose top k to actually interact with environment.")
    
    
    return parser
    pass