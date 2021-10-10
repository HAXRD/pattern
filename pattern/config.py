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
    parser.add_argument("--use_eval", action="store_true", default=False,
                        help="by default False, will not use evaluation; otherwise with evaluation.")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="number of episodes for evaluation.")
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
    parser.add_argument("--policy_distributional", action="store_true", default=False,
                        help="by default false, use distributional policy; otherwise use deterministic policy.")
    parser.add_argument("--emulator_replay_per", action="store_true", default=False,
                        help="by default false, use prioritized replay for emulator memory; otherwise use naive one.")
    parser.add_argument("--use_emulator_pt", action="store_false", default=True,
                        help="by default True, use stored ckpt to as φ_0; else interact with environment (that has no site-specific data) to collect experience and train a baseline env emulator as φ_0")
    parser.add_argument("--emulator_pt", type=str, default='./base_emulator.pt',
                        help="file name to env emulator φ0 ckpt")
    parser.add_argument("--num_emulator_warmup", type=int, default=100,
                        help="number of episodes to emulate to get φ_0")
    parser.add_argument("--emulator_replay_size", type=int, default=10,
                        help="replay size for emulator memory.")
    parser.add_argument("--emulator_lr", type=float, default=0.001,
                        help="lr for emulator.")
    parser.add_argument("--policy_lr", type=float, default=0.0001, help="lr for policy")
    parser.add_argument("--policy_replay_size", type=int, default=10000,
                        help="replay size for policy memory.")
    parser.add_argument("--num_planning_random_warmup", type=int, default=5,
                        help="number of episodes to use random permutations as ABS patterns.")
    parser.add_argument("--num_planning_random", type=int, default=2<<4,
                        help="size of different planning ABS patterns with random generation.")
    parser.add_argument("--num_planning_with_policy", type=int, default=2<<4,
                        help="size of different planning ABS patterns with policy prediction variations.")
    parser.add_argument("--planning_top_k", type=int,default=2<<3,
                        help="choose top k to actually interact with environment.")
    parser.add_argument("--planning_batch_size", type=int, default=32,
                        help="batch size for planning with emulator.")
    parser.add_argument("--num_base_env_episodes", type=int, default=100,
                        help="number of base environment episode to get a base emulator")
    parser.add_argument("--num_base_emulator_epochs", type=int, default=2,
                        help="number of epochs to train base emulator.")
    parser.add_argument("--num_base_emulator_batch_size", type=int, default=32,
                        help="batch_size for training base emulator.")
    parser.add_argument("--num_env_episodes", type=int, default=100,
                        help="number of environment episode to train.")
    parser.add_argument("--least_emulator_buffer_size", type=int, default=100,
                        help="least amount of transitions for emulator to start training.")
    parser.add_argument("--least_policy_buffer_size", type=int, default=32,
                        help="least amount of transitions for policy to start training")
    parser.add_argument("--num_train_policy", type=int, default=1,
                        help="number of repeats to train policy.")
    parser.add_argument("--policy_batch_size", type=int, default=16,
                        help="batch_size for training policy.")
    parser.add_argument("--num_train_emulator", type=int, default=5,
                        help="number of train repeat for training emulator.")
    parser.add_argument("--emulator_batch_size", type=int, default=64,
                        help="batch size for training emulator.")

    # intervals
    parser.add_argument("--save_interval", type=int, default=100,
                        help="number of episodes for saving.")
    parser.add_argument("--eval_interval", type=int, default=500,
                        help="number of episodes for evaluations.")
    parser.add_argument("--log_interval", type=int, default=1,
                        help="number of episodes for logging.")

    # eval
    parser.add_argument("--use_activation_oriented_policy_sample", action="store_false", default=True,
                        help="by default use activation oriented policy sample; otherwise not.")

    return parser
    pass