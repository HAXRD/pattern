# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import os
import sys
import torch
import numpy as np
import random
import wandb
import socket
import setproctitle
from pathlib import Path

from pattern.config import get_config
from pattern.envs.sse.SSE_env import SSEEnv
from pattern.runner.sse_runner import SSERunner as Runner
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv


def make_base_env(all_args):
    seed = all_args.seed
    env = SSEEnv(all_args, seed)
    env.seed(seed)
    return env

def make_train_env(all_args):
    seed = all_args.seed * 10 + 1
    env = SSEEnv(all_args, seed)
    env.seed(seed)
    return env

def make_eval_env(all_args):
    seed = all_args.seed * 100 + 13
    env = SSEEnv(all_args, seed)
    env.seed(seed)

def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        help="Which scenario to run on")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # cuda
    torch.set_num_threads(1)
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        print("chosse to use cpu...")
        device = torch.device("cpu")

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "results") / all_args.env_name / all_args.scenario_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.experiment_name) + "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        raise NotImplementedError

    setproctitle.setproctitle(str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)

    # env init
    if args.use_emulator_ckpt:
        base_env = None
    else:
        base_env = make_base_env(all_args)
    env = make_train_env(all_args)
    eval_env = make_eval_env(all_args) if all_args.use_eval else None

    config = {
        "all_args": all_args,
        "base_env": base_env,
        "env": env,
        "eval_env": eval_env,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    runner = Runner(config)
    runner.run()

    # post process
    env.close()
    if all_args.use_eval and eval_env is not env:
        eval_env.close()

    if all_args.use_wandb:
        run.finish()
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main(sys.argv[1:])