# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import os
import time
import torch
import math
import numpy as np
import contextlib
import wandb
from numpy.random import default_rng
from copy import deepcopy
from utils.loss import hybrid_mse

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def generate_1_sequence(size, MAX):
    rng = default_rng()
    sequence = rng.choice(MAX, size=size, replace=False)
    return sorted(sequence)

def _t2n(x):
    return x.detach().cpu().numpy()

class SSERunner(object):
    """Runner class to perform training, evaluation, and data collection for the SSEs."""
    def __init__(self, config):

        self.all_args  = config['all_args']
        self.base_env  = config['base_env']
        self.env       = config['env']
        self.eval_env  = config['eval_env']
        self.device    = config['device']

        # TODO: parameters
        # ...
        self.seed = self.all_args.seed
        self.world_len = self.all_args.world_len
        self.granularity = self.all_args.granularity
        self.K = int(self.world_len) // int(self.granularity)
        self.n_ABS = self.all_args.n_ABS
        self.use_emulator_pt = self.all_args.use_emulator_pt
        self.num_base_env_episodes = self.all_args.num_base_env_episodes
        self.num_base_emulator_epochs = self.all_args.num_base_emulator_epochs
        self.num_base_emulator_batch_size = self.all_args.num_base_emulator_batch_size
        self.top_o_activations = self.n_ABS + 10
        self.planning_batch_size = self.all_args.planning_batch_size
        self.planning_top_k = self.all_args.planning_top_k
        self.num_env_episodes = self.all_args.num_env_episodes
        self.num_planning_random_warmup = self.all_args.num_planning_random_warmup
        self.num_planning_random = self.all_args.num_planning_random
        self.num_planning_with_policy = self.all_args.num_planning_with_policy
        self.policy_distributional = self.all_args.policy_distributional
        self.emulator_replay_per = self.all_args.emulator_replay_per
        self.emulator_replay_size = self.all_args.emulator_replay_size
        self.policy_replay_size = self.all_args.policy_replay_size
        self.use_eval = self.all_args.use_eval
        self.use_wandb = self.all_args.use_wandb

        self.eval_episodes = self.all_args.eval_episodes
        self.use_activation_oriented_policy_sample = self.all_args.use_activation_oriented_policy_sample
        # TODO: interval
        # ...
        self.save_interval = self.all_args.save_interval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # TODO: dir
        # ...
        self.emulator_pt = self.all_args.emulator_pt

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)

        # TODO: env emulator φ
        # ...
        # from algorithms.emulator import Emulator
        from algorithms2.emulator import Emulator
        self.emulator = Emulator(self.all_args, self.device)

        # TODO: policy μ
        # ...
        # if self.policy_distributional == True:
        #     from algorithms.policy import DistributionalPolicy as Policy
        # else:
        #     from algorithms.policy import NaivePolicy as Policy
        from algorithms2.policy import Policy
        self.policy = Policy(self.all_args, self.device)

        # TODO: transitions for φ
        # ...
        if self.emulator_replay_per == True:
            from utils.replay import EmulatorPrioritizedReplay as Replay
        else:
            from utils.replay import EmulatorNaiveReplay as Replay
        self.emulator_buffer = Replay(self.K, self.emulator_replay_size)

        # TODO: transitions for μ
        # ...
        from utils.replay import PolicyReplay
        self.policy_buffer = PolicyReplay(self.K, self.policy_replay_size)

    def run(self):
        """"""
        # get a base emulator φ
        if not self.use_emulator_pt:
            self.emulator_pretrain()
        self.emulator_load()

        start = time.time()
        episodes = int(self.num_env_episodes)

        for episode in range(episodes):
            # reset env
            GU_pattern = self.env.reset() # (K, K)
            # plan with different strategies
            if episode < self.num_planning_random_warmup: # randomly sample different ABS patterns
                planning_size = self.num_planning_random
                planning_size, planning_ABS_patterns = self.random_sample_ABS_patterns(planning_size) # (planning_size, K, K)
            else: # use policy μ to predict an ABS pattern, then sample near this patterns.
                planning_size = self.num_planning_with_policy
                planning_size, planning_ABS_patterns = self.policy_sample_ABS_patterns(GU_pattern, planning_size)
            GU_patterns = np.repeat(np.expand_dims(GU_pattern, 0), planning_size, axis=0)
            assert planning_ABS_patterns.shape == (planning_size, self.K, self.K), f'{planning_ABS_patterns.shape}'
            assert GU_patterns.shape == (planning_size, self.K, self.K), f'{GU_patterns.shape}'

            # get indices of top k transitions
            top_k_GU_patterns, top_k_ABS_patterns, top_k_pred_CGU_patterns = self.plan(GU_patterns, planning_ABS_patterns) # each has a shape of (top_k, K, K)
            top_k_CGU_patterns = np.zeros_like(top_k_pred_CGU_patterns)
            for _CGU_pattern, _ABS_pattern in zip(top_k_CGU_patterns, top_k_ABS_patterns):
                _CGU_pattern = self.env.step(_ABS_pattern)

            # store transition to replay μ
            best_data = top_k_GU_patterns[0], top_k_ABS_patterns[0], top_k_pred_CGU_patterns[0], top_k_CGU_patterns[0]
            self.policy_buffer.add(best_data)

            # store transition to replay φ
            for _GU_pattern, _ABS_pattern, _pred_CGU_pattern, _CGU_pattern in zip(top_k_GU_patterns, top_k_ABS_patterns, top_k_pred_CGU_patterns, top_k_CGU_patterns):
                data = _GU_pattern, _ABS_pattern, _pred_CGU_pattern, _CGU_pattern
                self.emulator_buffer.add(data)

            # train the emulator and policy
            emulator_loss, policy_loss = self.train()
            if emulator_loss is not None:
                print(f'[emulator train] episode {episode}/{episodes}, emulator loss {emulator_loss}')
            if policy_loss is not None:
                print(f'[policy train] episode {episode}/{episodes}, policy loss {policy_loss}')

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(f"[whole process ({'random' if episode < self.num_planning_random_warmup else 'policy'})] episode {episode}/{episodes}, EPS {episode / (end - start)}")

                self.log_train(emulator_loss, policy_loss, episode)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(episode)

    def emulator_load(self):
        assert os.path.exists(self.emulator_pt)
        assert os.path.isfile(self.emulator_pt)
        emulator_state_dict = torch.load(self.emulator_pt)
        self.emulator.model.load_state_dict(emulator_state_dict)
        print(f'[emulator load] loaded base emulator φ...')

    def emulator_pretrain(self):
        print('[emulator pretrain] starting...')
        from utils.replay import EmulatorNaiveReplay
        replay = EmulatorNaiveReplay(self.K, self.num_base_env_episodes)
        episodes = int(self.num_base_env_episodes)

        for _episode in range(episodes):
            GU_pattern = self.base_env.reset() # (K, K)
            _, ABS_pattern = self.random_sample_ABS_patterns(1)
            ABS_pattern = ABS_pattern.reshape(self.K, self.K) # sample 1 pattern (K, K)
            CGU_pattern = self.base_env.step(ABS_pattern) # (K, K)
            data = GU_pattern, ABS_pattern, None, CGU_pattern
            replay.add(data)
            print(f'[emulator pretrain: collecting] collected memory {replay.size}/{replay.max_size}, progress {_episode + 1}/{episodes}')

        min_train_loss = math.inf
        epochs = self.num_base_emulator_epochs
        for epoch in range(epochs):
            train_loss = 0.
            replay.shuffle()
            train_loss = self.emulator.train(replay)
            print(f'[emulator pretrain: training] progress {epoch + 1}/{epochs} \t loss: {train_loss:.6f}')
            if train_loss < min_train_loss:
                min_train_loss = train_loss
                torch.save(self.emulator.model.state_dict(), 'base_emulator.pt')
                print(f'[emulator pretrain: training] updated best_emulator.pt file')
        print(f'[emulator pretrain] done...')

    def random_sample_ABS_patterns(self, planning_size):
        """
        :return: planning_ABS_patterns: shape==(planning_size, K, K)
        """
        # generate `planning_size` unique pattern-indices lists
        planning_ABS_pattern_idcs = set()
        counter = 0
        while len(planning_ABS_pattern_idcs) < planning_size:
            counter += 1
            planning_ABS_pattern_idcs.add(tuple(generate_1_sequence(self.n_ABS, self.K * self.K)))
            if counter >= planning_size * 2:
                break
        planning_size = len(planning_ABS_pattern_idcs)
        planning_ABS_pattern_idcs = np.array([list(item) for item in list(planning_ABS_pattern_idcs)]).astype(np.int32)
        # convert indices into patterns
        planning_ABS_patterns = np.zeros((planning_size, self.K * self.K), dtype=np.float32)
        for idc, p in zip(planning_ABS_pattern_idcs, planning_ABS_patterns):
            p[idc] = 1.
        planning_ABS_patterns = planning_ABS_patterns.reshape(planning_size, self.K, self.K)
        return (
            planning_size,
            planning_ABS_patterns
        )

    @torch.no_grad() # DEBUG
    def policy_sample_ABS_patterns(self, GU_pattern, planning_size):
        """
        :param GU_pattern  : shape==(K, K)
        :param planning_size: planning size
        :return: planning_ABS_patterns: shape==(planning_size, K, K)
        """
        GU_pattern = torch.FloatTensor(GU_pattern).to(self.device).view(1, 1, self.K, self.K) # (1, 1, K, K)
        pred_ABS_pattern = _t2n(self.policy.model(GU_pattern)) # (1, 1, K, K)
        # use base pattern to generate variations
        base_pred_ABS_pattern = pred_ABS_pattern.reshape(self.K * self.K) # (K * K)

        planning_ABS_patterns = np.zeros((planning_size, self.K*self.K), dtype=np.float32)
        if self.use_activation_oriented_policy_sample:
            sorted_activation_idcs = np.argsort(-base_pred_ABS_pattern) # (K * K)
            top_o_sorted_activation_idcs = np.repeat(sorted_activation_idcs[:self.top_o_activations].reshape(1, self.top_o_activations), planning_size, axis=0) # (planning_size, top_o)
            # sample indices of planning_ABS_patterns indices
            sampled_idcs = set()
            counter = 0
            while len(sampled_idcs) < planning_size:
                counter += 1
                sampled_idcs.add(tuple(generate_1_sequence(self.n_ABS, self.top_o_activations)))
                if counter >= planning_size*2:
                    break
            planning_size = len(sampled_idcs)
            sampled_idcs = np.array([list(item) for item in list(sampled_idcs)]).astype(np.int32) # (planning_size, n_ABS)

            selected_idcs = np.zeros_like(sampled_idcs)
            idx = np.arange(planning_size)
            for i, idc, tar in zip(idx, sampled_idcs, top_o_sorted_activation_idcs):
                selected_idcs[i] = tar[idc]
            for pattern, idcs in zip(planning_ABS_patterns, selected_idcs):
                pattern[idcs] = 1.
            planning_ABS_patterns = planning_ABS_patterns.reshape(planning_size, self.K, self.K)
        else:
            raise NotImplementedError
        return (
            planning_size,
            planning_ABS_patterns
        )


    @torch.no_grad()
    def plan(self, GU_patterns, planning_ABS_patterns):
        """
        Use emulator to predict CGU_patterns for env, then select top k to
        :param GU_patterns          : shape==(planning_size, K, K)
        :param planning_ABS_patterns: shape==(planning_size, K, K)
        :return: (
            top_k_GU_patterns : shape==(planning_size, K, K)
            top_k_ABS_patterns: shape==(planning_size, K, K)
            top_k_pred_CGU_patterns: shape==(planning_size, K, K)
        )
        """
        planning_size = GU_patterns.shape[0]
        
        xs = np.stack(
            (GU_patterns, planning_ABS_patterns), axis=1) # (planning_size, 2, K, K)
        
        # feed in the emulator to get predicted CGU_patterns
        if planning_size < self.planning_batch_size: # predict directly
            ys = _t2n(self.emulator.model(torch.FloatTensor(xs).to(self.device))).squeeze() # (planning_size, K, K)
        else: # partition into chunks
            batch_size = self.planning_batch_size
            # n_chunks = math.ceil(float(planning_size)/float(batch_size))
            xs_chunks = [xs[i: i+batch_size] for i in range(0, len(xs), batch_size)]
            ys_chunks = [] # predicted CGU_patterns in chunk
            for item in xs_chunks:
                out = _t2n(self.emulator.model(torch.FloatTensor(item).to(self.device))).squeeze() # (batch_size, K, K)
                ys_chunks.append(out)
            ys = np.concatenate(ys_chunks) # (planning_size, K, K)
            assert ys.shape == (planning_size, self.K, self.K), f"{(planning_size, self.K, self.K)}"
        pred_CGU_patterns = ys.reshape(planning_size, self.K, self.K)
        pred_CRs = np.sum(pred_CGU_patterns.reshape(planning_size, -1), axis=-1) # (planning_size,)
        top_k_idcs = np.argsort(-pred_CRs, axis=-1)[:self.planning_top_k]

        top_k_GU_patterns  = GU_patterns[top_k_idcs]
        top_k_ABS_patterns = planning_ABS_patterns[top_k_idcs]
        top_k_pred_CGU_patterns = pred_CGU_patterns[top_k_idcs]

        return (
            top_k_GU_patterns,
            top_k_ABS_patterns,
            top_k_pred_CGU_patterns
        )

    def train(self):
        emulator_loss = self.emulator.train(self.emulator_buffer)
        policy_loss = self.policy.train(self.policy_buffer)
        return (
            emulator_loss,
            policy_loss
        )

    def save(self):
        torch.save(self.emulator.model.state_dict(), 'emulator.pt')
        torch.save(self.policy.model.state_dict(), 'policy.pt')
        print(f'[save] saved emulator & policy pts...')


    def log_train(self, emulator_loss, policy_loss, episode):
        if self.use_wandb:
            print(f'[log] logging to wandb...')
            if emulator_loss is not None:
                wandb.log({'emulator_loss': emulator_loss}, episode)
            if policy_loss is not None:
                wandb.log({'policy_loss': policy_loss}, episode)
        else:
            pass

    @torch.no_grad()
    def eval(self, curr_episode):
        print(f'[eval] starting eval...')
        with temp_seed(self.seed+100000):
            best_CRs = []
            episodes = self.eval_episodes
            for episode in range(self.eval_episodes):
                # reset all env
                GU_pattern = self.eval_env.reset() # (K, K)

                # planning
                planning_size = self.num_planning_with_policy
                planning_size, planning_ABS_patterns = self.policy_sample_ABS_patterns(GU_pattern, planning_size) # (planning_size, K, K)
                GU_patterns = np.repeat(np.expand_dims(GU_pattern, 0), planning_size, axis=0) # (planning_size, K, K)
                assert planning_ABS_patterns.shape == (planning_size, self.K, self.K), f"{planning_ABS_patterns.shape}"
                assert GU_patterns.shape == (planning_size, self.K, self.K)

                _, top_k_ABS_patterns, _ = self.plan(GU_patterns, planning_ABS_patterns)
                top_k_CGU_patterns = np.zeros_like(top_k_ABS_patterns) # (planning_size, K, K)
                for idx, _ABS_pattern in enumerate(top_k_ABS_patterns):
                    _CGU_pattern = self.eval_env.step(_ABS_pattern)
                    top_k_CGU_patterns[idx] = _CGU_pattern

                top_k_CRs = np.sum(top_k_CGU_patterns.reshape(planning_size, -1), axis=-1) # (planning_size)
                sorted_idcs = np.argsort(-top_k_CRs)
                top_CR = top_k_CRs[sorted_idcs][0]

                best_CRs.append(top_CR)
                print(f'[eval] progress {episode + 1}/{episodes}')

            print(f'==============================')
            print(f'[eval] first 15 episode CRs: {best_CRs[:15]}')
            print(f'[eval] mean CRs: {np.mean(best_CRs)}')
            print(f'==============================')
            if self.use_wandb:
                wandb.log({'eval_CRs': best_CRs}, curr_episode)
                wandb.log({'mean_eval_CRs': np.mean(best_CRs)}, curr_episode)
