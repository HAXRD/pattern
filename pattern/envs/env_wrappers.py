# This code is from openai baseline
# https://github.com/openai/baselines/tree/master/baselines/common/vec_env

import numpy as np
from multiprocessing import Process, Pipe

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            CGU_pattern = env.step(data)
            remote.send(CGU_pattern)
        elif cmd == 'reset':
            GU_pattern = env.reset()
            remote.send(GU_pattern)
        elif cmd == 'render':
            if data == 'human':
                env.render(mode=data)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        else:
            raise NotImplementedError

class CloudpickleWrapper(object):
    """
    Use cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class VecEnv(object):
    """
    An abstract asynchronous, vectorized environment.
    """
    def __init__(self, num_envs):
        self.num_envs = num_envs

    def reset(self):
        """
        Reset all the environments and return an array of
        GU_patterns, or a tuple of GU_pattern arrays.
        if step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    def step_async(self, ABS_patterns):
        """
        Tell all the environments to start taking a step
        with the given ABS_patterns.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns CGU_patterns: an array of CGU_patterns.
        """
        pass

    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, ABS_patterns):
        self.step_async(ABS_patterns)
        return self.step_wait()

    def render(self, mode='human'):
        pass

class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns))
        self.ABS_patterns = None

    def step_async(self, ABS_patterns):
        self.ABS_patterns = ABS_patterns

    def step_wait(self):
        results = [env.step(ABS_pattern) for (ABS_pattern, env) in zip(self.ABS_patterns, self.envs)]
        CGU_patterns = map(np.array, zip(*results))

        self.ABS_patterns = None
        return CGU_patterns

    def reset(self):
        GU_patterns = [env.reset() for env in self.envs]
        return np.array(GU_patterns)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode='human'):
        if mode == 'human':
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns):
        """
        Custom SubproceVecEnv for asynchronous multiprocessing.
        :param env_fns: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        VecEnv.__init__(self, len(env_fns))

    def step_async(self, ABS_patterns):
        for remote, ABS_pattern in zip(self.remotes, ABS_patterns):
            remote.send(('step', ABS_pattern))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        CGU_patterns = results
        return np.stack(CGU_patterns)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode='human'):
        for remote in self.remotes:
            remote.send(('render', mode))
            print('send to render')

    def __len__(self):
        return self.nenvs