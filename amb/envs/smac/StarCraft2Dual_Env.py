import numpy as np
import atexit
import portpicker
from multiprocessing import Process, Pipe

from .StarCraft2_Env import StarCraft2Env
from .multiagentenv import MultiAgentEnv


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)


def process_env(env: StarCraft2Env, pipe):
    while True:
        # command
        command = pipe.recv()
        ret = None
        if command[0] == "reset":
            ret = env.reset()
        elif command[0] == "step":
            ret = env.step(command[1])
        elif command[0] == "seed":
            ret = env.seed(command[1])
        elif command[0] == "close":
            ret = env.close()
            pipe.close()
            break
        elif command[0] == "save_replay":
            ret = env.save_replay()
        elif command[0] == "get_env_info":
            ret = env.get_env_info()
        pipe.send(ret)


class StarCraft2DualEnv(MultiAgentEnv):
    def __init__(self, args, **kwargs):
        ports = [portpicker.pick_unused_port() for _ in range(4)]
        self.r = int(args["reverse_team"])
        del args["reverse_team"]
        self.host_env = StarCraft2Env(args, **kwargs, host=True, ports=ports)
        self.client_env = StarCraft2Env(args, **kwargs, host=False, ports=ports)
        self.host_pipe, self.host_child_pipe = Pipe()
        self.client_pipe, self.client_child_pipe = Pipe()
        self.p_host_env = Process(target=process_env, args=(self.host_env, self.host_child_pipe))
        self.p_client_env = Process(target=process_env, args=(self.client_env, self.client_child_pipe))
        self.p_host_env.daemon = True
        self.p_client_env.daemon = True
        self.p_host_env.start()
        self.p_client_env.start()

        self.host_pipe.send(["get_env_info"])
        self.client_pipe.send(["get_env_info"])
        data = list(zip(self.host_pipe.recv(), self.client_pipe.recv()))
        self.observation_space = [data[0][self.r], data[0][1-self.r]]
        self.share_observation_space = [data[1][self.r], data[1][1-self.r]]
        self.action_space = [data[2][self.r], data[2][1-self.r]]

        self.n_angels = data[3][self.r]
        self.n_demons = data[3][1-self.r]
        self.n_agents = self.n_angels + self.n_demons

    def seed(self, seed):
        """Returns reward, terminated, info."""
        self.host_pipe.send(["seed", seed])
        self.client_pipe.send(["seed", seed])
        self.host_pipe.recv()
        self.client_pipe.recv()

    def step(self, actions):
        """Returns reward, terminated, info."""
        self.host_pipe.send(["step", actions[self.r]])
        self.client_pipe.send(["step", actions[1-self.r]])
        if self.r:
            obs, share_obs, rewards, dones, infos, available_actions \
                = list(zip(self.client_pipe.recv(), self.host_pipe.recv()))
        else:
            obs, share_obs, rewards, dones, infos, available_actions \
                = list(zip(self.host_pipe.recv(), self.client_pipe.recv()))
        obs = [np.stack(obs[i], axis=0) for i in range(2)]
        share_obs = [np.stack(share_obs[i], axis=0) for i in range(2)]
        rewards = [np.stack(rewards[i], axis=0) for i in range(2)]
        dones = [np.stack(dones[i], axis=0) for i in range(2)]
        available_actions = [np.stack(available_actions[i], axis=0) for i in range(2)]
        return obs, share_obs, rewards, dones, infos, available_actions

    def reset(self):
        """Returns initial observations and states."""
        self.host_pipe.send(["reset"])
        self.client_pipe.send(["reset"])
        if self.r:
            obs, share_obs, available_actions = list(zip(self.client_pipe.recv(), self.host_pipe.recv()))
        else:
            obs, share_obs, available_actions = list(zip(self.host_pipe.recv(), self.client_pipe.recv()))
        obs = [np.stack(obs[i], axis=0) for i in range(2)]
        share_obs = [np.stack(share_obs[i], axis=0) for i in range(2)]
        available_actions = [np.stack(available_actions[i], axis=0) for i in range(2)]

        return obs, share_obs, available_actions

    def close(self):
        self.host_pipe.send(["close"])
        self.client_pipe.send(["close"])
        self.host_pipe.close()
        self.client_pipe.close()

    def render(self):
        """Use save_replay instead"""
        pass

    def save_replay(self):
        """Save a replay."""
        self.host_pipe.send(["save_replay"])
        return self.host_pipe.recv()
