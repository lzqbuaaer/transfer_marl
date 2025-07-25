import numpy as np
import atexit
import portpicker
from multiprocessing import Process, Pipe
from multiprocessing.connection import wait

from amb.envs.smacv2.smacv2_env import SMACv2Env
from amb.envs.smacv2.core.multiagentenv import MultiAgentEnv

from pysc2.lib import protocol


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


def process_env(env: SMACv2Env, pipe):
    while True:
        # command
        command = pipe.recv()
        ret = None
        try:
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
            elif command[0] == "get_stats":
                ret = env.get_stats()
        except (protocol.ProtocolError, protocol.ConnectionError) as e:
            import traceback
            traceback.print_exc()
            pipe.send(f"Error: {e}")
        finally:
            pipe.send(ret)


class SMACv2DualEnv(MultiAgentEnv):
    def __init__(self, args, **kwargs):
        ports = [portpicker.pick_unused_port() for _ in range(4)]
        self.r = int(args["reverse_team"])
        del args["reverse_team"]
        self.args = args
        self.multi_map_alignment = args.get("multi_map_alignment", False)
        self.kwargs = kwargs
        
        self.host_args, self.client_args = self.args.copy(), self.args.copy()
        if self.multi_map_alignment and (not self.r):
            self.host_multi_map_alignment = True
            self.client_multi_map_alignment = False
        elif self.multi_map_alignment:
            self.host_multi_map_alignment = False
            self.client_multi_map_alignment = True
        else:
            self.host_multi_map_alignment = False
            self.client_multi_map_alignment = False
        if not self.r:
            self.host_obs_align_v1, self.client_obs_align_v1 = args["angel_obs_align_v1"], args["demon_obs_align_v1"]
        else:
            self.host_obs_align_v1, self.client_obs_align_v1 = args["demon_obs_align_v1"], args["angel_obs_align_v1"]
        self.host_env = SMACv2Env(self.host_args, host=True, ports=ports, 
                                  multi_map_alignment=self.host_multi_map_alignment,
                                  obs_align_v1=self.host_obs_align_v1)
        self.client_env = SMACv2Env(self.client_args, host=False, ports=ports, 
                                    multi_map_alignment=self.client_multi_map_alignment,
                                    obs_align_v1=self.client_obs_align_v1)
        self.host_pipe, self.host_child_pipe = Pipe()
        self.client_pipe, self.client_child_pipe = Pipe()
        self.p_host_env = Process(target=process_env, args=(self.host_env, self.host_child_pipe))
        self.p_client_env = Process(target=process_env, args=(self.client_env, self.client_child_pipe))
        self.p_host_env.daemon = True
        self.p_client_env.daemon = True
        self.p_host_env.start()
        self.p_client_env.start()

        self.seed(0)
        self.host_pipe.send(["get_env_info"])
        self.client_pipe.send(["get_env_info"])
        data = list(zip(self.host_pipe.recv(), self.client_pipe.recv()))
        self.observation_space = [data[0][self.r], data[0][1-self.r]]
        self.share_observation_space = [data[1][self.r], data[1][1-self.r]]
        self.action_space = [data[2][self.r], data[2][1-self.r]]
        self.obs_own_feat = [data[4][self.r], data[4][1-self.r]]
        self.obs_enemy_feat = [data[5][self.r], data[5][1-self.r]]
        self.obs_ally_feat = [data[6][self.r], data[6][1-self.r]]

        self.n_angels = data[3][self.r]
        self.n_demons = data[3][1-self.r]
        self.n_agents = self.n_angels + self.n_demons
        self._seed = 0
        
    def force_restart(self):
        self.p_host_env.terminate()
        self.p_client_env.terminate()
        self.p_host_env.join()
        self.p_client_env.join()
        self.host_pipe.close()
        self.client_pipe.close()
        ports = [portpicker.pick_unused_port() for _ in range(4)]
        self.host_env = SMACv2Env(self.host_args, host=True, ports=ports, 
                                  multi_map_alignment=self.host_multi_map_alignment,
                                  obs_align_v1=self.host_obs_align_v1)
        self.client_env = SMACv2Env(self.client_args, host=False, ports=ports, 
                                    multi_map_alignment=self.client_multi_map_alignment,
                                    obs_align_v1=self.client_obs_align_v1)
        self.host_pipe, self.host_child_pipe = Pipe()
        self.client_pipe, self.client_child_pipe = Pipe()
        self.p_host_env = Process(target=process_env, args=(self.host_env, self.host_child_pipe))
        self.p_client_env = Process(target=process_env, args=(self.client_env, self.client_child_pipe))
        self.p_host_env.daemon = True
        self.p_client_env.daemon = True
        self.p_host_env.start()
        self.p_client_env.start()
        self.seed(self._seed)
        

    def seed(self, seed):
        """Returns reward, terminated, info."""
        self._seed = seed
        self.host_pipe.send(["seed", seed])
        self.client_pipe.send(["seed", seed])
        self.host_pipe.recv()
        self.client_pipe.recv()

    def step(self, actions):
        """Returns reward, terminated, info."""
        self.host_pipe.send(["step", actions[self.r]])
        self.client_pipe.send(["step", actions[1-self.r]])
        
        pipes = [self.host_pipe, self.client_pipe]
        recvs = [None, None]
        recved_num = 0
        if_error = False
        while recved_num < len(pipes):
            ready_pipes = wait(pipes)
            for ready_pipe in ready_pipes:
                ready_index = pipes.index(ready_pipe)
                recv = ready_pipe.recv()
                if (recv is not None) and isinstance(recv, str) \
                    and 'Error' == recv[:5]:
                    if_error = True
                    print(recv)
                    break
                recvs[ready_index] = recv
                recved_num += 1
        if not if_error:       
            if self.r:
                obs, share_obs, rewards, dones, infos, available_actions \
                    = list(zip(recvs[1], recvs[0]))
            else:
                obs, share_obs, rewards, dones, infos, available_actions \
                    = list(zip(recvs[0], recvs[1]))
            obs = [np.stack(obs[i], axis=0) for i in range(2)]
            share_obs = [np.stack(share_obs[i], axis=0) for i in range(2)]
            rewards = [np.stack(rewards[i], axis=0) for i in range(2)]
            dones = [np.stack(dones[i], axis=0) for i in range(2)]
            available_actions = [np.stack(available_actions[i], axis=0) for i in range(2)]
        else:
            self.force_restart()
            obs, share_obs, available_actions = self.reset()
            rewards = [[[0]] * self.n_angels, [[0]] * self.n_demons]
            rewards = [np.stack(rewards[i], axis=0) for i in range(2)]
            dones = [np.ones((self.n_angels), dtype=bool), np.ones((self.n_demons), dtype=bool)]
            info_host = info_client = {"battle_won": False, "dead_allies": 0, 
                                        "dead_enemies": 0, "episode_limit": False}
            if self.r:
                infos = [[info_client] * self.n_angels, [info_host] * self.n_demons]
            else:
                infos = [[info_host] * self.n_angels, [info_client] * self.n_demons]

        return obs, share_obs, rewards, dones, infos, available_actions

    def reset(self):
        """Returns initial observations and states."""
        while True:
            self.host_pipe.send(["reset"])
            self.client_pipe.send(["reset"])
            
            pipes = [self.host_pipe, self.client_pipe]
            recvs = [None, None]
            recved_num = 0
            if_error = False
            while recved_num < len(pipes):
                ready_pipes = wait(pipes)
                for ready_pipe in ready_pipes:
                    ready_index = pipes.index(ready_pipe)
                    recv = ready_pipe.recv()
                    if (recv is not None) and isinstance(recv, str) \
                        and 'Error' == recv[:5]:
                        if_error = True
                        print(recv)
                        break
                    recvs[ready_index] = recv
                    recved_num += 1
            if not if_error:
                if self.r:
                    obs, share_obs, available_actions = list(zip(recvs[1], recvs[0]))
                else:
                    obs, share_obs, available_actions = list(zip(recvs[0], recvs[1]))
                break
            else:
                self.force_restart()
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
