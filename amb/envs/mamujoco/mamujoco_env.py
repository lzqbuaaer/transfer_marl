import copy
import importlib
import logging
import numpy as np
import supersuit as ss
from gymnasium.spaces import Box
from gymnasium_robotics.envs.multiagent_mujoco.mamujoco_v0 import parallel_env as mujoco_env

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)


class MAMujocoEnv:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.scenario = args["scenario"]

        self.cur_step = 0
        self.env = ss.pad_observations_v0(mujoco_env(**self.args))
        
        self.env.reset()

        self.n_agents = self.env.num_agents
        self.agents = self.env.agents
        self.share_observation_space = self.repeat(Box(
            low=-np.inf, high=np.inf, shape=self.env.state().shape, dtype=np.float64
        ))
        self.observation_space = [self.env.observation_space(agent) for agent in self.agents]
        self.action_space = [self.env.action_space(agent) for agent in self.agents]
        self._seed = 0

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        obs, rew, term, trunc, info = self.env.step(self.wrap(actions))
        self.cur_step += 1
        if self.cur_step == 1001:
            trunc = {agent: True for agent in self.agents}
            for agent in self.agents:
                info[agent]["bad_transition"] = True
        dones = {agent: term[agent] or trunc[agent] for agent in self.agents}
        s_obs = self.repeat(self.env.state())
        total_reward = sum([rew[agent] for agent in self.agents])
        rewards = [[total_reward]] * self.n_agents
        return (
            self.unwrap(obs),
            s_obs,
            rewards,
            self.unwrap(dones),
            self.unwrap(info),
            self.get_avail_actions(),
        )

    def reset(self):
        """Returns initial observations and states"""
        self._seed += 1
        self.cur_step = 0
        obs = self.unwrap(self.env.reset(seed=self._seed)[0])
        s_obs = self.repeat(self.env.state())
        return obs, s_obs, self.get_avail_actions()

    def get_avail_actions(self):
        return None

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return None

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self._seed = seed

    def wrap(self, l):
        d = {}
        for i, agent in enumerate(self.agents):
            d[agent] = l[i]
        return d

    def unwrap(self, d):
        l = []
        for agent in self.agents:
            l.append(d[agent])
        return l

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
