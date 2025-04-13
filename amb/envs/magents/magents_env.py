import numpy as np

from magent2.environments import battle_v4

class MAgentsEnv:
    def __init__(self, args):
        self.map_size = args["map_size"]
        self.minimap_mode = args["minimap_mode"]
        self.env_reward_args = args["env_reward_args"]
        self.max_cycles = args["max_cycles"]
        self.extra_features = args["extra_features"]
        self.render_mode = args.get("render_mode", False)
        self._seed = args.get("seed", 0)
        
        self.env = battle_v4.parallel_env(map_size=self.map_size, 
                                          max_cycles=self.max_cycles, 
                                          minimap_mode=self.minimap_mode,
                                          extra_features=self.extra_features,
                                          render_mode=self.render_mode,
                                          seed=self._seed,
                                          **self.env_reward_args)
        self.agents = self.env.agents
        self.share_observation_space = [self.env.state_space]
        self.observation_space = [self.env.observation_space(agent) for agent in self.agents]
        self.action_space = [self.env.action_space(agent) for agent in self.agents]
        self.episode_limit = self.max_cycles
        self.available_actions = [[1] * self.env.action_space(agent).n for agent in self.agents]
        self.n_agents = len(self.agents)
    
    def step(self, actions):
        all_actions = {agent: actions[i] for i, agent in enumerate(self.agents)}
        observations, rewards, terminations, truncations, infos = self.env.step(all_actions=all_actions)
        observations = [observations.get(agent, np.zeros(self.observation_space[i].shape)) for i, agent in enumerate(self.agents)]
        states = [self.env.state() for _ in self.agents]
        rewards = [rewards.get(agent, 0) for agent in self.agents]
        dones = [terminations.get(agent, True) or (self.env.frames >= self.env.max_cycles) for agent in self.agents]
        infos = [infos.get(agent, {}) for agent in self.agents]
        return observations, states, rewards, dones, infos, self.available_actions

    def reset(self):
        observations = self.env.reset()
        observations = [observations.get(agent, np.zeros(self.observation_space[i].shape)) for i, agent in enumerate(self.agents)]
        states = [self.env.state() for _ in self.agents]
        return observations, states, self.available_actions
    
    def seed(self, seed):
        self._seed = seed
        self.env.seed(seed=seed)
    
    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()