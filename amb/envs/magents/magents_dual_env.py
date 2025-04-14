import numpy as np

from magent2.environments import battle_v4

class MAgentsDualEnv:
    def __init__(self, args):
        self.map_size = args["map_size"]
        self.minimap_mode = args["minimap_mode"]
        self.env_reward_args = args["env_reward_args"]
        self.max_cycles = args["max_cycles"]
        self.extra_features = args["extra_features"]
        self.render_mode = args.get("render_mode", False)
        self._seed = args.get("seed", 0)
        self.r = int(args["reverse_team"])
        
        self.env = battle_v4.parallel_env(map_size=self.map_size, 
                                          max_cycles=self.max_cycles, 
                                          minimap_mode=self.minimap_mode,
                                          extra_features=self.extra_features,
                                          render_mode=self.render_mode,
                                          seed=self._seed,
                                          **self.env_reward_args)
        self.agents = self.env.agents
        self.angel_agents_ids = self.env.env.get_agent_id(self.env.handles[self.r])
        self.demon_agents_ids = self.env.env.get_agent_id(self.env.handles[1 - self.r])
        self.angel_agents = [self.agents[id] for id in self.env.env.get_agent_id(self.env.handles[self.r])]
        self.demon_agents = [self.agents[id] for id in self.env.env.get_agent_id(self.env.handles[1-self.r])]
        self.share_observation_space = [[self.env.state_space], [self.env.state_space]]
        self.observation_space = [[self.env.observation_space(agent) for agent in self.angel_agents], \
                                  [self.env.observation_space(agent) for agent in self.demon_agents]]
        self.action_space = [[self.env.action_space(agent) for agent in self.angel_agents], 
                             [self.env.action_space(agent) for agent in self.demon_agents]]
        self.episode_limit = self.max_cycles
        self.available_actions_angel = [[1] * self.env.action_space(agent).n for agent in self.angel_agents]
        self.available_actions_demon = [[1] * self.env.action_space(agent).n for agent in self.demon_agents]
        self.available_actions = [self.available_actions_angel, self.available_actions_demon]
        self.available_actions = [np.stack(self.available_actions[i], axis=0) for i in range(2)]
        self.n_agents = len(self.agents)
        self.n_angels = len(self.angel_agents)
        self.n_demons = len(self.demon_agents)
        self.obs_own_feat = [self.observation_space[0][0].shape[2], self.observation_space[1][0].shape[2]]
        self.obs_enemy_feat = [self.observation_space[0][0].shape[2], self.observation_space[1][0].shape[2]]
        self.obs_ally_feat = [self.observation_space[0][0].shape[2], self.observation_space[1][0].shape[2]]
    
    def step(self, actions):
        angel_actions = {agent: actions[0][i] for i, agent in enumerate(self.angel_agents)}
        demon_actions = {agent: actions[1][i] for i, agent in enumerate(self.demon_agents)}
        all_actions = angel_actions | demon_actions
        observations, rewards, terminations, truncations, infos = self.env.step(all_actions=all_actions)
        observations = [[observations.get(agent, np.zeros(self.observation_space[0][i].shape)) for i, agent in enumerate(self.angel_agents)], 
                        [observations.get(agent, np.zeros(self.observation_space[1][i].shape)) for i, agent in enumerate(self.demon_agents)]]
        observations = [np.stack(observations[i], axis=0) for i in range(2)]
        states = [[self.env.state() for _ in self.angel_agents], [self.env.state() for _ in self.demon_agents]]
        states = [np.stack(states[i], axis=0) for i in range(2)]
        rewards = [[[rewards.get(agent, 0)] for agent in self.angel_agents], [[rewards.get(agent, 0)] for agent in self.demon_agents]]
        rewards = [np.stack(rewards[i], axis=0) for i in range(2)]
        dones = [[terminations.get(agent, True) or (self.env.frames >= self.env.max_cycles) for agent in self.angel_agents], 
                 [terminations.get(agent, True) or (self.env.frames >= self.env.max_cycles) for agent in self.demon_agents]]
        dones = [np.stack(dones[i], axis=0) for i in range(2)]
        infos = [[infos.get(agent, {}) for agent in self.angel_agents], 
                 [infos.get(agent, {}) for agent in self.demon_agents]]
        return observations, states, rewards, dones, infos, self.available_actions

    def reset(self):
        observations = self.env.reset()
        observations = [[observations.get(agent, np.zeros(self.observation_space[0][i].shape)) for i, agent in enumerate(self.angel_agents)], 
                        [observations.get(agent, np.zeros(self.observation_space[1][i].shape)) for i, agent in enumerate(self.demon_agents)]]
        observations = [np.stack(observations[i], axis=0) for i in range(2)]
        states = [[self.env.state() for _ in self.angel_agents], [self.env.state() for _ in self.demon_agents]]
        states = [np.stack(states[i], axis=0) for i in range(2)]
        return observations, states, self.available_actions
    
    def seed(self, seed):
        self._seed = seed
        self.env.seed(seed=seed)
    
    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()