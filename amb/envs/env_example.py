class EnvExample:
    # spaces are a list of gym.Spaces, and the length of spaces equals to the number of agents.
    observation_space = None
    share_observation_space = None
    action_space = None

    # an positive integer to reflect the number of agents.
    n_agents = None

    def seed(self, seed):
        """Set the random seed of this environment. Called before reset()."""
        return

    def reset(self):
        """Reset the environment.
        obs and share_obs are numpy.ndarrays that have shape (num_agents, vshape).
        available_actions are 0-1 numpy.ndarrays that have shape (num_agents, action_num) for discrete action spaces,
        and None for continuous action spaces.
        """
        return obs, share_obs, available_actions

    def step(self, actions):
        """Process a step of the environment.
        actions: numpy.ndarray (num_agents, action_shape). Actions must be 2-dimentional.

        obs, share_obs: numpy.ndarray (num_agents, vshape)
        rewards: numpy.ndarray (num_agents, 1). Rewards for different agents can be different.
        dones: boolean numpy.ndarray (num_agents, 1). True when an episode is done or the time is out of limit, else False.
        infos: list of dict, e.g., [{}, {}]
        available_actions: 0-1 numpy.ndarray (num_agents, action_num) or None.
        """
        return obs, share_obs, rewards, dones, infos, available_actions
    
    def render(self, mode):
        """render for environment.
        When mode is rgb_array, should return the rendered environment image in RGB values.
        """
        if mode == "rgb_array":
            return array 
        if mode == "human":
            return
        
    def close(self):
        """Close the environment and release the resources."""
        return
