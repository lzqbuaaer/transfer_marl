import os
import torch
from amb.agents.base_agent import BaseAgent
from amb.models.actor.ppo_actor import PPOActor

class PPOAgent(BaseAgent):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu"), ally_num=2, agent_type="victim", llm_env_prior=None, manual_env_prior=None):
        # save arguments
        self.args = args
        self.device = device
        self.ally_num = ally_num
        self.agent_type = agent_type

        self.obs_space = obs_space
        self.act_space = act_space
        print(llm_env_prior, manual_env_prior)

        self.actor = PPOActor(args, self.obs_space, self.act_space, device=self.device, llm_env_prior=llm_env_prior, manual_env_prior=manual_env_prior)

    def forward(self, obs, rnn_states, masks, available_actions=None):
        action_dist, rnn_states = self.actor(obs, rnn_states, masks, available_actions)
        
        return action_dist, rnn_states

    @torch.no_grad()
    def sample(self, obs, available_actions=None):
        action_dist = self.actor.sample(obs, available_actions)
        actions = action_dist.sample()

        return actions, action_dist

    @torch.no_grad()
    def perform(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        action_dist, rnn_states = self.actor(obs, rnn_states, masks, available_actions)
        actions = (action_dist.mode if deterministic else action_dist.sample())

        return actions, rnn_states
    
    @torch.no_grad()
    def collect(self, obs, rnn_states, masks, available_actions=None, t=0):
        action_dist, rnn_states = self.actor(obs, rnn_states, masks, available_actions)
        actions = action_dist.sample()
        action_log_probs = action_dist.log_probs(actions)

        return actions, action_log_probs, rnn_states
    
    def restore(self, path):
        state_dict = torch.load(os.path.join(path, "actor.pth"))
        self.actor.load_state_dict(state_dict)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))

    def prep_training(self):
        self.actor.train()

    def prep_rollout(self):
        self.actor.eval()
