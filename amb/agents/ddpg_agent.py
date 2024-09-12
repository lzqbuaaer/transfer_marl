import os
import torch
from amb.agents.base_agent import BaseAgent
from amb.models.actor.ddpg_actor import DDPGActor

class DDPGAgent(BaseAgent):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        # save arguments
        self.args = args
        self.device = device

        self.obs_space = obs_space
        self.act_space = act_space

        self.actor = DDPGActor(args, self.obs_space, self.act_space, self.device)

    def forward(self, obs, rnn_states, masks, available_actions=None):
        action_dist, rnn_states = self.actor(obs, rnn_states, masks, available_actions)
        
        return action_dist, rnn_states

    @torch.no_grad()
    def perform(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        action_dist, rnn_states = self.actor(obs, rnn_states, masks, available_actions)
        if self.actor.action_type == "Discrete":
            actions = action_dist.mode if deterministic else action_dist.sample()
            actions = actions.argmax(dim=-1, keepdim=True)
        elif self.actor.action_type == "Box":
            actions = action_dist.mean

        return actions, rnn_states
    
    @torch.no_grad()
    def sample(self, obs, available_actions=None):
        action_dist = self.actor.sample(obs, available_actions)
        if self.actor.action_type == "Discrete":
            actions_onehot = action_dist.sample()
            actions = actions_onehot.argmax(dim=-1, keepdim=True)
        elif self.actor.action_type == "Box":
            actions_onehot = action_dist.sample()
            actions = actions_onehot.clone()

        return actions, actions_onehot
    
    @torch.no_grad()
    def collect(self, obs, rnn_states, masks, available_actions=None, t=0):
        action_dist, rnn_states = self.actor(obs, rnn_states, masks, available_actions)
        if self.actor.action_type == "Discrete":
            actions_onehot = action_dist.sample()   # here sample() means mode + noise
            actions = actions_onehot.argmax(dim=-1, keepdim=True)
        elif self.actor.action_type == "Box":
            actions_onehot = action_dist.sample()
            actions = actions_onehot.clone()

        return actions, actions_onehot, rnn_states
    
    def restore(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pth")))

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))

    def prep_training(self):
        self.actor.train()

    def prep_rollout(self):
        self.actor.eval()
