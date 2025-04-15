import os
import torch
from amb.agents.base_agent import BaseAgent
from amb.models.actor.ppo_actor import PPOActor
from amb.models.belief.transformer_belief import TransformerBelief

class PPOAgent(BaseAgent):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu"), ally_num=2, agent_type="victim"):
        # save arguments
        self.args = args
        self.device = device
        self.ally_num = ally_num
        self.agent_type = agent_type

        self.obs_space = obs_space
        self.act_space = act_space

        self.actor = PPOActor(args, self.obs_space, self.act_space, device=self.device)
        
        self.env_belief = args.get("env_belief", False)
        if self.env_belief:
            self.belief = TransformerBelief(args, device=device)
        self.actor_divide_conquer = args.get("actor_divide_conquer", False)
        self.actor_use_dt2gs = args.get("actor_use_dt2gs", False)

    def forward(self, obs, rnn_states, masks, available_actions=None, env_belief=None, previous_skills=None):
        actions, rnn_states = self.actor(obs, rnn_states, masks, available_actions=available_actions, env_belief=env_belief, previous_skills=previous_skills)
        if self.actor_use_dt2gs:
            actions, _ = actions
        if self.actor_divide_conquer:
            actions, _, _ = actions
        action_dist = actions
        
        return action_dist, rnn_states
    
    def forward_belief(self, obs, last_reward, last_obs, last_belief, rnn_states, masks):
        assert self.env_belief
        belief = self.belief(obs, last_reward, last_obs, last_belief, rnn_states, masks)
        return belief

    @torch.no_grad()
    def sample(self, obs, available_actions=None):
        action_dist = self.actor.sample(obs, available_actions)
        actions = action_dist.sample()

        return actions, action_dist

    @torch.no_grad()
    def perform(self, obs, rnn_states, masks, available_actions=None, env_belief=None, previous_skills=None, deterministic=False):
        actions, rnn_states = self.actor(obs, rnn_states, masks, available_actions=available_actions, env_belief=env_belief, previous_skills=previous_skills, deterministic=deterministic)
        
        if self.actor_use_dt2gs:
            actions, skills = actions
        if self.actor_divide_conquer:
            actions, chosen, _ = actions
        action_dist = actions
        actions = (action_dist.mode if deterministic else action_dist.sample())
        if self.actor_divide_conquer:
            actions = (actions, chosen)
        if self.actor_use_dt2gs:
            actions = (actions, skills)
        return actions, rnn_states
    
    @torch.no_grad()
    def collect(self, obs, rnn_states, masks, available_actions=None, env_belief=None, previous_skills=None, t=0):
        actions, rnn_states = self.actor(obs, rnn_states, masks, available_actions=available_actions, env_belief=env_belief, previous_skills=previous_skills)
        
        if self.actor_use_dt2gs:
            actions, skills = actions
        if self.actor_divide_conquer:
            actions, chosen, chosen_log_prob = actions
        action_dist = actions
        actions = action_dist.sample()
        action_log_probs = action_dist.log_probs(actions)
        if self.actor_divide_conquer:
            actions = (actions, chosen)
            action_log_probs = action_log_probs + chosen_log_prob
        if self.actor_use_dt2gs:
            actions = (actions, skills)
        return actions, action_log_probs, rnn_states
    
    def restore(self, path):
        state_dict = torch.load(os.path.join(path, "actor.pth"))
        self.actor.load_state_dict(state_dict)
        
        if self.env_belief:
            state_dict_belief = torch.load(os.path.join(path, "belief.pth"))
            self.belief.load_state_dict(state_dict_belief)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))
        if self.env_belief:
            torch.save(self.belief.state_dict(), os.path.join(path, "belief.pth"))

    def prep_training(self):
        self.actor.train()
        if self.env_belief:
            self.belief.train()

    def prep_rollout(self):
        self.actor.eval()
        if self.env_belief:
            self.belief.train()
