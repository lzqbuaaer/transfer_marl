import os
import torch
from amb.agents.base_agent import BaseAgent
from amb.models.actor.ppo_actor import PPOActor

class PPOAgent(BaseAgent):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu"), ally_num=2, agent_type="victim", env_prior=None):
        # save arguments
        self.args = args
        self.device = device
        self.ally_num = ally_num
        self.agent_type = agent_type

        self.obs_space = obs_space
        self.act_space = act_space
        print(env_prior)

        self.actor = PPOActor(args, self.obs_space, self.act_space, device=self.device, env_prior=env_prior)

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
        keys = list(state_dict.keys())

        if self.agent_type == "adv_traitor":
            self.actor.load_state_dict(state_dict)
            return

        ally_num = self.ally_num # TODO
        ori_size = state_dict[keys[0]].shape[0]
        target_size = self.obs_space[0]
        ally_feat_size = target_size - ori_size - 1 
        offset = ally_feat_size * ally_num if self.agent_type == "adv_victim" else 0
        add_zero_num = 1 if self.agent_type == "adv_victim" else 0
        if target_size > ori_size:
            # 改掉[0, 1, 2]三维
            state_dict[keys[0]] = \
                torch.concatenate([state_dict[keys[0]][:offset], \
                    torch.zeros(ally_feat_size, device="cuda"), \
                    state_dict[keys[0]][offset:], \
                    torch.zeros(add_zero_num, device="cuda")], dim=0)
            state_dict[keys[1]] = \
                torch.concatenate([state_dict[keys[1]][:offset], \
                    torch.zeros(ally_feat_size, device="cuda"), \
                    state_dict[keys[1]][offset:], \
                    torch.zeros(add_zero_num, device="cuda")], dim=0)
            state_dict[keys[2]] = \
                torch.concatenate([state_dict[keys[2]][:, :offset], \
                    torch.zeros(self.args["hidden_sizes"][0], ally_feat_size, device="cuda"), \
                    state_dict[keys[2]][:, offset:], \
                    torch.zeros(self.args["hidden_sizes"][0], add_zero_num, device="cuda")], dim=-1)
        elif target_size < ori_size:
            raise Exception("PPOAgent restore function is transferring from large to small.")
        self.actor.load_state_dict(state_dict)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))

    def prep_training(self):
        self.actor.train()

    def prep_rollout(self):
        self.actor.eval()
