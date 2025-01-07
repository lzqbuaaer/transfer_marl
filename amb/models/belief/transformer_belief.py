import torch
import torch.nn as nn
from amb.models.base.transformers import Transformer
from amb.utils.env_utils import check


class TransformerBelief(nn.Module):
    def __init__(self, args, device=torch.device("cpu")):
        super(TransformerBelief, self).__init__()
        self.args = args
        self.hidden_sizes = args["hidden_sizes"]
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        self.env_belief_dim = args.get("env_belief_dim", 0)
        
        self.heads = args.get("obs_transformer_heads", 1)
        self.depth = args.get("obs_transformer_depth", 2)
        
        self.own_feat = args.get("obs_own_feat", 5)
        self.own_feat_length = self.own_feat
        self.enemy_feat = args.get("obs_enemy_feat", 5)
        self.enemy_feat_length = self.args["n_enemies"] * self.enemy_feat
        self.ally_feat = args.get("obs_ally_feat", 5)
        self.ally_feat_length = (self.args["n_agents"] - 1) * self.ally_feat
        
        self.own_feat_token_embedding = nn.Linear(self.own_feat, self.hidden_sizes[-1])
        self.enemy_feat_token_embedding = nn.Linear(self.enemy_feat, self.hidden_sizes[-1])
        self.ally_feat_token_embedding = nn.Linear(self.ally_feat, self.hidden_sizes[-1])
        self.reward_embedding = nn.Linear(1, self.hidden_sizes[-1])


        self.transformer = Transformer(emb=self.hidden_sizes[-1], heads=self.heads, 
                                       depth=self.depth, output_dim=self.env_belief_dim)

        self.to(device)

    def forward(self, obs, last_reward, last_obs, last_belief, rnn_states, masks):
        obs = check(obs).to(**self.tpdv)
        last_reward = check(last_reward).to(**self.tpdv)
        last_obs = check(last_obs).to(**self.tpdv)
        last_belief = check(last_belief).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        assert obs.shape[-1] == self.ally_feat_length + self.enemy_feat_length + self.own_feat_length
        assert last_obs.shape[-1] == obs.shape[-1]
            
        obs_own = obs[..., self.ally_feat_length + self.enemy_feat_length:].unsqueeze(-2)
        obs_own_embedding = self.own_feat_token_embedding(obs_own)
        obs_enemy = obs[..., self.ally_feat_length: self.ally_feat_length + self.enemy_feat_length].reshape(*obs.shape[:-1], -1, self.enemy_feat)
        obs_enemy_embedding = self.enemy_feat_token_embedding(obs_enemy)
        obs_ally = obs[..., :self.ally_feat_length].reshape(*obs.shape[:-1], -1, self.ally_feat)
        obs_ally_embedding = self.ally_feat_token_embedding(obs_ally)
        
        reward_embedding = self.reward_embedding(last_reward).unsqueeze(-2)
        real_output = torch.cat([reward_embedding, obs_own_embedding, obs_enemy_embedding, obs_ally_embedding], dim=-2)
        
        last_obs_own = last_obs[..., self.ally_feat_length + self.enemy_feat_length:].unsqueeze(-2)
        last_obs_own_embedding = self.own_feat_token_embedding(last_obs_own)
        last_obs_enemy = last_obs[..., self.ally_feat_length: self.ally_feat_length + self.enemy_feat_length].reshape(*last_obs.shape[:-1], -1, self.enemy_feat)
        last_obs_enemy_embedding = self.enemy_feat_token_embedding(last_obs_enemy)
        last_obs_ally = last_obs[..., :self.ally_feat_length].reshape(*last_obs.shape[:-1], -1, self.ally_feat)
        last_obs_ally_embedding = self.ally_feat_token_embedding(last_obs_ally)
        last_obs_embedding = torch.cat([last_obs_own_embedding, last_obs_enemy_embedding, last_obs_ally_embedding])
        
        if last_obs_embedding.shape[0] == rnn_states.shape[0]:
            rnn_states = rnn_states * masks.squeeze(-1).view(-1, 1, 1).repeat(1, self.recurrent_n, rnn_states.shape[-1])
            actor_outputs, rnn_states, _ = self.transformer(src=last_obs_embedding, tgt=real_output, src_h=rnn_states)
        else:
            T = int(last_obs_embedding.shape[0] / rnn_states.shape[0])
            last_obs_embedding = last_obs_embedding.view(T, rnn_states.shape[0], *last_obs_embedding.shape[1:])
            real_output = real_output.view(T, rnn_states.shape[0], *real_output.shape[1:])
            masks = masks.view(T, rnn_states.shape[0])
            actor_outputs = []
            for t in range(T):
                rnn_states = rnn_states * masks[t].view(-1, 1, 1).repeat(1, self.recurrent_n, rnn_states.shape[-1])
                actor_output, rnn_states, _ = self.transformer.forward(src=last_obs_embedding[t], tgt=real_output[t], src_h=rnn_states)
                actor_outputs.append(actor_output)
            actor_outputs = torch.cat(actor_outputs, dim=0)
                        
        prob_reward = torch.sigmoid(actor_outputs[..., 0, :])
        prob_obs_trans = torch.sigmoid(torch.mean(actor_outputs[..., 1:, :], dim=-2))
        prob = prob_reward * prob_obs_trans
        
        return prob / prob.sum(dim=-1, keepdim=True), rnn_states