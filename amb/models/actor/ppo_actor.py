import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.distributions import Categorical, Uniform
from amb.models.base.cnn import CNNLayer
from amb.models.base.mlp import MLPBase
from amb.models.base.transformers import Encoder
from amb.models.base.env import EnvLayer
from amb.models.base.rnn import RNNLayer
from amb.models.base.act import ACTLayer
from amb.utils.env_utils import check, get_shape_from_obs_space, get_onehot_shape_from_act_space
from amb.models.base.distributions import FixedCategorical


class PPOActor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(PPOActor, self).__init__()
        self.args = args
        self.gain = args["gain"]
        self.hidden_sizes = args["hidden_sizes"]
        self.initialization_method = args["initialization_method"]
        self.activation_func = args["activation_func"]
        self.action_space = action_space

        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        self.actor_use_updet = args.get("actor_use_updet", False)
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        self.env_belief = args.get("env_belief", False)
        self.env_belief_dim = args.get("env_belief_dim", 0)
        print(self.env_belief, self.env_belief_dim)
        self.actor_divide_conquer = args.get("actor_divide_conquer", False)

        obs_shape = get_shape_from_obs_space(obs_space)
        self.act_shape = get_onehot_shape_from_act_space(self.action_space)

        if self.env_belief:
            self.static_env_net = EnvLayer(args)

        if len(obs_shape) == 3:
            self.cnn = CNNLayer(
                obs_shape,
                self.hidden_sizes,
                self.initialization_method,
                self.activation_func,
            )
            input_dim = self.cnn.output_size
        else:
            self.cnn = nn.Identity()
            input_dim = obs_shape[0]

        if not self.actor_use_updet:
            self.base = MLPBase(args, input_dim)

            if self.use_recurrent_policy:
                self.rnn = RNNLayer(
                    self.hidden_sizes[-1],
                    self.hidden_sizes[-1],
                    self.recurrent_n,
                    self.initialization_method,
                )
                
            self.act = ACTLayer(
                action_space,
                self.hidden_sizes[-1],
                self.initialization_method,
                self.gain,
                args,
            )
        else:
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
            
            self.transformer = Encoder(emb=self.hidden_sizes[-1], heads=self.heads, 
                                       depth=self.depth)
            self.act = nn.Linear(self.hidden_sizes[-1], 6)
            
            if self.actor_divide_conquer:
                self.agent_relative = Encoder(emb=self.hidden_sizes[-1], heads=self.heads,
                                              depth=self.depth)

        self.action_type = action_space.__class__.__name__
        if self.action_type == "Box":
            self.low = torch.tensor(action_space.low).to(**self.tpdv)
            self.high = torch.tensor(action_space.high).to(**self.tpdv)

        self.to(device)

    def sample(self, obs, available_actions=None):
        # obs_alignment
        obs = check(obs).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        
        if self.action_type == "Box":
            actor_out = torch.ones((obs.shape[0], self.act_shape)).to(**self.tpdv)
            action_dist = Uniform(actor_out * self.low, actor_out * self.high)
        elif self.action_type == "Discrete" and available_actions is not None:
            actor_out = torch.ones((obs.shape[0], self.act_shape)).to(**self.tpdv)
            actor_out[available_actions == 0] = -1e10   
            action_dist = Categorical(logits=actor_out)        
        return action_dist

    def forward(self, obs, rnn_states, masks, available_actions=None, env_belief=None, deterministic=False, chosen_specify=None):
        # obs_alignment
        obs = check(obs).to(**self.tpdv)
        
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if env_belief is not None:
            env_belief = check(env_belief).to(**self.tpdv)
        
        # action_alignment
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if not self.actor_use_updet:
            actor_features = self.base(self.cnn(obs))

            if self.use_recurrent_policy:
                actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        else:
            assert obs.shape[-1] == self.ally_feat_length + self.enemy_feat_length + self.own_feat_length
            
            obs_own = obs[..., self.ally_feat_length + self.enemy_feat_length:].unsqueeze(-2)
            obs_own_embedding = self.own_feat_token_embedding(obs_own)
            obs_enemy = obs[..., self.ally_feat_length: self.ally_feat_length + self.enemy_feat_length].reshape(*obs.shape[:-1], -1, self.enemy_feat)
            obs_enemy_embedding = self.enemy_feat_token_embedding(obs_enemy)
            obs_ally = obs[..., :self.ally_feat_length].reshape(*obs.shape[:-1], -1, self.ally_feat)
            obs_ally_embedding = self.ally_feat_token_embedding(obs_ally)
            obs_embedding = torch.cat([obs_own_embedding, obs_enemy_embedding, obs_ally_embedding], dim=-2)
            
            # Calculate the relationship between agents in order to divide and conquer
            if self.actor_divide_conquer:
                if self.use_recurrent_policy:
                    rnn_states_dc = rnn_states.clone()
                    if obs_embedding.shape[0] == rnn_states_dc.shape[0]:
                        rnn_states_dc = rnn_states_dc * masks.squeeze(-1).view(-1, 1, 1).repeat(1, self.recurrent_n, rnn_states_dc.shape[-1])
                        output_dc = self.transformer.forward(obs_embedding, rnn_states_dc, None)
                        relationships = output_dc[:, :-self.recurrent_n, :]
                    else:
                        T = int(obs_embedding.shape[0] / rnn_states_dc.shape[0])
                        obs_embedding = obs_embedding.view(T, rnn_states_dc.shape[0], *obs_embedding.shape[1:])
                        masks = masks.view(T, rnn_states_dc.shape[0])
                        relationships = []
                        for t in range(T):
                            rnn_states_dc = rnn_states_dc * masks[t].view(-1, 1, 1).repeat(1, self.recurrent_n, rnn_states_dc.shape[-1])
                            output_dc = self.transformer.forward(obs_embedding[t], rnn_states_dc, None)
                            relationships.append(output_dc[:, :-self.recurrent_n, :])
                            rnn_states_dc = output_dc[:, -self.recurrent_n:, :]
                        relationships = torch.cat(relationships, dim=0)
                else:
                    relationships = self.transformer.forward(obs_embedding, None, None)
                reference = relationships[:, 0, :]
                comparisons = relationships[:, 1:, :]
                cosine_sim = F.cosine_similarity(reference.unsqueeze(1), comparisons, dim=-1)
                probs = (cosine_sim + 1) / 2    # [batch, entity_num - 1]
                if chosen_specify is not None:
                    chosen = chosen_specify.bool()
                elif not deterministic:
                    chosen = torch.bernoulli(probs).bool()
                else:
                    chosen = probs > 0.5
                chosen_prob = torch.where(chosen, probs, 1 - probs).prod(dim=-1).unsqueeze(-1)  # [batch, 1]
                if self.use_recurrent_policy:
                    chosen_mask = torch.ones(obs_embedding.shape[0], obs_embedding.shape[1] + self.recurrent_n).bool().to(**self.tpdv)
                else:
                    chosen_mask = torch.ones(obs_embedding.shape[0], obs_embedding.shape[1]).bool().to(**self.tpdv)
                chosen_mask[:, 1: (1 + chosen.shape[1])] = chosen
                if available_actions is not None:
                    available_actions[:, 6:][chosen[:, :self.args["n_enemies"]]==False] = 0.
            else:
                chosen_mask = None
                            
            if self.use_recurrent_policy:
                if obs_embedding.shape[0] == rnn_states.shape[0]:
                    rnn_states = rnn_states * masks.squeeze(-1).view(-1, 1, 1).repeat(1, self.recurrent_n, rnn_states.shape[-1])
                    output = self.transformer.forward(obs_embedding, rnn_states, chosen_mask)
                    actor_features = output[:, :-self.recurrent_n, :]
                    rnn_states = output[:, -self.recurrent_n:, :]
                else:
                    T = int(obs_embedding.shape[0] / rnn_states.shape[0])
                    obs_embedding = obs_embedding.view(T, rnn_states.shape[0], *obs_embedding.shape[1:])
                    masks = masks.view(T, rnn_states.shape[0])
                    actor_features = []
                    for t in range(T):
                        rnn_states = rnn_states * masks[t].view(-1, 1, 1).repeat(1, self.recurrent_n, rnn_states.shape[-1])
                        actor_feature = self.transformer.forward(obs_embedding[t], rnn_states, chosen_mask)
                        actor_features.append(actor_feature[:, :-self.recurrent_n, :])
                        rnn_states = actor_feature[:, -self.recurrent_n:, :]
                    actor_features = torch.cat(actor_features, dim=0)
                    
            else:
                actor_features = self.transformer.forward(obs_embedding, None, chosen_mask)

            
        if self.env_belief:
            env_features = self.static_env_net(env_belief)
            total_features = actor_features + env_features.unsqueeze(-2)
        else:
            total_features = actor_features

        if not self.actor_use_updet:
            action_dist = self.act(total_features, available_actions)
        else:
            basic_actions = self.act(total_features[..., 0, :])

            # each enemy has an output Q
            enemies_actions = []
            for i in range(self.args['n_enemies']):
                enemy_action = self.act(total_features[:, 1 + i, :])
                enemy_action = torch.mean(enemy_action, dim=-1)
                enemies_actions.append(enemy_action)
            enemies_actions = torch.stack(enemies_actions, dim=-1)

            logits = torch.cat((basic_actions, enemies_actions), dim=-1)
            if available_actions is not None:
                logits[available_actions == 0] = -1e10
            action_dist = FixedCategorical(logits=logits)
        
        if self.actor_divide_conquer:
            return (action_dist, chosen, chosen_prob), rnn_states
        else:
            return action_dist, rnn_states