import torch
import torch.nn as nn
import os
import numpy as np
from torch.distributions import Categorical, Uniform
from amb.models.base.cnn import CNNLayer
from amb.models.base.mlp import MLPBase
from amb.models.base.transformers import Transformer
from amb.models.base.env import EnvLayer
from amb.models.base.rnn import RNNLayer
from amb.models.base.act import ACTLayer
from amb.utils.env_utils import check, get_shape_from_obs_space, get_onehot_shape_from_act_space
from amb.models.base.distributions import FixedCategorical


class PPOActor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu"), llm_env_prior=None, manual_env_prior=None):
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
        # self.env_prior = env_prior
        # if self.env_prior is None:
        #     assert self.args.get("env_prior_length", 0) == 0
        # else:
        #     assert len(self.env_prior) == self.args.get("env_prior_length", 0)
        self.manual_env_prior = manual_env_prior
        if self.manual_env_prior is None:
            assert self.args.get("manual_env_prior_length", 0) == 0
        else:
            assert len(self.manual_env_prior) == self.args.get("manual_env_prior_length", 0)
        self.llm_env_prior = llm_env_prior
        if self.llm_env_prior is None:
            assert self.args.get("llm_env_prior_length", 0) == 0
        else:
            assert len(self.llm_env_prior) == self.args.get("llm_env_prior_length", 0)

        # # obs_alignment
        # self.obs_align = args["obs_state_align"] if "obs_state_align" in args else False
        # self.obs_align_len = args["obs_align_len"] if "obs_align_len" in args else 0
        obs_shape = get_shape_from_obs_space(obs_space)
        # if self.obs_align:
        #     obs_shape = [self.obs_align_len]

        # # action_alignment
        # self.action_space_align = args["action_space_align"] if "action_space_align" in args else False
        # self.action_align_len = args["action_align_len"] if "action_align_len" in args else 0
        self.act_shape = get_onehot_shape_from_act_space(self.action_space)
        # if self.action_space_align:
        #     self.act_shape = self.action_align_len
        
        if self.manual_env_prior is not None:
            self.manual_embedding_net = nn.Embedding(len(self.manual_env_prior), self.args["manual_embedding_length"])

        if self.args["static_env_net"]:
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
                self.hidden_sizes[-1] + self.args.get("env_hidden_size", 128) \
                    if self.args["static_env_net"] else self.hidden_sizes[-1],
                self.initialization_method,
                self.gain,
                args,
            )
        else:
            self.heads = args.get("obs_transformer_heads", 1)
            self.depth = args.get("obs_transformer_depth", 2)
            
            self.token_dim = args.get("obs_token_dim", 5)
            
            self.own_feat = args.get("obs_own_feat", 5)
            self.own_feat_length = self.own_feat
            self.enemy_feat = args.get("obs_enemy_feat", 5)
            self.enemy_feat_length = self.args["n_enemies"] * self.enemy_feat
            self.ally_feat = args.get("obs_ally_feat", 5)
            self.ally_feat_length = (self.args["n_agents"] - 1) * self.ally_feat
            
            self.own_feat_token_embedding = nn.Linear(self.own_feat, self.hidden_sizes[-1])
            self.enemy_feat_token_embedding = nn.Linear(self.enemy_feat, self.hidden_sizes[-1])
            self.ally_feat_token_embedding = nn.Linear(self.ally_feat, self.hidden_sizes[-1])
            
            self.transformer = Transformer(input_dim=self.token_dim, emb=self.hidden_sizes[-1], 
                                           heads=self.heads, depth=self.depth, output_dim=self.hidden_sizes[-1])
            self.act = nn.Linear(self.hidden_sizes[-1], 6)

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

    def forward(self, obs, rnn_states, masks, available_actions=None):
        # obs_alignment
        obs = check(obs).to(**self.tpdv)
        
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
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
            
            if self.use_recurrent_policy:
                if obs_embedding.shape[0] == rnn_states.shape[0]:
                    rnn_states = rnn_states * masks.squeeze(-1).view(-1, 1, 1).repeat(1, self.recurrent_n, rnn_states.shape[-1])
                    output = self.transformer.forward_embedding(obs_embedding, rnn_states, None)
                    actor_features = output[:, :-self.recurrent_n, :]
                    rnn_states = output[:, -self.recurrent_n:, :]
                else:
                    T = int(obs_embedding.shape[0] / rnn_states.shape[0])
                    obs_embedding = obs_embedding.view(T, rnn_states.shape[0], *obs_embedding.shape[1:])
                    masks = masks.view(T, rnn_states.shape[0])
                    actor_features = []
                    for t in range(T):
                        rnn_states = rnn_states * masks[t].view(-1, 1, 1).repeat(1, self.recurrent_n, rnn_states.shape[-1])
                        actor_feature = self.transformer.forward_embedding(obs_embedding[t], rnn_states, None)
                        actor_features.append(actor_feature[:, :-self.recurrent_n, :])
                        rnn_states = actor_feature[:, -self.recurrent_n:, :]
                    actor_features = torch.cat(actor_features, dim=0)
                    
            else:
                actor_features = self.transformer.forward_embedding(obs_embedding, None, None)

        if self.args["static_env_net"]:
            # assert self.env_prior is not None
            # env_prior = self.env_prior.repeat(actor_features.shape[0], 1)
            # env_features = self.static_env_net(env_prior)
            assert self.llm_env_prior is not None or self.manual_env_prior is not None
            env_prior = None
            if self.manual_env_prior is not None:
                manual_index = torch.arange(len(self.manual_env_prior)).repeat(actor_features.shape[0], 1).to(**self.tpdv).long()
                # print(manual_index.dtype)
                manual_embedded = self.manual_embedding_net(manual_index)
                manual_env_prior = self.manual_env_prior.repeat(actor_features.shape[0], 1).unsqueeze(1).float()
                env_prior = torch.bmm(manual_env_prior, manual_embedded)
                env_prior = torch.squeeze(env_prior)
            if self.llm_env_prior is not None:
                llm_env_prior = self.llm_env_prior.repeat(actor_features.shape[0], 1)
                if self.manual_env_prior is None:
                    env_prior = llm_env_prior
                else:
                    env_prior = torch.concatenate([env_prior, llm_env_prior], dim=-1)
            
            env_features = self.static_env_net(env_prior)
            # TODO: env belief posterior for UPDeT
            assert not self.actor_use_updet
            total_features = actor_features + env_features
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
        
        return action_dist, rnn_states