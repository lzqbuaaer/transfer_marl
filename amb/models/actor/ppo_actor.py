import torch
import torch.nn as nn
import os
import numpy as np
from torch.distributions import Categorical, Uniform
from amb.models.base.cnn import CNNLayer
from amb.models.base.mlp import MLPBase
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

        # obs_alignment
        self.obs_align = args["obs_state_align"] if "obs_state_align" in args else False
        self.obs_align_len = args["obs_align_len"] if "obs_align_len" in args else 0
        obs_shape = get_shape_from_obs_space(obs_space)
        if self.obs_align:
            obs_shape = [self.obs_align_len]

        # action_alignment
        self.action_space_align = args["action_space_align"] if "action_space_align" in args else False
        self.action_align_len = args["action_align_len"] if "action_align_len" in args else 0
        self.act_shape = get_onehot_shape_from_act_space(self.action_space)
        if self.action_space_align:
            self.act_shape = self.action_align_len

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

        self.base = MLPBase(args, input_dim)

        if self.use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_sizes[-1],
                self.hidden_sizes[-1],
                self.recurrent_n,
                self.initialization_method,
            )
        
        if self.manual_env_prior is not None:
            self.manual_embedding_net = nn.Embedding(len(self.manual_env_prior), self.args["manual_embedding_length"])

        if self.args["static_env_net"]:
            self.static_env_net = EnvLayer(args)

        self.act = ACTLayer(
            action_space,
            self.hidden_sizes[-1] + self.args.get("env_hidden_size", 128) \
                if self.args["static_env_net"] else self.hidden_sizes[-1],
            self.initialization_method,
            self.gain,
            args,
        )

        self.action_type = action_space.__class__.__name__
        if self.action_type == "Box":
            self.low = torch.tensor(action_space.low).to(**self.tpdv)
            self.high = torch.tensor(action_space.high).to(**self.tpdv)

        self.to(device)

    def sample(self, obs, available_actions=None):
        # obs_alignment
        obs = check(obs).to(**self.tpdv)
        if self.obs_align:
            thread_num = obs.shape[0]
            obs_add_len = self.obs_align_len - obs.shape[1]
            obs = torch.cat([obs, torch.zeros(thread_num, obs_add_len).to(**self.tpdv)], dim=1)
        # print(f"In PPOActor's sample function, obs's shape is now {obs.shape}.")

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
        if self.obs_align:
            thread_num = obs.shape[0]
            obs_add_len = self.obs_align_len - obs.shape[1]
            obs = torch.cat([obs, torch.zeros(thread_num, obs_add_len).to(**self.tpdv)], dim=1)
        # print(f"In PPOActor's forward function, obs's shape is now {obs.shape}.")
        
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        # action_alignment
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv) # （10, 10）==> (10, 41)
        if self.action_space_align:
            available_actions = torch.cat([available_actions, torch.zeros(available_actions.shape[0], self.action_align_len - available_actions.shape[1]).to(**self.tpdv)], dim=1)

        actor_features = self.base(self.cnn(obs))

        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

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
            # print(env_prior)
            env_features = self.static_env_net(env_prior)
            # # env_tensor = torch.zeros()
            # # env_features = self.static_env_net(env_tensor)
            # map_name = self.args["map_name"]
            # # 10 * 30
            # env_ori_tensor = torch.zeros(actor_features.shape[0], 30)
            # if map_name == "4m_vs_3m":
            #     indices = [0, 10, 20]
            #     values = [1, 3, 4]
            #     env_ori_tensor[:, indices] = torch.tensor(values, dtype=torch.float).view(1, -1)
            # elif map_name == "9m_vs_8m":
            #     indices = [0, 10, 20]
            #     values = [1, 9, 8]
            #     env_ori_tensor[:, indices] = torch.tensor(values, dtype=torch.float).view(1, -1)
            # elif map_name == "6m":
            #     indices = [0, 10, 20]
            #     values = [1, 6, 6]
            #     env_ori_tensor[:, indices] = torch.tensor(values, dtype=torch.float).view(1, -1)
            # env_tensor = check(env_ori_tensor).to(**self.tpdv)
            # # 10 * 128
            # env_features = self.static_env_net(env_tensor)
            total_features = torch.concatenate([actor_features, env_features], dim=-1)
            
            # todo: total_features = torch.concatenate([actor_features, env_features, llm_features], dim=-1)
        else:
            total_features = actor_features

        # action_alignment restore
        action_dist = self.act(total_features, available_actions)
        if self.action_space_align:
            action_dist = FixedCategorical(logits=action_dist.logits[:, :get_onehot_shape_from_act_space(self.action_space)])
        return action_dist, rnn_states