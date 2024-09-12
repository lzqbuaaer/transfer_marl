import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from amb.utils.env_utils import check, get_shape_from_obs_space, get_onehot_shape_from_act_space


class QTranBase(nn.Module):
    def __init__(self, args, num_agents, obs_space, act_sapce, device=th.device("cpu")):
        super(QTranBase, self).__init__()

        self.args = args
        self.n_agents = num_agents
        self.n_actions = get_onehot_shape_from_act_space(act_sapce[0])
        self.state_dim = int(np.prod(get_shape_from_obs_space(obs_space)))
        self.arch = args["qtran_arch"] # QTran architecture
        self.embed_dim = args["mixing_embed_dim"]
        self.rnn_hidden_dim = args["hidden_sizes"][-1]
        self.tpdv = dict(dtype=th.float32, device=device)

        # Q(s,u)
        if self.arch == "coma_critic":
            # Q takes [state, u] as input
            q_input_size = self.state_dim + self.n_actions
        elif self.arch == "qtran_paper":
            # Q takes [state, agent_action_observation_encodings]
            q_input_size = self.state_dim + self.rnn_hidden_dim + self.n_actions
        else:
            raise Exception("{} is not a valid QTran architecture".format(self.arch))

        self.q_input_size = q_input_size

        if self.args["network_size"] == "small":
            self.Q = nn.Sequential(nn.Linear(q_input_size, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, 1))

            # V(s)
            self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, 1))
            ae_input = self.rnn_hidden_dim + self.n_actions
            self.action_encoding = nn.Sequential(nn.Linear(ae_input, ae_input),
                                                 nn.ReLU(),
                                                 nn.Linear(ae_input, ae_input))
        elif self.args["network_size"] == "big":
            self.Q = nn.Sequential(nn.Linear(q_input_size, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, 1))
            # V(s)
            self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, 1))
            ae_input = self.rnn_hidden_dim + self.n_actions
            self.action_encoding = nn.Sequential(nn.Linear(ae_input, ae_input),
                                                 nn.ReLU(),
                                                 nn.Linear(ae_input, ae_input))
        else:
            assert False

    def forward(self, batch, hidden_states, actions=None):
        self = self.to(**self.tpdv)
        bs = batch["obs"].shape[0]
        hidden_states = hidden_states.squeeze(2).to(**self.tpdv) # [B, N, H]
        states = check(batch["share_obs"]).to(**self.tpdv) # [B, N, V]
        # ! 全局状态share_obs对于不同的agent为什么是不同的值呢？

        if actions is None:
            # Use the actions taken by the agents
            actions = check(batch["actions_onehot"]).to(**self.tpdv) # [B, N, V]
        else:
            actions = actions.to(**self.tpdv)

        if self.arch == "coma_critic":
            inputs = th.cat([states, actions], dim=2)
        elif self.arch == "qtran_paper":
            agent_state_action_input = th.cat([hidden_states, actions], dim=2)
            agent_state_action_encoding = self.action_encoding(agent_state_action_input.reshape(bs * self.n_agents, -1)).reshape(bs, self.n_agents, -1)
            # agent_state_action_encoding = agent_state_action_encoding.sum(dim=1) # Sum across agents
            inputs = th.cat([states, agent_state_action_encoding], dim=2)

        # [batch_size, num_agents, num_actions] -> [batch_size, 1]
        inputs = inputs.reshape(-1, self.q_input_size)
        q_outputs = self.Q(inputs).reshape(-1, self.n_agents, 1)

        states = states.reshape(-1, self.state_dim)
        v_outputs = self.V(states).reshape(-1, self.n_agents, 1)

        return q_outputs, v_outputs
