import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from amb.utils.env_utils import check, get_shape_from_obs_space, get_onehot_shape_from_act_space


class COMACritic(nn.Module):
    def __init__(self, args, num_agents, obs_spaces, share_obs_space, act_spaces, device=th.device("cpu")):
        super(COMACritic, self).__init__()

        self.args = args
        self.n_actions = get_onehot_shape_from_act_space(act_spaces[0])
        self.n_agents = num_agents
        self.device = device

        input_shape = self._get_input_shape(obs_spaces, share_obs_space, act_spaces,)
        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.n_actions)

    def forward(self, batch):
        self = self.to(self.device)
        inputs = self._build_inputs(batch)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _build_inputs(self, batch):
        bs = batch["obs"].shape[0]
        inputs = []
        # state
        inputs.append(check(batch["share_obs"]).to(self.device))

        # observation
        inputs.append(check(batch["obs"]).to(self.device))

        # actions (masked out by agent)
        actions = check(batch["actions_onehot"]).view(bs, 1, -1).repeat(1, self.n_agents, 1).to(self.device)
        agent_mask = (1 - th.eye(self.n_agents, device=self.device))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
        inputs.append(actions * agent_mask.unsqueeze(0))

        # last actions
        ori_actions = check(batch["actions_onehot"]).to(self.device)
        last_actions = th.cat([th.zeros_like(ori_actions[:, 0:1]), ori_actions[:, :-1]], dim=1)
        last_actions = last_actions.view(bs, 1, -1).repeat(1, self.n_agents, 1)
        inputs.append(last_actions)

        inputs.append(th.eye(self.n_agents, device=self.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, obs_spaces, share_obs_space, act_spaces,):
        # state
        input_shape = int(np.prod(get_shape_from_obs_space(share_obs_space)))
        # observation
        input_shape += int(np.prod(get_shape_from_obs_space(obs_spaces[0])))
        # actions and last actions
        input_shape += get_onehot_shape_from_act_space(act_spaces[0]) * self.n_agents * 2
        # agent id
        input_shape += self.n_agents
        return input_shape