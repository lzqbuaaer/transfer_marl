import torch
import torch.nn as nn
from torch.distributions import OneHotCategorical, Normal, Uniform
from amb.models.base.cnn import CNNLayer
from amb.models.base.mlp import MLPBase
from amb.models.base.rnn import RNNLayer
from amb.utils.env_utils import check, get_shape_from_obs_space, get_onehot_shape_from_act_space
from amb.utils.model_utils import get_active_func, init, get_init_method


class DDPGActor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(DDPGActor, self).__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.args = args
        self.hidden_sizes = args["hidden_sizes"]
        self.activation_func = args["activation_func"]
        self.final_activation_func = args["final_activation_func"]
        self.initialization_method = args["initialization_method"]
        self.expl_noise = args["expl_noise"]

        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        init_method = get_init_method(self.initialization_method)

        obs_shape = get_shape_from_obs_space(obs_space)
        self.act_shape = get_onehot_shape_from_act_space(action_space)

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
        
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        
        self.out = init_(nn.Linear(self.hidden_sizes[-1], self.act_shape))

        self.action_type = action_space.__class__.__name__
        if self.action_type == "Box":
            self.low = torch.tensor(action_space.low).to(**self.tpdv)
            self.high = torch.tensor(action_space.high).to(**self.tpdv)
            self.scale = (self.high - self.low) / 2
            self.mean = (self.high + self.low) / 2
            self.final_activation = get_active_func(self.final_activation_func)

        self.to(device)

    def sample(self, obs, available_actions=None):
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if self.action_type == "Box":
            actor_out = torch.ones((obs.shape[0], self.act_shape)).to(**self.tpdv)
            action_dist = Uniform(actor_out * self.low, actor_out * self.high)
        elif self.action_type == "Discrete" and available_actions is not None:
            actor_out = torch.ones((obs.shape[0], self.act_shape)).to(**self.tpdv)
            actor_out[available_actions == 0] = -1e10   
            action_dist = OneHotCategorical(logits=actor_out)        

        return action_dist

    def forward(self, obs, rnn_states, masks, available_actions=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(self.cnn(obs))
        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        actor_out = self.out(actor_features)

        if self.action_type == "Box":
            actor_out = self.scale * self.final_activation(actor_out) + self.mean
            action_dist = Normal(actor_out, self.expl_noise)
        elif self.action_type == "Discrete" and available_actions is not None:
            actor_out[available_actions == 0] = -1e10   
            action_dist = OneHotCategorical(logits=actor_out)        

        return action_dist, rnn_states
