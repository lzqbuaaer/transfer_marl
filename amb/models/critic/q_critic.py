import torch
import torch.nn as nn
from amb.models.base.cnn import CNNLayer
from amb.models.base.mlp import MLPBase
from amb.models.base.rnn import RNNLayer
from amb.utils.env_utils import check, get_shape_from_obs_space
from amb.utils.model_utils import init, get_init_method


def get_combined_dim(cent_obs_feature_dim, act_spaces):
    combined_dim = cent_obs_feature_dim
    for space in act_spaces:
        if space.__class__.__name__ == "Box":
            combined_dim += space.shape[0]
        elif space.__class__.__name__ == "Discrete":
            combined_dim += space.n
    return combined_dim


class QCritic(nn.Module):
    """Q Network for critic."""
    def __init__(self, args, cent_obs_space, act_spaces, device=torch.device("cpu")):
        super(QCritic, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.initialization_method = args["initialization_method"]
        self.activation_func = args["activation_func"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = get_init_method(self.initialization_method)

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        if len(cent_obs_shape) == 3:
            self.cnn = CNNLayer(
                cent_obs_shape,
                self.hidden_sizes,
                self.initialization_method,
                self.activation_func,
            )
            input_dim = self.cnn.output_size
        else:
            self.cnn = nn.Identity()
            input_dim = cent_obs_shape[0]
        
        input_dim = get_combined_dim(input_dim, act_spaces)
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
        
        self.q_out = init_(nn.Linear(self.hidden_sizes[-1], 1))

        self.to(device)

    def forward(self, cent_obs, actions, rnn_states, masks):
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        actions = actions.reshape(actions.shape[0], -1)

        critic_features = self.cnn(cent_obs)
        critic_features = self.base(torch.cat([critic_features, actions], dim=-1))
        if self.use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        q_values = self.q_out(critic_features)

        return q_values, rnn_states
