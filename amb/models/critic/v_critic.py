import torch
import torch.nn as nn
from amb.models.base.cnn import CNNLayer
from amb.models.base.mlp import MLPBase
from amb.models.base.env import EnvLayer
from amb.models.base.rnn import RNNLayer
from amb.utils.env_utils import check, get_shape_from_obs_space
from amb.utils.model_utils import init, get_init_method


class VCritic(nn.Module):
    """V Network. Outputs value function predictions given global states."""

    def __init__(self, args, cent_obs_space, device=torch.device("cpu"), env_prior=None):
        """Initialize VCritic model.
        Args:
            args: (dict) arguments containing relevant model information.
            cent_obs_space: (gym.Space) centralized observation space.
            device: (torch.device) specifies the device to run on (cpu/gpu).        
        """
        super(VCritic, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.initialization_method = args["initialization_method"]
        self.activation_func = args["activation_func"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = get_init_method(self.initialization_method)
        self.state_align = args["obs_state_align"] if "obs_state_align" in args else False
        self.state_align_len = args["state_align_len"] if "state_align_len" in args else 0
        self.use_static_env_net = args.get("static_env_net", False)
        
        self.env_prior = env_prior
        if self.env_prior is None:
            assert args.get("env_prior_length", 0) == 0
        else:
            assert len(self.env_prior) == args.get("env_prior_length", 0)
        
        # TODO: state_alignment
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        if self.state_align:
            cent_obs_shape = [self.state_align_len]

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
        
        if self.use_static_env_net:
            self.static_env_net = EnvLayer(args)
            self.v_out = init_(nn.Linear(self.hidden_sizes[-1] + args.get("env_hidden_size", 128), 1))
        else:
            self.v_out = init_(nn.Linear(self.hidden_sizes[-1], 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """Compute actions from the given inputs.
        Args:
            cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.
        Returns:
            values: (torch.Tensor) value function predictions.
            rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        # TODO: state_alignment
        cent_obs = check(cent_obs).to(**self.tpdv)
        if self.state_align:
            thread_num = cent_obs.shape[0]
            state_add_len = self.state_align_len - cent_obs.shape[1]
            cent_obs = torch.cat([cent_obs, torch.zeros(thread_num, state_add_len).to(**self.tpdv)], dim=1)

        # 12000的来源：20 * 3 * 200
        # if cent_obs.shape[0] != 20:
        #     print(f"In VCritic's sample function, state's shape is now {cent_obs.shape}.")

        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(self.cnn(cent_obs))
        if self.use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
            
        if self.use_static_env_net:
            assert self.env_prior is not None
            env_prior = self.env_prior.repeat(critic_features.shape[0], 1)
            env_features = self.static_env_net(env_prior)
            critic_features = torch.concatenate([critic_features, env_features], dim=-1)
        
        values = self.v_out(critic_features)

        return values, rnn_states
