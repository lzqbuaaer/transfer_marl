import torch.nn as nn
from amb.models.base.distributions import Categorical, DiagGaussian


class ACTLayer(nn.Module):
    """MLP Module to compute actions."""

    def __init__(
        self, action_space, inputs_dim, initialization_method, gain, args=None
    ):
        """Initialize ACTLayer.
        Args:
            action_space: (gym.Space) action space.
            inputs_dim: (int) dimension of network input.
            initialization_method: (str) initialization method.
            gain: (float) gain of the output layer of the network.
            args: (dict) arguments relevant to the network.
        """
        super(ACTLayer, self).__init__()
        self.action_type = action_space.__class__.__name__
        self.action_space_align = args["action_space_align"] if "action_space_align" in args else False
        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            # TODO: action_alignment
            if self.action_space_align:
                action_dim = args["action_align_len"]
            self.action_out = Categorical(
                inputs_dim, action_dim, initialization_method, gain
            )
        elif action_space.__class__.__name__ == "Box":
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(
                inputs_dim, action_dim, initialization_method, gain, args
            )

    def forward(self, x, available_actions=None):
        """Compute actions and action logprobs from given input.
        Args:
            x: (torch.Tensor) input to network.
            available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
            deterministic: (bool) whether to sample from action distribution or return the mode.
        Returns:
            actions: (torch.Tensor) actions to take.
            action_dist: (torch.distributions.Distribution) distributions of actions.
        """
        action_dist = self.action_out(x, available_actions)

        return action_dist
