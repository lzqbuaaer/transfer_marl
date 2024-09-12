"""Modify standard PyTorch distributions so they to make compatible with this codebase."""
import torch
import torch.nn as nn
from amb.utils.model_utils import init, get_init_method


class OneHotEpsilonGreedy:
    def __init__(self, logits, t, avail_actions=None, eps_start=1.0, eps_finish=0.05, eps_anneal_time=100000):
        self.logits = logits
        self.t = t
        self.eps_start = eps_start
        self.eps_finish = eps_finish
        self.eps_anneal_time = eps_anneal_time
        self.avail_actions = avail_actions
        delta = (self.eps_start - self.eps_finish) / self.eps_anneal_time

        if self.avail_actions is not None:
            self.logits[self.avail_actions==0] = -1e10

        self.epsilon = max(self.eps_finish, self.eps_start - delta * self.t)

    def sample(self):
        random_logits = torch.ones_like(self.logits)

        if self.avail_actions is not None:
            random_logits[self.avail_actions==0] = -1e10

        random_actions = torch.distributions.Categorical(logits=random_logits).sample()
        masked_actions = self.logits.argmax(dim=-1)

        random_numbers = torch.rand_like(self.logits[..., 0])
        pick_random = (random_numbers < self.epsilon).long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_actions

        return torch.nn.functional.one_hot(picked_actions, num_classes=self.logits.shape[-1])
    
    @property
    def mode(self):
        masked_logits = self.logits.clone()

        if self.avail_actions is not None:
            masked_logits[self.avail_actions==0] = -1e10

        masked_actions = masked_logits.argmax(dim=-1)

        return torch.nn.functional.one_hot(masked_actions, num_classes=self.logits.shape[-1])


class OneHotMultinomial:
    def __init__(self, logits, t, avail_actions=None, eps_start=1.0, eps_finish=0.05, eps_anneal_time=100000):
        self.logits = logits
        self.t = t
        self.eps_start = eps_start
        self.eps_finish = eps_finish
        self.eps_anneal_time = eps_anneal_time
        self.avail_actions = avail_actions
        delta = (self.eps_start - self.eps_finish) / self.eps_anneal_time

        if self.avail_actions is not None:
            self.logits[self.avail_actions==0] = -1e10

        self.epsilon = max(self.eps_finish, self.eps_start - delta * self.t)

    def sample(self):
        random_logits = torch.ones_like(self.logits)

        if self.avail_actions is not None:
            random_logits[self.avail_actions==0] = -1e10
        
        # softmax
        # random_logits = torch.exp(random_logits) / torch.exp(random_logits).sum(dim=-1, keepdim=True)

        random_actions = torch.distributions.Categorical(logits=random_logits).sample()

        return torch.nn.functional.one_hot(random_actions, num_classes=self.logits.shape[-1])
    
    @property
    def mode(self):
        masked_logits = self.logits.clone()

        if self.avail_actions is not None:
            masked_logits[self.avail_actions==0] = -1e10

        masked_actions = masked_logits.argmax(dim=-1)

        return torch.nn.functional.one_hot(masked_actions, num_classes=self.logits.shape[-1])
    

class FixedCategorical(torch.distributions.Categorical):
    """Modify standard PyTorch Categorical."""
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return super().log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    @property
    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

class FixedNormal(torch.distributions.Normal):
    """Modify standard PyTorch Normal."""
    def log_probs(self, actions):
        return super().log_prob(actions)

    def entropy(self):
        return super().entropy().sum(-1)

    @property
    def mode(self):
        return self.mean


class Categorical(nn.Module):
    """A linear layer followed by a Categorical distribution."""
    # 这里的init_method可能需要更改一下，
    # 因为行为维数扩增以后，我们希望无用的维度的权重偏小
    def __init__(self, num_inputs, num_outputs, initialization_method="orthogonal_", gain=0.01):
        super(Categorical, self).__init__()
        init_method = get_init_method(initialization_method)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    """A linear layer followed by a Diagonal Gaussian distribution."""
    def __init__(
        self,
        num_inputs,
        num_outputs,
        initialization_method="orthogonal_",
        gain=0.01,
        args=None,
    ):
        super(DiagGaussian, self).__init__()

        init_method = get_init_method(initialization_method)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        if args is not None:
            self.std_x_coef = args["std_x_coef"]
            self.std_y_coef = args["std_y_coef"]
        else:
            self.std_x_coef = 1.0
            self.std_y_coef = 0.5
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        log_std = torch.ones(num_outputs) * self.std_x_coef
        self.log_std = torch.nn.Parameter(log_std)

    def forward(self, x, available_actions=None):
        action_mean = self.fc_mean(x)
        action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef
        return FixedNormal(action_mean, action_std)
