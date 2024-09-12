import torch
from amb.agents.base_agent import BaseAgent
from amb.utils.env_utils import check


class IGS:
    """Iterative Gradient Sign Attack"""
    def __init__(self, args, act_space, device=torch.device("cpu")):
        self.args = args
        self.act_space = act_space
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.action_type = self.act_space.__class__.__name__

        self.epsilon = args["perturb_epsilon"]
        self.num_iters = args["perturb_iters"]
        self.adaptive_alpha = args["adaptive_alpha"]
        if self.adaptive_alpha:
            self.alpha = self.epsilon / self.num_iters
        else:
            self.alpha = args["perturb_alpha"]

        self.criterion = args.get("criterion", "default")
        if self.criterion != "default":
            self.criterion = eval(f"torch.nn.{self.criterion}()")
        else:
            if self.action_type == "Discrete":
                self.criterion = torch.nn.CrossEntropyLoss()
            elif self.action_type == "Box":
                self.criterion = torch.nn.MSELoss()

    def perturb(self, agent: BaseAgent, obs, rnn_states, masks, available_actions=None, target_action=None):
        targeted_attack = False
        obs = check(obs).to(**self.tpdv)
        if target_action is not None:
            target_action = check(target_action).to(**self.tpdv)
            targeted_attack = True
        else:
            action_dist, _ = agent.forward(obs, rnn_states, masks, available_actions)
            target_action = action_dist.mode

        # the input of target_action should have the same shape with action_dist.mode
        if self.action_type == "Discrete":
            if target_action.shape[-1] == 1:
                target_action = target_action.squeeze(-1)
            else:
                target_action = target_action.argmax(dim=-1)
            
        obs_adv = obs.detach().clone() + torch.clamp(torch.randn_like(obs) * self.epsilon, -self.epsilon, self.epsilon)
        for _ in range(self.num_iters):
            obs_adv.requires_grad_(True)
            action_dist, _ = agent.forward(obs_adv, rnn_states, masks, available_actions)
            if self.action_type == "Discrete":
                actor_out = action_dist.logits
            elif self.action_type == "Box":
                actor_out = action_dist.mean

            loss = self.criterion(actor_out, target_action.detach())
            grad = torch.autograd.grad(loss, obs_adv)[0]

            if targeted_attack:
                delta = torch.clamp(obs_adv - self.alpha * grad.sign() - obs, -self.epsilon, self.epsilon)
            else:
                delta = torch.clamp(obs_adv + self.alpha * grad.sign() - obs, -self.epsilon, self.epsilon)
            obs_adv = (obs + delta).detach()

        return obs_adv