import torch
from amb.models.actor.q_actor import QActor
from amb.models.base.distributions import OneHotMultinomial
from amb.utils.env_utils import check


class COMAActor(QActor):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super().__init__(args, obs_space, action_space, device)

    def forward(self, obs, rnn_states, masks, available_actions=None, t=0, test_mode=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(self.cnn(obs))
        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        q_values = self.out(actor_features)

        if self.action_type == "Box":
            raise f"Box action space is not supported for {self.__class__.__name__}"
        
        if self.action_type == "Discrete":
            action_dist = OneHotMultinomial(q_values, t, available_actions, self.epsilon_start, self.epsilon_finish, self.epsilon_anneal_time)

        return action_dist, rnn_states
