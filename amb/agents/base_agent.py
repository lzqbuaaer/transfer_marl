class BaseAgent:
    def forward(self, obs, rnn_states, masks, available_actions=None):
        raise NotImplementedError
    
    def sample(self, obs, available_actions=None):
        raise NotImplementedError
    
    def perform(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        raise NotImplementedError
    
    def collect(self, obs, rnn_states, masks, available_actions=None, t=0):
        raise NotImplementedError
    
    def restore(self, path):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def prep_training(self):
        raise NotImplementedError

    def prep_rollout(self):
        raise NotImplementedError