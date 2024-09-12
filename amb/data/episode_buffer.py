import torch
import numpy as np
from copy import deepcopy

from amb.utils.trans_utils import _dimalign

class EpisodeBuffer:
    def __init__(self, args, buffer_size, scheme, num_agents=None):
        # scheme: vshape(required), dtype, init_value, offset
        self.scheme = scheme
        self.episode_length = args["episode_length"]
        self.buffer_size = buffer_size
        self.num_agents = num_agents

        if self.num_agents:
            for key in scheme:
                scheme[key]["vshape"] = (num_agents, *scheme[key]["vshape"])

        self.gamma = args.get("gamma", 0.99)
        self.gae_lambda = args.get("gae_lambda", 0.95)
        self.use_gae = args.get("use_gae", True)
        self.use_proper_time_limits = args.get("use_proper_time_limits", True)

        self.n_step = args.get("n_step", 1)
        self.gamma = args.get("gamma", 0.99)

        tmp_scheme = {}
        for key in self.scheme:
            extra = self.scheme[key].get("extra", [])
            if "sample_next" in extra:
                tmp_scheme["next_" + key] = deepcopy(self.scheme[key])
                tmp_scheme["next_" + key]["extra"].remove("sample_next")
        
        self.scheme.update(tmp_scheme)

        self.scheme["filled"] = {"vshape": (), "dtype": np.int32, "offset": 0, "init_value": 0, "extra": []}
        self.reset()

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data[key]
        else:
            return {k: self.data[k][key] for k in self.data}
        
    def reset(self):
        self.data = {}
        self.current_size = 0
        
        for key in self.scheme:
            vshape = self.scheme[key]["vshape"]
            dtype = self.scheme[key].get("dtype", np.float32)
            init_value = self.scheme[key].get("init_value", 0)
            offset = self.scheme[key].get("offset", 0)
            extra = self.scheme[key].get("extra", [])
            if "more_length" in extra:
                self.data[key] = np.ones((self.buffer_size, self.episode_length + abs(offset) + 1, *vshape), dtype=dtype) * init_value
            else:
                self.data[key] = np.ones((self.buffer_size, self.episode_length + abs(offset), *vshape), dtype=dtype) * init_value
            
    def insert(self, data, t):
        assert "filled" in data, "'filled' is needed to be inserted in episode buffer!"
        n = data["filled"].shape[0]
        for key in data:
            d = data[key]
            dtype = self.scheme[key].get("dtype", np.float32)
            offset = self.scheme[key].get("offset", 0)
            extra = self.scheme[key].get("extra", [])
            if isinstance(d, torch.Tensor):
                d = d.detach().cpu().numpy()
            if not isinstance(d, np.ndarray):
                d = np.array(d)
            d = d.astype(dtype)
            d = d.reshape(n, *self.scheme[key]["vshape"])

            if self.current_size + n > self.buffer_size:
                right = self.current_size + n - self.buffer_size
                left = n - right
                self.data[key][self.current_size:, t+offset] = d[:left]
                self.data[key][:right, t+offset] = d[left:]
                if "sample_next" in extra:
                    self.data["next_" + key][self.current_size:, max(t+offset-self.n_step, 0):t+offset] = d[:left, np.newaxis].copy()
                    self.data["next_" + key][:right, max(t+offset-self.n_step, 0)] = d[left:, np.newaxis].copy()
            else:
                self.data[key][self.current_size:self.current_size+n, t+offset] = d
                if "sample_next" in extra:
                    self.data["next_" + key][self.current_size:self.current_size+n, max(t+offset-self.n_step, 0):t+offset] = d[:, np.newaxis].copy()

    def after_update(self):
        self.data["filled"] = np.zeros((self.buffer_size, self.episode_length), dtype=np.int32)
        for key in self.data:
            offset = self.scheme[key].get("offset", 0)
            if offset > 0:
                for i in range(offset):
                    self.data[key][:, i] = self.data[key][:, i-offset].copy()

    def init_batch(self, data):
        for key in data:
            d = data[key]
            dtype = self.scheme[key].get("dtype", np.float32)
            if isinstance(d, torch.Tensor):
                d = d.detach().cpu().numpy()
            if not isinstance(d, np.ndarray):
                d = np.array(d)
            d = d.astype(dtype)
            d = d.reshape(d.shape[0], *self.scheme[key]["vshape"])
            n = d.shape[0]

            if self.current_size + n > self.buffer_size:
                right = self.current_size + n - self.buffer_size
                left = n - right
                self.data[key][self.current_size:, 0] = d[:left]
                self.data[key][:right, 0] = d[left:]
            else:
                self.data[key][self.current_size:self.current_size+n, 0] = d

    def get_timesteps(self, n):
        if self.current_size + n > self.buffer_size:
            right = self.current_size + n - self.buffer_size
            filled = np.concatenate([self.data["filled"][self.current_size:], self.data["filled"][:right]], axis=0)
        else:
            filled = self.data["filled"][self.current_size:self.current_size+n]

        return int(filled.sum())
    
    def move(self, n):
        # move to blank area
        self.current_size = (self.current_size + n) % self.buffer_size

        for key in self.data:
            init_value = self.scheme[key].get("init_value", 0)
            if self.current_size + n > self.buffer_size:
                right = self.current_size + n - self.buffer_size
                self.data[key][self.current_size:] = init_value
                self.data[key][:right] = init_value
            else:
                self.data[key][self.current_size:self.current_size+n] = init_value

    def compute_returns(self, next_values, value_normalizer=None):
        assert "rewards" in self.data
        assert "masks" in self.data   # RNN termination
        assert "returns" in self.data
        assert "value_preds" in self.data
        if self.use_proper_time_limits:  # consider the difference between truncation and termination
            assert "bad_masks" in self.data
            if self.use_gae:  # use GAE
                self.data["value_preds"][:, -1] = next_values
                gae = 0
                for step in reversed(range(self.data["rewards"].shape[1])):
                    if value_normalizer is not None:  # use PopArt
                        delta = (
                            self.data["rewards"][:, step]
                            + self.gamma
                            * value_normalizer.denormalize(self.data["value_preds"][:, step + 1])
                            * self.data["masks"][:, step + 1]
                            - value_normalizer.denormalize(self.data["value_preds"][:, step])
                        )
                        gae = delta + self.gamma * self.gae_lambda * self.data["masks"][:, step + 1] * gae
                        gae = self.data["bad_masks"][:, step + 1] * gae
                        self.data["returns"][:, step] = gae + value_normalizer.denormalize(
                            self.data["value_preds"][:, step]
                        )
                    else:  # do not use PopArt
                        delta = (
                            self.data["rewards"][:, step]
                            + self.gamma * self.data["value_preds"][:, step + 1] * self.data["masks"][:, step + 1]
                            - self.data["value_preds"][:, step]
                        )
                        gae = delta + self.gamma * self.gae_lambda * self.data["masks"][:, step + 1] * gae
                        gae = self.data["bad_masks"][:, step + 1] * gae
                        self.data["returns"][:, step] = gae + self.data["value_preds"][:, step]
            else:  # do not use GAE
                self.data["returns"][:, -1] = next_values
                for step in reversed(range(self.data["rewards"].shape[1])):
                    if value_normalizer is not None:  # use PopArt
                        self.data["returns"][:, step] = (
                            self.data["returns"][:, step + 1] * self.gamma * self.data["masks"][:, step + 1]
                            + self.data["rewards"][:, step]
                        ) * self.data["bad_masks"][:, step + 1] + (
                            1 - self.data["bad_masks"][:, step + 1]
                        ) * value_normalizer.denormalize(
                            self.data["value_preds"][:, step]
                        )
                    else:  # do not use PopArt
                        self.data["returns"][:, step] = (
                            self.data["returns"][:, step + 1] * self.gamma * self.data["masks"][:, step + 1]
                            + self.data["rewards"][:, step]) * self.data["bad_masks"][:, step + 1] + (
                            1 - self.data["bad_masks"][:, step + 1]) * self.data["value_preds"][:, step]
        else:  # do not consider the difference between truncation and termination, i.e. all done episodes are terminated
            if self.use_gae:  # use GAE
                self.data["value_preds"][:, -1] = next_values
                gae = 0
                for step in reversed(range(self.data["rewards"].shape[0])):
                    if value_normalizer is not None:  # use PopArt
                        delta = (
                            self.data["rewards"][:, step]
                            + self.gamma
                            * value_normalizer.denormalize(self.data["value_preds"][:, step + 1])
                            * self.data["masks"][:, step + 1]
                            - value_normalizer.denormalize(self.data["value_preds"][:, step])
                        )
                        gae = delta + self.gamma * self.gae_lambda * self.data["masks"][:, step + 1] * gae
                        self.data["returns"][:, step] = gae + value_normalizer.denormalize(
                            self.data["value_preds"][:, step]
                        )
                    else:  # do not use PopArt
                        delta = (
                            self.data["rewards"][:, step]
                            + self.gamma * self.data["value_preds"][:, step + 1] * self.data["masks"][:, step + 1]
                            - self.data["value_preds"][:, step]
                        )
                        gae = delta + self.gamma * self.gae_lambda * self.data["masks"][:, step + 1] * gae
                        self.data["returns"][:, step] = gae + self.data["value_preds"][:, step]
            else:  # do not use GAE
                self.data["returns"][:, -1] = next_values
                for step in reversed(range(self.data["rewards"].shape[0])):
                    self.data["returns"][:, step] = (
                        self.data["returns"][:, step + 1] * self.gamma * self.data["masks"][:, step + 1]
                        + self.data["rewards"][:, step]
                    )

    def compute_nstep_rewards(self, n):
        assert "rewards" in self.data
        assert "gammas" in self.data
        if self.current_size + n > self.buffer_size:
            right = self.current_size + n - self.buffer_size
            rewards = np.concatenate([self.data["rewards"][self.current_size:], self.data["rewards"][:right]], axis=0)
            filled = np.concatenate([self.data["filled"][self.current_size:], self.data["filled"][:right]], axis=0)
        else:
            rewards = self.data["rewards"][self.current_size:self.current_size+n]
            filled = self.data["filled"][self.current_size:self.current_size+n]

        filled = _dimalign(filled, rewards)
        gammas = np.ones_like(rewards)

        length = self.data["rewards"].shape[1] - 1
        for step in reversed(range(length)):
            rewards[:, step] += rewards[:, step + 1] * self.gamma * filled[:, step + 1]
            gammas[:, step] = gammas[:, step + 1] + filled[:, step + 1]

        for step in range(length + 1 - self.n_step):
            rewards[:, step] -= rewards[:, step + self.n_step] * filled[:, step + self.n_step] * (self.gamma ** self.n_step)
            gammas[:, step] -= gammas[:,  step + self.n_step] * filled[:,  step + self.n_step]

        if self.current_size + n > self.buffer_size:
            right = self.current_size + n - self.buffer_size
            left = n - right
            self.data["rewards"][self.current_size:] = rewards[:left]
            self.data["rewards"][:right] = rewards[left:]
            self.data["gammas"][self.current_size:] = gammas[:left]
            self.data["gammas"][:right] = gammas[left:]
        else:
            self.data["rewards"][self.current_size:self.current_size+n] = rewards
            self.data["gammas"][self.current_size:self.current_size+n] = gammas

    def step_generator(self, num_mini_batch, mini_batch_size=None):
        total_timesteps = np.sum(self.data["filled"])
        if mini_batch_size is None:
            assert total_timesteps >= num_mini_batch
            mini_batch_size = total_timesteps // num_mini_batch

        index = np.where(self.data["filled"] == 1)
        rand = torch.randperm(total_timesteps).numpy()
        sampler = [rand[i*mini_batch_size: (i+1)*mini_batch_size] for i in range(num_mini_batch)]
        
        for indices in sampler:
            sampled_data = {}
            for key in self.data:
                index_indices = (index[0][indices], index[1][indices])
                sampled_data[key] = self.data[key][index_indices]

            yield sampled_data

    def episode_generator(self, num_mini_batch, mini_batch_size):
        total_episodes = np.sum(self.data["filled"][:, 0])
        rand = torch.randperm(total_episodes).numpy()
        sampler = [rand[i*mini_batch_size: (i+1)*mini_batch_size] for i in range(num_mini_batch)]

        for indices in sampler:
            max_t_filled = int(self.data["filled"][indices].sum(axis=1).max())
            sampled_data = {}
            for key in self.data:
                extra = self.scheme[key].get("extra", [])
                if "rnn_state" in extra:
                    d = self.data[key][indices, 0]
                    # [B, V]
                else:
                    d = self.data[key][indices, :max_t_filled]
                    # [B, T, V] -> [T, B, V] -> [T*B, V]
                    d = np.swapaxes(d, 0, 1)
                    d = d.reshape(-1, *d.shape[2:])
                sampled_data[key] = d

            yield sampled_data
        
    def chunk_generator(self, num_mini_batch, chunk_length):
        # only used for fully fiiled buffer
        assert chunk_length > 1
        assert self.episode_length % chunk_length == 0
        assert np.sum(self.data["filled"]) == self.buffer_size * self.episode_length
        data_chunks = self.buffer_size * self.episode_length // chunk_length
        mini_batch_size = data_chunks // num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i*mini_batch_size: (i+1)*mini_batch_size] for i in range(num_mini_batch)]

        for indices in sampler:
            sampled_data = {}
            for key in self.data:
                extra = self.scheme[key].get("extra", [])
                d = self.data[key][:, :self.episode_length]
                # [N, T, V] -> [N*T, V]
                d = d.reshape(-1, *d.shape[2:])
                data_batch = []
                if "rnn_state" in extra:
                    for index in indices:
                        ind = index * chunk_length
                        data_batch.append(d[ind])
                    # [[V] * B] -> [B, V]
                    data_batch = np.stack(data_batch, axis=0)
                else:
                    for index in indices:
                        ind = index * chunk_length
                        data_batch.append(d[ind:ind+chunk_length])
                    # [[L, V] * B] -> [L, B, V] -> [L*B, V]
                    data_batch = np.stack(data_batch, axis=1)
                    data_batch = data_batch.reshape(-1, *data_batch.shape[2:])
                sampled_data[key] = data_batch

            yield sampled_data