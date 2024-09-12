"""Base runner for off-policy algorithms."""
import torch
import numpy as np
from amb.data.episode_buffer import EpisodeBuffer
from amb.runners.dual.base_runner import BaseRunner
from amb.utils.trans_utils import _t2n
from amb.utils.env_utils import (
    get_shape_from_obs_space,
    get_shape_from_act_space,
    get_onehot_shape_from_act_space
)


class OffPolicyRunner(BaseRunner):
    """Base runner for off-policy algorithms."""

    def __init__(self, args, algo_args, env_args):
        """Initialize the dual/OffPolicyRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        super(OffPolicyRunner, self).__init__(args, algo_args, env_args)
        self.warmup_steps = self.algo_args["angel"]['warmup_steps']

        self.current_timestep = 0  # total iteration
        self.last_log = -1
        self.last_eval = -1
        self.last_train = -1

        if self.algo_args["angel"]['use_render'] is False:  # train, not render
            scheme = {
                "obs": {"vshape": get_shape_from_obs_space(self.envs.observation_space[0][0]), "offset": 1, "extra": ["sample_next"]},
                "rnn_states_actor": {"vshape": (self.angel_recurrent_n, self.angel_rnn_hidden_size), "extra": ["rnn_state"]},
                "share_obs": {"vshape": get_shape_from_obs_space(self.envs.share_observation_space[0][0]), "offset": 1, "extra": ["sample_next"]},
                "rnn_states_critic": {"vshape": (self.angel_recurrent_n, self.angel_rnn_hidden_size), "extra": ["rnn_state"]},
                "actions": {"vshape": (get_shape_from_act_space(self.envs.action_space[0][0]),)},
                "actions_onehot": {"vshape": (get_onehot_shape_from_act_space(self.envs.action_space[0][0]),)},
                "rewards": {"vshape": (1,)},
                "gammas": {"vshape": (1,), "init_value": 1},
                "dones_env": {"vshape": (1,)},
                "trunc_env": {"vshape": (1,)},
                "masks": {"vshape": (1,), "offset": 1, "init_value": 1},
                "active_masks": {"vshape": (1,), "offset": 1, "init_value": 1},
            }
            if self.action_type == "Discrete":
                scheme["available_actions"] = {"vshape": (self.envs.action_space[0][0].n,), "offset": 1, "init_value": 1, "extra": ["sample_next"]}
            self.buffer = EpisodeBuffer(algo_args["angel"], self.algo_args["angel"]["buffer_size"], scheme, num_agents=self.num_angels)

        self.restore()

    def init_batch(self):
        obs, share_obs, available_actions = self.envs.reset()
        data = {
            "obs": obs[0].copy(),
            "share_obs": share_obs[0].copy()
        }
        if "available_actions" in self.buffer.data:
            data["available_actions"] = available_actions[0].copy()
        self.buffer.init_batch(data)
        return obs, share_obs, available_actions

    def run(self):
        """Run the training (or rendering) pipeline."""
        if self.algo_args["angel"]['use_render'] is True:
            self.render()
            return
        update_num = int(  # update number per train
            self.algo_args["angel"]['update_per_train'] * self.algo_args["angel"]['train_interval']
        )
        self.logger.init()

        self.logger.episode_init(0) 
        self.eval()

        while self.current_timestep < self.algo_args["angel"]['num_env_steps']:
            self.logger.episode_init(self.current_timestep)  # logger callback at the beginning of each episode
            obs, share_obs, available_actions = self.init_batch()
            self.algo.prep_rollout()

            angel_rnn_states = np.zeros((self.n_rollout_threads, self.num_angels, self.angel_recurrent_n, self.angel_rnn_hidden_size), dtype=np.float32)
            angel_masks = np.ones((self.n_rollout_threads, self.num_angels, 1), dtype=np.float32)
            filled = np.ones((self.n_rollout_threads,), dtype=np.float32)

            demon_rnn_states = np.zeros((self.n_rollout_threads, self.num_demons, self.demon_recurrent_n, self.demon_rnn_hidden_size), dtype=np.float32)
            demon_masks = np.ones((self.n_rollout_threads, self.num_demons, 1), dtype=np.float32)

            for step in range(self.episode_length):
                angel_actions, actions_onehot, angel_rnn_states = self.collect(obs[0], angel_rnn_states, angel_masks, available_actions[0])
                
                demon_actions_collector = []
                for agent_id in range(self.num_demons):
                    demon_actions, temp_rnn_state = self.demons[agent_id].perform(
                        obs[1][:, agent_id],
                        demon_rnn_states[:, agent_id],
                        demon_masks[:, agent_id],
                        available_actions[1][:, agent_id]
                        if available_actions[1][0] is not None else None,
                        deterministic=False
                    )
                    demon_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                    demon_actions_collector.append(_t2n(demon_actions))
                demon_actions = np.stack(demon_actions_collector, axis=1)
                
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step((angel_actions, demon_actions), filled)
                data = {
                    "obs": obs[0], "share_obs": share_obs[0], "rewards": rewards[0], "dones": dones[0],
                    "infos": infos[0], "actions": angel_actions, "actions_onehot": actions_onehot, "filled": filled
                }
                if "available_actions" in self.buffer.data:
                    data.update({"available_actions": available_actions[0]})

                self.logger.per_step(data)  # logger callback at each step
                self.insert(data, step)  # insert data into buffer

                # Here is the main difference from on-policy runner: 
                # we only sample episodic data, and the terminated episodes are thrown.
                filled = filled * (1 - np.all(dones[0], axis=1))
                if np.sum(filled) == 0:
                    break

            self.buffer.compute_nstep_rewards(self.n_rollout_threads)

            if self.current_timestep > self.warmup_steps:
                if self.current_timestep - self.last_train >= self.algo_args["angel"]['train_interval']:
                    self.last_train = self.current_timestep
                    self.algo.prep_training()
                    if self.algo_args["angel"]['use_linear_lr_decay']:  # linear decay of learning rate
                        self.algo.lr_decay(self.current_timestep, self.algo_args["angel"]['num_env_steps'])
                    for _ in range(update_num):
                        # self.algo.current_train_step = self.current_timestep
                        actor_train_infos, critic_train_info = self.algo.train(self.buffer)
                    
                # log information
                if self.current_timestep - self.last_log >= self.algo_args["angel"]['log_interval']:
                    self.last_log = self.current_timestep
                    self.logger.episode_log(actor_train_infos, critic_train_info, [self.buffer])

                # eval
                if self.current_timestep - self.last_eval >= self.algo_args["angel"]['eval_interval']:
                    self.last_eval = self.current_timestep
                    if self.algo_args["angel"]['use_eval']:
                        self.eval()
                    self.save()
            
            self.current_timestep += self.buffer.get_timesteps(self.n_rollout_threads)
            self.buffer.move(self.n_rollout_threads)

    @torch.no_grad()
    def collect(self, obs, rnn_states, masks, available_actions):
        """Collect actions from actors considering warmup."""
        action_collector = []
        action_onehot_collector = []
        rnn_state_collector = []

        for agent_id in range(self.num_angels):
            if self.current_timestep <= self.warmup_steps:
                action, action_onehot = self.angels[agent_id].sample(
                    obs[:, agent_id], available_actions[:, agent_id]
                    if "available_actions" in self.buffer.data else None)
                rnn_state_collector.append(rnn_states[:, agent_id])
            else:
                action, action_onehot, rnn_state = self.angels[agent_id].collect(
                    obs[:, agent_id],
                    rnn_states[:, agent_id],
                    masks[:, agent_id],
                    available_actions[:, agent_id]
                    if "available_actions" in self.buffer.data else None,
                    t=self.current_timestep)
                rnn_state_collector.append(_t2n(rnn_state))

            action_collector.append(_t2n(action))
            action_onehot_collector.append(_t2n(action_onehot))

        actions = np.stack(action_collector, axis=1)
        actions_onehot = np.stack(action_onehot_collector, axis=1)
        rnn_states = np.stack(rnn_state_collector, axis=1)

        return actions, actions_onehot, rnn_states
    
    def insert(self, data, step):
        """Insert data into buffer.
           obs, share_obs, rewards, dones, infos, available_actions, 
           actions, actions_onehot, filled
           rnn_states and masks are not inserted, just used as their initial values.
        """
        dones_env = np.all(data["dones"], axis=1)
        data["dones_env"] = np.expand_dims(dones_env, 1).repeat(self.num_angels, 1)
        data["trunc_env"] = np.zeros((self.n_rollout_threads, self.num_angels, 1), dtype=np.float32)
        for i in range(self.n_rollout_threads):
            for j in range(self.num_angels):
                if "bad_transition" in data["infos"][i][j] and data["infos"][i][j]["bad_transition"] == True:
                    data["trunc_env"][i, j] = 1

        data["active_masks"] = np.ones((self.n_rollout_threads, self.num_angels, 1), dtype=np.float32)
        data["active_masks"][data["dones"]==True] = 0

        del data["infos"]
        del data["dones"]

        self.buffer.insert(data, step)


