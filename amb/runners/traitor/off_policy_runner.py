"""Base runner for off-policy algorithms."""
import torch
import numpy as np
from amb.data.episode_buffer import EpisodeBuffer
from amb.runners.traitor.base_runner import BaseRunner
from amb.utils.trans_utils import _t2n, gather, scatter
from amb.utils.env_utils import (
    get_shape_from_obs_space,
    get_shape_from_act_space,
    get_onehot_shape_from_act_space
)


class OffPolicyRunner(BaseRunner):
    """Base runner for off-policy algorithms."""

    def __init__(self, args, algo_args, env_args):
        """Initialize the traitor/OffPolicyRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        super(OffPolicyRunner, self).__init__(args, algo_args, env_args)
        self.warmup_steps = self.algo_args['train']['warmup_steps']

        self.current_timestep = 0  # total iteration
        self.last_log = -1
        self.last_eval = -1
        self.last_train = -1

        if self.algo_args['train']['use_render'] is False:  # train, not render
            scheme = {
                "obs": {"vshape": get_shape_from_obs_space(self.envs.observation_space[0]), "offset": 1, "extra": ["sample_next"]},
                "rnn_states_actor": {"vshape": (self.recurrent_n, self.rnn_hidden_size), "extra": ["rnn_state"]},
                "share_obs": {"vshape": get_shape_from_obs_space(self.envs.share_observation_space[0]), "offset": 1, "extra": ["sample_next"]},
                "rnn_states_critic": {"vshape": (self.recurrent_n, self.rnn_hidden_size), "extra": ["rnn_state"]},
                "actions": {"vshape": (get_shape_from_act_space(self.envs.action_space[0]),)},
                "actions_onehot": {"vshape": (get_onehot_shape_from_act_space(self.envs.action_space[0]),)},
                "adv_agent_ids": {"vshape": (1,), "offset": 0},
                "rewards": {"vshape": (1,)},
                "gammas": {"vshape": (1,), "init_value": 1},
                "dones_env": {"vshape": (1,)},
                "trunc_env": {"vshape": (1,)},
                "masks": {"vshape": (1,), "offset": 1, "init_value": 1},
                "active_masks": {"vshape": (1,), "offset": 1, "init_value": 1},
            }
            if self.action_type == "Discrete":
                scheme["available_actions"] = {"vshape": (self.envs.action_space[0].n,), "offset": 1, "init_value": 1, "extra": ["sample_next"]}
            self.buffer = EpisodeBuffer(algo_args["train"], self.algo_args["train"]["buffer_size"], scheme, num_agents=self.num_adv_agents)

        self.restore()

    def init_batch(self):
        obs, share_obs, available_actions = self.envs.reset()
        adv_agent_ids = np.stack([self.get_certain_adv_ids() for _ in range(self.n_rollout_threads)], axis=0)

        data = {
            "adv_agent_ids": adv_agent_ids.copy(),
            "obs": gather(obs, adv_agent_ids, axis=1).copy(),
            "share_obs": gather(share_obs, adv_agent_ids, axis=1).copy()
        }
        if "available_actions" in self.buffer.data:
            data["available_actions"] = gather(available_actions, adv_agent_ids, axis=1).copy()
        self.buffer.init_batch(data)
        return obs, share_obs, available_actions, adv_agent_ids

    def run(self):
        """Run the training (or rendering) pipeline."""
        if self.algo_args['train']['use_render'] is True:
            self.render()
            return
        update_num = int(  # update number per train
            self.algo_args['train']['update_per_train'] * self.algo_args['train']['train_interval']
        )
        self.logger.init()

        self.logger.episode_init(0) 
        self.eval()
        self.eval_adv()

        while self.current_timestep < self.algo_args['train']['num_env_steps']:
            self.logger.episode_init(self.current_timestep)  # logger callback at the beginning of each episode
            obs, share_obs, available_actions, adv_agent_ids = self.init_batch()
            self.algo.prep_rollout()

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.victim_recurrent_n, self.victim_rnn_hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            filled = np.ones((self.n_rollout_threads,), dtype=np.float32)

            adv_rnn_states = np.zeros((self.n_rollout_threads, self.num_adv_agents, self.recurrent_n, self.rnn_hidden_size), dtype=np.float32)
            adv_masks = np.ones((self.n_rollout_threads, self.num_adv_agents, 1), dtype=np.float32)
            
            for step in range(self.episode_length):
                adv_actions, adv_actions_onehot, adv_rnn_states = self.collect(
                    gather(obs, adv_agent_ids, 1), 
                    adv_rnn_states, 
                    adv_masks, 
                    gather(available_actions, adv_agent_ids, 1)
                    if available_actions[0] is not None else None)

                actions_collector = []
                for agent_id in range(self.num_agents):
                    actions, temp_rnn_state = self.victims[agent_id].perform(
                        obs[:, agent_id],
                        rnn_states[:, agent_id],
                        masks[:, agent_id],
                        available_actions[:, agent_id]
                        if available_actions[0] is not None else None,
                        deterministic=False
                    )
                    rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                    actions_collector.append(_t2n(actions))
                actions = np.stack(actions_collector, axis=1)

                actions = scatter(actions, adv_agent_ids, adv_actions, axis=1)

                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions, filled)
                data = {
                    "obs": gather(obs, adv_agent_ids, 1), "share_obs": gather(share_obs, adv_agent_ids, 1), 
                    "rewards": gather(rewards, adv_agent_ids, 1), "dones": dones, "infos": infos, 
                    "actions": adv_actions, "actions_onehot": adv_actions_onehot, "filled": filled,
                    "adv_agent_ids": adv_agent_ids
                }
                if "available_actions" in self.buffer.data:
                    data.update({"available_actions": gather(available_actions, adv_agent_ids, 1)})

                data["rewards"] = self.get_adv_rewards(data)

                self.logger.per_step(data)  # logger callback at each step
                self.insert(data, step)  # insert data into buffer

                # Here is the main difference from on-policy runner: 
                # we only sample episodic data, and the terminated episodes are thrown.
                filled = filled * (1 - np.all(dones, axis=1))
                if np.sum(filled) == 0:
                    break

            self.buffer.compute_nstep_rewards(self.n_rollout_threads)

            if self.current_timestep > self.warmup_steps:
                if self.current_timestep - self.last_train >= self.algo_args['train']['train_interval']:
                    self.last_train = self.current_timestep
                    self.algo.prep_training()
                    if self.algo_args['train']['use_linear_lr_decay']:  # linear decay of learning rate
                        self.algo.lr_decay(self.current_timestep, self.algo_args['train']['num_env_steps'])
                    for _ in range(update_num):
                        actor_train_infos, critic_train_info = self.algo.train(self.buffer)
                    
                # log information
                if self.current_timestep - self.last_log >= self.algo_args['train']['log_interval']:
                    self.last_log = self.current_timestep
                    self.logger.episode_log(actor_train_infos, critic_train_info, [self.buffer])

                # eval
                if self.current_timestep - self.last_eval >= self.algo_args['train']['eval_interval']:
                    self.last_eval = self.current_timestep
                    if self.algo_args['train']['use_eval']:
                        self.eval()
                        self.eval_adv()
                    self.save()
            
            self.current_timestep += self.buffer.get_timesteps(self.n_rollout_threads)
            self.buffer.move(self.n_rollout_threads)

    @torch.no_grad()
    def collect(self, obs, rnn_states, masks, available_actions):
        """Collect actions from actors considering warmup."""
        action_collector = []
        action_onehot_collector = []
        rnn_state_collector = []

        for agent_id in range(self.num_adv_agents):
            if self.current_timestep <= self.warmup_steps:
                action, action_onehot = self.agents[agent_id].sample(
                    obs[:, agent_id], available_actions[:, agent_id]
                    if "available_actions" in self.buffer.data else None)
                rnn_state_collector.append(rnn_states[:, agent_id])
            else:
                action, action_onehot, rnn_state = self.agents[agent_id].collect(
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
        data["dones_env"] = np.expand_dims(dones_env, 1).repeat(self.num_adv_agents, 1)
        data["trunc_env"] = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        for i in range(self.n_rollout_threads):
            for j in range(self.num_agents):
                if "bad_transition" in data["infos"][i][j] and data["infos"][i][j]["bad_transition"] == True:
                    data["trunc_env"][i, j] = 1
        data["trunc_env"] = gather(data["trunc_env"], data["adv_agent_ids"], 1)

        data["active_masks"] = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        data["active_masks"][data["dones"]==True] = 0
        data["active_masks"] = gather(data["active_masks"], data["adv_agent_ids"], 1)
        
        del data["infos"]
        del data["dones"]

        self.buffer.insert(data, step)


