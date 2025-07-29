import os
import torch
import numpy as np
from amb.data.episode_buffer import EpisodeBuffer
from amb.runners.dual.base_runner import BaseRunner
from amb.runners.dual.on_policy_runner import OnPolicyRunner
from amb.utils.popart import PopArt
from amb.utils.trans_utils import _t2n
from amb.utils.env_utils import (
    get_shape_from_obs_space,
    get_shape_from_act_space,
)


class OnPolicySelfPlayRunner(OnPolicyRunner):
    def __init__(self, args, algo_args, env_args):
        """Initialize the dual/OnPolicyRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        super(OnPolicySelfPlayRunner, self).__init__(args, algo_args, env_args)

        if self.algo_args["angel"]['use_render'] is False:  # train, not render
            self.demon_buffers = []
            for agent_id in range(self.num_demons):
                scheme = {
                    "obs": {"vshape": get_shape_from_obs_space(self.envs.observation_space[1][agent_id]), "offset": 1},
                    "rnn_states_actor": {"vshape": (self.demon_recurrent_n, self.demon_rnn_hidden_size), "offset": 1, "extra": ["rnn_state"]},
                    "share_obs": {"vshape": get_shape_from_obs_space(self.envs.share_observation_space[1][0]), "offset": 1},
                    "rnn_states_critic": {"vshape": (self.demon_recurrent_n, self.demon_rnn_hidden_size), "offset": 1, "extra": ["rnn_state"]},
                    "actions": {"vshape": (get_shape_from_act_space(self.envs.action_space[1][agent_id]),), "offset": 0},
                    "action_log_probs": {"vshape": (get_shape_from_act_space(self.envs.action_space[1][agent_id]),), "offset": 0},
                    "value_preds": {"vshape": (1,), "offset": 0, "extra": ["more_length"]},
                    "rewards": {"vshape": (1,), "offset": 0},
                    "returns": {"vshape": (1,), "offset": 0, "extra": ["more_length"]},
                    "advantages": {"vshape": (1,), "offset": 0},
                    "masks": {"vshape": (1,), "offset": 1, "init_value": 1},
                    "active_masks": {"vshape": (1,), "offset": 1, "init_value": 1},
                    "bad_masks": {"vshape": (1,), "offset": 1, "init_value": 1},
                }
                if self.demon_env_belief:
                    self.demon_env_belief_dim = self.algo_args["demon"]["env_belief_dim"]
                    scheme["rnn_states_belief"] = {"vshape": (self.demon_recurrent_n, self.demon_rnn_hidden_size), "offset": 1, "extra": ["rnn_state", "sample_next"]}
                    scheme["belief"] = {"vshape": (self.demon_env_belief_dim,), "offset": 0}
                    scheme["obs"]["extra"] = ["sample_next"]
                    scheme["masks"]["extra"] = ["sample_next"]
                if self.demon_actor_divide_conquer:
                    obs_shape = self.envs.observation_space[1][0] if isinstance(self.envs.observation_space[1][0], list) else self.envs.observation_space[1][0].shape
                    if len(obs_shape) >= 3:
                        scheme["chosens"] = {"vshape": (obs_shape[0] * obs_shape[1] - 1,), "offset": 0}
                    else:
                        scheme["chosens"] = {"vshape": (self.num_angels + self.num_demons - 1,), "offset": 0}
                if self.demon_actor_use_dt2gs:
                    scheme["previous_skills"] = {"vshape": (self.demon_actor_skills_num,), "offset": 1}
                if self.action_type == "Discrete":
                    scheme["available_actions"] = {"vshape": (self.envs.action_space[1][agent_id].n,), "offset": 1, "init_value": 1}
                algo_args["demon"]["episode_length"] = algo_args["angel"]["episode_length"]
                self.demon_buffers.append(EpisodeBuffer(algo_args["demon"], self.n_rollout_threads, scheme))

            if self.algo_args["demon"]["use_popart"] is True:
                self.demon_value_normalizer = PopArt(1, device=self.device)
            else:
                self.demon_value_normalizer = None

        self.restore()

    def init_batch(self):
        """initialize the replay buffer."""
        obs, share_obs, available_actions = self.envs.reset()
        for agent_id in range(self.num_angels):
            data = {
                "obs": obs[0][:, agent_id].copy(),
                "share_obs": share_obs[0][:, agent_id].copy()
            }
            if "available_actions" in self.buffers[agent_id].data:
                data["available_actions"] = available_actions[0][:, agent_id].copy()
            self.buffers[agent_id].init_batch(data)
            
        for agent_id in range(self.num_demons):
            data = {
                "obs": obs[1][:, agent_id].copy(),
                "share_obs": share_obs[1][:, agent_id].copy()
            }
            if "available_actions" in self.demon_buffers[agent_id].data:
                data["available_actions"] = available_actions[1][:, agent_id].copy()
            self.demon_buffers[agent_id].init_batch(data)
        return obs, share_obs, available_actions

    def run(self):
        """Run the training (or rendering) pipeline."""
        if self.algo_args["angel"].get('matter_transfer_test', False) or \
            self.algo_args["demon"].get('matter_transfer_test', False):
            if self.algo_args["angel"]['use_render'] is False:
                self.logger.init()
                self.logger.episode_init(0)
            print("Searching for the proper factors in transfer test of MATTER...")
            self.eval(few_shot_learning_mode=True)
            
        if "eval_only" in self.algo_args['angel'] and self.algo_args['angel']['eval_only']:
            print("[[EVAL MODE]]")
            self.logger.init()  # logger callback at the beginning of training
            self.logger.episode_init(0)
            for _ in range(self.algo_args['angel']['eval_times']):
                self.eval()
            return

        if self.algo_args["angel"]['use_render'] is True:
            self.render()
            return
        print("start running")
        self.logger.init()  # logger callback at the beginning of training

        self.logger.episode_init(0) 
        self.eval()

        obs, share_obs, available_actions = self.init_batch()
        if self.env_belief:
            self.bayesian_update = np.zeros((self.n_rollout_threads), dtype=bool)
        if self.demon_env_belief:
            self.demon_bayesian_update = np.zeros((self.n_rollout_threads), dtype=bool)

        episodes = int(self.algo_args["angel"]['num_env_steps']) // self.algo_args["angel"]['episode_length'] // self.algo_args["angel"]['n_rollout_threads']
        
        for episode in range(1, episodes + 1):
            if self.algo_args["angel"]['use_linear_lr_decay']:  # linear decay of learning rate
                self.algo.lr_decay(episode, episodes)
            if self.algo_args["angel"]['use_linear_lr_decay']:  # linear decay of learning rate
                self.demon_algo.lr_decay(episode, episodes)

            self.logger.episode_init(episode * self.algo_args["angel"]['episode_length'] * self.algo_args["angel"]['n_rollout_threads'])  # logger callback at the beginning of each episode

            self.algo.prep_rollout()

            for step in range(self.algo_args["angel"]['episode_length']):
                # Sample actions from actors and values from critics
                if self.env_belief:
                    values, angel_actions, action_log_probs, angel_rnn_states, rnn_states_critic, beliefs, rnn_states_belief = self.collect(step)
                else:
                    values, angel_actions, action_log_probs, angel_rnn_states, rnn_states_critic = self.collect(step)
                
                if self.actor_use_dt2gs:
                    angel_actions, angel_skills = angel_actions
                if self.actor_divide_conquer:
                    angel_actions, angel_chosens = angel_actions
                    
                if self.demon_env_belief:
                    demon_values, demon_actions, demon_action_log_probs, demon_rnn_states, demon_rnn_states_critic, demon_beliefs, demon_rnn_states_belief = self.collect_demon(step)
                else:
                    demon_values, demon_actions, demon_action_log_probs, demon_rnn_states, demon_rnn_states_critic = self.collect_demon(step)
                
                if self.demon_actor_use_dt2gs:
                    demon_actions, demon_skills = demon_actions
                if self.demon_actor_divide_conquer:
                    demon_actions, demon_chosens = demon_actions
                    
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step((angel_actions, demon_actions))
                if self.env_belief:
                    self.bayesian_update[:] = True
                if self.demon_env_belief:
                    self.demon_bayesian_update[:] = True

                assert self.num_angels == rewards[0].shape[1]
                assert self.num_demons == rewards[1].shape[1]
                assert rewards[0].shape[0] == rewards[1].shape[0]
                for process_id in range(rewards[1].shape[0]):
                    rewards[1][process_id, :, :] = -np.mean(rewards[0][process_id])
                
                filled = np.ones((self.n_rollout_threads, self.num_angels), dtype=np.float32)
                demon_filled = np.ones((self.n_rollout_threads, self.num_demons), dtype=np.float32)

                data = {
                    "obs": obs[0], "share_obs": share_obs[0], "rewards": rewards[0], "dones": dones[0],
                    "infos": infos[0], "value_preds": values, "actions": angel_actions, "action_log_probs": action_log_probs,
                    "rnn_states_actor": angel_rnn_states, "rnn_states_critic": rnn_states_critic, "filled": filled
                }
                if self.actor_divide_conquer:
                    data.update({"chosens": angel_chosens})
                if self.actor_use_dt2gs:
                    data.update({"previous_skills": angel_skills})
                if self.env_belief:
                    data.update({"belief": beliefs, "rnn_states_belief": rnn_states_belief})
                if "available_actions" in self.buffers[0].data:
                    data.update({"available_actions": available_actions[0]})
                
                self.logger.per_step(data)  # logger callback at each step
                self.insert(data, step)  # insert data into buffer
                
                demon_data = {
                    "obs": obs[1], "share_obs": share_obs[1], "rewards": rewards[1], "dones": dones[1],
                    "infos": infos[1], "value_preds": demon_values, "actions": demon_actions, "action_log_probs": demon_action_log_probs,
                    "rnn_states_actor": demon_rnn_states, "rnn_states_critic": demon_rnn_states_critic, "filled": demon_filled
                }
                if self.demon_actor_divide_conquer:
                    demon_data.update({"chosens": demon_chosens})
                if self.demon_actor_use_dt2gs:
                    demon_data.update({"previous_skills": demon_skills})
                if self.demon_env_belief:
                    demon_data.update({"belief": demon_beliefs, "rnn_states_belief": demon_rnn_states_belief})
                if "available_actions" in self.demon_buffers[0].data:
                    demon_data.update({"available_actions": available_actions[1]})
                
                self.insert_demon(demon_data, step)  # insert data into buffer

            # compute return and update network
            value_collector = []
            for agent_id in range(self.num_angels):
                value, _ = self.critic(
                    self.buffers[agent_id].data["share_obs"][:, step],
                    self.buffers[agent_id].data["rnn_states_critic"][:, step],
                    self.buffers[agent_id].data["masks"][:, step],
                )
                value_collector.append(_t2n(value))
            next_values = np.stack(value_collector, axis=1)

            self.algo.prep_training()
            actor_train_infos, critic_train_info = self.train(next_values)
            
            demon_value_collector = []
            for agent_id in range(self.num_demons):
                demon_value, _ = self.demon_critic(
                    self.demon_buffers[agent_id].data["share_obs"][:, step],
                    self.demon_buffers[agent_id].data["rnn_states_critic"][:, step],
                    self.demon_buffers[agent_id].data["masks"][:, step],
                )
                demon_value_collector.append(_t2n(demon_value))
            demon_next_values = np.stack(demon_value_collector, axis=1)

            self.demon_algo.prep_training()
            demon_actor_train_infos, demon_critic_train_info = self.train_demon(demon_next_values)

            # log information
            if episode % self.algo_args["angel"]['log_interval'] == 0:
                self.logger.episode_log(actor_train_infos, critic_train_info, self.buffers, 
                                        demon_actor_train_infos, demon_critic_train_info, self.demon_buffers)

            # eval
            if episode % self.algo_args["angel"]['eval_interval'] == 0:
                if self.algo_args["angel"]['use_eval']:
                    self.eval()
                self.save(episode // self.algo_args["angel"]['eval_interval'])

            for buffer in self.buffers:
                buffer.after_update()
            for buffer in self.demon_buffers:
                buffer.after_update()

    @torch.no_grad()
    def collect_demon(self, step):
        """Collect actions and values from actors and critics."""
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        value_collector = []
        rnn_state_critic_collector = []
        if self.demon_env_belief:
            belief_collector = []
            rnn_state_belief_collector = []
        if self.demon_actor_divide_conquer:
            chosen_collector = []
        if self.demon_actor_use_dt2gs:
            skills_collector = []

        for agent_id in range(self.num_demons):
            if self.demon_env_belief:
                belief, rnn_state_belief = self.demons[agent_id].forward_belief(
                    self.demon_buffers[agent_id].data["obs"][:, step],
                    self.demon_buffers[agent_id].data["rewards"][:, step - 1],
                    self.demon_buffers[agent_id].data["obs"][:, step - 1],
                    self.demon_buffers[agent_id].data["belief"][:, step - 1],
                    self.demon_buffers[agent_id].data["rnn_states_belief"][:, step],
                    self.demon_buffers[agent_id].data["masks"][:, step],
                )
                belief_np = _t2n(belief)
                belief_np[self.demon_bayesian_update == False, :] = self.demon_env_prior
                belief_collector.append(belief_np)
                rnn_state_belief_np = self.demon_buffers[agent_id].data["rnn_states_belief"][:, step]
                rnn_state_belief_np[self.demon_bayesian_update == True] = _t2n(rnn_state_belief)[self.demon_bayesian_update == True]
                rnn_state_belief_collector.append(rnn_state_belief_np)
            else:
                belief_np = None

            action, action_log_prob, rnn_state = self.demons[agent_id].collect(
                self.demon_buffers[agent_id].data["obs"][:, step],
                self.demon_buffers[agent_id].data["rnn_states_actor"][:, step],
                self.demon_buffers[agent_id].data["masks"][:, step],
                self.demon_buffers[agent_id].data["available_actions"][:, step]
                if "available_actions" in self.demon_buffers[agent_id].data else None,
                env_belief = self.demon_env_belief_ground_truth[:, agent_id] if (self.demon_env_belief and self.demon_env_belief_matter) else belief_np,
                previous_skills = self.demon_buffers[agent_id].data["previous_skills"][:, step]
                if self.demon_actor_use_dt2gs else None
            )
            if self.demon_actor_use_dt2gs:
                action, skill = action
                skills_collector.append(_t2n(skill))
            if self.demon_actor_divide_conquer:
                action, chosen = action
                chosen_collector.append(_t2n(chosen))
            value, rnn_state_critic = self.demon_critic(
                self.demon_buffers[agent_id].data["share_obs"][:, step],
                self.demon_buffers[agent_id].data["rnn_states_critic"][:, step],
                self.demon_buffers[agent_id].data["masks"][:, step],
            )
            
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            value_collector.append(_t2n(value))
            rnn_state_critic_collector.append(_t2n(rnn_state_critic))

        actions = np.stack(action_collector, axis=1)
        if self.demon_actor_divide_conquer:
            chosens = np.stack(chosen_collector, axis=1)
            actions = (actions, chosens)
        if self.demon_actor_use_dt2gs:
            skills = np.stack(skills_collector, axis=1)
            actions = (actions, skills)
        action_log_probs = np.stack(action_log_prob_collector, axis=1)
        rnn_states = np.stack(rnn_state_collector, axis=1)
        values = np.stack(value_collector, axis=1)
        rnn_states_critic = np.stack(rnn_state_critic_collector, axis=1)
        if self.demon_env_belief:
            beliefs = np.stack(belief_collector, axis=1)
            rnn_states_belief = np.stack(rnn_state_belief_collector, axis=1)
            return values, actions, action_log_probs, rnn_states, rnn_states_critic, beliefs, rnn_states_belief
        else:
            return values, actions, action_log_probs, rnn_states, rnn_states_critic
    
    def insert_demon(self, data, step):
        """Insert data into buffer.
           obs, share_obs, rewards, dones, infos, available_actions, values, 
           actions, action_log_probs, rnn_states_actor, rnn_states_critic
        """
        dones_env = np.all(data["dones"], axis=1)
        data["rnn_states_actor"][dones_env==True] = 0
        data["rnn_states_critic"][dones_env==True] = 0
        if self.demon_env_belief:
            self.demon_bayesian_update[dones_env == True] = False
            data["rnn_states_belief"][dones_env==True] = 0
        if self.demon_actor_use_dt2gs:
            data["previous_skills"][dones_env==True] = 0

        data["masks"] = np.ones((self.n_rollout_threads, self.num_demons, 1), dtype=np.float32)
        data["masks"][dones_env==True] = 0

        data["active_masks"] = np.ones((self.n_rollout_threads, self.num_demons, 1), dtype=np.float32)
        data["active_masks"][data["dones"]==True] = 0
        data["active_masks"][dones_env==True] = 1

        data["bad_masks"] = np.ones((self.n_rollout_threads, self.num_demons, 1), dtype=np.float32)
        for i in range(self.n_rollout_threads):
            for j in range(self.num_demons):
                if "bad_transition" in data["infos"][i][j] and data["infos"][i][j]["bad_transition"] == True:
                    data["bad_masks"][i, j] = 0

        del data["infos"]
        del data["dones"]

        # print({k: data[k].shape for k in data})
        for agent_id in range(self.num_demons):
            self.demon_buffers[agent_id].insert({k: data[k][:, agent_id] for k in data}, step)

    def train_demon(self, next_values):
        """Training procedure for MAPPO."""
        advantages = []
        for agent_id in range(self.num_demons):
            self.demon_buffers[agent_id].compute_returns(next_values[:, agent_id], self.demon_value_normalizer)
            if self.value_normalizer is not None:
                advantage = self.demon_buffers[agent_id].data["returns"][:, :-1] - self.demon_value_normalizer.denormalize(self.demon_buffers[agent_id].data["value_preds"])[:, :-1]
            else:
                advantage = self.demon_buffers[agent_id].data["returns"][:, :-1] - self.demon_buffers[agent_id].data["value_preds"][:, :-1]
            advantages.append(advantage)
        advantages = np.stack(advantages, axis=2)

        active_masks_collector = [self.demon_buffers[i].data["active_masks"] for i in range(self.num_demons)]
        active_masks_array = np.stack(active_masks_collector, axis=2)
        advantages_copy = advantages.copy()
        advantages_copy[active_masks_array[:, :-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for agent_id in range(self.num_demons):
            self.demon_buffers[agent_id]["advantages"][:] = advantages[:, :, agent_id].copy()

        # update belief
        belief_train_infos = []
        if self.demon_env_belief:
            if self.demon_share_param:
                belief_train_info = self.demon_algo.share_param_train_belief(self.demon_buffers)
                for _ in range(self.num_demons):
                    belief_train_infos.append(belief_train_info)
            else:
                for agent_id in range(self.num_demons):
                    belief_train_info = self.demon_algo.train_belief(self.demon_buffers[agent_id], agent_id)
                    belief_train_infos.append(belief_train_info)

        # update actors
        actor_train_infos = []
        if self.demon_share_param:
            actor_train_info = self.demon_algo.share_param_train_actor(self.demon_buffers)
            for _ in range(self.num_demons):
                actor_train_infos.append(actor_train_info)
        else:
            for agent_id in range(self.num_demons):
                actor_train_info = self.demon_algo.train_actor(self.demon_buffers[agent_id], agent_id)
                actor_train_infos.append(actor_train_info)
        if self.demon_env_belief:
            for agent_id in range(self.num_demons):
                actor_train_infos[agent_id].update(belief_train_infos[agent_id])

        # update critic
        critic_train_info = self.demon_algo.train_critic(self.demon_buffers, self.demon_value_normalizer)

        return actor_train_infos, critic_train_info

    def save(self, time_step=None):
        """Save model parameters."""
        if time_step is not None:
            save_dir = os.path.join(self.save_dir, str(time_step))
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = self.save_dir
        super().save(time_step=time_step)
        self.demon_algo.save(os.path.join(save_dir, "demon"))
        if self.demon_value_normalizer is not None:
            torch.save(
                self.demon_value_normalizer.state_dict(),
                str(save_dir) + "/demon_value_normalizer.pth",
            )

    def restore(self):
        """Restore model parameters."""
        super().restore()
        if (self.algo_args["demon"]['model_dir'] is not None) and self.algo_args["demon"].get('load_critic', True):
            if self.algo_args["demon"]['use_render'] is False and self.demon_value_normalizer is not None:
                value_normalizer_state_dict = torch.load(
                    str(self.algo_args["demon"]['model_dir']) + "/demon_value_normalizer.pth"
                )
                self.demon_value_normalizer.load_state_dict(value_normalizer_state_dict)