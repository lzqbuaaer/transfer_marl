import os
import time
import torch
import numpy as np
import setproctitle
from amb.algorithms import ALGO_REGISTRY
from amb.envs import LOGGER_REGISTRY
from amb.utils.trans_utils import _t2n
from amb.utils.env_utils import (
    make_eval_env,
    make_train_env,
    make_render_env,
    set_seed,
)
from amb.utils.model_utils import init_device
from amb.utils.config_utils import init_dir, save_config, get_task_name


class BaseRunner:
    def __init__(self, args, algo_args, env_args):
        """Initialize the dual/BaseRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        self.angel_rnn_hidden_size = algo_args["angel"]["hidden_sizes"][-1]
        self.angel_recurrent_n = algo_args["angel"]["recurrent_n"]
        self.demon_rnn_hidden_size = algo_args["demon"]["hidden_sizes"][-1]
        self.demon_recurrent_n = algo_args["demon"]["recurrent_n"]

        self.angel_share_param = algo_args["angel"]['share_param']
        self.demon_share_param = algo_args["demon"]['share_param']
        
        self.episode_length = algo_args["angel"]["episode_length"]
        self.n_rollout_threads = algo_args["angel"]["n_rollout_threads"]
        self.n_eval_rollout_threads = algo_args["angel"]['n_eval_rollout_threads']

        set_seed(algo_args["angel"])
        self.device = init_device(algo_args["angel"])
        self.task_name = get_task_name(args["env"], env_args)
        if not self.algo_args["angel"]['use_render']:
            self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
                args["env"],
                env_args,
                args["angel"] + "-" + args["demon"],
                args["exp_name"],
                args["run"],
                algo_args["angel"]["seed"],
                logger_path=algo_args["angel"]["log_dir"],
            )
            save_config(args, algo_args, env_args, self.run_dir)
        setproctitle.setproctitle(
            str(args["angel"]) + "-" + str(args["demon"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
        )

        # set the config of env
        if self.algo_args["angel"]['use_render']:  # make envs for rendering
            (
                self.envs,
                self.manual_render,
                self.manual_delay,
                self.env_num,
            ) = make_render_env(args["env"], algo_args["angel"]["seed"], env_args)
        else:  # make envs for training and evaluation
            self.envs = make_train_env(
                args["env"],
                algo_args["angel"]["seed"],
                algo_args["angel"]["n_rollout_threads"],
                env_args,
            )
            self.eval_envs = (
                make_eval_env(
                    args["env"],
                    algo_args["angel"]["seed"],
                    algo_args["angel"]["n_eval_rollout_threads"],
                    env_args,
                )
                if algo_args["angel"]["use_eval"]
                else None
            )
        # self.num_agents = self.envs.n_agents
        self.num_angels = self.envs.n_angels
        self.num_demons = self.envs.n_demons

        self.action_type = self.envs.action_space[0][0].__class__.__name__

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space, self.action_type)

        if self.algo_args["angel"]['use_render'] is False:
            self.logger = LOGGER_REGISTRY[args["env"]](
                args, algo_args, env_args, self.num_angels, self.num_demons, self.writter, self.run_dir
            )

        # algorithm
        self.algo = ALGO_REGISTRY[args["angel"]](
            algo_args["angel"],
            self.num_angels,
            self.envs.observation_space[0],
            self.envs.share_observation_space[0][0],
            self.envs.action_space[0],
            device=self.device,
            agent_type="adv_traitor",
        )
        self.angels = self.algo.agents
        self.critic = self.algo.critic

        self.demons = []
        if self.demon_share_param:
            agent = ALGO_REGISTRY[args["demon"]].create_agent(
                algo_args["demon"],
                self.envs.observation_space[1][0],
                self.envs.action_space[1][0],
                device=self.device,
                agent_type="adv_victim",
            )
            agent.prep_rollout()
            for agent_id in range(self.num_demons):
                self.demons.append(agent)
        else:
            for agent_id in range(self.num_demons):
                agent = ALGO_REGISTRY[args["demon"]].create_agent(
                    algo_args["demon"],
                    self.envs.observation_space[1][agent_id],
                    self.envs.action_space[1][agent_id],
                    device=self.device,
                    agent_type="adv_victim",
                )
                agent.prep_rollout()
                self.demons.append(agent)

    def run(self):
        raise NotImplementedError

    @torch.no_grad()
    def eval(self):
        """Evaluate the model. All algorithms should fit this evaluation pipeline."""
        self.algo.prep_rollout()

        self.logger.eval_init(self.n_eval_rollout_threads)  # logger callback at the beginning of evaluation
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_angel_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_angels, self.angel_recurrent_n, self.angel_rnn_hidden_size), dtype=np.float32)
        eval_demon_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_demons, self.demon_recurrent_n, self.demon_rnn_hidden_size), dtype=np.float32)

        eval_angel_masks = np.ones((self.n_eval_rollout_threads, self.num_angels, 1), dtype=np.float32)
        eval_demon_masks = np.ones((self.n_eval_rollout_threads, self.num_demons, 1), dtype=np.float32)

        while True:
            eval_angel_actions_collector = []
            for agent_id in range(self.num_angels):
                eval_actions, temp_rnn_state = self.angels[agent_id].perform(
                    eval_obs[0][:, agent_id],
                    eval_angel_rnn_states[:, agent_id],
                    eval_angel_masks[:, agent_id],
                    eval_available_actions[0][:, agent_id]
                    if eval_available_actions[0][0] is not None else None,
                    deterministic=True,
                )
                eval_angel_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_angel_actions_collector.append(_t2n(eval_actions))
            eval_angel_actions = np.array(eval_angel_actions_collector).transpose(1, 0, 2)

            eval_demon_actions_collector = []
            for agent_id in range(self.num_demons):
                eval_actions, temp_rnn_state = self.demons[agent_id].perform(
                    eval_obs[1][:, agent_id],
                    eval_demon_rnn_states[:, agent_id],
                    eval_demon_masks[:, agent_id],
                    eval_available_actions[1][:, agent_id]
                    if eval_available_actions[1][0] is not None else None,
                    deterministic=True,
                )
                eval_demon_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_demon_actions_collector.append(_t2n(eval_actions))
            eval_demon_actions = np.array(eval_demon_actions_collector).transpose(1, 0, 2)
            
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step((eval_angel_actions, eval_demon_actions))

            # MARL versus MARL, recording rewards of the opponents.
            assert self.num_angels == eval_rewards[0].shape[1]
            assert self.num_demons == eval_rewards[1].shape[1]
            assert eval_rewards[0].shape[0] == eval_rewards[1].shape[0]
            for process_id in range(eval_rewards[0].shape[0]):
                eval_rewards[0][process_id, :, :] = np.mean(eval_rewards[1][process_id])

            eval_data = (eval_obs[0], eval_share_obs[0], eval_rewards[0], eval_dones[0], eval_infos[0], eval_available_actions[0])
            self.logger.eval_per_step(eval_data)  # logger callback at each step of evaluation

            eval_dones_env = np.all(eval_dones[0], axis=1)

            eval_angel_rnn_states[eval_dones_env == True] = 0
            eval_demon_rnn_states[eval_dones_env == True] = 0

            eval_angel_masks = np.ones((self.n_eval_rollout_threads, self.num_angels, 1), dtype=np.float32)
            eval_demon_masks = np.ones((self.n_eval_rollout_threads, self.num_demons, 1), dtype=np.float32)
            eval_angel_masks[eval_dones_env == True] = 0
            eval_demon_masks[eval_dones_env == True] = 0

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(eval_i)  # logger callback when an episode is done

            if eval_episode >= self.algo_args["angel"]["eval_episodes"]:
                self.logger.eval_log(eval_episode)  # logger callback at the end of evaluation
                break

    @torch.no_grad()
    def render(self):
        """Render the model"""
        print("start rendering")
        self.algo.prep_rollout()

        for _ in range(self.algo_args["angel"]['render_episodes']):
            eval_angel_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_angels, self.angel_recurrent_n, self.angel_rnn_hidden_size), dtype=np.float32)
            eval_demon_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_demons, self.demon_recurrent_n, self.demon_rnn_hidden_size), dtype=np.float32)

            eval_angel_masks = np.ones((self.n_eval_rollout_threads, self.num_angels, 1), dtype=np.float32)
            eval_demon_masks = np.ones((self.n_eval_rollout_threads, self.num_demons, 1), dtype=np.float32)

            eval_obs, _, eval_available_actions = self.envs.reset()
            rewards = 0
            while True:
                eval_obs = [np.expand_dims(np.array(eval_obs[i]), axis=0) for i in range(2)]
                if eval_available_actions is not None:
                    eval_available_actions = [np.expand_dims(np.array(eval_available_actions[i]), axis=0) for i in range(2)]
                    
                eval_angel_actions_collector = []
                for agent_id in range(self.num_angels):
                    eval_actions, temp_rnn_state = self.angels[agent_id].perform(
                        eval_obs[0][:, agent_id],
                        eval_angel_rnn_states[:, agent_id],
                        eval_angel_masks[:, agent_id],
                        eval_available_actions[0][:, agent_id]
                        if eval_available_actions[0][0] is not None else None,
                        deterministic=True,
                    )
                    eval_angel_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                    eval_angel_actions_collector.append(_t2n(eval_actions))
                eval_angel_actions = np.array(eval_angel_actions_collector).transpose(1, 0, 2)

                eval_demon_actions_collector = []
                for agent_id in range(self.num_demons):
                    eval_actions, temp_rnn_state = self.demons[agent_id].perform(
                        eval_obs[1][:, agent_id],
                        eval_demon_rnn_states[:, agent_id],
                        eval_demon_masks[:, agent_id],
                        eval_available_actions[1][:, agent_id]
                        if eval_available_actions[1][0] is not None else None,
                        deterministic=True,
                    )
                    eval_demon_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                    eval_demon_actions_collector.append(_t2n(eval_actions))
                eval_demon_actions = np.array(eval_demon_actions_collector).transpose(1, 0, 2)
                
                eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.envs.step((eval_angel_actions[0], eval_demon_actions[0]))
                # rewards += eval_rewards[0][0]
                # MARL versus MARL, recording rewards of the opponents.
                rewards += eval_rewards[1][0]
                if self.manual_render:
                    self.envs.render()
                if self.manual_delay:
                    time.sleep(0.1)
                eval_dones_env = np.all(eval_dones[0])
                if eval_dones_env:
                    print(f'total reward of this episode: {rewards}')
                    break
                
        if "smac" in self.args["env"]:  # replay for smac, no rendering
            if "v2" in self.args["env"]:
                self.envs.env.save_replay()
            else:
                self.envs.save_replay()

    def restore(self):
        """Restore the model"""
        if self.algo_args['angel']['model_dir'] is not None:  # restore model
            print("Restore angel model from", self.algo_args['angel']['model_dir'])
            # self.algo.restore(str(self.algo_args['angel']['model_dir']))
            self.algo.restore(os.path.join(self.algo_args['angel']['model_dir'], 'angel'))

        if self.algo_args['demon']['model_dir'] is not None:  # restore model
            print("Restore demon model from", self.algo_args['demon']['model_dir'])
            if self.demon_share_param:
                self.demons[0].restore(str(self.algo_args['demon']['model_dir']))
            else:
                for agent_id in range(self.num_demons):
                    self.demons[agent_id].restore(os.path.join(self.algo_args['demon']['model_dir'], str(agent_id)))

    def save(self):
        """Save the model"""
        self.algo.save(os.path.join(self.save_dir, "angel"))

        # if self.demon_share_param:
        #     self.demons[0].save(os.path.join(self.save_dir, "demon"))
        # else:
        #     for agent_id in range(self.num_demons):
        #         self.demons[agent_id].save(os.path.join(self.save_dir, "demon", str(agent_id)))

    def close(self):
        """Close environment, writter, and log file."""
        if self.algo_args["angel"]['use_render']:
            self.envs.close()
        else:
            self.envs.close()
            if self.algo_args["angel"]["use_eval"] and self.eval_envs is not self.envs:
                self.eval_envs.close()
            self.writter.close()
            self.logger.close()