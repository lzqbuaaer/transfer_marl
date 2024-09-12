import os
import nni
import socket
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
        """Initialize the single/BaseRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        self.rnn_hidden_size = algo_args["train"]["hidden_sizes"][-1]
        self.recurrent_n = algo_args["train"]["recurrent_n"]
        
        self.episode_length = algo_args["train"]["episode_length"]
        self.n_rollout_threads = algo_args["train"]["n_rollout_threads"]
        self.n_eval_rollout_threads = algo_args['train']['n_eval_rollout_threads']

        self.share_param = algo_args["train"]['share_param']

        set_seed(algo_args["train"])
        self.device = init_device(algo_args["train"])
        self.task_name = get_task_name(args["env"], env_args)
        if not self.algo_args['train']['use_render']:
            self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
                args["env"],
                env_args,
                args["algo"],
                args["exp_name"],
                args["run"],
                algo_args["train"]["seed"],
                logger_path=algo_args["train"]["log_dir"],
            )
            save_config(args, algo_args, env_args, self.run_dir)

            # init wandb and save config
            wandb_dir = self.run_dir
            wandb_name = (
                args["env"]
                + "_"
                + get_task_name(args["env"], env_args)
                + "_"
                + args["run"]
                + "_"
                + args["algo"]
                + "_seed-"
                + str(algo_args["train"]["seed"])
                + "_"
                + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            )
            if "use_wandb" in algo_args["train"] and algo_args["train"]["use_wandb"]:
                # nni wandb path
                if algo_args["train"]["log_dir"] == "#nni_dynamic":
                    wandb_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"])
                    wandb_name = nni.get_trial_id()

                import wandb

                wandb.init(
                    project=args["exp_name"],
                    name=wandb_name,
                    config={
                        "args": args,
                        "algo_args": algo_args,
                        "env_args": env_args,
                    },
                    notes=socket.gethostname(),
                    entity="adv_marl_benchmark",
                    dir=wandb_dir,
                    job_type="training",
                )

        setproctitle.setproctitle(str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"]))

        # set the config of env
        if self.algo_args['train']['use_render']:  # make envs for rendering
            (
                self.envs,
                self.manual_render,
                self.manual_delay,
                self.env_num,
            ) = make_render_env(args["env"], algo_args["train"]["seed"], env_args)
        else:  # make envs for training and evaluation
            self.envs = make_train_env(
                args["env"],
                algo_args["train"]["seed"],
                algo_args["train"]["n_rollout_threads"],
                env_args,
            )
            self.eval_envs = (
                make_eval_env(
                    args["env"],
                    algo_args["train"]["seed"],
                    algo_args["train"]["n_eval_rollout_threads"],
                    env_args,
                )
                if algo_args["train"]["use_eval"]
                else None
            )
        self.num_agents = self.envs.n_agents
        self.action_type = self.envs.action_space[0].__class__.__name__

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space, self.action_type)

        if self.algo_args['train']['use_render'] is False:
            self.logger = LOGGER_REGISTRY[args["env"]](
                args, algo_args, env_args, self.num_agents, self.writter, self.run_dir
            )

        # algorithm
        if self.share_param:
            for agent_id in range(1, self.num_agents):
                assert (
                    self.envs.observation_space[agent_id] == self.envs.observation_space[0]
                ), "Agents have heterogeneous observation spaces, parameter sharing is not valid."
                assert (
                    self.envs.action_space[agent_id] == self.envs.action_space[0]
                ), "Agents have heterogeneous action spaces, parameter sharing is not valid."

        self.algo = ALGO_REGISTRY[args["algo"]](
            algo_args["train"],
            self.num_agents,
            self.envs.observation_space,
            self.envs.share_observation_space[0],
            self.envs.action_space,
            device=self.device,
        )

        self.agents = self.algo.agents
        self.critic = self.algo.critic

    def run(self):
        raise NotImplementedError

    @torch.no_grad()
    def eval(self):
        """Evaluate the model. All algorithms should fit this evaluation pipeline."""
        self.algo.prep_rollout()

        # logger callback at the beginning of evaluation
        self.logger.eval_init(self.n_eval_rollout_threads)
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, self.num_agents, self.recurrent_n, self.rnn_hidden_size),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                eval_actions, temp_rnn_state = self.agents[agent_id].perform(
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                    deterministic=True,
                )
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)

            eval_data = (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            )
            # logger callback at each step of evaluation
            
            # 这个函数没有用到available_actions的相关信息
            self.logger.eval_per_step(eval_data)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = 0

            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = 0

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    # logger callback when an episode is done
                    self.logger.eval_thread_done(eval_i)

            if eval_episode >= self.algo_args["train"]["eval_episodes"]:
                self.logger.eval_log(eval_episode)  # logger callback at the end of evaluation
                break

    @torch.no_grad()
    def render(self):
        """Render the model"""
        print("start rendering")
        self.algo.prep_rollout()

        for _ in range(self.algo_args['train']['render_episodes']):
            eval_rnn_states = np.zeros((self.env_num, self.num_agents, self.recurrent_n, self.rnn_hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.env_num, self.num_agents, 1), dtype=np.float32)

            eval_obs, _, eval_available_actions = self.envs.reset()
            rewards = 0
            while True:
                eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                if eval_available_actions is not None:
                    eval_available_actions = np.expand_dims(np.array(eval_available_actions), axis=0)

                eval_actions_collector = []
                for agent_id in range(self.num_agents):
                    eval_actions, temp_rnn_state = self.agents[agent_id].perform(
                        eval_obs[:, agent_id],
                        eval_rnn_states[:, agent_id],
                        eval_masks[:, agent_id],
                        eval_available_actions[:, agent_id] if eval_available_actions is not None else None,
                        deterministic=True,
                    )
                    eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                    eval_actions_collector.append(_t2n(eval_actions))
                eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

                eval_obs, _, eval_rewards, eval_dones, _, eval_available_actions = self.envs.step(eval_actions[0])
                rewards += eval_rewards[0][0]
                if self.manual_render:
                    self.envs.render()
                if self.manual_delay:
                    time.sleep(0.1)
                eval_dones_env = np.all(eval_dones)
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
        if self.algo_args['train']['model_dir'] is not None:  # restore model
            print("Restore model from", self.algo_args['train']['model_dir'])
            self.algo.restore(str(self.algo_args['train']['model_dir']))

    def save(self):
        """Save the model"""
        self.algo.save(str(self.save_dir))

    def close(self):
        """Close environment, writter, and log file."""
        if self.algo_args['train']['use_render']:
            self.envs.close()
        else:
            self.envs.close()
            if self.algo_args["train"]["use_eval"] and self.eval_envs is not self.envs:
                self.eval_envs.close()
            self.writter.close()
            self.logger.close()
