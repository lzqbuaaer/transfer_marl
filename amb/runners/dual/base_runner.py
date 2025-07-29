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

        self.reverse_team = self.env_args['reverse_team']
        self.use_minus_opponent_reward = algo_args["angel"].get("use_minus_opponent_reward", True)
        self.eval_use_minus_opponent_reward = algo_args["angel"].get("eval_use_minus_opponent_reward", True)
        
        self.angel_rnn_hidden_size = algo_args["angel"]["hidden_sizes"][-1]
        self.angel_recurrent_n = algo_args["angel"]["recurrent_n"]
        self.demon_rnn_hidden_size = algo_args["demon"]["hidden_sizes"][-1]
        self.demon_recurrent_n = algo_args["demon"]["recurrent_n"]

        self.angel_share_param = algo_args["angel"]['share_param']
        self.demon_share_param = algo_args["demon"]['share_param']
        
        self.episode_length = algo_args["angel"]["episode_length"]
        self.n_rollout_threads = algo_args["angel"]["n_rollout_threads"]
        self.n_eval_rollout_threads = algo_args["angel"]['n_eval_rollout_threads']
        
        # Transferable networks for angel
        self.env_belief = algo_args["angel"].get("env_belief", False)
        self.env_belief_matter = algo_args["angel"].get("env_belief_matter", False)
        self.actor_divide_conquer = algo_args["angel"].get("actor_divide_conquer", False)
        self.actor_use_dt2gs = algo_args["angel"].get("actor_use_dt2gs", False)
        if self.actor_use_dt2gs:
            self.angel_actor_skills_num = algo_args["angel"].get("actor_skills_num", 4)
        if self.actor_divide_conquer:
            assert algo_args["angel"].get("actor_use_updet", False), \
                "When 'actor_divide_conquer' is set to <True>, 'actor_use_updet' must be set to <True> also!"
        if self.env_belief:
            env_prior_path = algo_args["angel"].get("env_prior_path", "./env_prior.npy")
            if os.path.exists(env_prior_path):
                self.env_prior = np.load(env_prior_path)
                self.env_belief_dim = len(self.env_prior)
                algo_args["angel"]["env_belief_dim"] = self.env_belief_dim
            else:
                import warnings
                warnings.warn(f"Env prior path {env_prior_path} does not exist! Switch param 'env_prior' to False.")
                self.env_belief = False
                algo_args["angel"]["env_belief"] = False
                
        # Transferable networks for demons
        self.demon_env_belief = algo_args["demon"].get("env_belief", False)
        self.demon_env_belief_matter = algo_args["demon"].get("env_belief_matter", False)
        self.demon_actor_divide_conquer = algo_args["demon"].get("actor_divide_conquer", False)
        self.demon_actor_use_dt2gs = algo_args["demon"].get("actor_use_dt2gs", False)
        if self.demon_actor_use_dt2gs:
            self.demon_actor_skills_num = algo_args["demon"].get("actor_skills_num", 4)
        if self.demon_actor_divide_conquer:
            assert algo_args["demon"].get("actor_use_updet", False), \
                "When 'actor_divide_conquer' is set to <True>, 'actor_use_updet' must be set to <True> also!"
        if self.demon_env_belief:
            env_prior_path = algo_args["demon"].get("env_prior_path", "./env_prior.npy")
            if os.path.exists(env_prior_path):
                self.demon_env_prior = np.load(env_prior_path)
                self.demon_env_belief_dim = len(self.demon_env_prior)
                algo_args["demon"]["env_belief_dim"] = self.demon_env_belief_dim
            else:
                import warnings
                warnings.warn(f"Env prior path {env_prior_path} does not exist! Switch param 'env_prior' to False.")
                self.demon_env_belief = False
                algo_args["demon"]["env_belief"] = False

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
            self.eval_envs = (
                make_eval_env(
                    args["env"],
                    algo_args["angel"]["seed"],
                    algo_args["angel"]["n_eval_rollout_threads"],
                    env_args,
                )
                if self.algo_args["angel"]['matter_transfer_test']
                else None
            )
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
        algo_args["angel"]["n_agents"] = self.num_angels
        algo_args["angel"]["n_enemies"] = self.num_demons
        algo_args["demon"]["n_agents"] = self.num_demons
        algo_args["demon"]["n_enemies"] = self.num_angels

        self.action_type = self.envs.action_space[0][0].__class__.__name__

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space, self.action_type)
        obs_own_feat, obs_enemy_feat, obs_ally_feat = self.envs.obs_own_feat, self.envs.obs_enemy_feat, self.envs.obs_ally_feat
        print("obs_own_feat: ", obs_own_feat)
        print("obs_enemy_feat: ", obs_enemy_feat)
        print("obs_ally_feat: ", obs_ally_feat)
        algo_args["angel"]["obs_own_feat"] = obs_own_feat[0]
        algo_args["angel"]["obs_enemy_feat"] = obs_enemy_feat[0]
        algo_args["angel"]["obs_ally_feat"] = obs_ally_feat[0]
        algo_args["demon"]["obs_own_feat"] = obs_own_feat[1]
        algo_args["demon"]["obs_enemy_feat"] = obs_enemy_feat[1]
        algo_args["demon"]["obs_ally_feat"] = obs_ally_feat[1]

        if self.algo_args["angel"]['use_render'] is False:
            self.logger = LOGGER_REGISTRY[args["env"]](
                args, algo_args, env_args, self.num_angels, self.num_demons, self.writter, self.run_dir
            )
            
        if self.env_belief and self.env_belief_matter:
            self.eval_angel_env_belief_ground_truth = np.zeros((self.n_eval_rollout_threads, self.num_angels, self.env_belief_dim), dtype=np.float32)
            self.eval_angel_env_belief_ground_truth[:] = self.env_prior
            self.angel_env_belief_ground_truth = np.zeros((self.n_rollout_threads, self.num_angels, self.env_belief_dim), dtype=np.float32)
            self.angel_env_belief_ground_truth[:] = self.env_prior
        if self.demon_env_belief and self.demon_env_belief_matter:
            self.eval_demon_env_belief_ground_truth = np.zeros((self.n_eval_rollout_threads, self.num_demons, self.demon_env_belief_dim), dtype=np.float32)
            self.eval_demon_env_belief_ground_truth[:] = self.demon_env_prior
            self.demon_env_belief_ground_truth = np.zeros((self.n_rollout_threads, self.num_demons, self.demon_env_belief_dim), dtype=np.float32)
            self.demon_env_belief_ground_truth[:] = self.demon_env_prior

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
        if args["run"] == 'dual_self_play':
            self.demon_algo = ALGO_REGISTRY[args["demon"]](
                algo_args["demon"],
                self.num_demons,
                self.envs.observation_space[1],
                self.envs.share_observation_space[1][0],
                self.envs.action_space[1],
                device=self.device,
                agent_type="adv_traitor",
            )
            self.demons = self.demon_algo.agents
            self.demon_critic = self.demon_algo.critic
        else:
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
    def eval(self, few_shot_learning_mode=False):
        if few_shot_learning_mode:
            # TODO few-shot learning for demon.
            assert (self.env_belief and self.env_belief_matter) or \
                (self.demon_env_belief and self.demon_env_belief_matter)
        """Evaluate the model. All algorithms should fit this evaluation pipeline."""
        self.algo.prep_rollout()

        if self.algo_args["angel"]['use_render'] is False:
            self.logger.eval_init(self.n_eval_rollout_threads)  # logger callback at the beginning of evaluation
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        last_obs = eval_obs
        last_reward = np.zeros((self.n_eval_rollout_threads, self.num_angels, 1))
        last_reward_demon = np.zeros((self.n_eval_rollout_threads, self.num_demons, 1))

        eval_angel_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_angels, self.angel_recurrent_n, self.angel_rnn_hidden_size), dtype=np.float32)
        eval_demon_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_demons, self.demon_recurrent_n, self.demon_rnn_hidden_size), dtype=np.float32)
        if self.env_belief:
            eval_angel_rnn_states_belief = np.zeros((self.n_eval_rollout_threads, self.num_angels, self.angel_recurrent_n, self.angel_rnn_hidden_size), dtype=np.float32)
            eval_angel_env_belief = np.zeros((self.n_eval_rollout_threads, self.num_angels, self.env_belief_dim), dtype=np.float32)
            eval_angel_env_belief[:] = self.env_prior
            eval_angel_bayesian_update = np.zeros((self.n_eval_rollout_threads), dtype=bool)
            if few_shot_learning_mode:
                eval_angel_env_belief_list = [self.env_prior.copy() for _ in range(self.n_eval_rollout_threads * self.num_angels)]
        else:
            eval_angel_env_belief = None
        if self.actor_use_dt2gs:
            eval_angel_previous_skills = np.zeros((self.n_eval_rollout_threads, self.num_angels, self.angel_actor_skills_num))
            
        if self.demon_env_belief:
            eval_demon_rnn_states_belief = np.zeros((self.n_eval_rollout_threads, self.num_demons, self.demon_recurrent_n, self.demon_rnn_hidden_size), dtype=np.float32)
            eval_demon_env_belief = np.zeros((self.n_eval_rollout_threads, self.num_demons, self.demon_env_belief_dim), dtype=np.float32)
            eval_demon_env_belief[:] = self.demon_env_prior
            eval_demon_bayesian_update = np.zeros((self.n_eval_rollout_threads), dtype=bool)
            if few_shot_learning_mode:
                eval_demon_env_belief_list = [self.demon_env_prior.copy() for _ in range(self.n_eval_rollout_threads * self.num_demons)]
        else:
            eval_demon_env_belief = None
        if self.actor_use_dt2gs:
            eval_demon_previous_skills = np.zeros((self.n_eval_rollout_threads, self.num_demons, self.demon_actor_skills_num))

        eval_angel_masks = np.ones((self.n_eval_rollout_threads, self.num_angels, 1), dtype=np.float32)
        eval_demon_masks = np.ones((self.n_eval_rollout_threads, self.num_demons, 1), dtype=np.float32)

        while True:
            eval_angel_actions_collector = []
            for agent_id in range(self.num_angels):
                if self.env_belief and (not self.env_belief_matter or few_shot_learning_mode):
                    env_belief, rnn_state_belief = self.angels[agent_id].forward_belief(
                        eval_obs[0][:, agent_id],
                        last_reward[:, agent_id],
                        last_obs[0][:, agent_id],
                        eval_angel_env_belief[:, agent_id],
                        eval_angel_rnn_states_belief[:, agent_id],
                        eval_angel_masks[:, agent_id],
                    )
                    eval_angel_env_belief[eval_angel_bayesian_update == True, agent_id] = _t2n(env_belief)[eval_angel_bayesian_update == True]
                    eval_angel_rnn_states_belief[eval_angel_bayesian_update == True, agent_id] = _t2n(rnn_state_belief)[eval_angel_bayesian_update == True]
                    if few_shot_learning_mode:
                        eval_angel_env_belief_list.extend([eval_angel_env_belief[i, agent_id].copy() 
                                                           for i in range(self.n_eval_rollout_threads) if eval_angel_bayesian_update[i]])
                eval_actions, temp_rnn_state = self.angels[agent_id].perform(
                    eval_obs[0][:, agent_id],
                    eval_angel_rnn_states[:, agent_id],
                    eval_angel_masks[:, agent_id],
                    eval_available_actions[0][:, agent_id]
                    if eval_available_actions[0][0] is not None else None,
                    env_belief = (self.eval_angel_env_belief_ground_truth[:, agent_id] if self.env_belief_matter 
                                  else eval_angel_env_belief[:, agent_id]) if self.env_belief else None,
                    previous_skills = eval_angel_previous_skills[:, agent_id] if self.actor_use_dt2gs else None,
                    deterministic=True,
                )
                if self.actor_use_dt2gs:
                    eval_actions, skills = eval_actions
                    eval_angel_previous_skills[:, agent_id] = _t2n(skills)
                if self.actor_divide_conquer:
                    eval_actions, _ = eval_actions
                eval_angel_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_angel_actions_collector.append(_t2n(eval_actions))
            eval_angel_actions = np.array(eval_angel_actions_collector).transpose(1, 0, 2)

            eval_demon_actions_collector = []
            for agent_id in range(self.num_demons):
                if self.demon_env_belief and (not self.demon_env_belief_matter or few_shot_learning_mode):
                    env_belief, rnn_state_belief = self.demons[agent_id].forward_belief(
                        eval_obs[1][:, agent_id],
                        last_reward_demon[:, agent_id],
                        last_obs[1][:, agent_id],
                        eval_demon_env_belief[:, agent_id],
                        eval_demon_rnn_states_belief[:, agent_id],
                        eval_demon_masks[:, agent_id],
                    )
                    eval_demon_env_belief[eval_demon_bayesian_update == True, agent_id] = _t2n(env_belief)[eval_demon_bayesian_update == True]
                    eval_demon_rnn_states_belief[eval_demon_bayesian_update == True, agent_id] = _t2n(rnn_state_belief)[eval_demon_bayesian_update == True]
                    if few_shot_learning_mode:
                        eval_demon_env_belief_list.extend([eval_demon_env_belief[i, agent_id].copy() 
                                                           for i in range(self.n_eval_rollout_threads) if eval_demon_bayesian_update[i]])
                eval_actions, temp_rnn_state = self.demons[agent_id].perform(
                    eval_obs[1][:, agent_id],
                    eval_demon_rnn_states[:, agent_id],
                    eval_demon_masks[:, agent_id],
                    eval_available_actions[1][:, agent_id]
                    if eval_available_actions[1][0] is not None else None,
                    env_belief = (self.eval_demon_env_belief_ground_truth[:, agent_id] if self.demon_env_belief_matter 
                                  else eval_demon_env_belief[:, agent_id]) if self.demon_env_belief else None,
                    previous_skills = eval_demon_previous_skills[:, agent_id] if self.demon_actor_use_dt2gs else None,
                    deterministic=True,
                )
                if self.demon_actor_use_dt2gs:
                    eval_actions, skills = eval_actions
                    eval_demon_previous_skills[:, agent_id] = _t2n(skills)
                if self.demon_actor_divide_conquer:
                    eval_actions, _ = eval_actions
                eval_demon_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_demon_actions_collector.append(_t2n(eval_actions))
            eval_demon_actions = np.array(eval_demon_actions_collector).transpose(1, 0, 2)
            
            last_obs = eval_obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step((eval_angel_actions, eval_demon_actions))
            if self.env_belief:
                eval_angel_bayesian_update[:] = True
            if self.demon_env_belief:
                eval_demon_bayesian_update[:] = True

            # MARL versus MARL, recording rewards of the opponents.
            assert self.num_angels == eval_rewards[0].shape[1]
            assert self.num_demons == eval_rewards[1].shape[1]
            assert eval_rewards[0].shape[0] == eval_rewards[1].shape[0]
            if self.reverse_team and self.eval_use_minus_opponent_reward:   # Dual attack: flip the reward while training, here is the same
                for process_id in range(eval_rewards[0].shape[0]):
                    eval_rewards[0][process_id, :, :] = np.mean(eval_rewards[1][process_id])
                last_reward = -eval_rewards[0]
            else:
                last_reward = eval_rewards[0]
            last_reward_demon = eval_rewards[1]

            eval_data = (eval_obs[0], eval_share_obs[0], eval_rewards[0], eval_dones[0], eval_infos[0], eval_available_actions[0])
            if self.algo_args["angel"]['use_render'] is False:
                self.logger.eval_per_step(eval_data)  # logger callback at each step of evaluation

            eval_dones_env = np.all(eval_dones[0], axis=1)
            eval_demon_dones_env = np.all(eval_dones[1], axis=1)

            eval_angel_rnn_states[eval_dones_env == True] = 0
            eval_demon_rnn_states[eval_dones_env == True] = 0
            if self.actor_use_dt2gs:
                eval_angel_previous_skills[eval_dones_env == True] = 0
            if self.demon_actor_use_dt2gs:
                eval_demon_previous_skills[eval_demon_dones_env == True] = 0
            if self.env_belief:
                eval_angel_rnn_states_belief[eval_dones_env == True] = 0
                eval_angel_bayesian_update[eval_dones_env == True] = False
                eval_angel_env_belief[eval_dones_env == True, :] = self.env_prior
                if few_shot_learning_mode:
                    eval_angel_env_belief_list.extend([self.env_prior.copy() for i in range(self.n_eval_rollout_threads * self.num_angels) 
                                                       if eval_angel_bayesian_update[i % self.n_eval_rollout_threads]])
            if self.demon_env_belief:
                eval_demon_rnn_states_belief[eval_demon_dones_env == True] = 0
                eval_demon_bayesian_update[eval_demon_dones_env == True] = False
                eval_demon_env_belief[eval_demon_dones_env == True, :] = self.demon_env_prior
                if few_shot_learning_mode:
                    eval_demon_env_belief_list.extend([self.demon_env_prior.copy() for i in range(self.n_eval_rollout_threads * self.num_demons) 
                                                       if eval_demon_bayesian_update[i % self.n_eval_rollout_threads]])

            eval_angel_masks = np.ones((self.n_eval_rollout_threads, self.num_angels, 1), dtype=np.float32)
            eval_demon_masks = np.ones((self.n_eval_rollout_threads, self.num_demons, 1), dtype=np.float32)
            eval_angel_masks[eval_dones_env == True] = 0
            eval_demon_masks[eval_dones_env == True] = 0

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    if self.algo_args["angel"]['use_render'] is False:
                        self.logger.eval_thread_done(eval_i)  # logger callback when an episode is done

            if eval_episode >= (self.algo_args["angel"]["eval_episodes"] if not few_shot_learning_mode 
                                else self.algo_args["angel"]["matter_transfer_few_shot_episodes"]):
                if self.algo_args["angel"]['use_render'] is False:
                    self.logger.eval_log(eval_episode)  # logger callback at the end of evaluation
                if few_shot_learning_mode:
                    self.env_prior = np.stack(eval_angel_env_belief_list, axis=0).mean(axis=0)
                    self.eval_angel_env_belief_ground_truth[:] = self.env_prior
                    self.angel_env_belief_ground_truth[:] = self.env_prior
                    
                    self.demon_env_prior = np.stack(eval_demon_env_belief_list, axis=0).mean(axis=0)
                    self.eval_demon_env_belief_ground_truth[:] = self.demon_env_prior
                    self.demon_env_belief_ground_truth[:] = self.demon_env_prior
                break

    @torch.no_grad()
    def render(self):
        """Render the model"""
        print("start rendering")
        self.algo.prep_rollout()

        for _ in range(self.algo_args["angel"]['render_episodes']):
            eval_angel_rnn_states = np.zeros((1, self.num_angels, self.angel_recurrent_n, self.angel_rnn_hidden_size), dtype=np.float32)
            eval_demon_rnn_states = np.zeros((1, self.num_demons, self.demon_recurrent_n, self.demon_rnn_hidden_size), dtype=np.float32)
            if self.env_belief:
                eval_angel_rnn_states_belief = np.zeros((1, self.num_angels, self.angel_recurrent_n, self.angel_rnn_hidden_size), dtype=np.float32)
                eval_angel_env_belief = np.zeros((1, self.num_angels, self.env_belief_dim), dtype=np.float32)
                eval_angel_env_belief[:] = self.env_prior
                eval_bayesian_update = np.zeros((1), dtype=bool)
            else:
                eval_angel_env_belief = None
            if self.actor_use_dt2gs:
                eval_angel_previous_skills = np.zeros((1, self.num_angels, self.angel_actor_skills_num))

            eval_angel_masks = np.ones((1, self.num_angels, 1), dtype=np.float32)
            eval_demon_masks = np.ones((1, self.num_demons, 1), dtype=np.float32)

            eval_obs, _, eval_available_actions = self.envs.reset()
            last_obs = [np.expand_dims(np.array(eval_obs[i]), axis=0) for i in range(2)]
            last_reward = np.zeros((1, self.num_angels, 1))
            
            rewards = 0
            while True:
                eval_obs = [np.expand_dims(np.array(eval_obs[i]), axis=0) for i in range(2)]
                if eval_available_actions is not None:
                    eval_available_actions = [np.expand_dims(np.array(eval_available_actions[i]), axis=0) for i in range(2)]
                    
                eval_angel_actions_collector = []
                for agent_id in range(self.num_angels):
                    if self.env_belief and (not self.env_belief_matter):
                        env_belief, rnn_state_belief = self.angels[agent_id].forward_belief(
                            eval_obs[0][:, agent_id],
                            last_reward[:, agent_id],
                            last_obs[0][:, agent_id],
                            eval_angel_env_belief[:, agent_id],
                            eval_angel_rnn_states_belief[:, agent_id],
                            eval_angel_masks[:, agent_id],
                        )
                        eval_angel_env_belief[eval_bayesian_update == True, agent_id] = _t2n(env_belief)[eval_bayesian_update == True]
                        eval_angel_rnn_states_belief[eval_bayesian_update == True, agent_id] = _t2n(rnn_state_belief)[eval_bayesian_update == True]
                    
                    eval_actions, temp_rnn_state = self.angels[agent_id].perform(
                        eval_obs[0][:, agent_id],
                        eval_angel_rnn_states[:, agent_id],
                        eval_angel_masks[:, agent_id],
                        eval_available_actions[0][:, agent_id]
                        if eval_available_actions[0][0] is not None else None,
                        env_belief = (self.eval_angel_env_belief_ground_truth[:1, agent_id] if self.env_belief_matter 
                                    else eval_angel_env_belief[:, agent_id]) if self.env_belief else None,
                        previous_skills = eval_angel_previous_skills[:, agent_id] if self.actor_use_dt2gs else None,
                        deterministic=True,
                    )
                    if self.actor_use_dt2gs:
                        eval_actions, skills = eval_actions
                        eval_angel_previous_skills[:, agent_id] = _t2n(skills)
                    if self.actor_divide_conquer:
                        eval_actions, chosen = eval_actions
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
                
                last_obs = eval_obs
                eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.envs.step((eval_angel_actions[0], eval_demon_actions[0]))
                if self.env_belief:
                    eval_bayesian_update[:] = True
                eval_rewards = [np.expand_dims(np.array(eval_rewards[i]), axis=0) for i in range(2)]
                if self.reverse_team and self.eval_use_minus_opponent_reward:   # Dual attack: flip the reward while training, here is the same
                    for process_id in range(eval_rewards[0].shape[0]):
                        eval_rewards[0][process_id, :, :] = np.mean(eval_rewards[1][process_id])
                    last_reward = -eval_rewards[0]
                else:
                    last_reward = eval_rewards[0]
                    
                    # np.expand_dims(np.array(eval_obs[i]), axis=0)
                # rewards += eval_rewards[0][0]
                # MARL versus MARL, recording rewards of the opponents.
                rewards += eval_rewards[0][0][1][0]
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
            try:
                self.algo.restore(os.path.join(self.algo_args['angel']['model_dir'], 'angel'))
            except Exception as e1:
                self.algo.restore(str(self.algo_args['angel']['model_dir']))            

        if self.args["run"] == 'dual_self_play':
            print("Restore demon model from", self.algo_args['demon']['model_dir'])
            # self.algo.restore(str(self.algo_args['angel']['model_dir']))
            try:
                self.demon_algo.restore(os.path.join(self.algo_args['demon']['model_dir'], 'demon'))
            except Exception as e1:
                self.demon_algo.restore(str(self.algo_args['demon']['model_dir']))
        else:
            if self.algo_args['demon']['model_dir'] is not None:  # restore model
                print("Restore demon model from", self.algo_args['demon']['model_dir'])
                if self.demon_share_param:
                    self.demons[0].restore(str(self.algo_args['demon']['model_dir']))
                else:
                    for agent_id in range(self.num_demons):
                        self.demons[agent_id].restore(os.path.join(self.algo_args['demon']['model_dir'], str(agent_id)))

    def save(self, time_step=None):
        """Save the model"""
        if time_step is not None:
            save_dir = os.path.join(self.save_dir, str(time_step))
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = self.save_dir
        self.algo.save(os.path.join(save_dir, "angel"))

        # if self.demon_share_param:
        #     self.demons[0].save(os.path.join(self.save_dir, "demon"))
        # else:
        #     for agent_id in range(self.num_demons):
        #         self.demons[agent_id].save(os.path.join(self.save_dir, "demon", str(agent_id)))

    def close(self):
        """Close environment, writter, and log file."""
        if self.algo_args["angel"]['use_render']:
            self.envs.close()
            if self.algo_args["angel"]['matter_transfer_test']:
                self.eval_envs.close()
        else:
            self.envs.close()
            if self.algo_args["angel"]["use_eval"] and self.eval_envs is not self.envs:
                self.eval_envs.close()
            self.writter.close()
            self.logger.close()