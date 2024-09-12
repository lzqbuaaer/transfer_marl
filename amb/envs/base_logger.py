"""Base logger."""

import time
import os
import numpy as np
import nni
from amb.utils.trans_utils import _dimalign


class BaseLogger:
    """Base logger class.
    Used for logging information in the on-policy training pipeline.
    """

    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        """Initialize the logger."""
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args
        self.task_name = self.get_task_name()
        self.num_agents = num_agents
        self.writter = writter
        self.run_dir = run_dir
        self.log_file = open(os.path.join(run_dir, "progress.txt"), "w", encoding='utf-8')
        if args["run"] == "perturbation":
            self.adv_file = open("./perturbation_rewards.txt", "a", encoding="utf-8")
        
        # wandb
        self.wandb = None
        if "use_wandb" in algo_args["train"] and algo_args["train"]["use_wandb"]:
            import wandb
            self.wandb = wandb

    def get_task_name(self):
        """Get the task name."""
        raise NotImplementedError
    
    def get_average_step_reward(self, buffers):
        t = buffers[0].current_size
        rewards = buffers[0].data["rewards"][t:t+self.algo_args["train"]["n_rollout_threads"]]
        filled = buffers[0].data["filled"][t:t+self.algo_args["train"]["n_rollout_threads"]]
        filled = _dimalign(filled, rewards)
        average_rewards = (rewards * filled).sum() / filled.sum()
        return average_rewards

    def init(self):
        """Initialize the logger."""
        self.start = time.time()
        self.train_episode_rewards = np.zeros(self.algo_args["train"]["n_rollout_threads"])
        self.done_episodes_rewards = []
        self.one_episode_len = np.zeros(self.algo_args["train"]["n_rollout_threads"], dtype=np.int32)
        self.episode_lens = []

    def episode_init(self, timestep):
        """Initialize the logger for each episode."""
        self.timestep = timestep

    def per_step(self, data):
        """Process data per step."""
        rewards = data["rewards"]
        dones = data["dones"]
        filled = data["filled"]
        if len(filled.shape) > 1:
            filled = filled[:, 0]
        dones_env = np.all(dones, axis=1) * filled
        reward_env = np.mean(rewards, axis=1).flatten() * filled
        self.train_episode_rewards += reward_env
        for t in range(self.algo_args["train"]["n_rollout_threads"]):
            if filled[t] :
                self.one_episode_len[t] += 1
                if dones_env[t]:
                    self.done_episodes_rewards.append(self.train_episode_rewards[t].copy())
                    self.train_episode_rewards[t] = 0
                    self.episode_lens.append(self.one_episode_len[t].copy())
                    self.one_episode_len[t] = 0

    def episode_log(self, actor_train_infos, critic_train_info, buffers):
        """Log information for each episode."""
        self.end = time.time()
        print(
            "\n[Env] {} [Task] {} [Algo] {} [Exp] {}. Total timesteps {}/{}, FPS {}.".format(
                self.args["env"],
                self.task_name,
                self.args["algo"],
                self.args["exp_name"],
                self.timestep,
                self.algo_args['train']['num_env_steps'],
                int(self.timestep / (self.end - self.start)),
            )
        )

        average_episode_len = np.mean(self.episode_lens) if len(self.episode_lens) > 0 else 0.0
        self.writter.add_scalar("env/ep_length_mean", average_episode_len, self.timestep)
        self.wandb and self.wandb.log({"env/ep_length_mean": average_episode_len}, step=self.timestep)

        aver_episode_rewards = np.mean(self.done_episodes_rewards)
        critic_train_info["average_step_rewards"] = aver_episode_rewards / average_episode_len
        self.writter.add_scalar("env/train_episode_rewards", aver_episode_rewards, self.timestep)
        self.wandb and self.wandb.log({"env/train_episode_rewards": aver_episode_rewards}, step=self.timestep)

        self.log_train(actor_train_infos, critic_train_info)

        print(
            "Train-time average step reward is {:.4f}, average episode length is {:.4f}, average episode reward is {:.4f}.".format(
                aver_episode_rewards / average_episode_len,
                average_episode_len,
                aver_episode_rewards
            )
        )

        self.done_episodes_rewards = []
        self.episode_lens = []

    def eval_init(self, n_eval_rollout_threads):
        """Initialize the logger for evaluation."""
        self.eval_episode_rewards = []
        self.one_episode_rewards = []
        self.n_eval_rollout_threads = n_eval_rollout_threads
        for eval_i in range(n_eval_rollout_threads):
            self.one_episode_rewards.append([])
            self.eval_episode_rewards.append([])

    def eval_per_step(self, eval_data):
        """Log evaluation information per step."""
        (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
        ) = eval_data
        for eval_i in range(self.n_eval_rollout_threads):
            self.one_episode_rewards[eval_i].append(eval_rewards[eval_i])
        self.eval_infos = eval_infos

    def eval_thread_done(self, tid):
        """Log evaluation information."""
        self.eval_episode_rewards[tid].append(np.sum(self.one_episode_rewards[tid], axis=0))
        self.one_episode_rewards[tid] = []

    def eval_log(self, eval_episode):
        """Log evaluation information."""
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards]
        )
        eval_env_infos = {
            "eval_return_mean": self.eval_episode_rewards,
            "eval_return_std": [np.std(self.eval_episode_rewards)],
        }
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print("Evaluation average episode reward is {}.\n".format(eval_avg_rew))
        # nni report
        nni.report_intermediate_result(eval_avg_rew)
        
        if self.args["run"] == "single":
            self.log_file.write(
                ",".join(map(str, [self.timestep, eval_avg_rew])) + "\n"
            )
            self.log_file.flush()

    def eval_log_adv(self, eval_episode):
        """Log evaluation information."""
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards]
        )
        eval_env_infos = {
            "eval_adv_return_mean": self.eval_episode_rewards,
            "eval_adv_return_std": [np.std(self.eval_episode_rewards)],
        }
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print("Evaluation adv average episode reward is {}.\n".format(eval_avg_rew))
        if self.args["run"] == "perturbation":
            self.adv_file.write(
                ",".join(map(str, [
                    self.algo_args["train"]["perturb_epsilon"], 
                    self.algo_args["train"]["perturb_iters"], 
                    self.algo_args["train"]["adaptive_alpha"], 
                    self.algo_args["train"]["perturb_alpha"], 
                    eval_avg_rew])) + "\n"
            )
            self.adv_file.flush()
        elif self.args["run"] == "traitor":
            self.log_file.write(
                ",".join(map(str, [self.timestep, eval_avg_rew])) + "\n"
            )
            self.log_file.flush()

    def log_train(self, actor_train_infos, critic_train_info):
        """Log training information."""
        # log actor
        for agent_id in range(self.num_agents):
            for k, v in actor_train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalar(agent_k, v, self.timestep)
                self.wandb and self.wandb.log({agent_k: v}, step=self.timestep)
        # log critic
        for k, v in critic_train_info.items():
            critic_k = "critic/" + k
            self.writter.add_scalar(critic_k, v, self.timestep)
            self.wandb and self.wandb.log({critic_k: v}, step=self.timestep)

    def log_env(self, env_infos):
        """Log environment information."""
        for k, v in env_infos.items():
            if len(v) > 0:
                self.writter.add_scalar("env/{}".format(k), np.mean(v), self.timestep)
                self.wandb and self.wandb.log({"env/{}".format(k): np.mean(v)}, step=self.timestep)

    def close(self):
        """Close the logger."""
        self.log_file.close()
