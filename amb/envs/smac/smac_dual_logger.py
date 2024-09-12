import time
from functools import reduce
import numpy as np
from amb.envs.dual_logger import DualLogger


class SMACDualLogger(DualLogger):

    def __init__(self, args, algo_args, env_args, num_angels, num_demons, writter, run_dir):
        super(SMACDualLogger, self).__init__(args, algo_args, env_args, num_angels, num_demons, writter, run_dir)
        self.win_key = "won"
        self.infos = [{} for i in range(self.algo_args["angel"]["n_rollout_threads"])]

    def get_task_name(self):
        return self.env_args["map_name"]

    def init(self):
        super().init()
        self.last_battles_game = np.zeros(
            self.algo_args["angel"]["n_rollout_threads"], dtype=np.float32
        )
        self.last_battles_won = np.zeros(
            self.algo_args["angel"]["n_rollout_threads"], dtype=np.float32
        )

    def per_step(self, data):
        super().per_step(data)
        infos = data["infos"]
        filled = data["filled"]
        if len(filled.shape) > 1:
            filled = filled[:, 0]
        for i in range(self.algo_args["angel"]["n_rollout_threads"]):
            if filled[i]:
                self.infos[i] = infos[i]

    def episode_log(self, actor_train_infos, critic_train_info, buffers):
        for agent_id in range(len(buffers)):
            actor_train_infos[agent_id]["dead_ratio"] = 1 - buffers[agent_id].data["active_masks"].sum() / (
                len(buffers) * reduce(lambda x, y: x * y, list(buffers[agent_id].data["active_masks"].shape)))
        super().episode_log(actor_train_infos, critic_train_info, buffers)

        battles_won = []
        battles_game = []
        incre_battles_won = []
        incre_battles_game = []

        for i, info in enumerate(self.infos):
            if "battles_won" in info[0].keys():
                battles_won.append(info[0]["battles_won"])
                incre_battles_won.append(
                    info[0]["battles_won"] - self.last_battles_won[i]
                )
            if "battles_game" in info[0].keys():
                battles_game.append(info[0]["battles_game"])
                incre_battles_game.append(
                    info[0]["battles_game"] - self.last_battles_game[i]
                )

        incre_win_rate = np.sum(incre_battles_won) / np.sum(incre_battles_game) if np.sum(incre_battles_game) > 0 else 0.0
        self.writter.add_scalar("env/incre_win_rate", incre_win_rate, self.timestep)

        self.last_battles_game = battles_game
        self.last_battles_won = battles_won

        print(
            "Increase games {:.4f}, win rate on these games is {:.4f}".format(
                np.sum(incre_battles_game),
                incre_win_rate,
            )
        )

    def eval_init(self, n_eval_rollout_threads):
        super().eval_init(n_eval_rollout_threads)
        self.eval_battles_won = 0

    def eval_thread_done(self, tid):
        super().eval_thread_done(tid)
        if self.eval_infos[tid][0][self.win_key] == True:
            self.eval_battles_won += 1

    def eval_log(self, eval_episode):
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards])
        eval_win_rate = self.eval_battles_won / eval_episode
        eval_env_infos = {
            "eval_return_mean": self.eval_episode_rewards,
            "eval_return_std": [np.std(self.eval_episode_rewards)],
            "eval_win_rate": [eval_win_rate],
        }
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print(
            "Evaluation win rate is {}, evaluation average episode reward is {}.\n".format(
                eval_win_rate, eval_avg_rew
            )
        )
        if self.args["run"] == "dual":
            self.log_file.write(
                ",".join(map(str, [self.timestep, eval_avg_rew, eval_win_rate]))
                + "\n"
            )
            self.log_file.flush()

    def eval_log_adv(self, eval_episode):
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards])
        eval_win_rate = self.eval_battles_won / eval_episode
        eval_env_infos = {
            "eval_adv_return_mean": self.eval_episode_rewards,
            "eval_adv_return_std": [np.std(self.eval_episode_rewards)],
            "eval_adv_win_rate": [eval_win_rate],
        }
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print(
            "Evaluation adv win rate is {}, evaluation adv average episode reward is {}.\n".format(
                eval_win_rate, eval_avg_rew
            )
        )
        if self.args["run"] == "perturbation":
            self.adv_file.write(
                ",".join(map(str, [
                    self.algo_args["angel"]["perturb_epsilon"], 
                    self.algo_args["angel"]["perturb_iters"], 
                    self.algo_args["angel"]["adaptive_alpha"], 
                    self.algo_args["angel"]["perturb_alpha"], 
                    eval_avg_rew, eval_win_rate])) + "\n"
            )
            self.adv_file.flush()
        elif self.args["run"] == "traitor":
            self.log_file.write(
                ",".join(map(str, [self.timestep, eval_avg_rew, eval_win_rate]))
                + "\n"
            )
            self.log_file.flush()
