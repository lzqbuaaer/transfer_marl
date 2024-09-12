import numpy as np
from amb.envs.base_logger import BaseLogger


class FootballLogger(BaseLogger):

    def get_task_name(self):
        return self.env_args["env_name"]

    def eval_init(self, n_eval_rollout_threads):
        super().eval_init(n_eval_rollout_threads)
        self.eval_episode_cnt = 0
        self.eval_score_cnt = 0

    def eval_thread_done(self, tid):
        super().eval_thread_done(tid)
        self.eval_episode_cnt += 1
        if self.eval_infos[tid][0]["score_reward"] > 0:
            self.eval_score_cnt += 1

    def eval_log(self, eval_episode):
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards]
        )
        eval_score_rate = self.eval_score_cnt / self.eval_episode_cnt
        eval_env_infos = {
            "eval_return_mean": self.eval_episode_rewards,
            "eval_return_std": [np.std(self.eval_episode_rewards)],
            "eval_score_rate": [eval_score_rate],
        }
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print(
            "Evaluation average episode reward is {}, evaluation score rate is {}.\n".format(
                eval_avg_rew, eval_score_rate
            )
        )
        if self.args["run"] == "single":
            self.log_file.write(
                ",".join(map(str, [self.timestep, eval_avg_rew, eval_score_rate]))
                + "\n"
            )
            self.log_file.flush()

    def eval_log_adv(self, eval_episode):
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards]
        )
        eval_score_rate = self.eval_score_cnt / self.eval_episode_cnt
        eval_env_infos = {
            "eval_adv_return_mean": self.eval_episode_rewards,
            "eval_adv_return_std": [np.std(self.eval_episode_rewards)],
            "eval_adv_score_rate": [eval_score_rate],
        }
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print(
            "Evaluation adv average episode reward is {}, evaluation adv score rate is {}.\n".format(
                eval_avg_rew, eval_score_rate
            )
        )
        if self.args["run"] == "perturbation":
            self.adv_file.write(
                ",".join(map(str, [
                    self.algo_args["train"]["perturb_epsilon"], 
                    self.algo_args["train"]["perturb_iters"], 
                    self.algo_args["train"]["adaptive_alpha"], 
                    self.algo_args["train"]["perturb_alpha"], 
                    eval_avg_rew, eval_score_rate])) + "\n"
            )
            self.adv_file.flush()
        elif self.args["run"] == "traitor":
            self.log_file.write(
                ",".join(map(str, [self.timestep, eval_avg_rew, eval_score_rate]))
                + "\n"
            )
            self.log_file.flush()