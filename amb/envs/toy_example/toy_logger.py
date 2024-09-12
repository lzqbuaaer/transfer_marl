from amb.envs.base_logger import BaseLogger


class ToyLogger(BaseLogger):

    def get_task_name(self):
        return "state{}_action{}".format(self.env_args["obs_last_state"], self.env_args["obs_last_action"])
