from amb.envs.base_logger import BaseLogger


class MAgentsLogger(BaseLogger):

    def get_task_name(self):
        return "size{}x{}".format(self.env_args["map_size"], self.env_args["map_size"])
