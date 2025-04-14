from amb.envs.dual_logger import DualLogger


class MAgentsDualLogger(DualLogger):

    def get_task_name(self):
        return "size{}x{}".format(self.env_args["map_size"], self.env_args["map_size"])
