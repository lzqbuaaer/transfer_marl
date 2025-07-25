from amb.envs.smac.smac_dual_logger import SMACDualLogger


class SMACv2DualLogger(SMACDualLogger):

    def __init__(self, args, algo_args, env_args, num_angels, num_demons, writter, run_dir):
        super(SMACv2DualLogger, self).__init__(args, algo_args, env_args, num_angels, num_demons, writter, run_dir)
        self.win_key = "battle_won"
