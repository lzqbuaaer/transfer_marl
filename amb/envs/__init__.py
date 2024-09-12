import socket
from absl import flags
from amb.envs.smac.smac_logger import SMACLogger
from amb.envs.smac.smac_dual_logger import SMACDualLogger
from amb.envs.smacv2.smacv2_logger import SMACv2Logger
from amb.envs.mamujoco.mamujoco_logger import MAMuJoCoLogger
from amb.envs.pettingzoo_mpe.pettingzoo_mpe_logger import PettingZooMPELogger
from amb.envs.gym.gym_logger import GYMLogger
from amb.envs.football.football_logger import FootballLogger
from amb.envs.toy_example.toy_logger import ToyLogger

FLAGS = flags.FLAGS
FLAGS(["train_sc.py"])

LOGGER_REGISTRY = {
    "smac": SMACLogger,
    "smac_dual": SMACDualLogger,
    "mamujoco": MAMuJoCoLogger,
    "pettingzoo_mpe": PettingZooMPELogger,
    "gym": GYMLogger,
    "football": FootballLogger,
    "smacv2": SMACv2Logger,
    "toy": ToyLogger,
}
