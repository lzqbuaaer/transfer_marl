from amb.algorithms.maddpg import MADDPG
from amb.algorithms.mappo import MAPPO
from amb.algorithms.igs import IGS
from amb.algorithms.q import Q
from amb.algorithms.qtran import QTran
from amb.algorithms.coma import COMA

ALGO_REGISTRY = {
    "maddpg": MADDPG,
    "mappo": MAPPO,
    "igs": IGS,
    "qmix": Q,
    "vdn": Q,
    "iql": Q,
    "qtran": QTran,
    "coma": COMA,
}
