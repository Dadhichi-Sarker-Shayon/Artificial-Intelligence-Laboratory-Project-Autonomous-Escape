from .base import Agent
from .rovers import ROVERS, FloodFillRover, AStarRover, DStarLiteRover, MinimaxRover, FuzzyRover
from .aliens import ALIENS, RandomAlien, GreedyAlien, CoreAlien, MCTSAlien

ROVER_BY_NAME = {c.name: c for c in ROVERS}
ALIEN_BY_NAME = {c.name: c for c in ALIENS}


def make_rover(name, seed=0):
    return ROVER_BY_NAME[name]()


def make_alien(name, seed=0, **kw):
    cls = ALIEN_BY_NAME[name]
    if cls in (RandomAlien, MCTSAlien):
        kw.setdefault('seed', seed)
    return cls(**kw)
