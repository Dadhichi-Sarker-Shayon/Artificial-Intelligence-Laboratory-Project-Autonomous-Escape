from .base import Agent
from .rovers import ROVERS, FloodFillRover, AStarRover, DStarLiteRover, MinimaxRover, FuzzyRover
from .aliens import ALIENS, RandomAlien, GreedyAlien, CoreAlien, MCTSAlien

ROVER_BY_NAME = {c.name: c for c in ROVERS}
ALIEN_BY_NAME = {c.name: c for c in ALIENS}


def make_rover(name, seed=0):
    import random
    rover = ROVER_BY_NAME[name]()
    rover.rng = random.Random(seed)          # breaks ties between equally good routes
    return rover


def make_alien(name, seed=0, **kw):
    import random
    cls = ALIEN_BY_NAME[name]
    if cls in (RandomAlien, MCTSAlien):
        kw.setdefault('seed', seed)
    alien = cls(**kw)
    alien.rng = getattr(alien, 'rng', None) or random.Random(seed)    # every alien varies its choices
    if hasattr(alien, 'greedy'):
        alien.greedy.rng = random.Random(seed + 1)
    return alien
