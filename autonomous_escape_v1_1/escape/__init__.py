"""Autonomous Escape: a rover-vs-alien search game on Phobos."""
from .config import Rules, DEFAULT, OPEN, SOFT, HARD, INF
from .mapgen import generate, GameMap
from .state import GameState, step, ROVER, ALIEN
