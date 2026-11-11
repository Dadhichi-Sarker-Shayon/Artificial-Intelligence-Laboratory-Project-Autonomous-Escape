"""Game rules. Defaults are the values balanced in simulation (see README)."""
from dataclasses import dataclass

OPEN, SOFT, HARD = 0, 1, 2
INF = float('inf')


@dataclass(frozen=True)
class Rules:
    size: int = 21              # grid is size x size, odd
    alien_every: int = 3        # alien acts once every N rover turns
    alien_budget: int = 8       # total alien actions per game
    fuel_mult: float = 2.0      # starting fuel = fuel_mult x shortest route
    move_cost: int = 1          # move or wait
    shot_cost: int = 2          # laser shot on an adjacent soft boulder
    debris_damage: int = 6      # fuel lost when debris lands on the rover
    debris_delay: int = 2       # rover turns between launch and landing
    radar: int = 5              # rover sensor radius (circle, in tiles)
    rock_fill: float = 0.55     # share of the field that starts as rock
    protect_radius: int = 2     # no-build zone around base and pad (Chebyshev)
    min_routes: int = 3         # vertex-disjoint routes required between zones
    max_routes: int = 6         # ...but not wide open: keep real choke points
    min_length: int = 30        # shortest base-to-pad route, in moves

    def enter_cost(self, b):
        """Fuel to get onto a tile in state b (shoot first if soft)."""
        if b == HARD:
            return INF
        return self.move_cost + (self.shot_cost if b == SOFT else 0)


DEFAULT = Rules()
