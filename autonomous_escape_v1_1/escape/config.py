"""Game rules. Defaults are the values balanced in simulation (see README)."""
from dataclasses import dataclass

OPEN, SOFT, HARD = 0, 1, 2
INF = float('inf')


@dataclass(frozen=True)
class Rules:
    size: int = 51              # grid is size x size, odd (v1.1: five giant maps)
    alien_every: int = 10       # alien acts once every N rover turns (v1.1: long maps)
    alien_budget: int = 10      # total alien actions per game
    fuel_mult: float = 2.0      # starting fuel = fuel_mult x shortest route
    move_cost: int = 1          # move or wait
    shot_cost: int = 2          # laser shot on an adjacent soft boulder
    debris_damage: int = 6      # fuel lost when debris lands on the rover
    debris_delay: int = 2       # rover turns between launch and landing
    radar: int = 5              # rover sensor radius (circle, in tiles)
    rock_fill: float = 0.45     # share of the field that starts as rock
    channels: int = 10          # winding channels carved from base to pad
    waypoints: int = 1          # random detour points per channel (meandering corridors)
    ridges: int = 4             # long rock ridges with alternating gaps (switchbacks)
    protect_radius: int = 2     # no-build zone around base and pad (Chebyshev)
    min_routes: int = 5         # vertex-disjoint routes required between zones
    max_routes: int = 12        # ...but not wide open: keep real choke points
    min_length: int = 120       # shortest base-to-pad route, in moves

    def enter_cost(self, b):
        """Fuel to get onto a tile in state b (shoot first if soft)."""
        if b == HARD:
            return INF
        return self.move_cost + (self.shot_cost if b == SOFT else 0)


DEFAULT = Rules()
