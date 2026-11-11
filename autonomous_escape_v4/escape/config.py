"""Game rules. Defaults are the values balanced in simulation (see README)."""
from dataclasses import dataclass

OPEN, SOFT, HARD = 0, 1, 2
INF = float('inf')


@dataclass(frozen=True)
class Rules:
    size: int = 29              # grid is size x size, odd (v4: bigger field)
    alien_every: int = 3        # alien acts once every N rover turns
    alien_budget: int = 14      # total alien actions per game (v4: bigger field)
    fuel_mult: float = 1.8      # starting fuel = fuel_mult x shortest route
    move_cost: int = 1          # move or wait
    shot_cost: int = 2          # laser shot on an adjacent soft boulder
    debris_damage: int = 6      # fuel lost when debris lands on the rover
    debris_delay: int = 2       # rover turns between launch and landing
    radar: int = 5              # rover sensor radius (circle, in tiles)
    rock_fill: float = 0.5      # share of the field that starts as rock
    channels: int = 6           # winding channels carved from base to pad
    protect_radius: int = 2     # no-build zone around base and pad (Chebyshev)
    min_routes: int = 4         # vertex-disjoint routes required between zones
    max_routes: int = 10        # ...but not wide open: keep real choke points
    min_length: int = 44        # shortest base-to-pad route, in moves
    # optional rover abilities (campaign); off by default so the AI benchmark is unchanged
    scan_cost: int = 0          # fuel for a radar pulse; 0 = ability not available
    scan_radius: int = 10       # a pulse maps every boulder this far away (circle, tiles)
    shields: int = 0            # debris hits the shield absorbs per game

    def enter_cost(self, b):
        """Fuel to get onto a tile in state b (shoot first if soft)."""
        if b == HARD:
            return INF
        return self.move_cost + (self.shot_cost if b == SOFT else 0)


DEFAULT = Rules()
