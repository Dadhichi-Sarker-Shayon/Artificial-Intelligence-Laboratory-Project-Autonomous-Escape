"""Game rules. Defaults are the values balanced in simulation (see README)."""
from dataclasses import dataclass

OPEN, SOFT, HARD = 0, 1, 2
INF = float('inf')


@dataclass(frozen=True)
class Rules:
    size: int = 51              # grid is size x size, odd (v1.1: five giant maps)
    alien_every: int = 10       # alien acts once every N rover turns (v1.1: long maps)
    alien_budget: int = 10      # total alien actions per game (each game: one more or less)
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
    style: str = 'ridges'       # rock layout: ridges | rings | bars | islands | scatter
    cluster: int = 1            # rock clumping (blur passes): 0 scattered, 1 clumps, 3 islands
    protect_radius: int = 2     # no-build zone around base and pad (Chebyshev)
    min_routes: int = 5         # vertex-disjoint routes required between zones
    max_routes: int = 12        # ...but not wide open: keep real choke points
    min_length: int = 120       # shortest base-to-pad route, in moves
    random_events: bool = True  # strike timing, debris delay and damage vary each game

    def enter_cost(self, b):
        """Fuel to get onto a tile in state b (shoot first if soft)."""
        if b == HARD:
            return INF
        return self.move_cost + (self.shot_cost if b == SOFT else 0)


DEFAULT = Rules()


# Each of the five maps has its own rock layout style (a new layout every game)
STYLE_RULES = {
    'ridges': dict(style='ridges', alien_every=5, alien_budget=10, ridges=4, rock_fill=0.45, channels=10, waypoints=1, cluster=1),
    'rings': dict(style='rings', alien_every=5, alien_budget=12, ridges=4, rock_fill=0.28, channels=8, waypoints=1, cluster=1, min_length=92),
    'bars': dict(style='bars', alien_every=5, alien_budget=10, ridges=5, rock_fill=0.3, channels=8, waypoints=1, cluster=1, min_length=92),
    'islands': dict(style='islands', alien_every=5, alien_budget=14, ridges=0, rock_fill=0.42, channels=6, waypoints=2, cluster=3,
                    min_length=92, max_routes=30),
    'scatter': dict(style='scatter', alien_every=5, alien_budget=12, ridges=0, rock_fill=0.3, channels=4, waypoints=2, cluster=0,
                    min_length=92, max_routes=30),
}


def rules_for(style, **extra):
    from dataclasses import replace
    return replace(DEFAULT, **{**STYLE_RULES[style], **extra})


def random_rules(game_seed, gmap):
    """Rules for one game, drawn at random: how often the alien strikes, how many
    strikes it gets (sized to this map's narrowest choke point, so it can win
    some games and lose others), and how much fuel the rover starts with."""
    import random
    from dataclasses import replace
    from .graph import disjoint_routes
    rng = random.Random(game_seed * 7 + 3)
    cut = disjoint_routes(gmap, 8)
    return replace(DEFAULT, size=gmap.size, alien_every=rng.randint(3, 7),
                   alien_budget=max(2, 2 * cut + rng.randint(-1, 2)),   # around what a full seal needs
                   fuel_mult=round(rng.uniform(1.6, 2.2), 2))
