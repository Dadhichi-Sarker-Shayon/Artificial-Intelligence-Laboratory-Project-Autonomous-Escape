import random

import pytest

from escape import GameState, SOFT, HARD, INF, ROVER, ALIEN
from escape.agents import ROVERS, ALIENS, DStarLiteRover, CoreAlien, GreedyAlien, FuzzyRover
from escape.agents.rovers import DStarLite
from escape.bench import play, wilson
from escape.graph import astar


@pytest.mark.parametrize('rover', [c.name for c in ROVERS])
@pytest.mark.parametrize('alien', [c.name for c in ALIENS])
def test_every_pairing_finishes(rover, alien):
    row = play(101, rover, alien, alien_kw={'iterations': 30} if alien == 'MCTS' else None)
    assert row['winner'] in (ROVER, ALIEN)
    assert row['turns'] > 0


def test_dstar_lite_matches_astar_after_changes():
    s = GameState.new(12)
    rng = random.Random(3)
    belief = {}
    planner = DStarLite(s.map, s.rules, s.rover, belief)
    pos = s.rover
    for step in range(15):
        planner.observe(pos, belief)
        planner.compute()
        want, path = astar(s.map, belief, pos, s.map.goal, s.rules)
        assert planner.cost() == want
        if len(path) < 3:
            break
        # move one step, then change a few tiles near the route
        pos = path[1] if belief.get(path[1]) is None else pos
        for p in rng.sample(path[2:], min(2, len(path) - 2)):
            if not s.map.protected(p):
                belief[p] = rng.choice([SOFT, HARD])


def test_unaided_rover_wins_without_alien():
    s = GameState.new(9)
    s.budget = 0
    from escape.agents import AStarRover
    r = AStarRover()
    while s.result is None:
        s.apply_rover(r.act(s.rover_view()))
    assert s.result == ROVER


def test_core_beats_greedy_on_sample():
    seeds = range(200, 215)
    core = sum(play(sd, 'A*', 'Core')['winner'] == ALIEN for sd in seeds)
    greedy = sum(play(sd, 'A*', 'Greedy')['winner'] == ALIEN for sd in seeds)
    assert core > greedy


def test_fuzzy_caution_responds_to_inputs():
    f = FuzzyRover()
    assert f.decide_caution(0.05, 5, 10) < 0.35          # no slack: sprint
    assert f.decide_caution(0.6, 2, 10) > 0.6            # slack and a short detour: play safe
    assert f.decide_caution(0.3, 8, 1) > f.decide_caution(0.3, 8, 10)  # debris nearby


def test_wilson_interval():
    lo, hi = wilson(80, 100)
    assert lo == pytest.approx(0.7112, abs=1e-3) and hi == pytest.approx(0.8667, abs=1e-3)
