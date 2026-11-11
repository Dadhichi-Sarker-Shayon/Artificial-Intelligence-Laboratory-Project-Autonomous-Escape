import random

import pytest

from escape import GameState, Rules, SOFT, HARD, INF, ROVER, ALIEN
from escape.fastsim import Sim
from escape.graph import astar, cost_to_go, disjoint_routes, manhattan
from escape.mapgen import generate


@pytest.fixture
def s():
    return GameState.new(100)


def free_neighbor(s):
    return next(v for v in s.map.nbrs[s.rover] if not s.blk.get(v))


def test_maps_are_deterministic_and_fair():
    a, b = generate(7), generate(7)
    assert a.walls == b.walls
    for seed in range(20):
        m = generate(seed)
        assert disjoint_routes(m, 3) >= 3
        assert m.is_open(m.start) and m.is_open(m.goal)


def test_starting_fuel(s):
    assert s.fuel0 == round(s.rules.fuel_mult * s.shortest)
    assert s.fuel == s.fuel0


def test_move_and_wait_cost_one(s):
    f = s.fuel
    s.apply_rover(('move', free_neighbor(s)))
    assert s.fuel == f - 1
    s.apply_rover(('wait', None))
    assert s.fuel == f - 2


def test_laser_destroys_soft_boulder(s):
    v = free_neighbor(s)
    s.blk[v] = SOFT
    f = s.fuel
    pos = s.rover
    s.apply_rover(('shoot', v))
    assert v not in s.blk and s.rover == pos and s.fuel == f - 2 and s.shots == 1


def test_cannot_shoot_hard_or_walk_into_boulders(s):
    v = free_neighbor(s)
    s.blk[v] = HARD
    assert ('shoot', v) not in s.rover_actions()
    assert ('move', v) not in s.rover_actions()
    pos = s.rover
    s.apply_rover(('move', v))          # illegal -> treated as a wait
    assert s.rover == pos


def test_alien_acts_every_third_turn(s):
    for _ in range(2):
        s.apply_rover(('wait', None))
        assert s.turn == ROVER
    s.apply_rover(('wait', None))
    assert s.turn == ALIEN


def far_tile(s, dist=6):
    return next(p for p in s.map.open_tiles
                if s.alien_can_target(p) and manhattan(p, s.rover) >= dist)


def test_debris_lands_after_delay_and_hardens(s):
    p = far_tile(s)
    s.turn = ALIEN
    s.apply_alien(('boulder', p))
    assert s.blk[p] == SOFT and s.budget == s.rules.alien_budget - 1
    s.turn = ALIEN
    s.apply_alien(('debris', p))
    s.apply_rover(('wait', None))
    assert s.blk[p] == SOFT and p in s.pending
    s.apply_rover(('wait', None))
    assert s.blk[p] == HARD and p not in s.pending


def test_debris_on_open_tile_makes_soft_boulder(s):
    p = far_tile(s)
    s.turn = ALIEN
    s.apply_alien(('debris', p))
    s.apply_rover(('wait', None))
    s.apply_rover(('wait', None))
    assert s.blk[p] == SOFT


def test_debris_hit_costs_fuel(s):
    # put debris two tiles away, then walk onto it as it lands
    path = astar(s.map, {}, s.rover, s.map.goal, s.rules)[1]
    p = path[2]
    s.pending[p] = 2
    f = s.fuel
    s.apply_rover(('move', path[1]))
    s.apply_rover(('move', path[2]))
    assert s.hits == 1 and s.fuel == f - 2 - s.rules.debris_damage and p not in s.blk


def test_alien_target_rules(s):
    for v in s.map.nbrs[s.rover]:
        assert not s.alien_can_target(v)              # next to the rover
    assert not s.alien_can_target(s.map.goal)        # protected pad
    p = far_tile(s)
    s.blk[p] = HARD
    assert not s.legal_alien(('debris', p))
    q = far_tile(s)
    s.blk[q] = SOFT
    assert not s.legal_alien(('boulder', q)) and s.legal_alien(('debris', q))


def test_hold_does_not_spend_budget(s):
    s.turn = ALIEN
    s.apply_alien(None)
    assert s.budget == s.rules.alien_budget


def test_sealed_ends_game(s):
    from escape.graph import min_vertex_cut
    size, cut = min_vertex_cut(s.map, {}, [s.rover], s.map.in_goal_zone, s.alien_can_target)
    for p in cut:
        s.blk[p] = HARD
    s.turn = ALIEN
    s.apply_alien(None)
    assert s.result == ALIEN and s.reason == 'Every route sealed'


def test_score_is_fuel_fraction():
    s = GameState.new(3)
    path = astar(s.map, {}, s.rover, s.map.goal, s.rules)[1]
    for p in path[1:]:
        if s.turn == ALIEN:
            s.apply_alien(None)
        s.apply_rover(('move', p))
    assert s.result == ROVER
    assert s.score == pytest.approx(100 * (s.fuel0 - s.shortest) / s.fuel0)


def test_cost_to_go_matches_astar():
    m = generate(11)
    r = Rules()
    rng = random.Random(1)
    blk = {p: rng.choice([SOFT, HARD]) for p in rng.sample(m.open_tiles, 25) if not m.protected(p)}
    d = cost_to_go(m, blk, m.goal, r)
    for p in rng.sample(m.open_tiles, 30):
        if blk.get(p) == HARD:
            continue
        assert d.get(p, INF) == astar(m, blk, p, m.goal, r)[0]


@pytest.mark.parametrize('seed', [5, 6, 7])
def test_fastsim_matches_gamestate(seed):
    """Random play: the search simulator must follow the real rules exactly."""
    rng = random.Random(seed)
    s = GameState.new(seed)
    sim = Sim.from_state(s)
    while s.result is None:
        if s.turn == ALIEN:
            acts = s.alien_actions() + [None]
            a = rng.choice(acts)
            s.apply_alien(a)
            sim.apply_alien(a)
        else:
            a = rng.choice(s.rover_actions())
            s.apply_rover(a)
            sim.apply_rover(a)
        assert (sim.pos, sim.fuel, sim.tick, sim.budget, sim.blk, sim.pending, sim.turn) == \
               (s.rover, s.fuel, s.tick, s.budget, s.blk, s.pending, s.turn)
        assert sim.result == s.result


def test_rover_view_hides_far_boulders(s):
    p = far_tile(s, dist=s.rules.radar + 3)
    s.blk[p] = SOFT
    s.sense()
    assert p not in s.rover_view().belief


def test_rover_sees_far_debris_and_where_it_lands(s):
    p = far_tile(s, dist=s.rules.radar + 3)
    s.pending[p] = 1
    s.sense()
    assert s.rover_view().pending == {p: 1}
    s.apply_rover(('wait', None))
    assert s.blk[p] == SOFT and s.rover_view().belief[p] == SOFT


def test_radar_is_a_circle(s):
    r0, c0 = s.rover
    R = s.rules.radar
    assert s.in_radar((r0 + R, c0)) and s.in_radar((r0 + 3, c0 + 4))
    assert not s.in_radar((r0 + R, c0 + 1))
