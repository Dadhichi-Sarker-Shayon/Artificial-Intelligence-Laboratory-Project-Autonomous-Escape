"""Cross-check our min vertex cut against networkx's max-flow."""
import random

import networkx as nx
import pytest

from escape import GameState, SOFT, HARD
from escape.graph import min_vertex_cut, bfs_moves


def nx_cut_size(s, cuttable):
    G = nx.DiGraph()
    big = 10 ** 6
    for p in s.map.open_tiles:
        if s.blk.get(p) == HARD:
            continue
        G.add_edge((p, 0), (p, 1), capacity=1 if cuttable(p) else big)
        for q in s.map.nbrs[p]:
            if s.blk.get(q) != HARD:
                G.add_edge((p, 1), (q, 0), capacity=big)
        if s.map.in_goal_zone(p):
            G.add_edge((p, 1), 'T', capacity=big)
    G.add_edge('S', (s.rover, 0), capacity=big)
    return nx.maximum_flow_value(G, 'S', 'T')


@pytest.mark.parametrize('seed', range(8))
def test_min_cut_matches_networkx_and_separates(seed):
    rng = random.Random(seed)
    s = GameState.new(300 + seed)
    for p in rng.sample(s.map.open_tiles, 30):
        if not s.map.protected(p) and p != s.rover:
            s.blk[p] = rng.choice([SOFT, HARD])
    size, cut = min_vertex_cut(s.map, s.blk, [s.rover], s.map.in_goal_zone, s.alien_can_target, limit=50)
    want = nx_cut_size(s, s.alien_can_target)
    if want >= 10 ** 6:
        assert cut == []                      # no cut exists
        return
    assert size == want == len(cut)
    if size:
        walled = {**s.blk, **{p: HARD for p in cut}}
        reach = bfs_moves(s.map, walled, s.rover)
        assert not any(s.map.in_goal_zone(p) for p in reach)
