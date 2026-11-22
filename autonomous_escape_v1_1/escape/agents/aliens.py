"""Alien agents. They see the whole GameState and must not modify it."""
import math
import random
import time

from ..config import OPEN, SOFT, INF
from ..graph import astar, bfs_moves, min_vertex_cut
from ..fastsim import Sim, ROVER, ALIEN
from .base import Agent


def action_for(state, p):
    """Boulder on an open tile, debris on a soft one (to harden it)."""
    return ('boulder', p) if state.blk.get(p, OPEN) == OPEN else ('debris', p)


def rover_path(state):
    _, path = astar(state.map, state.blk, state.rover, state.map.goal, state.rules)
    return path[1:]


class RandomAlien(Agent):
    """Dormant tier: any legal tile, no plan."""
    name = 'Random'
    side = ALIEN

    def __init__(self, seed=None):
        super().__init__()
        self.rng = random.Random(seed)

    def act(self, s):
        tiles = [p for p in s.map.open_tiles if s.alien_can_target(p)]
        return action_for(s, self.rng.choice(tiles)) if tiles else None


class GreedyAlien(Agent):
    """Reactive tier: block (or harden) the next legal tile on the rover's
    cheapest route."""
    name = 'Greedy'
    side = ALIEN

    def act(self, s):
        for p in rover_path(s):
            if s.alien_can_target(p):
                return action_for(s, p)
        return None


class CoreAlien(Agent):
    """Core tier: sink-side minimum vertex cut + race test.

    1. Max-flow finds the fewest tiles separating the rover from the pad zone,
       taking the cut nearest the pad (the rover reaches it last).
    2. Race test: each open cut tile needs 2 actions (boulder, then debris to
       harden), a soft one needs 1. Seal only if that finishes before the
       rover can arrive (with one alien turn of slack).
    3. Work through the cut, most urgent tile first.
    4. Otherwise fall back to greedy blocking, which burns rover fuel and time.
    """
    name = 'Core'
    side = ALIEN

    def __init__(self, max_cut=6):
        super().__init__()
        self.max_cut = max_cut
        self.greedy = GreedyAlien()

    def plan(self, s):
        """Returns (cut, actions_needed, seal_time, arrival_time)."""
        cuttable = lambda p: s.alien_can_target(p) or (p in s.pending and not s.map.protected(p))
        size, cut = min_vertex_cut(s.map, s.blk, [s.rover], s.map.in_goal_zone, cuttable, limit=self.max_cut)
        self.nodes += len(s.map.open_tiles)
        if not cut:
            return [], 0, INF, 0
        moves = bfs_moves(s.map, s.blk, s.rover)
        need = 0
        for p in cut:
            b = s.blk.get(p, OPEN)
            if p in s.pending:
                need += 0 if b == SOFT else 1
            else:
                need += 2 if b == OPEN else 1
        seal = need * s.rules.alien_every + s.rules.debris_delay
        arrive = min(moves.get(p, INF) for p in cut)
        return cut, need, seal, arrive

    def act(self, s):
        cut, need, seal, arrive = self.plan(s)
        if cut and seal <= arrive + s.rules.alien_every:
            moves = bfs_moves(s.map, s.blk, s.rover)
            for p in sorted(cut, key=lambda p: moves.get(p, INF)):
                if s.alien_can_target(p):
                    self.info = f'sealing cut of {len(cut)}: {seal} vs rover {arrive}'
                    return action_for(s, p)
        self.info = f'cut {len(cut)} too slow ({seal} vs {arrive}): blocking' if cut else 'no cut: blocking'
        return self.greedy.act(s)


class _Node:
    __slots__ = ('children', 'untried', 'n', 'w')

    def __init__(self, actions):
        self.children = {}
        self.untried = list(actions)
        self.n = 0
        self.w = 0.0


class MCTSAlien(Agent):
    """UCT over the alien's decisions. The rover is modelled as a greedy
    cost-to-go follower on the true map; rollouts use a cheap block-the-route
    policy and are cut off after `horizon` alien turns, scored by how far the
    rover's route cost exceeds its fuel."""
    name = 'MCTS'
    side = ALIEN

    def __init__(self, iterations=150, time_limit=0.6, horizon=4, c=0.7, seed=0):
        super().__init__()
        self.iterations, self.time_limit = iterations, time_limit
        self.horizon, self.c = horizon, c
        self.rng = random.Random(seed)
        self.core = CoreAlien()

    def candidates(self, sim, extra_tiles=()):
        tiles = list(extra_tiles)
        tiles += sim.path_ahead(14)
        tiles += [p for p, b in sim.blk.items() if b == SOFT]
        acts, seen = [None], set()
        for p in tiles:
            if p in seen or not sim.can_target(p):
                continue
            seen.add(p)
            b = sim.blk.get(p, OPEN)
            acts.append(('boulder', p) if b == OPEN else ('debris', p))
            if b == OPEN and len(acts) < 10:
                acts.append(('debris', p))
            if len(acts) >= 24:
                break
        return acts

    def _advance_rover(self, sim):
        while sim.result is None and sim.turn == ROVER:
            sim.apply_rover(sim.greedy_rover_action())

    def _reward(self, sim):
        if sim.result == ALIEN:
            return 1.0
        if sim.result == ROVER:
            return 0.0
        margin = sim.fuel - sim.route_cost()
        return 1.0 / (1.0 + math.exp(margin / 4.0))

    def _rollout(self, sim, turns):
        while sim.result is None and turns > 0:
            self._advance_rover(sim)
            if sim.result is not None:
                break
            path = [p for p in sim.path_ahead(8) if sim.can_target(p)]
            r = self.rng.random()
            if path and r < 0.6:
                p = path[0] if r < 0.4 else self.rng.choice(path)
                sim.apply_alien(('boulder', p) if sim.blk.get(p, OPEN) == OPEN else ('debris', p))
            else:
                softs = [p for p, b in sim.blk.items() if b == SOFT and sim.can_target(p)]
                sim.apply_alien(('debris', self.rng.choice(softs)) if softs else None)
            turns -= 1
        return self._reward(sim)

    def act(self, s):
        cache, counter = {}, [0]
        root_sim = Sim.from_state(s, cache, counter)
        cut, *_ = self.core.plan(s)
        root = _Node(self.candidates(root_sim, cut))
        t0 = time.perf_counter()
        it = 0
        while it < self.iterations and time.perf_counter() - t0 < self.time_limit:
            it += 1
            sim = root_sim.clone()
            node, path, depth = root, [root], 0
            # selection
            while not node.untried and node.children and sim.result is None:
                logn = math.log(node.n)
                a, node = max(node.children.items(), key=lambda kv: kv[1].w / kv[1].n
                              + self.c * math.sqrt(logn / kv[1].n))
                sim.apply_alien(a)
                self._advance_rover(sim)
                path.append(node)
                depth += 1
            # expansion
            if node.untried and sim.result is None:
                a = node.untried.pop(self.rng.randrange(len(node.untried)))
                sim.apply_alien(a)
                self._advance_rover(sim)
                child = _Node(self.candidates(sim) if sim.result is None else [])
                node.children[a] = child
                path.append(child)
                depth += 1
            reward = self._rollout(sim, max(self.horizon - depth, 0))
            for n in path:
                n.n += 1
                n.w += reward
        self.nodes += counter[0] + it
        if not root.children:
            return None
        best_a, best = max(root.children.items(), key=lambda kv: kv[1].n)
        self.info = f'{it} rollouts, best win est. {best.w / best.n:.2f}'
        return best_a


ALIENS = [RandomAlien, GreedyAlien, CoreAlien, MCTSAlien]
