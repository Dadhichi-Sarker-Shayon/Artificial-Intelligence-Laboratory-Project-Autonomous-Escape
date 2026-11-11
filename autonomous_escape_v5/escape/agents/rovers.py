"""Rover agents. Each one sees only rover_view(): walls, its own boulder
memory, incoming debris, fuel and public counters."""
import heapq
import math

import numpy as np

from ..config import OPEN, SOFT, HARD, INF
from ..graph import astar, bfs_moves, manhattan
from ..fastsim import Sim, ROVER
from .base import Agent, path_action


class FloodFillRover(Agent):
    """R0 baseline: breadth-first flood from the pad, ignoring fuel costs.
    Treats boulders as walls unless that leaves no way through."""
    name = 'Flood Fill'
    side = ROVER

    def act(self, v):
        goal = v.map.goal
        dist = bfs_moves(v.map, v.belief, goal, passable=lambda b: b == OPEN)
        if v.rover not in dist:
            dist = bfs_moves(v.map, v.belief, goal)
        self.nodes += len(dist)
        here = dist.get(v.rover)
        if here is None:
            return ('wait', None)
        steps = [n for n in v.map.nbrs[v.rover] if dist.get(n, INF) == here - 1]
        steps.sort(key=lambda n: v.belief.get(n, OPEN))   # open tiles first
        return path_action(v, steps[0] if steps else None)


class AStarRover(Agent):
    """R1 baseline: weighted A* on the belief map, replanned every turn.
    Soft boulders cost a shot plus a move; hard boulders are walls."""
    name = 'A*'
    side = ROVER

    def act(self, v):
        c = [0]
        _, path = astar(v.map, v.belief, v.rover, v.map.goal, v.rules, counter=c)
        self.nodes += c[0]
        return path_action(v, path[1] if len(path) > 1 else None)


class DStarLite:
    """D* Lite (Koenig & Likhachev 2002) on the grid, planning from the pad
    back to the rover so replanning after map changes is incremental."""

    def __init__(self, gmap, rules, start, belief):
        self.m, self.rules = gmap, rules
        self.goal = gmap.goal
        self.start = self.last = start
        self.blk = dict(belief)
        self.km = 0
        self.g, self.rhs = {}, {self.goal: 0}
        self.U, self.inq = [], {}
        self.expanded = 0
        self._push(self.goal)

    def _c(self, v):
        return self.rules.enter_cost(self.blk.get(v, OPEN))

    def _key(self, s):
        m = min(self.g.get(s, INF), self.rhs.get(s, INF))
        return (m + manhattan(self.start, s) + self.km, m)

    def _push(self, s):
        k = self._key(s)
        self.inq[s] = k
        heapq.heappush(self.U, (k, s))

    def _update(self, u):
        if u != self.goal:
            self.rhs[u] = min((self._c(s) + self.g.get(s, INF) for s in self.m.nbrs[u]), default=INF)
        if self.g.get(u, INF) != self.rhs.get(u, INF):
            self._push(u)
        else:
            self.inq.pop(u, None)

    def compute(self):
        while self.U:
            k, u = self.U[0]
            if self.inq.get(u) != k:
                heapq.heappop(self.U)
                continue
            if not (k < self._key(self.start)
                    or self.rhs.get(self.start, INF) != self.g.get(self.start, INF)):
                break
            heapq.heappop(self.U)
            k_new = self._key(u)
            if k < k_new:
                self.inq[u] = k_new
                heapq.heappush(self.U, (k_new, u))
                continue
            self.expanded += 1
            del self.inq[u]
            if self.g.get(u, INF) > self.rhs.get(u, INF):
                self.g[u] = self.rhs[u]
                for s in self.m.nbrs[u]:
                    self._update(s)
            else:
                self.g[u] = INF
                for s in self.m.nbrs[u] + [u]:
                    self._update(s)

    def observe(self, rover, belief):
        """Move the start and apply any boulder changes the rover saw."""
        self.start = rover
        changed = [p for p in set(self.blk) | set(belief) if self.blk.get(p, OPEN) != belief.get(p, OPEN)]
        if changed:
            self.km += manhattan(self.last, self.start)
            self.last = self.start
            self.blk = dict(belief)
            for p in changed:
                for u in self.m.nbrs[p]:
                    self._update(u)

    def cost(self):
        return self.g.get(self.start, INF)

    def next_tile(self):
        best, best_v = None, INF
        for s in self.m.nbrs[self.start]:
            val = self._c(s) + self.g.get(s, INF)
            if val < best_v:
                best, best_v = s, val
        return best


class DStarLiteRover(Agent):
    """R2: same decisions as A*, but replans incrementally. Compare its
    nodes-expanded count with A* in the benchmark."""
    name = 'D* Lite'
    side = ROVER

    def reset(self):
        super().reset()
        self.planner = None

    def act(self, v):
        if getattr(self, 'planner', None) is None or self.planner.m is not v.map:
            self.planner = DStarLite(v.map, v.rules, v.rover, v.belief)
        else:
            self.planner.observe(v.rover, v.belief)
        before = self.planner.expanded
        self.planner.compute()
        self.nodes += self.planner.expanded - before
        return path_action(v, self.planner.next_tile())


class MinimaxRover(Agent):
    """R3 anticipatory rover: alpha-beta over its own actions and the alien's
    next strike, on the belief map. Leaf value = fuel left after the cheapest
    remaining route (higher is better for the rover)."""
    name = 'Minimax'
    side = ROVER

    def __init__(self, depth=4, alien_branch=6):
        super().__init__()
        self.depth = depth
        self.alien_branch = alien_branch

    def act(self, v):
        cache, counter = {}, [0]
        root = Sim.from_view(v, cache, counter)
        best, best_val = ('wait', None), -INF
        alpha = -INF
        for a in self._ordered(root):
            child = root.clone()
            child.apply_rover(a)
            val = self._value(child, self.depth - 1, alpha, INF)
            if val > best_val:
                best, best_val = a, val
            alpha = max(alpha, val)
        self.nodes += counter[0]
        self.info = f'look-ahead value {best_val:.0f}'
        return best

    def _leaf(self, s):
        if s.result == ROVER or s.pos == s.m.goal:
            return 1000 + s.fuel
        d = s.route_cost()
        if s.result is not None or d == INF or d > s.fuel:
            return -1000 + (s.fuel - (d if d != INF else 100))
        return s.fuel - d

    def _ordered(self, s):
        d = s.dist()
        return sorted(s.rover_moves(), key=lambda a: (
            s.r.enter_cost(s.blk.get(a[1], OPEN)) + d.get(a[1], INF) if a[1] else INF))

    def _alien_options(self, s):
        opts = [None]
        for p in s.path_ahead(10):
            if not s.can_target(p):
                continue
            b = s.blk.get(p, OPEN)
            opts.append(('boulder', p) if b == OPEN else ('debris', p))
            if b == OPEN:
                opts.append(('debris', p))
            if len(opts) > self.alien_branch:
                break
        return opts

    def _value(self, s, depth, alpha, beta):
        self.nodes += 1
        if s.result is not None or depth <= 0:
            return self._leaf(s)
        if s.turn == ROVER:
            val = -INF
            for a in self._ordered(s):
                c = s.clone()
                c.apply_rover(a)
                val = max(val, self._value(c, depth - 1, alpha, beta))
                alpha = max(alpha, val)
                if alpha >= beta:
                    break
            return val
        val = INF
        for a in self._alien_options(s):
            c = s.clone()
            c.apply_alien(a)
            val = min(val, self._value(c, depth, alpha, beta))
            beta = min(beta, val)
            if alpha >= beta:
                break
        return val


# --------------------------------------------------------------------- fuzzy
try:
    import skfuzzy as fuzz
except ImportError:  # small stand-in so the game still runs without scipy
    class fuzz:
        @staticmethod
        def trimf(x, abc):
            a, b, c = abc
            y = np.zeros_like(x, dtype=float)
            if b > a:
                y = np.where((x >= a) & (x <= b), (x - a) / (b - a), y)
            if c > b:
                y = np.where((x >= b) & (x <= c), (c - x) / (c - b), y)
            y[x == b] = 1.0
            return y

        @staticmethod
        def interp_membership(x, mf, v):
            return float(np.interp(v, x, mf))

        @staticmethod
        def defuzz(x, mf, mode):
            s = mf.sum()
            return float((x * mf).sum() / s) if s > 0 else 0.5


class FuzzyRover(Agent):
    """R5: Mamdani fuzzy controller sets a caution level in [0, 1] from
    fuel margin, detour length and incoming debris. A* then plans with
    extra cost on soft boulders (they may harden) and debris landing zones,
    scaled by caution: so the same planner sprints, detours or dodges."""
    name = 'Fuzzy'
    side = ROVER

    SOFT_FEAR = 6     # max extra cost on a soft boulder
    DEBRIS_FEAR = 15  # max extra cost on a debris landing tile

    def __init__(self):
        super().__init__()
        self.xm = np.linspace(0, 1, 101)      # fuel margin (slack / fuel)
        self.xd = np.linspace(0, 20, 101)     # detour length (fuel)
        self.xn = np.linspace(0, 10, 101)     # distance to incoming debris on route
        self.xc = np.linspace(0, 1, 101)      # caution
        t = fuzz.trimf
        self.margin = {'low': t(self.xm, [0, 0, 0.25]), 'mid': t(self.xm, [0.1, 0.3, 0.5]),
                       'high': t(self.xm, [0.35, 1, 1])}
        self.detour = {'short': t(self.xd, [0, 0, 6]), 'mid': t(self.xd, [3, 8, 13]),
                       'long': t(self.xd, [10, 20, 20])}
        self.near = {'near': t(self.xn, [0, 0, 3]), 'far': t(self.xn, [2, 10, 10])}
        self.caution = {'low': t(self.xc, [0, 0, 0.4]), 'mid': t(self.xc, [0.2, 0.5, 0.8]),
                        'high': t(self.xc, [0.6, 1, 1])}

    def decide_caution(self, margin, detour, near):
        mu = lambda sets, x, v: {k: fuzz.interp_membership(x, mf, v) for k, mf in sets.items()}
        m = mu(self.margin, self.xm, min(max(margin, 0), 1))
        d = mu(self.detour, self.xd, min(detour, 20))
        n = mu(self.near, self.xn, min(near, 10))
        rules = [
            (m['low'], 'low'),                           # no slack: sprint
            (min(m['high'], d['short']), 'high'),         # cheap to play safe
            (min(m['mid'], d['mid']), 'mid'),
            (min(m['mid'], d['short']), 'mid'),
            (d['long'], 'low'),                           # detour too long: shoot through
            (n['near'], 'high'),                          # debris about to land on route
            (min(m['high'], n['far'], d['mid']), 'mid'),
        ]
        agg = np.zeros_like(self.xc)
        for w, out in rules:
            agg = np.maximum(agg, np.minimum(w, self.caution[out]))
        if agg.sum() == 0:
            return 0.0
        return float(fuzz.defuzz(self.xc, agg, 'centroid'))

    def act(self, v):
        c = [0]
        cost, path = astar(v.map, v.belief, v.rover, v.map.goal, v.rules, counter=c)
        softs = {p for p, b in v.belief.items() if b == SOFT}
        avoid = {p: 1000 for p in softs}
        cost_avoid, _ = astar(v.map, v.belief, v.rover, v.map.goal, v.rules, extra=avoid, counter=c)
        detour = 20 if cost_avoid >= 1000 or cost == INF else cost_avoid - cost
        on_route = set(path)
        near = min((manhattan(v.rover, p) for p in v.pending
                    if p in on_route or any(q in on_route for q in v.map.nbrs.get(p, ()))), default=10)
        margin = (v.fuel - cost) / max(v.fuel, 1) if cost != INF else 0
        k = self.decide_caution(margin, detour, near)
        extra = {p: k * self.SOFT_FEAR for p in softs}
        for p, ttl in v.pending.items():
            extra[p] = extra.get(p, 0) + k * self.DEBRIS_FEAR * (2 if ttl == 1 else 1)
        _, path = astar(v.map, v.belief, v.rover, v.map.goal, v.rules, extra=extra, counter=c)
        self.nodes += c[0]
        self.info = f'caution {k:.2f}'
        return path_action(v, path[1] if len(path) > 1 else None)


ROVERS = [FloodFillRover, AStarRover, DStarLiteRover, MinimaxRover, FuzzyRover]
