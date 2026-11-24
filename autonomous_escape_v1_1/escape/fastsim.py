"""A lean copy of the rules for search (minimax, MCTS).

Same rules as GameState, minus logging and fog bookkeeping. Route costs come
from a reverse-Dijkstra map cached per boulder layout, so a rollout step is
O(1) until the layout changes. tests/test_rules.py checks it against
GameState move by move.
"""
from .config import OPEN, SOFT, HARD, INF
from .graph import cost_to_go, manhattan

ROVER, ALIEN = 'rover', 'alien'


class Sim:
    __slots__ = ('m', 'r', 'pos', 'fuel', 'tick', 'budget', 'blk', 'pending',
                 'result', 'turn', 'cache', 'counter')

    def __init__(self, m, rules, pos, fuel, tick, budget, blk, pending, turn=ROVER,
                 cache=None, counter=None):
        self.m, self.r = m, rules
        self.pos, self.fuel, self.tick, self.budget = pos, fuel, tick, budget
        self.blk, self.pending = blk, pending
        self.result = None
        self.turn = turn
        self.cache = {} if cache is None else cache
        self.counter = counter if counter is not None else [0]

    @classmethod
    def from_state(cls, s, cache=None, counter=None):
        return cls(s.map, s.rules, s.rover, s.fuel, s.tick, s.budget, dict(s.blk),
                   dict(s.pending), s.turn, cache, counter)

    @classmethod
    def from_view(cls, v, cache=None, counter=None):
        return cls(v.map, v.rules, v.rover, v.fuel, v.tick, v.budget, dict(v.belief),
                   dict(v.pending), ROVER, cache, counter)

    def clone(self):
        s = Sim(self.m, self.r, self.pos, self.fuel, self.tick, self.budget,
                dict(self.blk), dict(self.pending), self.turn, self.cache, self.counter)
        s.result = self.result
        return s

    def dist(self):
        key = frozenset(self.blk.items())
        d = self.cache.get(key)
        if d is None:
            d = cost_to_go(self.m, self.blk, self.m.goal, self.r, counter=self.counter)
            self.cache[key] = d
        return d

    def route_cost(self):
        return self.dist().get(self.pos, INF)

    # rover ------------------------------------------------------------------
    def rover_moves(self):
        acts = []
        for v in self.m.nbrs[self.pos]:
            b = self.blk.get(v, OPEN)
            if b == OPEN:
                acts.append(('move', v))
            elif b == SOFT:
                acts.append(('shoot', v))
        acts.append(('wait', None))
        return acts

    def apply_rover(self, a):
        kind, t = a
        if kind == 'move':
            self.pos = t
            self.fuel -= self.r.move_cost
        elif kind == 'shoot':
            del self.blk[t]
            self.fuel -= self.r.shot_cost
        else:
            self.fuel -= self.r.move_cost
        self.tick += 1
        for p in sorted(self.pending):
            self.pending[p] -= 1
            if self.pending[p] > 0:
                continue
            del self.pending[p]
            if p == self.pos:
                self.fuel -= self.r.debris_damage
            elif self.blk.get(p, OPEN) == SOFT:
                self.blk[p] = HARD
            else:
                self.blk[p] = SOFT
        if not self.check_end() and self.budget > 0 and self.tick % self.r.alien_every == 0:
            self.turn = ALIEN

    # alien ------------------------------------------------------------------
    def can_target(self, p):
        return (not self.m.protected(p) and manhattan(p, self.pos) > 1
                and p not in self.pending and self.blk.get(p, OPEN) != HARD)

    def apply_alien(self, a):
        if a is not None:
            kind, p = a
            self.budget -= 1
            if kind == 'boulder':
                self.blk[p] = SOFT
            else:
                self.pending[p] = self.r.debris_delay
        self.turn = ROVER
        self.check_end()

    def check_end(self):
        if self.pos == self.m.goal:
            self.result = ROVER
        elif self.fuel <= 0:
            self.result = ALIEN
        else:
            c = self.route_cost()
            if c == INF or c > self.fuel:
                self.result = ALIEN
        return self.result is not None

    def greedy_rover_action(self):
        """Step down the cost-to-go map (shoot if the next tile is soft)."""
        d = self.dist()
        best, best_v = ('wait', None), INF
        for v in self.m.nbrs[self.pos]:
            b = self.blk.get(v, OPEN)
            if b == HARD:
                continue
            val = self.r.enter_cost(b) + d.get(v, INF)
            if val < best_v:
                best_v = val
                best = ('shoot', v) if b == SOFT else ('move', v)
        return best

    def path_ahead(self, limit=12):
        """The rover's cheapest route from here (true layout), as tiles."""
        d = self.dist()
        out, p, seen = [], self.pos, {self.pos}
        while p != self.m.goal and len(out) < limit:
            nxt = min((v for v in self.m.nbrs[p] if v not in seen),
                      key=lambda v: self.r.enter_cost(self.blk.get(v, OPEN)) + d.get(v, INF),
                      default=None)
            if nxt is None or d.get(nxt, INF) == INF:
                break
            out.append(nxt)
            seen.add(nxt)
            p = nxt
        return out
