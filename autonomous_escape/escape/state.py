"""Game state and rules engine.

Turn order (one "tick" = one rover action):
  1. rover acts: move (1 fuel), wait (1 fuel) or shoot an adjacent soft
     boulder (shot_cost fuel; the boulder is destroyed, the rover stays put)
  2. incoming debris counts down; on landing it hardens a soft boulder,
     makes a soft boulder on an open tile, or costs the rover debris_damage
     fuel if it is standing there
  3. every `alien_every` ticks, while budget lasts, the alien may act once:
     drop a boulder on an open tile, or fling debris at an open/soft tile
The game ends when the rover reaches the pad (rover wins), runs out of fuel,
or can no longer afford any route to the pad (alien wins).

What the rover knows: the ridge walls (orbital map), boulders inside its
circular radar (remembered afterwards), and all incoming debris. Phobos has no
air, so a glowing debris streak is visible anywhere in the sky, and the rover
sees where it lands. Boulder drops outside the radar stay hidden.
"""
from types import SimpleNamespace

from .config import OPEN, SOFT, HARD, INF, DEFAULT
from .graph import astar, manhattan, in_radar
from . import mapgen

ROVER, ALIEN = 'rover', 'alien'


class GameState:
    def __init__(self, gmap, rules=DEFAULT):
        self.map = gmap
        self.rules = rules
        self.rover = gmap.start
        self.blk = {}          # true boulders: tile -> SOFT/HARD
        self.pending = {}      # incoming debris: tile -> turns until landing
        self.belief = {}       # rover memory of boulders: tile -> SOFT/HARD
        self.seen_pending = {}
        L, _ = astar(gmap, {}, gmap.start, gmap.goal, rules)
        self.shortest = L
        self.fuel0 = self.fuel = round(rules.fuel_mult * L)
        self.budget = rules.alien_budget
        self.tick = 0
        self.turn = ROVER      # whose decision is next
        self.shots = 0
        self.hits = 0
        self.alien_moves = 0
        self.result = None     # None, ROVER or ALIEN
        self.reason = ''
        self.trail = [self.rover]
        self.log = []
        self.sense()

    @classmethod
    def new(cls, seed, rules=DEFAULT):
        return cls(mapgen.generate(seed, rules), rules)

    # ---------------------------------------------------------------- views
    def clone(self):
        s = object.__new__(GameState)
        s.__dict__.update(self.__dict__)
        s.blk = dict(self.blk)
        s.pending = dict(self.pending)
        s.belief = dict(self.belief)
        s.seen_pending = dict(self.seen_pending)
        s.trail = list(self.trail)
        s.log = []
        return s

    def in_radar(self, p):
        return in_radar(p, self.rover, self.rules.radar)

    def sense(self):
        """Refresh the rover's belief inside its radar."""
        R = self.rules.radar
        r0, c0 = self.rover
        for r in range(r0 - R, r0 + R + 1):
            for c in range(c0 - R, c0 + R + 1):
                p = (r, c)
                if not in_radar(p, self.rover, R):
                    continue
                b = self.blk.get(p, OPEN)
                if b:
                    self.belief[p] = b
                else:
                    self.belief.pop(p, None)
        self.seen_pending = dict(self.pending)   # debris streaks are visible anywhere

    def rover_view(self):
        """What the rover is allowed to know: walls (orbital map), its own
        memory of boulders, all incoming debris, and public counters."""
        return SimpleNamespace(
            map=self.map, rules=self.rules, rover=self.rover, fuel=self.fuel,
            fuel0=self.fuel0, belief=dict(self.belief), pending=dict(self.seen_pending),
            tick=self.tick, budget=self.budget, shots=self.shots)

    # ---------------------------------------------------------------- rover
    def rover_actions(self):
        acts = []
        for v in self.map.nbrs[self.rover]:
            b = self.blk.get(v, OPEN)
            if b == OPEN:
                acts.append(('move', v))
            elif b == SOFT:
                acts.append(('shoot', v))
        acts.append(('wait', None))
        return acts

    def apply_rover(self, action):
        assert self.result is None and self.turn == ROVER, 'not the rover turn'
        kind, tgt = action
        if action not in self.rover_actions():
            kind, tgt = 'wait', None   # illegal actions become a wait
        if kind == 'move':
            self.rover = tgt
            self.fuel -= self.rules.move_cost
            self.trail.append(tgt)
        elif kind == 'shoot':
            del self.blk[tgt]
            self.fuel -= self.rules.shot_cost
            self.shots += 1
            self._event(f'Laser destroys boulder at {tgt}')
        else:
            self.fuel -= self.rules.move_cost
        self.tick += 1
        self._land_debris()
        self.sense()
        if self._check_end():
            return
        if self.budget > 0 and self.tick % self.rules.alien_every == 0:
            self.turn = ALIEN

    def _land_debris(self):
        for p in sorted(self.pending):
            self.pending[p] -= 1
            if self.pending[p] > 0:
                continue
            del self.pending[p]
            if p == self.rover:
                self.fuel -= self.rules.debris_damage
                self.hits += 1
                self._event(f'Debris hits the rover! -{self.rules.debris_damage} fuel')
            elif self.blk.get(p, OPEN) == SOFT:
                self.blk[p] = HARD
                self._event(f'Debris hardens boulder at {p}')
            else:
                self.blk[p] = SOFT
                self._event(f'Debris forms a boulder at {p}')
            if p in self.blk:
                self.belief[p] = self.blk[p]       # the rover watched it land

    # ---------------------------------------------------------------- alien
    def alien_can_target(self, p):
        """Tile rules shared by both alien actions."""
        return (self.map.is_open(p) and not self.map.protected(p)
                and manhattan(p, self.rover) > 1 and p not in self.pending
                and self.blk.get(p, OPEN) != HARD)

    def legal_alien(self, action):
        if action is None:
            return True
        kind, p = action
        if not self.alien_can_target(p):
            return False
        if kind == 'boulder':
            return self.blk.get(p, OPEN) == OPEN
        return kind == 'debris'

    def alien_actions(self):
        acts = []
        for p in self.map.open_tiles:
            if self.alien_can_target(p):
                if self.blk.get(p, OPEN) == OPEN:
                    acts.append(('boulder', p))
                acts.append(('debris', p))
        return acts

    def apply_alien(self, action):
        """action: ('boulder', tile), ('debris', tile) or None to hold."""
        assert self.result is None and self.turn == ALIEN, 'not the alien turn'
        if action is not None and self.legal_alien(action):
            kind, p = action
            self.budget -= 1
            self.alien_moves += 1
            if kind == 'boulder':
                self.blk[p] = SOFT
                self._event(f'Alien drops a boulder at {p}')
            else:
                self.pending[p] = self.rules.debris_delay
                self._event(f'Alien flings debris at {p}')
        self.turn = ROVER
        self.sense()
        self._check_end()

    # ---------------------------------------------------------------- end
    def route_cost(self):
        """True cheapest fuel cost from the rover to the pad."""
        c, _ = astar(self.map, self.blk, self.rover, self.map.goal, self.rules)
        return c

    def _check_end(self):
        if self.rover == self.map.goal:
            self.result, self.reason = ROVER, 'Reached the launch pad'
        elif self.fuel <= 0:
            self.result, self.reason = ALIEN, 'Out of fuel'
        else:
            c = self.route_cost()
            if c == INF:
                self.result, self.reason = ALIEN, 'Every route sealed'
            elif c > self.fuel:
                self.result, self.reason = ALIEN, 'Not enough fuel for any route'
        if self.result:
            self._event(self.reason)
        return self.result is not None

    @property
    def score(self):
        """Escape score: 100 x fuel left / starting fuel, 0 if the rover lost."""
        if self.result != ROVER:
            return 0.0
        return 100.0 * max(self.fuel, 0) / self.fuel0

    def _event(self, text):
        self.log.append((self.tick, text))


def step(state, rover_agent, alien_agent):
    """Advance one decision using AI agents. Returns the action taken."""
    if state.result:
        return None
    if state.turn == ALIEN:
        a = alien_agent.act(state)
        state.apply_alien(a)
    else:
        a = rover_agent.act(state.rover_view())
        state.apply_rover(a)
    return a
