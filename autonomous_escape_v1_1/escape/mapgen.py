"""Phobos map generator: an open debris field. Impassable tiles are rock
fields and deep craters grown by a cellular automaton, so the gaps between
them wind naturally like the concept art (no visible maze). The launch pad
sits on a cleared crater floor. Every map is checked for at least
`rules.min_routes` vertex-disjoint routes from the base to the pad, and for
a few real choke points so the alien has something to fight over."""
import random
from dataclasses import dataclass, field

from .config import DEFAULT

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass
class GameMap:
    size: int
    walls: list            # walls[r][c] -> True for impassable rock / crater
    start: tuple
    goal: tuple
    seed: int
    protect: int
    nbrs: dict = field(default_factory=dict)
    open_tiles: list = field(default_factory=list)

    def __post_init__(self):
        n = self.size
        self.open_tiles = [(r, c) for r in range(n) for c in range(n) if not self.walls[r][c]]
        for p in self.open_tiles:
            r, c = p
            self.nbrs[p] = [(r + dr, c + dc) for dr, dc in DIRS
                            if 0 <= r + dr < n and 0 <= c + dc < n and not self.walls[r + dr][c + dc]]

    def is_open(self, p):
        r, c = p
        return 0 <= r < self.size and 0 <= c < self.size and not self.walls[r][c]

    def in_start_zone(self, p):
        return max(abs(p[0] - self.start[0]), abs(p[1] - self.start[1])) <= self.protect

    def in_goal_zone(self, p):
        return max(abs(p[0] - self.goal[0]), abs(p[1] - self.goal[1])) <= self.protect

    def protected(self, p):
        return self.in_start_zone(p) or self.in_goal_zone(p)


def _blur(a):
    """3x3 box blur with edge padding (pure python, tiny grids)."""
    n = len(a)
    out = [[0.0] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            t = k = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = min(n - 1, max(0, r + dr)), min(n - 1, max(0, c + dc))
                    t += a[rr][cc]
                    k += 1
            out[r][c] = t / k
    return out


def _field(n, rng, fill):
    """Rock fields: smooth random noise, the densest `fill` share becomes rock."""
    a = [[rng.random() for _ in range(n)] for _ in range(n)]
    a = _blur(a)
    vals = sorted(v for row in a for v in row)
    cut = vals[int((1 - fill) * len(vals))]
    return [[v >= cut for v in row] for row in a]


def _ridges(n, rng, count):
    """Long rock ridges across the start-to-pad diagonal, each with two gaps.
    The gaps alternate between the ends and the middle, so every route has to
    switch back and forth across the field instead of cutting straight over."""
    if not count:
        return set(), set()
    ridge, gaps = set(), set()
    lo, hi = 6, 2 * (n - 1) - 10
    for k in range(count):
        s = round(lo + (hi - lo) * (k + 1) / (count + 1))
        cells = [(r, s - r) for r in range(n) if 0 <= s - r < n]
        L = len(cells)
        if k % 2 == 0:                       # both gaps at one end ...
            centres = [int(L * rng.uniform(0.04, 0.1)), int(L * rng.uniform(0.16, 0.24))]
        else:                                # ... then the other: a serpentine
            centres = [int(L * rng.uniform(0.76, 0.84)), int(L * rng.uniform(0.9, 0.96))]
        for i, (r, c) in enumerate(cells):
            hole = any(abs(i - g) <= 4 for g in centres)
            for dr, dc in ((0, 0), (0, 1), (1, 0)):
                q = (r + dr, c + dc)
                if 0 <= q[0] < n and 0 <= q[1] < n:
                    (gaps if hole else ridge).add(q)
        for g in centres:                        # clear an apron so each gap is a real opening
            gr, gc = cells[min(g, L - 1)]
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    q = (gr + dr, gc + dc)
                    if 0 <= q[0] < n and 0 <= q[1] < n and abs(dr + dc) >= 2:
                        gaps.add(q)
    return ridge - gaps, gaps


def _carve(walls, rng, start, goal, channels, waypoints=0, blocked=frozenset()):
    """Wind channels from the base to the pad through the rock, each one
    avoiding the earlier ones. With waypoints, every channel detours through
    random points first, so the corridors meander across the whole field."""
    import heapq
    n = len(walls)
    used = set()

    def path(a, b, w):
        dist, prev, pq = {a: 0}, {}, [(0, a)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == b:
                break
            if d > dist[u]:
                continue
            for dr, dc in DIRS:
                v = (u[0] + dr, u[1] + dc)
                if 0 <= v[0] < n and 0 <= v[1] < n and d + w[v] < dist.get(v, 1e9):
                    dist[v], prev[v] = d + w[v], u
                    heapq.heappush(pq, (d + w[v], v))
        out, v = [], b
        while v != a:
            out.append(v)
            v = prev[v]
        return out

    for _ in range(channels):
        w = {(r, c): (1e6 if (r, c) in blocked else
                      1 + 5 * rng.random() + (4 if walls[r][c] else 0) + (9 if (r, c) in used else 0))
             for r in range(n) for c in range(n)}
        stops = [start] + [(rng.randrange(2, n - 2), rng.randrange(2, n - 2)) for _ in range(waypoints)] + [goal]
        for a, b in zip(stops, stops[1:]):
            for v in path(a, b, w):
                walls[v[0]][v[1]] = False
                used.add(v)


def _keep_reachable(walls, start):
    """Fill open pockets the rover could never reach (they render as rock)."""
    n = len(walls)
    seen, stack = {start}, [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS:
            q = (r + dr, c + dc)
            if 0 <= q[0] < n and 0 <= q[1] < n and not walls[q[0]][q[1]] and q not in seen:
                seen.add(q)
                stack.append(q)
    for r in range(n):
        for c in range(n):
            if (r, c) not in seen:
                walls[r][c] = True


def generate(seed, rules=DEFAULT):
    from .graph import disjoint_routes, astar
    n = rules.size
    assert n % 2 == 1 and n >= 11, 'size must be odd and >= 11'
    rng = random.Random(seed)
    start = (1, 1)
    goal = (n - 4, n - 4)
    R = rules.protect_radius
    while True:
        walls = _field(n, rng, rules.rock_fill)
        ridge, gaps = _ridges(n, rng, rules.ridges)
        for r, c in ridge:
            walls[r][c] = True
        for r, c in gaps:
            walls[r][c] = False
        _carve(walls, rng, start, goal, rules.channels, rules.waypoints, ridge)
        for centre in (start, goal):              # base clearing and crater floor
            for r in range(centre[0] - R, centre[0] + R + 1):
                for c in range(centre[1] - R, centre[1] + R + 1):
                    if 0 <= r < n and 0 <= c < n:
                        walls[r][c] = False
        _keep_reachable(walls, start)
        if walls[goal[0]][goal[1]]:
            continue
        gmap = GameMap(n, walls, start, goal, seed, R)
        routes = disjoint_routes(gmap, rules.max_routes + 1)
        if not rules.min_routes <= routes <= rules.max_routes:
            continue
        L, _ = astar(gmap, {}, start, goal, rules)
        if L >= rules.min_length:
            return gmap
