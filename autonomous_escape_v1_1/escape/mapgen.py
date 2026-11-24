"""Phobos map generator: an open debris field. Impassable tiles are rock
fields and deep craters grown by a cellular automaton, so the gaps between
them wind naturally like the concept art (no visible maze). The launch pad
sits on a cleared crater floor. Every map is checked for at least
`rules.min_routes` vertex-disjoint routes from the base to the pad, and for
a few real choke points so the alien has something to fight over."""
import math
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


def _field(n, rng, fill, cluster=1):
    """Rock fields: smooth random noise, the densest `fill` share becomes rock.
    cluster = blur passes: 0 scattered single boulders, 1 clumps, 3 big islands."""
    a = [[rng.random() for _ in range(n)] for _ in range(n)]
    for _ in range(cluster):
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


def _barrier(n, cells, centres, half, apron=3):
    """Wall cells with gaps around the given centre indices, and an open apron
    on both sides of each gap so every gap is a real passage."""
    wall, gaps = set(), set()
    for i, (r, c) in enumerate(cells):
        hole = any(abs(i - g) <= half for g in centres)
        for dr, dc in ((0, 0), (0, 1), (1, 0)):
            q = (r + dr, c + dc)
            if 0 <= q[0] < n and 0 <= q[1] < n:
                (gaps if hole else wall).add(q)
    for g in centres:
        gr, gc = cells[min(g, len(cells) - 1)]
        for dr in range(-apron, apron + 1):
            for dc in range(-apron, apron + 1):
                q = (gr + dr, gc + dc)
                if 0 <= q[0] < n and 0 <= q[1] < n:
                    gaps.add(q)
    return wall, gaps


def _rings(n, rng, count, goal):
    """Concentric crater rims in mid-map, each broken by a few gaps at random angles."""
    wall, gaps = set(), set()
    gr, gc = n // 2 + rng.randint(-3, 3), n // 2 + rng.randint(-3, 3)      # a crater field in mid-map
    lo, hi = 5, int(n * 0.45)
    radii = [round(lo + (hi - lo) * (k + rng.uniform(0.3, 0.7)) / count) for k in range(count)] if count else []
    for rad in radii:
        cells, seen = [], set()
        steps = int(2 * math.pi * rad * 2)
        for k in range(steps):
            a = 2 * math.pi * k / steps
            q = (round(gr - rad * math.sin(a)), round(gc - rad * math.cos(a)))
            if 0 <= q[0] < n and 0 <= q[1] < n and q not in seen:
                seen.add(q)
                cells.append(q)
        if len(cells) < 12:
            continue
        L = len(cells)
        k_gaps = max(2, L // 28)                       # gaps scale with the visible arc
        centres = [int(L * (k + rng.uniform(0.25, 0.75)) / k_gaps) for k in range(k_gaps)]
        w, g = _barrier(n, cells, centres, 3, apron=2)
        wall |= w
        gaps |= g
    return wall - gaps, gaps


def _bars(n, rng, count):
    """Long straight walls across the field (grooves), gaps staggered row to row."""
    wall, gaps = set(), set()
    for k in range(count):
        r = round((k + 1) * n / (count + 1))
        cells = [(r, c) for c in range(n)]
        side = k % 2
        centres = [int(n * rng.uniform(0.05, 0.25)) if side == 0 else int(n * rng.uniform(0.75, 0.95)),
                   int(n * rng.uniform(0.4, 0.6))]
        w, g = _barrier(n, cells, centres, 3)
        wall |= w
        gaps |= g
    return wall - gaps, gaps


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
    attempts = 0
    while True:
        attempts += 1
        walls = _field(n, rng, rules.rock_fill, rules.cluster)
        if rules.style == 'rings':
            ridge, gaps = _rings(n, rng, rules.ridges, goal)
        elif rules.style == 'bars':
            ridge, gaps = _bars(n, rng, rules.ridges)
        else:
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
        relax = attempts > 40                     # never loop forever on a hard style
        lo = 3 if relax else rules.min_routes
        open_style = rules.max_routes >= 30            # open styles: just make sure there are enough
        routes = disjoint_routes(gmap, lo if open_style else rules.max_routes + 1)
        if not lo <= routes <= (99 if relax or open_style else rules.max_routes):
            continue
        L, _ = astar(gmap, {}, start, goal, rules)
        if L >= (0 if relax else rules.min_length):
            return gmap


# ------------------------------------------------------------------ fully random complex maps
FEATURES = ('rings', 'bars', 'ridges', 'islands', 'scatter', 'plazas')


def _maze(n, rng, loops):
    """Recursive-backtracker maze on the odd cells (1-wide winding corridors and
    dead ends), then a share of the inner walls knocked out to make side loops."""
    walls = [[True] * n for _ in range(n)]
    sr, sc = 1 + 2 * rng.randrange((n - 1) // 2), 1 + 2 * rng.randrange((n - 1) // 2)
    walls[sr][sc] = False
    stack, seen = [(sr, sc)], {(sr, sc)}
    while stack:
        r, c = stack[-1]
        opts = [(r + 2 * dr, c + 2 * dc, dr, dc) for dr, dc in DIRS
                if 0 < r + 2 * dr < n - 1 and 0 < c + 2 * dc < n - 1 and (r + 2 * dr, c + 2 * dc) not in seen]
        if not opts:
            stack.pop()
            continue
        nr, nc, dr, dc = rng.choice(opts)
        walls[r + dr][c + dc] = walls[nr][nc] = False
        seen.add((nr, nc))
        stack.append((nr, nc))
    for r in range(1, n - 1):
        for c in range(1, n - 1):
            if walls[r][c] and (r % 2) != (c % 2) and rng.random() < loops:
                walls[r][c] = False
    return walls


def _endpoints(n, rng):
    """Base and pad on random opposite sides (or opposite corners)."""
    lo, hi = 2, n - 3
    side = rng.choice(['lr', 'tb', 'diag', 'anti'])
    if side == 'lr':
        a, b = (rng.randint(lo, hi), 2), (rng.randint(lo, hi), n - 3)
    elif side == 'tb':
        a, b = (2, rng.randint(lo, hi)), (n - 3, rng.randint(lo, hi))
    elif side == 'diag':
        a, b = (rng.randint(2, 6), rng.randint(2, 6)), (rng.randint(n - 7, n - 3), rng.randint(n - 7, n - 3))
    else:
        a, b = (rng.randint(2, 6), rng.randint(n - 7, n - 3)), (rng.randint(n - 7, n - 3), rng.randint(2, 6))
    return (a, b) if rng.random() < 0.5 else (b, a)


def _overlay(walls, n, rng, feature, start, goal):
    """Lay one feature over the maze. Barriers (rings, bars, ridges) come with
    gaps; islands and scatter add rock; plazas open up wide clearings."""
    if feature == 'rings':
        wall, gaps = _rings(n, rng, rng.randint(2, 4), goal)
    elif feature == 'bars':
        wall, gaps = _bars(n, rng, rng.randint(2, 4))
    elif feature == 'ridges':
        wall, gaps = _ridges(n, rng, rng.randint(2, 4))
    else:
        wall, gaps = set(), set()
        if feature == 'islands':
            for _ in range(rng.randint(3, 7)):
                cr, cc, rad = rng.randrange(n), rng.randrange(n), rng.uniform(2, 4.5)
                for r in range(max(0, int(cr - rad)), min(n, int(cr + rad) + 1)):
                    for c in range(max(0, int(cc - rad)), min(n, int(cc + rad) + 1)):
                        if (r - cr) ** 2 + (c - cc) ** 2 <= rad * rad * rng.uniform(0.7, 1.1):
                            wall.add((r, c))
        elif feature == 'scatter':
            for r in range(n):
                for c in range(n):
                    if not walls[r][c] and rng.random() < 0.06:
                        wall.add((r, c))
        elif feature == 'plazas':
            for _ in range(rng.randint(3, 6)):
                cr, cc = rng.randrange(n), rng.randrange(n)
                h, w = rng.randint(3, 7), rng.randint(3, 7)
                for r in range(max(1, cr - h // 2), min(n - 1, cr + h // 2 + 1)):
                    for c in range(max(1, cc - w // 2), min(n - 1, cc + w // 2 + 1)):
                        gaps.add((r, c))
    for r, c in wall:
        walls[r][c] = True
    for r, c in gaps:
        walls[r][c] = False


def generate_random(game_seed, min_routes=3, detour=1.1, attempts=150):
    """A brand-new complex map for one game: random size, random base and pad
    positions, a maze with side loops, and 2-3 random features on top.
    Rejected unless the shortest route is at least `detour` x the straight
    distance (a real maze), it has plenty of dead ends and junctions, and at
    least `min_routes` independent routes keep it fair."""
    from .graph import astar, disjoint_routes, manhattan
    from .config import DEFAULT
    rng = random.Random(game_seed)
    best = None
    for k in range(attempts):
        n = rng.choice(range(41, 62, 2))
        start, goal = _endpoints(n, rng)
        # a debris field: clustered rock with winding channels carved between base and pad,
        # plus one or two random barrier features (rings, walls, ridges) or islands / scatter
        walls = _field(n, rng, rng.uniform(0.42, 0.55), rng.choice((1, 1, 2)))
        feats = rng.sample(('rings', 'bars', 'ridges', 'islands', 'scatter'), rng.randint(1, 2))
        blocked = set()
        for f in feats:
            if f in ('rings', 'bars', 'ridges'):
                wall, gaps = (_rings(n, rng, rng.randint(2, 4), goal) if f == 'rings' else
                              _bars(n, rng, rng.randint(2, 4)) if f == 'bars' else _ridges(n, rng, rng.randint(2, 4)))
                for r, c in wall:
                    walls[r][c] = True
                for r, c in gaps:
                    walls[r][c] = False
                blocked |= wall
            else:
                _overlay(walls, n, rng, f, start, goal)
        _carve(walls, rng, start, goal, rng.randint(5, 9), rng.randint(0, 2), blocked)
        for r in range(n):                                   # solid outer rim
            walls[r][0] = walls[r][n - 1] = True
            walls[0][r] = walls[n - 1][r] = True
        R = DEFAULT.protect_radius
        for centre in (start, goal):                         # rough landing clearings, wider than
            cr = R + rng.uniform(2.0, 3.5)                    # the no-strike zone, so several corridors
            for r in range(centre[0] - 6, centre[0] + 7):    # lead in and out
                for c in range(centre[1] - 6, centre[1] + 7):
                    if 0 < r < n - 1 and 0 < c < n - 1 and                             (r - centre[0]) ** 2 + (c - centre[1]) ** 2 <= cr * cr * rng.uniform(0.8, 1.15):
                        walls[r][c] = False
        _keep_reachable(walls, start)
        if walls[goal[0]][goal[1]]:
            continue
        gmap = GameMap(n, walls, start, goal, game_seed, R)
        gmap.features = feats
        L, _ = astar(gmap, {}, start, goal, DEFAULT)
        straight = manhattan(start, goal)
        deg = [len(gmap.nbrs[p]) for p in gmap.open_tiles]
        dead = sum(d == 1 for d in deg)
        routes = disjoint_routes(gmap, min_routes)
        score = (L / straight, routes, dead)
        ok = L >= detour * straight and routes >= min_routes
        if ok:
            return gmap
        if routes >= 2 and (best is None or score > best[0]):
            best = (score, gmap)
    if best:
        return best[1]
    gmap = generate(game_seed)
    gmap.features = ['ridges']
    return gmap
