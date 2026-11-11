"""Search primitives on the Phobos grid.

Costs are "enter" costs: stepping onto a tile costs Rules.enter_cost(state of
that tile), so a soft boulder costs a shot plus a move and a hard one is a wall.
"""
import heapq
from collections import deque

from .config import INF, HARD


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def in_radar(p, centre, radius):
    """Circular sensor footprint: tile centres within `radius` tiles."""
    return (p[0] - centre[0]) ** 2 + (p[1] - centre[1]) ** 2 <= radius * radius


def astar(gmap, blk, src, dst, rules, extra=None, counter=None):
    """Cheapest route src -> dst. Returns (cost, path). path includes both ends."""
    if src == dst:
        return 0, [src]
    g = {src: 0}
    prev = {}
    pq = [(manhattan(src, dst), 0, src)]
    expanded = 0
    while pq:
        f, d, u = heapq.heappop(pq)
        if d > g[u]:
            continue
        expanded += 1
        if u == dst:
            break
        for v in gmap.nbrs[u]:
            w = rules.enter_cost(blk.get(v, 0))
            if w == INF:
                continue
            if extra:
                w += extra.get(v, 0)
            nd = d + w
            if nd < g.get(v, INF):
                g[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd + manhattan(v, dst), nd, v))
    if counter is not None:
        counter[0] += expanded
    if dst not in g:
        return INF, []
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return g[dst], path[::-1]


def cost_to_go(gmap, blk, root, rules, extra=None, counter=None):
    """dist[p] = cheapest cost from p to root (reverse Dijkstra)."""
    dist = {root: 0}
    pq = [(0, root)]
    expanded = 0
    while pq:
        d, v = heapq.heappop(pq)
        if d > dist[v]:
            continue
        expanded += 1
        w = rules.enter_cost(blk.get(v, 0))   # cost of stepping onto v
        if w == INF:
            continue
        if extra:
            w += extra.get(v, 0)
        for u in gmap.nbrs[v]:
            nd = d + w
            if nd < dist.get(u, INF):
                dist[u] = nd
                heapq.heappush(pq, (nd, u))
    if counter is not None:
        counter[0] += expanded
    return dist


def bfs_moves(gmap, blk, src, passable=lambda b: b != HARD):
    """Unweighted move distance from src over tiles where passable(state)."""
    dist = {src: 0}
    dq = deque([src])
    while dq:
        u = dq.popleft()
        for v in gmap.nbrs[u]:
            if v not in dist and passable(blk.get(v, 0)):
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist


def min_vertex_cut(gmap, blk, sources, is_sink, cuttable, limit=8):
    """Minimum set of tiles separating `sources` from tiles where is_sink(p).

    Only tiles where cuttable(p) can be in the cut; hard boulders are already
    walls. Returns (size, cut) where cut is the sink-side minimum cut (the one
    nearest the sinks), or (INF, []) if no cut of size <= limit exists.
    """
    BIG = 1 << 30
    S, T = 'S', 'T'
    cap = {}
    adj = {}

    def add(u, v, c):
        if (u, v) not in cap:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
            cap[(u, v)] = 0
            cap.setdefault((v, u), 0)
        cap[(u, v)] += c

    open_tiles = [p for p in gmap.open_tiles if blk.get(p, 0) != HARD]
    for p in open_tiles:
        add((p, 0), (p, 1), 1 if cuttable(p) else BIG)
        for q in gmap.nbrs[p]:
            if blk.get(q, 0) != HARD:
                add((p, 1), (q, 0), BIG)
        if is_sink(p):
            add((p, 1), T, BIG)
    for s in sources:
        add(S, (s, 0), BIG)

    flow = 0
    while True:
        par = {S: None}
        dq = deque([S])
        while dq and T not in par:
            u = dq.popleft()
            for v in adj.get(u, ()):
                if v not in par and cap[(u, v)] > 0:
                    par[v] = u
                    dq.append(v)
        if T not in par:
            break
        b = BIG
        v = T
        while par[v] is not None:
            b = min(b, cap[(par[v], v)])
            v = par[v]
        if b >= BIG:
            return INF, []
        v = T
        while par[v] is not None:
            u = par[v]
            cap[(u, v)] -= b
            cap[(v, u)] += b
            v = u
        flow += b
        if flow > limit:
            return INF, []
    # sink side: nodes that can still reach T in the residual graph
    reach = {T}
    dq = deque([T])
    while dq:
        v = dq.popleft()
        for u in adj.get(v, ()):
            if u not in reach and cap[(u, v)] > 0:
                reach.add(u)
                dq.append(u)
    cut = [p for p in open_tiles if (p, 1) in reach and (p, 0) not in reach]
    return flow, cut


def disjoint_routes(gmap, limit):
    """Number of vertex-disjoint routes from the start tile to the pad zone
    that the alien could cut (protected tiles cannot be cut), capped at limit.
    Counted from the start tile itself: base-zone tiles in other corridors
    do not give the rover extra ways out."""
    n, _ = min_vertex_cut(gmap, {}, [gmap.start], gmap.in_goal_zone,
                          lambda p: not gmap.protected(p), limit=limit)
    return limit if n == INF else n
