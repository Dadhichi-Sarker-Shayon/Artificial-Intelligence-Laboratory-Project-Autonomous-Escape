"""Natural rover driving for the 3D view.

The rules engine moves the rover one tile per turn; this class turns those
discrete moves into continuous driving: moves are queued as curve segments
(cubic Bezier through the tile centres, so corners are rounded), speed
follows a simple acceleration / braking profile along the whole queued path
(so held keys and AI moves chain without stopping), and turns on the spot
rotate at a limited rate. It reports how far the wheels rolled and how hard
the rover is steering, for the wheel animation.
"""
import math

import numpy as np


def heading_vec(h):
    r = math.radians(h)
    return -math.sin(r), math.cos(r)


def heading_of_vec(dx, dy):
    return math.degrees(math.atan2(-dx, dy))


def angle_diff(a, b):
    return (b - a + 180) % 360 - 180


class Segment:
    def __init__(self, kind, pts=None, h0=0.0, h1=0.0, reverse=False):
        self.kind = kind                    # 'path' or 'turn'
        self.reverse = reverse
        self.h0, self.h1 = h0, h1
        if pts is not None:
            self.pts = np.asarray(pts, float)
            d = np.hypot(*np.diff(self.pts, axis=0).T)
            self.cum = np.concatenate([[0], np.cumsum(d)])
            self.length = float(self.cum[-1])
        else:
            self.length = 0.0

    def at(self, s):
        i = int(np.clip(np.searchsorted(self.cum, s) - 1, 0, len(self.pts) - 2))
        a, b = self.pts[i], self.pts[i + 1]
        seg = self.cum[i + 1] - self.cum[i]
        t = 0 if seg < 1e-9 else (s - self.cum[i]) / seg
        p = a + (b - a) * t
        dx, dy = b - a
        h = heading_of_vec(dx, dy)
        return p, (h + 180) % 360 if self.reverse else h


class Motion:
    def __init__(self, xy, heading, vmax=2.2, acc=5.0, turn_rate=260.0):
        self.vmax, self.acc, self.turn_rate = vmax, acc, turn_rate
        self.snap(xy, heading)

    def snap(self, xy, heading):
        self.x, self.y = xy
        self.h = heading
        self.v = 0.0
        self.queue = []
        self.seg = None
        self.s = 0.0
        self.end_xy, self.end_h = (self.x, self.y), heading
        self.rolled = 0.0
        self.steer = 0.0

    # ------------------------------------------------------------ queueing
    def push_move(self, to_xy, reverse=False):
        sx, sy = self.end_xy
        tx, ty = to_xy
        dx, dy = tx - sx, ty - sy
        if reverse:
            pts = [(sx + dx * t, sy + dy * t) for t in np.linspace(0, 1, 8)]
            self.queue.append(Segment('path', pts, reverse=True))
            self.end_xy = (tx, ty)
            return
        want = heading_of_vec(dx, dy)
        turn = angle_diff(self.end_h, want)
        if abs(turn) > 120:                              # about-turn: stop and rotate first
            self.push_turn(want)
            turn = 0
        if abs(turn) < 5:
            pts = [(sx + dx * t, sy + dy * t) for t in np.linspace(0, 1, 8)]
        else:                                            # rounded corner
            fx, fy = heading_vec(self.end_h)
            L = math.hypot(dx, dy)
            p0 = np.array([sx, sy])
            p1 = p0 + np.array([fx, fy]) * L * 0.42
            p3 = np.array([tx, ty])
            p2 = p3 - np.array([dx, dy]) / L * L * 0.42
            t = np.linspace(0, 1, 18)[:, None]
            pts = ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)
        self.queue.append(Segment('path', pts))
        self.end_xy, self.end_h = (tx, ty), want

    def push_turn(self, h1):
        if abs(angle_diff(self.end_h, h1)) < 0.5:
            return
        self.queue.append(Segment('turn', h0=self.end_h, h1=h1))
        self.end_h = h1

    # ------------------------------------------------------------ state
    def remaining(self):
        """Distance (tiles) still to drive, counting turns as half a tile."""
        r = 0.0
        if self.seg is not None:
            r += (self.seg.length - self.s) if self.seg.kind == 'path' else 0.5
        for q in self.queue:
            r += q.length if q.kind == 'path' else 0.5
        return r

    def busy(self):
        return self.seg is not None or bool(self.queue)

    def _path_ahead(self):
        """Path length we can drive through without stopping (up to the next turn)."""
        r = self.seg.length - self.s
        for q in self.queue:
            if q.kind != 'path' or q.reverse != self.seg.reverse:
                break
            r += q.length
        return r

    def update(self, dt):
        self.rolled = 0.0
        h_before = self.h
        while dt > 1e-6:
            if self.seg is None:
                if not self.queue:
                    self.v = 0.0
                    self.h += angle_diff(self.h, self.end_h) * min(1.0, dt * 10)
                    break
                self.seg = self.queue.pop(0)
                self.s = 0.0
            seg = self.seg
            if seg.kind == 'turn':
                self.v = 0.0
                d = angle_diff(self.h, seg.h1)
                step = self.turn_rate * dt
                if abs(d) <= step:
                    self.h = seg.h1
                    dt -= abs(d) / self.turn_rate
                    self.seg = None
                else:
                    self.h += math.copysign(step, d)
                    dt = 0
                continue
            ahead = self._path_ahead()
            target = min(self.vmax, math.sqrt(2 * self.acc * max(ahead, 0)) + 0.25)
            if self.v < target:
                self.v = min(target, self.v + self.acc * dt)
            else:
                self.v = max(target, self.v - self.acc * 1.6 * dt)
            ds = self.v * dt
            if self.s + ds >= seg.length:
                used = (seg.length - self.s) / max(self.v, 1e-6)
                self.rolled += (seg.length - self.s) * (-1 if seg.reverse else 1)
                p, self.h = seg.at(seg.length)
                self.x, self.y = p
                self.seg = None
                dt -= used
                if not self.queue or self.queue[0].kind != 'path':
                    self.v = 0.0 if not self.queue else self.v
            else:
                self.s += ds
                self.rolled += ds * (-1 if seg.reverse else 1)
                p, self.h = seg.at(self.s)
                self.x, self.y = p
                dt = 0
        rate = angle_diff(h_before, self.h)
        self.steer += (max(-28.0, min(28.0, rate * 6)) - self.steer) * 0.25
        return self.x, self.y, self.h
