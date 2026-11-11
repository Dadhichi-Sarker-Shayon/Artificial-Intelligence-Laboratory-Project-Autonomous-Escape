"""Procedural Phobos ground for the board, styled after the concept art.

Open ground is dusty grey-brown regolith pocked with craters of every size.
Impassable tiles are not drawn as walls: each one becomes a cluster of real
3D boulders or a deep crater pit, so the field reads as natural terrain and
the grid stays hidden. Everything is a height field lit from the upper left
(hill shading + cast shadows); rock and dust have different colours.
Built once per map.
"""
import math
import random

import numpy as np
import pygame as pg
from scipy import ndimage

LIGHT = np.array([-0.55, -0.62, 0.56])
LIGHT = LIGHT / np.linalg.norm(LIGHT)

DUST = np.array([120, 106, 94], np.float32)    # regolith
DUST2 = np.array([96, 86, 80], np.float32)     # darker grey patches
ROCK = np.array([146, 140, 134], np.float32)   # boulders: grey like the art


def _noise(rng, h, w, cell):
    """Smooth value noise in [0, 1] with features about `cell` pixels wide.
    Coarse noise is made at reduced resolution and scaled up (much faster)."""
    f = 4 if cell >= 16 else 2 if cell >= 6 else 1
    hh, ww, c = -(-h // f), -(-w // f), max(1, cell // f)
    g = rng.random((hh // c + 2, ww // c + 2), np.float32)
    z = np.repeat(np.repeat(g, c, 0), c, 1)[:hh, :ww]
    z = ndimage.gaussian_filter(z, c * 0.5, mode='nearest')
    if f > 1:
        z = ndimage.zoom(z, f, order=1)[:h, :w]
    return (z - z.min()) / (np.ptp(z) + 1e-6)


def _fbm(rng, h, w, cells):
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for cell in cells:
        out += amp * _noise(rng, h, w, cell)
        total += amp
        amp *= 0.55
    out /= total
    return (out - out.min()) / (np.ptp(out) + 1e-6)


def _box(shape, cx, cy, R):
    h, w = shape
    x0, x1 = max(0, int(cx - R)), min(w, int(cx + R) + 1)
    y0, y1 = max(0, int(cy - R)), min(h, int(cy + R) + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    return (slice(y0, y1), slice(x0, x1)), xx - cx, yy - cy


def _crater(hf, dark, cx, cy, r, depth, darken=0.35):
    """Bowl with a raised rim; the floor is darker (shadowed dust)."""
    b = _box(hf.shape, cx, cy, r * 2.3 + 2)
    if b is None:
        return
    sl, dx, dy = b
    d = np.hypot(dx, dy) / r
    bowl = np.where(d < 1, -depth * (1 - d * d), 0)
    rim = depth * 0.3 * np.exp(-((d - 1) / 0.3) ** 2)
    hf[sl] += bowl + rim
    dark[sl] = np.maximum(dark[sl], darken * np.clip(1.15 - d, 0, 1))


def _boulder(hf, rock, rng, cx, cy, r, height):
    """Irregular rounded rock stamped into the height field."""
    b = _box(hf.shape, cx, cy, r * 1.4 + 2)
    if b is None:
        return
    sl, dx, dy = b
    th = np.arctan2(dy, dx)
    p1, p2 = rng.uniform(0, 6.3), rng.uniform(0, 6.3)
    reff = r * (1 + 0.18 * np.sin(3 * th + p1) + 0.1 * np.sin(5 * th + p2))
    d = np.hypot(dx, dy) / reff
    inside = d < 1
    bump = np.where(inside, height * np.sqrt(np.clip(1 - d * d, 0, 1)) ** 0.8, 0)
    hf[sl] += bump
    rock[sl] = np.maximum(rock[sl], np.clip((1 - d) * 6, 0, 1))


def build(gmap, tile):
    """Return the lit ground surface for this map at `tile` pixels per cell."""
    n = gmap.size
    S = n * tile
    rng = np.random.default_rng(gmap.seed * 9973 + 17)
    prng = random.Random(gmap.seed * 31 + 5)

    swell = _fbm(rng, S, S, (160, 80, 40))
    hf = 4 * swell + 0.6 * _noise(rng, S, S, 3)
    rock = np.zeros((S, S), np.float32)
    dark = np.zeros((S, S), np.float32)

    # craters everywhere on the open ground (purely cosmetic, shallow)
    gy, gx = gmap.goal
    _crater(hf, dark, (gx + 0.5) * tile, (gy + 0.5) * tile, 2.6 * tile, 5.0, 0.12)
    for _ in range(4):
        _crater(hf, dark, prng.uniform(0, S), prng.uniform(0, S), prng.uniform(50, 90), 3.0, 0.18)
    for _ in range(40):
        _crater(hf, dark, prng.uniform(0, S), prng.uniform(0, S), prng.uniform(8, 24), 3.5, 0.32)
    for _ in range(140):
        _crater(hf, dark, prng.uniform(0, S), prng.uniform(0, S), prng.uniform(2, 6), 1.4, 0.3)

    # impassable tiles: big crater pits over solid 2x2 blocks, boulder fields
    # everywhere else, placed freely so no grid shows through
    W = gmap.walls
    pit = set()
    for r in range(n - 1):
        for c in range(n - 1):
            block = {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
            if all(W[a][b] for a, b in block) and not block & pit and prng.random() < 0.22:
                pit |= block
                _crater(hf, dark, (c + 1 + prng.uniform(-0.15, 0.15)) * tile, (r + 1 + prng.uniform(-0.15, 0.15)) * tile,
                        tile * prng.uniform(0.8, 0.95), 16.0, 0.8)

    def is_wall(x, y):
        c, r = int(x // tile), int(y // tile)
        return 0 <= r < n and 0 <= c < n and W[r][c]

    for r in range(n):
        for c in range(n):
            if not W[r][c]:
                continue
            k = prng.randint(1, 3) if (r, c) in pit else prng.randint(4, 7)
            sizes = [prng.uniform(0.26, 0.4) * tile] if (r, c) not in pit else []
            sizes += [tile * min(0.3, prng.lognormvariate(-2.0, 0.45)) for _ in range(k)]
            for i, rr in enumerate(sizes):
                for _ in range(6):
                    x = (c + prng.uniform(-0.15, 1.15)) * tile
                    y = (r + prng.uniform(-0.15, 1.15)) * tile
                    if is_wall(x, y):
                        break
                else:
                    x, y = (c + 0.5) * tile, (r + 0.5) * tile
                if (r, c) in pit:
                    a = prng.uniform(0, 2 * math.pi)
                    x, y = (c + 0.5 + 0.45 * math.cos(a)) * tile, (r + 0.5 + 0.45 * math.sin(a)) * tile
                _boulder(hf, rock, prng, x, y, rr, rr * prng.uniform(0.9, 1.3))

    # loose pebbles on open ground: small enough to read as drivable
    for p in gmap.open_tiles:
        if gmap.in_goal_zone(p):
            continue
        for _ in range(prng.choice((0, 0, 1, 1, 2, 3))):
            x = (p[1] + prng.uniform(0.1, 0.9)) * tile
            y = (p[0] + prng.uniform(0.1, 0.9)) * tile
            rr = prng.uniform(1.4, 3.6)
            _boulder(hf, rock, prng, x, y, rr, rr * 1.1)

    # hill shading
    dy, dx = np.gradient(hf)
    norm = np.sqrt(dx * dx + dy * dy + 1)
    lam = (-dx * LIGHT[0] - dy * LIGHT[1] + LIGHT[2]) / norm
    shade = np.clip(lam / LIGHT[2], 0, 1.6)

    # cast shadows: something taller between this pixel and the sun
    pad = 40
    hp = np.pad(hf, pad, mode='edge')
    flat = math.hypot(LIGHT[0], LIGHT[1])
    tan_e = LIGHT[2] / flat
    occl = np.zeros_like(hf)
    for k in range(2, 30, 2):
        sx, sy = int(round(k * LIGHT[0] / flat)), int(round(k * LIGHT[1] / flat))
        sh = hp[pad + sy:pad + sy + S, pad + sx:pad + sx + S]
        occl = np.maximum(occl, sh - hf - k * tan_e)
    shadow = ndimage.gaussian_filter(np.clip(occl / 2.0, 0, 1), 1.2)

    # colour: dust with darker patches, pale rocks, dark crater floors, grain
    patch = _fbm(rng, S, S, (110, 50))[..., None]
    grain = _noise(rng, S, S, 2)[..., None]
    col = DUST * (1 - 0.8 * patch) + DUST2 * 0.8 * patch
    rk = rock[..., None]
    rock_tex = 0.85 + 0.3 * _noise(rng, S, S, 4)[..., None]
    col = col * (1 - rk) + ROCK * rock_tex * rk
    col *= (1 - dark[..., None])
    col *= 0.9 + 0.2 * grain

    light = (0.32 + 0.74 * shade) * (1 - 0.62 * shadow)
    img = col * light[..., None]
    xx, yy = np.meshgrid(np.arange(S, dtype=np.float32), np.arange(S, dtype=np.float32))
    edge = np.minimum(np.minimum(xx, S - 1 - xx), np.minimum(yy, S - 1 - yy))
    img *= np.clip(0.6 + edge / 70, 0, 1)[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    return pg.surfarray.make_surface(img.transpose(1, 0, 2))
