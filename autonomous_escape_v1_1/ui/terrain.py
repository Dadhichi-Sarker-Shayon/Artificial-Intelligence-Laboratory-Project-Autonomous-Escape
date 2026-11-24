"""Procedural Phobos ground for the board, styled after the concept art.

Open ground is regolith pocked with craters; impassable tiles become clusters
of boulders or deep crater pits, so the field reads as natural terrain and
the grid stays hidden. Everything is a height field lit by the sun (hill
shading + cast shadows).

Every map has its own THEME (palette, sun, light colour, signature feature),
and every game re-rolls the details from the game's random seed: the sun's
exact direction and height, colour shifts, where every crater, rock and
ridge feature lies. No two games look exactly alike.
"""
import math
import random

import numpy as np
import pygame as pg
from scipy import ndimage

THEMES = {
    # dust, darker dust, rock; sun azimuth (deg, 0 = from the top), elevation;
    # light tint; signature feature
    'iron': dict(label='Iron dust', dust=(140, 98, 74), dust2=(104, 70, 54), rock=(156, 124, 104),
                 azimuth=-40, elev=0.42, tint=(1.1, 0.98, 0.86), feature='giant_crater', craters=1.0, walls='boulders'),
    'basalt': dict(label='Dark basalt', dust=(84, 82, 84), dust2=(60, 59, 62), rock=(120, 118, 120),
                   azimuth=-55, elev=0.5, tint=(1.0, 1.0, 1.02), feature='grooves', craters=1.0, walls='pits'),
    'pale': dict(label='Pale dust plains', dust=(178, 166, 150), dust2=(150, 140, 128), rock=(190, 182, 172),
                 azimuth=-20, elev=0.62, tint=(1.04, 1.02, 0.97), feature='plains', craters=0.45, walls='boulders'),
    'boulder': dict(label='Boulder fields', dust=(120, 98, 80), dust2=(92, 76, 63), rock=(142, 128, 114),
                    azimuth=-70, elev=0.55, tint=(1.04, 0.99, 0.93), feature='boulders', craters=0.9, walls='boulders'),
    'ice': dict(label='Frost', dust=(112, 122, 138), dust2=(84, 92, 108), rock=(168, 180, 196),
                azimuth=60, elev=0.4, tint=(0.88, 0.97, 1.12), feature='frost', craters=1.0, walls='boulders'),
}


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


# ------------------------------------------------------------------ signature features
def _giant_crater(hf, dark, bright, prng, S, tile):
    """One huge old impact crater with bright ejecta rays across the field."""
    cx, cy = prng.uniform(0.25, 0.75) * S, prng.uniform(0.25, 0.75) * S
    R = prng.uniform(7, 10) * tile
    _crater(hf, dark, cx, cy, R, 22.0, 0.25)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    ang = np.arctan2(yy - cy, xx - cx)
    dist = np.hypot(xx - cx, yy - cy) / R
    rays = np.zeros((S, S), np.float32)
    for _ in range(prng.randint(9, 15)):
        a = prng.uniform(-math.pi, math.pi)
        width = prng.uniform(0.03, 0.07)
        diff = np.angle(np.exp(1j * (ang - a)))
        rays = np.maximum(rays, np.exp(-(diff / width) ** 2) * np.exp(-(dist - 1) / prng.uniform(1.5, 3)) *
                          (dist > 1))
    np.maximum(bright, 0.7 * rays, out=bright)


def _grooves(hf, dark, prng, S, tile):
    """Long parallel troughs, broken into segments, with chains of small pits."""
    ang = prng.uniform(0, math.pi)
    ca, sa = math.cos(ang), math.sin(ang)
    spacing = tile * prng.uniform(2.4, 3.4)
    width = tile * 0.24
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    u = xx * ca + yy * sa
    v = -xx * sa + yy * ca
    k = np.floor(u / spacing)
    f = u - (k + 0.5) * spacing
    phase = prng.uniform(0, 6.3)
    mask = np.clip((0.5 + 0.5 * np.sin(v / (tile * 4.2) + k * 2.37 + phase) - 0.3) * 3, 0, 1)
    trough = np.exp(-(f / width) ** 2) * mask
    hf -= 1.8 * trough
    np.maximum(dark, 0.1 * trough, out=dark)
    for kk in range(int(u.min() // spacing) - 1, int(u.max() // spacing) + 1):
        uu = (kk + 0.5) * spacing
        vv = float(v.min())
        while vv < v.max():
            vv += tile * prng.uniform(0.3, 0.6)
            if prng.random() < 0.45:
                continue
            x, y = uu * ca - vv * sa, uu * sa + vv * ca
            if 0 <= x < S and 0 <= y < S:
                _crater(hf, dark, x, y, prng.uniform(2.5, 5.5), 1.8, 0.3)


def _ripples(hf, prng, S, tile, rng):
    """Smooth plains with gentle wind-free 'ripples' of settled dust."""
    ang = prng.uniform(0, math.pi)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    u = xx * math.cos(ang) + yy * math.sin(ang)
    warp = _noise(rng, S, S, 60) * 18
    hf += 0.9 * np.sin((u + warp) / (tile * prng.uniform(0.45, 0.7)))
    hf *= 0.8


def build(gmap, tile, theme=None, seed=None):
    """Return the lit ground surface for this map at `tile` pixels per cell.
    `theme` names one of THEMES; `seed` (the game's random seed) re-rolls every
    detail, so the same map looks different from game to game."""
    T = THEMES.get(theme or 'iron', THEMES['iron'])
    n = gmap.size
    S = n * tile
    seed = gmap.seed * 9973 + 17 if seed is None else seed
    rng = np.random.default_rng(seed)
    prng = random.Random(seed * 31 + 5)

    # per-game variation of the theme: sun direction and height, colour shifts
    az = math.radians(T['azimuth'] + prng.uniform(-35, 35))
    elev = max(0.25, min(0.9, T['elev'] + prng.uniform(-0.08, 0.08)))
    flat_len = math.sqrt(1 - elev * elev)
    light = np.array([math.sin(az) * flat_len, -math.cos(az) * flat_len, elev], np.float32)

    def jitter(c, amt=0.07):
        return np.array([v * prng.uniform(1 - amt, 1 + amt) for v in c], np.float32)
    dust, dust2, rock_col = jitter(T['dust']), jitter(T['dust2']), jitter(T['rock'])
    tint = np.array(T['tint'], np.float32) * prng.uniform(0.96, 1.04)

    swell = _fbm(rng, S, S, (160, 80, 40))
    hf = 4 * swell + 0.6 * _noise(rng, S, S, 3)
    rock = np.zeros((S, S), np.float32)
    dark = np.zeros((S, S), np.float32)
    bright = np.zeros((S, S), np.float32)

    # craters on the open ground (cosmetic), scaled to the map's area
    area = (n / 21) ** 2 * T['craters'] * prng.uniform(0.8, 1.25)
    gy, gx = gmap.goal
    _crater(hf, dark, (gx + 0.5) * tile, (gy + 0.5) * tile, 2.6 * tile, 5.0, 0.12)
    for _ in range(int(4 * area)):
        _crater(hf, dark, prng.uniform(0, S), prng.uniform(0, S), prng.uniform(50, 90), 3.0, 0.18)
    for _ in range(int(40 * area)):
        _crater(hf, dark, prng.uniform(0, S), prng.uniform(0, S), prng.uniform(8, 24), 3.5, 0.32)
    for _ in range(int(140 * area)):
        _crater(hf, dark, prng.uniform(0, S), prng.uniform(0, S), prng.uniform(2, 6), 1.4, 0.3)

    feature = T['feature']
    if feature == 'giant_crater':
        _giant_crater(hf, dark, bright, prng, S, tile)
    elif feature == 'grooves':
        _grooves(hf, dark, prng, S, tile)
    elif feature == 'plains':
        _ripples(hf, prng, S, tile, rng)

    # impassable tiles: big crater pits over solid 2x2 blocks, boulder fields elsewhere
    W = gmap.walls
    style = T.get('walls', 'boulders')
    if style in ('mesa', 'ridge'):
        # raised plateaus (mesa) or smooth ice ridges over the impassable ground
        wall = np.kron(np.array(W, np.float32), np.ones((tile, tile), np.float32))
        wob = 0.6 * _noise(rng, S, S, 16) + 0.4 * _noise(rng, S, S, 6) - 0.5
        mask = (ndimage.gaussian_filter(wall, tile * 0.28) + 0.5 * wob) > 0.5       # rounded, natural outline
        soft = ndimage.gaussian_filter(mask.astype(np.float32), 2.2 if style == 'mesa' else 4.0)
        lump = _fbm(rng, S, S, (24, 10, 5))
        hf += soft * ((14 + 4 * lump) if style == 'mesa' else (10 + 10 * lump))
        np.maximum(rock, soft * (0.75 if style == 'mesa' else 0.9), out=rock)
    pit = set()
    pit_chance = {'boulders': 0.35 if feature == 'boulders' else 0.22, 'pits': 0.85,
                  'mesa': 0.05, 'ridge': 0.05}[style] * prng.uniform(0.8, 1.2)
    for r in range(n - 1):
        for c in range(n - 1):
            block = {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
            if all(W[a][b] for a, b in block) and not block & pit and prng.random() < pit_chance:
                pit |= block
                _crater(hf, dark, (c + 1 + prng.uniform(-0.15, 0.15)) * tile, (r + 1 + prng.uniform(-0.15, 0.15)) * tile,
                        tile * prng.uniform(0.8, 0.95), 16.0, 0.8)

    def is_wall(x, y):
        c, r = int(x // tile), int(y // tile)
        return 0 <= r < n and 0 <= c < n and W[r][c]

    extra = 3 if feature == 'boulders' else 0
    for r in range(n):
        for c in range(n):
            if not W[r][c]:
                continue
            k = prng.randint(1, 3) if (r, c) in pit else prng.randint(4, 7) + extra
            if style in ('mesa', 'ridge'):
                k = prng.randint(0, 2)                  # a few loose rocks on the plateau
            sizes = [prng.uniform(0.26, 0.4) * tile] if (r, c) not in pit and style not in ('mesa', 'ridge') else []
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
    pebbles = {'boulders': (1, 2, 2, 3, 4), 'plains': (0, 0, 0, 1)}.get(feature, (0, 0, 1, 1, 2, 3))
    for p in gmap.open_tiles:
        if gmap.in_goal_zone(p):
            continue
        for _ in range(prng.choice(pebbles)):
            x = (p[1] + prng.uniform(0.1, 0.9)) * tile
            y = (p[0] + prng.uniform(0.1, 0.9)) * tile
            rr = prng.uniform(1.4, 3.6)
            _boulder(hf, rock, prng, x, y, rr, rr * 1.1)

    # hill shading
    dy, dx = np.gradient(hf)
    norm = np.sqrt(dx * dx + dy * dy + 1)
    lam = (-dx * light[0] - dy * light[1] + light[2]) / norm
    shade = np.clip(lam / light[2], 0, 1.6)

    # cast shadows: something taller between this pixel and the sun
    pad = 60
    hp = np.pad(hf, pad, mode='edge')
    flat = math.hypot(light[0], light[1]) + 1e-6
    tan_e = light[2] / flat
    occl = np.zeros_like(hf)
    reach = int(min(56, 18 / max(tan_e, 0.3)))
    for k in range(2, reach, 2):
        sx, sy = int(round(k * light[0] / flat)), int(round(k * light[1] / flat))
        sh = hp[pad + sy:pad + sy + S, pad + sx:pad + sx + S]
        occl = np.maximum(occl, sh - hf - k * tan_e)
    shadow = ndimage.gaussian_filter(np.clip(occl / 2.0, 0, 1), 1.2)

    # colour: dust with darker patches, rocks, dark crater floors, bright rays, grain
    patch = _fbm(rng, S, S, (110, 50))[..., None]
    grain = _noise(rng, S, S, 2)[..., None]
    col = dust * (1 - 0.8 * patch) + dust2 * 0.8 * patch
    rk = rock[..., None]
    rock_tex = 0.85 + 0.3 * _noise(rng, S, S, 4)[..., None]
    col = col * (1 - rk) + rock_col * rock_tex * rk
    col *= (1 - dark[..., None])
    col *= 1 + bright[..., None]
    col *= 0.9 + 0.2 * grain

    lit = (0.32 + 0.74 * shade) * (1 - 0.62 * shadow)
    img = col * lit[..., None] * tint
    if feature == 'frost':                              # frost glints on sunlit rock
        glint = (rng.random((S, S)) > 0.985) & (rock > 0.5) & (shade > 1.0)
        img[glint] = np.minimum(255, img[glint] * 1.6 + 40)
        img += (rock * np.clip(shade - 0.9, 0, 1) * 22)[..., None] * np.array([0.8, 0.9, 1.0], np.float32)
    xx, yy = np.meshgrid(np.arange(S, dtype=np.float32), np.arange(S, dtype=np.float32))
    edge = np.minimum(np.minimum(xx, S - 1 - xx), np.minimum(yy, S - 1 - yy))
    img *= np.clip(0.6 + edge / 70, 0, 1)[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    return pg.surfarray.make_surface(img.transpose(1, 0, 2))
