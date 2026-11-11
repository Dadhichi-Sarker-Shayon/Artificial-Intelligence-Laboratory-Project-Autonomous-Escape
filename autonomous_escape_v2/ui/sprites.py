"""Sprites drawn in code at 4x and smooth-scaled down, so edges are soft."""
import math
import random

import pygame as pg

SS = 4   # supersampling factor


def _canvas(w, h):
    return pg.Surface((w * SS, h * SS), pg.SRCALPHA)


def _down(surf, w, h):
    return pg.transform.smoothscale(surf, (w, h))


def glow(radius, color, strength=110):
    """Soft round glow, brightest in the middle."""
    d = radius * 2
    s = pg.Surface((d, d), pg.SRCALPHA)
    steps = max(6, radius // 2)
    for i in range(steps):
        t = i / steps
        a = int(strength * (t ** 2))
        pg.draw.circle(s, (*color, a), (radius, radius), radius * (1 - t) + 1)
    return s


def rover(size=46):
    """Top-down rover facing +x, after the concept art: white chassis, blue
    solar deck, four wheels on struts with cyan rims, camera mast in front."""
    s = _canvas(size, size)
    c = size * SS / 2
    k = SS * size / 46
    strut = (205, 212, 224)
    for dx in (-1, 1):
        for dy in (-1, 1):
            wx, wy = c + dx * 14 * k, c + dy * 15 * k
            pg.draw.line(s, (60, 66, 80), (c + dx * 6 * k, c + dy * 6 * k), (wx, wy), int(4 * k))
            pg.draw.line(s, strut, (c + dx * 6 * k, c + dy * 6 * k), (wx, wy), int(2.2 * k))
            pg.draw.circle(s, (20, 24, 32), (wx, wy), 6.2 * k)
            pg.draw.circle(s, (110, 220, 245), (wx, wy), 6.2 * k, int(1.8 * k))
            pg.draw.circle(s, (230, 236, 244), (wx, wy), 2.4 * k)
    body = pg.Rect(0, 0, 25 * k, 20 * k)
    body.center = (c - 1 * k, c)
    pg.draw.rect(s, (40, 44, 56), body.move(0, 1.5 * k), border_radius=int(3 * k))
    pg.draw.rect(s, (238, 241, 246), body, border_radius=int(3 * k))
    panel = body.inflate(-4 * k, -4 * k)
    pg.draw.rect(s, (30, 60, 160), panel, border_radius=int(1.5 * k))
    for i in range(1, 4):
        x = panel.x + panel.w * i / 4
        pg.draw.line(s, (90, 132, 228), (x, panel.y), (x, panel.bottom), max(1, int(0.7 * k)))
    for j in (1, 2):
        y = panel.y + panel.h * j / 3
        pg.draw.line(s, (90, 132, 228), (panel.x, y), (panel.right, y), max(1, int(0.7 * k)))
    pg.draw.line(s, (190, 215, 255), (panel.x + k, panel.y + 1.2 * k), (panel.x + panel.w * 0.5, panel.y + 1.2 * k),
                 int(1 * k))
    mast = pg.Rect(0, 0, 6 * k, 8 * k)
    mast.midleft = (body.right - 2 * k, c)
    pg.draw.rect(s, (238, 241, 246), mast, border_radius=int(1.5 * k))
    pg.draw.rect(s, (250, 200, 70), mast.inflate(-2.4 * k, -2.4 * k), border_radius=int(1 * k))
    pg.draw.circle(s, (20, 20, 30), (mast.right - 1.6 * k, c), 1.6 * k)
    return _down(s, size, size)


def _rock_poly(rng, cx, cy, r, pts=10, rough=0.22):
    out = []
    for i in range(pts):
        a = 2 * math.pi * i / pts + rng.uniform(-0.18, 0.18)
        rr = r * (1 - rough + rng.uniform(0, rough))
        out.append((cx + rr * math.cos(a), cy + rr * 0.92 * math.sin(a)))
    return out


def _scaled(poly, cx, cy, f, ox, oy):
    return [(cx + (x - cx) * f + ox, cy + (y - cy) * f + oy) for x, y in poly]


def boulder(kind, variant, tile):
    """A loose boulder (soft) or a debris-fused, laser-proof one (hard).
    The sprite is 1.5 tiles wide so the shadow and glow can spill over."""
    rng = random.Random(variant * 7919 + (0 if kind == 1 else 1))
    size = int(tile * 1.5)
    s = _canvas(size, size)
    c = size * SS / 2
    r = tile * SS * (0.47 if kind == 1 else 0.49)
    poly = _rock_poly(rng, c, c, r)
    if kind == 1:   # alien-dropped: warm, scorched rock with a faint ember glow
        base, dark, lit, edge = (158, 112, 86), (86, 56, 44), (214, 170, 136), (64, 40, 32)
        g = glow(int(r * 1.3), (255, 140, 60), 85)
        s.blit(g, g.get_rect(center=(c, c)))
    else:
        base, dark, lit, edge = (88, 64, 62), (42, 30, 32), (138, 104, 96), (30, 20, 22)
        g = glow(int(r * 1.25), (255, 120, 40), 90)
        s.blit(g, g.get_rect(center=(c, c)))
    # shadow cast to the lower right
    pg.draw.polygon(s, (0, 0, 0, 105), _scaled(poly, c, c, 1.02, 0.16 * r, 0.2 * r))
    pg.draw.polygon(s, dark, poly)
    pg.draw.polygon(s, base, _scaled(poly, c, c, 0.86, -0.08 * r, -0.08 * r))
    pg.draw.polygon(s, lit, _scaled(poly, c, c, 0.48, -0.3 * r, -0.3 * r))
    for _ in range(3):                                       # pits
        a, d = rng.uniform(0, 2 * math.pi), rng.uniform(0.15, 0.55) * r
        px, py, pr = c + d * math.cos(a), c + d * math.sin(a), rng.uniform(0.07, 0.13) * r
        pg.draw.circle(s, dark, (px, py), pr)
        pg.draw.circle(s, lit, (px + pr * 0.4, py + pr * 0.4), pr * 0.5)
    pg.draw.polygon(s, edge, poly, int(SS * 0.9))
    if kind == 2:                                            # glowing fused cracks
        crack = [(c - 0.55 * r, c - 0.25 * r), (c - 0.15 * r, c + 0.05 * r), (c + 0.12 * r, c - 0.35 * r),
                 (c + 0.5 * r, c + 0.1 * r)]
        branch = [(c - 0.15 * r, c + 0.05 * r), (c - 0.05 * r, c + 0.55 * r)]
        for pts, w in ((crack, 0.13), (branch, 0.11)):
            pg.draw.lines(s, (255, 120, 40), False, pts, int(w * r))
            pg.draw.lines(s, (255, 226, 150), False, pts, int(w * r * 0.4))
    return _down(s, size, size)


def ufo(w=66):
    """The alien saucer: blue hull, gold running lights, violet dome."""
    h = int(w * 0.62)
    s = _canvas(w, h)
    W, H = w * SS, h * SS
    hull = pg.Rect(0, 0, W * 0.96, H * 0.42)
    hull.center = (W / 2, H * 0.62)
    dome = pg.Rect(0, 0, W * 0.46, H * 0.62)
    dome.midbottom = (W / 2, hull.centery + H * 0.04)
    pg.draw.ellipse(s, (210, 120, 235, 235), dome)
    pg.draw.ellipse(s, (248, 210, 255, 200), dome.inflate(-dome.w * 0.55, -dome.h * 0.6).move(-dome.w * 0.12, -dome.h * 0.18))
    pg.draw.ellipse(s, (22, 44, 120), hull.move(0, H * 0.05))
    pg.draw.ellipse(s, (44, 96, 210), hull)
    pg.draw.ellipse(s, (120, 170, 250), hull.inflate(-hull.w * 0.1, -hull.h * 0.55).move(0, -hull.h * 0.16))
    for i in range(9):
        a = math.pi * (0.1 + 0.8 * i / 8)
        x = W / 2 - math.cos(a) * hull.w * 0.42
        y = hull.centery + math.sin(a) * hull.h * 0.3
        pg.draw.circle(s, (255, 214, 90), (x, y), SS * 1.6)
    return _down(s, w, h)


def base(tile):
    """The damaged mining base: a landing deck with a habitat dome."""
    size = int(tile * 1.6)
    s = _canvas(size, size)
    c = size * SS / 2
    R = c * 0.86
    octo = [(c + R * math.cos(math.pi / 8 + i * math.pi / 4), c + R * math.sin(math.pi / 8 + i * math.pi / 4))
            for i in range(8)]
    pg.draw.polygon(s, (0, 0, 0, 110), [(x + 6 * SS, y + 7 * SS) for x, y in octo])
    pg.draw.polygon(s, (70, 74, 86), octo)
    pg.draw.polygon(s, (250, 200, 70), octo, int(SS * 2.2))
    inner = [(c + (x - c) * 0.72, c + (y - c) * 0.72) for x, y in octo]
    pg.draw.polygon(s, (96, 100, 114), inner)
    pg.draw.polygon(s, (130, 136, 152), inner, int(SS * 1.2))
    # habitat dome and solar wing in the corner (scorched on one side)
    dc = (c - R * 0.55, c - R * 0.55)
    pg.draw.circle(s, (0, 0, 0, 90), (dc[0] + 4 * SS, dc[1] + 5 * SS), R * 0.36)
    pg.draw.circle(s, (210, 214, 224), dc, R * 0.36)
    pg.draw.circle(s, (240, 244, 250), (dc[0] - R * 0.1, dc[1] - R * 0.1), R * 0.16)
    pg.draw.circle(s, (60, 40, 36), (dc[0] + R * 0.18, dc[1] + R * 0.14), R * 0.12)
    wing = pg.Rect(0, 0, R * 0.9, R * 0.28)
    wing.midleft = (dc[0] + R * 0.25, dc[1] - R * 0.42)
    pg.draw.rect(s, (40, 70, 160), wing)
    pg.draw.rect(s, (160, 170, 190), wing, int(SS * 0.8))
    return _down(s, size, size)
