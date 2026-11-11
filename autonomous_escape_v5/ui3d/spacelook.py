"""Version 5 visual upgrades for space, and the shuttle's flight physics.

* asteroid_template: cratered, lumpy, smoothly lit asteroids with colour
  variation (darker crater floors, brighter rims), shared by instancing.
* skydome: a procedural equirectangular sky (Milky Way band with dust lanes,
  two nebulae, thousands of stars) on an inward-facing sphere.
* mars_texture: albedo features, craters, a canyon scar and polar caps.
* Flight: the shuttle follows the planned path with inertia (a damped
  spring on position and heading), so it eases in and out, overshoots a
  touch and settles; lateral acceleration banks it, thrust dips the nose.
"""
import math
import os

import numpy as np
from panda3d.core import GeomNode, NodePath, Vec3
from scipy import ndimage

from . import meshes

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sky_cache.npy')


# ------------------------------------------------------------------ asteroids
def asteroid_arrays(rng, sub=3):
    v, f = meshes.sphere_arrays(sub)
    v = v.astype(np.float32)
    d = np.zeros(len(v), np.float32)
    for octave in range(4):                          # lumpy multi-scale shape
        for _ in range(5):
            k = rng.normal(size=3)
            k /= np.linalg.norm(k)
            freq = rng.uniform(1.2, 2.5) * 2 ** octave
            d += (0.5 ** octave) * rng.uniform(0.4, 1.0) * np.sin(v @ k * freq + rng.uniform(0, 6.3))
    d = 1 + 0.11 * d
    shade = np.ones(len(v), np.float32)
    for _ in range(rng.integers(4, 8)):              # impact craters
        c = rng.normal(size=3)
        c /= np.linalg.norm(c)
        size = rng.uniform(0.25, 0.6)
        ang = np.arccos(np.clip(v @ c, -1, 1)) / size
        bowl = np.where(ang < 1, -(1 - ang ** 2) * 0.16 * size, 0)
        rim = 0.05 * size * np.exp(-((ang - 1) / 0.25) ** 2)
        d += bowl + rim
        shade -= np.where(ang < 1, 0.18 * (1 - ang), 0) - 0.12 * np.exp(-((ang - 1) / 0.25) ** 2)
    stretch = np.array([1.0, rng.uniform(0.75, 1.0), rng.uniform(0.6, 0.9)], np.float32)
    p = v * d[:, None] * stretch
    k = rng.normal(size=3)
    patch = 0.88 + 0.12 * np.sin(v @ (k / np.linalg.norm(k)) * 3.1)
    return p.astype(np.float32), f, np.clip(shade * patch, 0.45, 1.25)


def asteroid_template(rng, colour):
    p, f, shade = asteroid_arrays(rng)
    nrm = meshes.smooth_normals(p, f)
    col = np.concatenate([np.clip(np.outer(shade, colour), 0, 1), np.ones((len(p), 1))], 1).astype(np.float32)
    return meshes.mesh('asteroid', p, nrm, f, col)


# ------------------------------------------------------------------ debris pieces
def debris_part(rng, centre):
    """A drifting fragment: rock chip, a bit of solar panel, or a piece of truss."""
    x, y, z = centre
    kind = rng.random()
    s = rng.uniform(0.03, 0.1)
    a = rng.uniform(0, 6.3)
    if kind < 0.18:                                  # solar panel shard
        pos, nrm, idx = meshes.box_arrays(s * 3, s * 0.15, s * 2, (0, 0, 0))
        col = np.tile(np.array([0.12, 0.22, 0.55, 1], np.float32), (len(pos), 1))
    elif kind < 0.3:                                 # truss / hull strip
        pos, nrm, idx = meshes.box_arrays(s * 3.5, s * 0.3, s * 0.3, (0, 0, 0))
        g = rng.uniform(0.45, 0.65)
        col = np.tile(np.array([g, g, g * 1.05, 1], np.float32), (len(pos), 1))
    else:                                            # rock chip
        p, f = meshes.rock_arrays(rng, 1, rough=0.5)
        pos, nrm, idx = meshes.flat_shade(p * s, f)
        g = rng.uniform(0.35, 0.55)
        col = np.tile(np.array([g, g * 0.95, g * 0.9, 1], np.float32), (len(pos), 1))
    R = _rot(rng.uniform(0, 360, 3))
    pos = pos @ R.T + np.array([x, y, z], np.float32)
    nrm = nrm @ R.T
    return pos.astype(np.float32), nrm.astype(np.float32), col, idx


def _rot(hpr):
    h, p, r = np.radians(hpr)
    Rz = np.array([[math.cos(h), -math.sin(h), 0], [math.sin(h), math.cos(h), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Ry = np.array([[math.cos(r), 0, math.sin(r)], [0, 1, 0], [-math.sin(r), 0, math.cos(r)]])
    return (Rz @ Rx @ Ry).astype(np.float32)


# ------------------------------------------------------------------ sky
def _fbm(rng, h, w, cells):
    out = np.zeros((h, w), np.float32)
    amp = 1.0
    for c in cells:
        g = rng.random((h // c + 2, w // c + 2)).astype(np.float32)
        z = ndimage.zoom(g, c, order=1)[:h, :w]
        out += amp * ndimage.gaussian_filter(z, c / 4, mode='wrap')
        amp *= 0.55
    return (out - out.min()) / (np.ptp(out) + 1e-6)


def sky_texture(w=2048, h=1024):
    if os.path.exists(CACHE):
        try:
            img = np.load(CACHE)
            if img.shape == (h, w, 3):
                return img
        except Exception:
            pass
    rng = np.random.default_rng(2189)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    lon = xx / w * 2 * np.pi
    lat = (0.5 - yy / h) * np.pi
    d = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)
    axis = np.array([0.3, -0.5, 0.81])
    axis /= np.linalg.norm(axis)
    band = np.exp(-(d @ axis / 0.22) ** 2)                       # the Milky Way's plane
    cloud = _fbm(rng, h, w, (128, 64, 32, 16, 8))
    lanes = _fbm(rng, h, w, (64, 32, 16))
    milky = band * (0.35 + 0.9 * cloud) * (1 - 0.75 * np.clip((lanes - 0.45) * 3, 0, 1))
    img = np.zeros((h, w, 3), np.float32)
    img += milky[..., None] * np.array([150, 135, 160], np.float32)
    for centre, colour, size in (((0.6, 0.7, 0.3), (190, 60, 150), 0.35), ((-0.7, -0.4, 0.55), (40, 140, 190), 0.3)):
        c = np.array(centre)
        c /= np.linalg.norm(c)
        neb = np.exp(-((1 - d @ c) / (size * 0.35)) ** 1.5) * _fbm(rng, h, w, (96, 48, 24, 12))
        img += (neb ** 1.6)[..., None] * np.array(colour, np.float32) * 0.9
    stars = rng.random((h, w))
    dens = 0.9985 - 0.0035 * band
    bright = np.where(stars > dens, rng.random((h, w)) ** 3, 0)
    small = bright[..., None] * np.array([255, 250, 240], np.float32)
    big = ndimage.gaussian_filter(np.where(stars > 0.99996, 1.0, 0.0), 1.2)[..., None] * np.array([2200, 2100, 2400])
    img += small + big
    img = np.clip(img, 0, 255).astype(np.uint8)
    try:
        np.save(CACHE, img)
    except OSError:
        pass
    return img


def uv_sphere(rows=48, cols=96, inward=True):
    lat = np.linspace(np.pi / 2, -np.pi / 2, rows + 1)
    lon = np.linspace(0, 2 * np.pi, cols + 1)
    LON, LAT = np.meshgrid(lon, lat)
    pos = np.stack([np.cos(LAT) * np.cos(LON), np.cos(LAT) * np.sin(LON), np.sin(LAT)], -1).reshape(-1, 3)
    uv = np.stack([LON / (2 * np.pi), 1 - (np.pi / 2 - LAT) / np.pi], -1).reshape(-1, 2)
    ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    a = (ii * (cols + 1) + jj).reshape(-1)
    b, c, dd = a + 1, a + cols + 1, a + cols + 2
    idx = np.concatenate([np.stack([a, c, dd], -1), np.stack([a, dd, b], -1)])
    if inward:
        idx = idx[:, ::-1]
    nrm = -pos if inward else pos
    return pos.astype(np.float32), nrm.astype(np.float32), idx, uv.astype(np.float32)


def skydome(sky):
    pos, nrm, idx, uv = uv_sphere()
    dome = meshes.mesh('skydome', pos * 480, nrm, idx, (1, 1, 1, 1), uv)
    dome.setTexture(meshes.texture(sky_texture(), 'sky'))
    dome.reparentTo(sky)
    dome.setBin('background', -1)
    dome.setDepthWrite(False)
    dome.setDepthTest(False)
    dome.setLightOff(1)
    dome.setShaderOff(1)
    return dome


def mars_texture(w=1024, h=512):
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = _fbm(rng, h, w, (128, 64, 32, 16, 8, 4))
    dark = _fbm(rng, h, w, (160, 80, 40))
    img = np.array([196, 104, 62], np.float32) * (0.75 + 0.35 * base[..., None])
    maria = np.clip((dark - 0.55) * 4, 0, 1)[..., None]                      # dark albedo regions
    img = img * (1 - 0.45 * maria)
    canyon = np.exp(-((yy - h * 0.53 - 18 * np.sin(xx / 60)) / 3.5) ** 2) * ((xx > w * 0.42) & (xx < w * 0.66))
    img *= (1 - 0.45 * canyon)[..., None]
    for _ in range(140):                                                       # craters
        cx, cy, r = rng.uniform(0, w), rng.uniform(h * 0.1, h * 0.9), rng.uniform(2, 14)
        dd = np.hypot(xx - cx, yy - cy) / r
        img *= (1 - 0.18 * np.clip(1 - dd, 0, 1) + 0.12 * np.exp(-((dd - 1) / 0.25) ** 2))[..., None]
    lat = np.abs(yy - h / 2) / (h / 2)
    cap = np.clip((lat - 0.86) * 9 + 0.3 * (base - 0.5), 0, 1)[..., None]
    img = img * (1 - cap) + np.array([238, 232, 226]) * cap
    return np.clip(img, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ bloom
def bloom(base):
    """Glow around bright things (engines, sun, lasers). Silently skipped if the
    graphics card cannot run the filter."""
    try:
        from direct.filter.CommonFilters import CommonFilters
        f = CommonFilters(base.win, base.cam)
        if f.setBloom(blend=(0.3, 0.4, 0.3, 0.0), mintrigger=0.72, maxtrigger=1.0, desat=0.2,
                      intensity=1.15, size='medium'):
            return f
    except Exception:
        pass
    return None


# ------------------------------------------------------------------ flight physics
def _adiff(a, b):
    return (b - a + 180) % 360 - 180


class Flight:
    """Damped-spring follower of the planned path: position and heading lag the
    guide a little, accelerate and settle smoothly (slightly underdamped, like
    thrusters correcting), and expose the accelerations that drive banking,
    nose pitch, engine glow and RCS jets."""

    def __init__(self, x, y, h, freq=4.2, damping=0.78, yaw_freq=5.0, yaw_damping=0.82):
        self.pos = np.array([x, y], float)
        self.vel = np.zeros(2)
        self.acc = np.zeros(2)
        self.h, self.hrate, self.hacc = h, 0.0, 0.0
        self.w, self.z = freq, damping
        self.wy, self.zy = yaw_freq, yaw_damping
        self.bank = self.pitch = 0.0

    def snap(self, x, y, h):
        self.pos[:] = (x, y)
        self.vel[:] = 0
        self.h, self.hrate = h, 0.0

    def update(self, tx, ty, th, dt):
        steps = max(1, int(dt / 0.008))
        sdt = dt / steps
        for _ in range(steps):
            err = np.array([tx, ty]) - self.pos
            self.acc = self.w ** 2 * err - 2 * self.z * self.w * self.vel
            self.vel += self.acc * sdt
            self.pos += self.vel * sdt
            dh = _adiff(self.h, th)
            self.hacc = self.wy ** 2 * dh - 2 * self.zy * self.wy * self.hrate
            self.hrate += self.hacc * sdt
            self.h += self.hrate * sdt
        r = math.radians(self.h)
        fwd = np.array([-math.sin(r), math.cos(r)])
        right = np.array([math.cos(r), math.sin(r)])
        a_fwd = float(self.acc @ fwd)
        a_side = float(self.acc @ right)
        speed = float(np.hypot(*self.vel))
        bank = max(-38.0, min(38.0, -a_side * 5.5 + self.hrate * 0.12))
        pitch = max(-14.0, min(14.0, -a_fwd * 2.2))
        k = min(1.0, dt * 8)
        self.bank += (bank - self.bank) * k
        self.pitch += (pitch - self.pitch) * k
        return self.pos[0], self.pos[1], self.h, a_fwd, speed
