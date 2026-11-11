"""Version 4 models: the six-wheeled rover, the alien saucer, the obstacle
families (rock, crystal, wreck), incoming projectiles and shatter fragments.
Everything faces +y and stands on z = 0."""
import math

import numpy as np
from panda3d.core import CardMaker, NodePath, Vec3

from . import meshes

STYLES = ('rock', 'crystal', 'wreck')


def glow_card(parent, tex, size, color, billboard=True):
    cm = CardMaker('glow')
    cm.setFrame(-size / 2, size / 2, -size / 2, size / 2)
    c = parent.attachNewNode(cm.generate())
    c.setTexture(tex)
    c.setColor(*color)
    if billboard:
        c.setBillboardPointEye()
    meshes.additive(c)
    return c


def bar(parent, a, b, thick, color, name='bar'):
    """A square beam from point a to point b."""
    a, b = Vec3(*a), Vec3(*b)
    L = (b - a).length()
    m = meshes.box(name, thick, L, thick, color)
    m.reparentTo(parent)
    m.setPos((a + b) / 2)
    m.lookAt(parent, b)
    return m


# ------------------------------------------------------------------ rover
class RoverModel:
    """PATHFINDER-7: rocker-bogie suspension, six cleated wheels that spin and
    steer, a camera mast with stereo eyes, high-gain dish and an RTG."""
    WHEEL_R = 0.075
    WHEELS = [(-0.25, 0.21), (0.25, 0.21), (-0.27, 0.0), (0.27, 0.0), (-0.25, -0.21), (0.25, -0.21)]

    def __init__(self, glow_tex):
        self.root = NodePath('rover')
        body = self.body = self.root.attachNewNode('body')
        white, grey, dark = (0.9, 0.91, 0.93, 1), (0.55, 0.57, 0.6, 1), (0.16, 0.17, 0.2, 1)
        gold, orange = (0.86, 0.66, 0.26, 1), (0.95, 0.5, 0.15, 1)
        meshes.box('chassis', 0.36, 0.48, 0.1, white, (0, 0, 0.25)).reparentTo(body)
        meshes.box('deck', 0.34, 0.44, 0.012, dark, (0, 0, 0.306)).reparentTo(body)
        for sx in (-1, 1):
            meshes.box('foil', 0.012, 0.4, 0.07, gold, (sx * 0.183, 0, 0.25)).reparentTo(body)
        meshes.box('bumper', 0.3, 0.03, 0.04, orange, (0, 0.245, 0.22)).reparentTo(body)
        meshes.box('belly', 0.3, 0.42, 0.04, dark, (0, 0, 0.19)).reparentTo(body)
        meshes.box('solar', 0.22, 0.2, 0.012, (0.12, 0.24, 0.62, 1), (-0.04, -0.07, 0.318)).reparentTo(body)
        for i in range(1, 4):
            meshes.box('cell', 0.004, 0.2, 0.014, (0.35, 0.5, 0.9, 1),
                       (-0.15 + i * 0.055, -0.07, 0.32)).reparentTo(body)
        # camera mast with a stereo head
        meshes.box('mast', 0.03, 0.03, 0.26, grey, (0.1, 0.17, 0.44)).reparentTo(body)
        self.head = body.attachNewNode('head')
        self.head.setPos(0.1, 0.18, 0.6)
        meshes.box('headbox', 0.15, 0.07, 0.07, white, (0, 0, 0)).reparentTo(self.head)
        for dx in (-0.04, 0.04):
            eye = meshes.cylinder('eye', 0.018, 0.02, dark, seg=14, cap_color=(0.05, 0.08, 0.12, 1))
            eye.setH(90)
            eye.setPos(dx, 0.04, 0)
            eye.reparentTo(self.head)
        glow_card(self.head, glow_tex, 0.05, (1, 0.5, 0.2, 1)).setPos(0.07, 0.03, 0.03)
        # high-gain dish and whip antenna
        post = meshes.box('post', 0.025, 0.025, 0.12, grey, (-0.1, -0.02, 0.37))
        post.reparentTo(body)
        dish = meshes.sphere('dish', (0.85, 0.86, 0.88, 1), sub=2)
        dish.setScale(0.085, 0.085, 0.018)
        dish.setPos(-0.1, -0.02, 0.44)
        dish.setP(35)
        dish.reparentTo(body)
        meshes.box('whip', 0.008, 0.008, 0.28, grey, (0.14, -0.2, 0.44)).reparentTo(body)
        # RTG at the back
        rtg = meshes.cylinder('rtg', 0.045, 0.16, dark, seg=14)
        rtg.setH(90)
        rtg.setR(35)
        rtg.setPos(0, -0.29, 0.29)
        rtg.reparentTo(body)
        for k in range(4):
            fin = meshes.box('fin', 0.008, 0.12, 0.05, grey, (0, 0, 0))
            fin.setPos(0, -0.29, 0.29)
            fin.setR(35)
            fin.setH(k * 45)
            fin.reparentTo(body)
        # headlights
        for dx in (-0.09, 0.09):
            glow_card(body, glow_tex, 0.07, (0.85, 0.95, 1, 0.9)).setPos(dx, 0.26, 0.24)
        # rocker-bogie suspension and wheels
        self.wheels = []
        for i, (x, y) in enumerate(self.WHEELS):
            pivot = self.root.attachNewNode('wheel_pivot')
            pivot.setPos(x, y, self.WHEEL_R)
            spin = pivot.attachNewNode('spin')
            tyre = meshes.cylinder('tyre', self.WHEEL_R, 0.06, (0.2, 0.2, 0.22, 1), seg=18,
                                   cap_color=(0.55, 0.56, 0.6, 1))
            tyre.reparentTo(spin)
            for k in range(10):
                a = 2 * math.pi * k / 10
                cl = meshes.box('cleat', 0.064, 0.012, 0.012, (0.32, 0.32, 0.35, 1))
                cl.setPos(0, self.WHEEL_R * math.cos(a), self.WHEEL_R * math.sin(a))
                cl.setP(math.degrees(a))
                cl.reparentTo(spin)
            self.wheels.append((pivot, spin, i < 2 or i >= 4))
        for sx in (-1, 1):
            hip = (sx * 0.2, 0.0, 0.24)
            bogie = (sx * 0.24, 0.1, 0.15)
            bar(body, hip, (sx * 0.24, -0.21, self.WHEEL_R + 0.02), 0.022, grey)
            bar(body, hip, bogie, 0.022, grey)
            bar(body, bogie, (sx * 0.24, 0.21, self.WHEEL_R + 0.02), 0.02, grey)
            bar(body, bogie, (sx * 0.26, 0.0, self.WHEEL_R + 0.02), 0.02, grey)
        self.spin_angle = 0.0

    def roll_wheels(self, distance, steer):
        """Spin every wheel by the distance driven; steer the corner wheels."""
        self.spin_angle -= math.degrees(distance / self.WHEEL_R)
        for pivot, spin, corner in self.wheels:
            spin.setP(self.spin_angle)
            front = pivot.getY() > 0
            pivot.setH((steer if front else -steer) if corner else 0)


# ------------------------------------------------------------------ alien saucer
class SaucerModel:
    """The alien craft: a layered disc hull with panel seams, a counter-rotating
    outer ring of hull plates and running lights, a glass dome with its pilot,
    three engine pods underneath, an antenna, and the weapon emitter."""

    def __init__(self, glow_tex):
        self.root = NodePath('saucer')
        self.tilt = self.root.attachNewNode('tilt')          # banks and pitches with flight
        t = self.tilt
        steel, dark, light = (0.36, 0.39, 0.48, 1), (0.13, 0.14, 0.2, 1), (0.7, 0.74, 0.84, 1)
        hull = meshes.sphere('hull', steel, sub=4)
        hull.setScale(0.95, 0.95, 0.16)
        hull.reparentTo(t)
        under = meshes.sphere('under', dark, sub=3)
        under.setScale(0.7, 0.7, 0.2)
        under.setZ(-0.05)
        under.reparentTo(t)
        top = meshes.sphere('top', light, sub=3)
        top.setScale(0.62, 0.62, 0.12)
        top.setZ(0.07)
        top.reparentTo(t)
        for r in (0.5, 0.75):                               # panel seams
            seam = meshes.ring('seam', r, r + 0.02, dark)
            seam.setZ(0.115 if r < 0.6 else 0.075)
            seam.reparentTo(t)
        self.ring = t.attachNewNode('ring')                  # outer ring of plates, spun separately
        for k in range(16):
            plate = meshes.box('plate', 0.2, 0.09, 0.05, light if k % 2 else steel, (0, 1.0, 0.01))
            plate.setH(360 * k / 16)
            plate.reparentTo(self.ring)
        self.lights = self.ring.attachNewNode('lights')
        for k in range(16):
            a = 2 * math.pi * (k + 0.5) / 16
            col = (1, 0.75, 0.25, 1) if k % 2 else (0.4, 0.95, 1, 1)
            glow_card(self.lights, glow_tex, 0.15, col).setPos(1.06 * math.cos(a), 1.06 * math.sin(a), 0.01)
        body = meshes.sphere('alien', (0.25, 0.5, 0.3, 1), sub=2)
        body.setScale(0.08, 0.07, 0.12)
        body.setZ(0.2)
        body.reparentTo(t)
        head = meshes.sphere('alienhead', (0.32, 0.62, 0.36, 1), sub=3)
        head.setScale(0.1, 0.085, 0.08)
        head.setZ(0.34)
        head.reparentTo(t)
        for dx in (-0.04, 0.04):
            eye = meshes.sphere('eye', (0.02, 0.02, 0.03, 1), sub=2)
            eye.setScale(0.03, 0.02, 0.022)
            eye.setPos(dx, 0.07, 0.35)
            eye.reparentTo(t)
        dome = meshes.sphere('dome', (0.65, 0.55, 1.0, 0.38), sub=3)
        dome.setScale(0.36, 0.36, 0.32)
        dome.setZ(0.12)
        meshes.translucent(dome)
        dome.reparentTo(t)
        meshes.box('antenna', 0.015, 0.015, 0.22, light, (0, 0, 0.52)).reparentTo(t)
        self.tip = glow_card(t, glow_tex, 0.12, (1, 0.3, 0.3, 1))
        self.tip.setZ(0.64)
        self.thrusters = []                                  # engine pods with blue glow
        for k in range(3):
            a = 2 * math.pi * k / 3 + 0.5
            px, py = 0.5 * math.cos(a), 0.5 * math.sin(a)
            pod = meshes.sphere('pod', dark, sub=2)
            pod.setScale(0.12, 0.12, 0.08)
            pod.setPos(px, py, -0.13)
            pod.reparentTo(t)
            g = glow_card(t, glow_tex, 0.35, (0.4, 0.75, 1, 0.8))
            g.setPos(px, py, -0.2)
            self.thrusters.append(g)
        self.emitter = glow_card(t, glow_tex, 0.9, (1, 0.5, 0.2, 0.6), billboard=False)
        self.emitter.setP(-90)
        self.emitter.setZ(-0.24)
        self.charge = glow_card(t, glow_tex, 0.6, (1, 0.55, 0.2, 1))
        self.charge.setZ(-0.27)
        self.charge.setScale(0.01)

    def animate(self, now, speed):
        self.ring.setH(-now * 35)
        self.lights.setH(now * 140)
        self.tip.setAlphaScale(1.0 if int(now * 3) % 2 else 0.2)
        for k, g in enumerate(self.thrusters):
            g.setScale(0.8 + 0.25 * math.sin(now * 20 + k) + min(1.0, speed * 0.15))


# ------------------------------------------------------------------ obstacles
def spike_arrays(r, h, tip, sides=6):
    """Hexagonal crystal prism with a pointed tip, standing on z = 0."""
    pos, idx = [], []
    ring0 = [(r * math.cos(2 * math.pi * k / sides), r * math.sin(2 * math.pi * k / sides)) for k in range(sides)]
    for k in range(sides):
        (x0, y0), (x1, y1) = ring0[k], ring0[(k + 1) % sides]
        b = len(pos)
        pos += [(x0, y0, 0), (x1, y1, 0), (x1, y1, h), (x0, y0, h)]
        idx += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
        b = len(pos)
        pos += [(x0, y0, h), (x1, y1, h), (0, 0, h + tip)]
        idx.append((b, b + 1, b + 2))
    return meshes.flat_shade(np.array(pos, np.float32), np.array(idx))


def _crystals(rng, colour, scale=1.0, count=7):
    root = NodePath('crystals')
    for k in range(count):
        h = rng.uniform(0.18, 0.5) * scale
        r = rng.uniform(0.04, 0.08) * scale
        p, n, i = spike_arrays(r, h, r * 2.2)
        shade = rng.uniform(0.85, 1.15)
        col = tuple(min(1, c * shade) for c in colour[:3]) + (colour[3],)
        c = meshes.mesh('spike', p, n, i, col)
        c.reparentTo(root)
        c.setPos(rng.uniform(-0.16, 0.16) * scale, rng.uniform(-0.16, 0.16) * scale, -0.02)
        c.setHpr(rng.uniform(0, 360), rng.uniform(-30, 30), rng.uniform(-30, 30))
    return root


def _rubble(root, rng, colour, n=7, spread=0.38):
    for k in range(n):
        m = meshes.rock('rubble', rng, colour, sub=1)
        s = rng.uniform(0.04, 0.09)
        m.setScale(s, s, s * 0.7)
        a = rng.uniform(0, 2 * math.pi)
        d = rng.uniform(0.22, spread)
        m.setPos(d * math.cos(a), d * math.sin(a), s * 0.3)
        m.reparentTo(root)


def _wreck(rng, hard):
    """A crashed satellite: bent solar wing, a fuel tank, a dish and a truss."""
    root = NodePath('wreck')
    metal = (0.2, 0.18, 0.18, 1) if hard else (0.74, 0.76, 0.8, 1)
    gold = (0.3, 0.16, 0.08, 1) if hard else (0.86, 0.66, 0.26, 1)
    solar = (0.1, 0.08, 0.12, 1) if hard else (0.14, 0.26, 0.62, 1)
    wing = root.attachNewNode('wing')
    meshes.box('wing', 0.42, 0.02, 0.22, solar).reparentTo(wing)
    for k in range(1, 4):
        meshes.box('cellline', 0.004, 0.024, 0.22, metal, (-0.21 + k * 0.105, 0, 0)).reparentTo(wing)
    wing.setPos(-0.05, 0.08, 0.13)
    wing.setHpr(rng.uniform(-40, 40), rng.uniform(-50, -20), rng.uniform(-20, 20))
    tank = meshes.cylinder('tank', 0.07, 0.26, gold, seg=14, cap_color=metal)
    tank.setPos(0.08, -0.1, 0.08)
    tank.setH(rng.uniform(0, 180))
    tank.reparentTo(root)
    dish = meshes.sphere('dish', metal, sub=2)
    dish.setScale(0.11, 0.11, 0.025)
    dish.setPos(-0.14, -0.12, 0.06)
    dish.setP(60)
    dish.reparentTo(root)
    bar(root, (-0.22, 0.12, 0.01), (0.12, -0.05, 0.3), 0.025, metal)
    bar(root, (0.12, -0.05, 0.3), (0.2, 0.12, 0.05), 0.02, metal)
    return root


def obstacle_model(style, hard, variant, glow_tex):
    """Alien obstacles. Loose (soft) ones are laserable: warm and marked with a
    pulsing alien ring. Fused (hard) ones are dark with glowing seams and the
    laser cannot cut them."""
    rng = np.random.default_rng(1000 + variant * 7 + STYLES.index(style))
    root = NodePath('obstacle')
    if style == 'rock':
        body = (0.2, 0.13, 0.12, 1) if hard else (0.56, 0.4, 0.31, 1)
        m = meshes.rock('rock', rng, body, sub=3)
        m.setScale(0.36, 0.33, 0.32)
        m.setZ(0.16)
        m.reparentTo(root)
        for k in range(2):
            sm = meshes.rock('rock2', rng, body, sub=2)
            s = rng.uniform(0.12, 0.17)
            a = rng.uniform(0, 2 * math.pi)
            sm.setScale(s, s, s * 0.8)
            sm.setPos(0.27 * math.cos(a), 0.27 * math.sin(a), s * 0.4)
            sm.reparentTo(root)
        _rubble(root, rng, body)
        glow = (1, 0.42, 0.1, 0.95) if hard else (1, 0.55, 0.25, 0.22)
        seam = (1, 0.55, 0.15, 1)
    elif style == 'crystal':
        _crystals(rng, (0.2, 0.25, 0.7, 1) if hard else (0.6, 0.86, 1, 1)).reparentTo(root)
        _rubble(root, rng, (0.45, 0.5, 0.6, 1), 5, 0.3)
        glow = (0.35, 0.5, 1, 0.9) if hard else (0.5, 0.85, 1, 0.4)
        seam = (0.6, 0.8, 1, 1)
    else:
        _wreck(rng, hard).reparentTo(root)
        _rubble(root, rng, (0.4, 0.4, 0.42, 1), 5, 0.35)
        glow = (1, 0.38, 0.08, 0.9) if hard else (1, 0.6, 0.3, 0.15)
        seam = (1, 0.6, 0.2, 1)
    glow_card(root, glow_tex, 1.25 if hard else 0.8, glow).setZ(0.22)
    if hard:
        for k in range(7):                     # glowing seams where it fused
            a = rng.uniform(0, 2 * math.pi)
            glow_card(root, glow_tex, rng.uniform(0.08, 0.16), seam).setPos(
                0.28 * math.cos(a), 0.28 * math.sin(a), rng.uniform(0.06, 0.32))
    else:                                      # alien marker: laserable
        mark = meshes.additive(meshes.ring('mark', 0.44, 0.47, (1, 0.55, 0.2, 0.7)))
        mark.setZ(0.02)
        mark.reparentTo(root)
        root.setPythonTag('mark', mark)
    return root


def projectile_model(style, glow_tex):
    """What the saucer throws: a meteor rock, an ice chunk or a scrap of hull."""
    rng = np.random.default_rng(7)
    root = NodePath('projectile')
    body = root.attachNewNode('tumble')
    if style == 'rock':
        m = meshes.rock('m', rng, (0.35, 0.25, 0.2, 1))
        m.setScale(0.2)
        heat = (1, 0.5, 0.15, 1)
    elif style == 'crystal':
        m = _crystals(rng, (0.7, 0.9, 1, 1), 0.45)
        heat = (0.5, 0.85, 1, 1)
    else:
        m = _wreck(rng, False)
        m.setScale(0.5)
        heat = (1, 0.6, 0.25, 1)
    m.reparentTo(body)
    glow_card(root, glow_tex, 0.9, heat[:3] + (0.9,))
    glow_card(root, glow_tex, 0.35, (1, 0.97, 0.9, 1))
    root.setPythonTag('heat', heat)
    return root, body


def fragment_model(style, rng):
    if style == 'crystal':
        c = meshes.cylinder('shard', 0.03, rng.uniform(0.08, 0.16), (0.62, 0.86, 1, 1), seg=6)
        return c
    if style == 'wreck':
        return meshes.box('scrap', rng.uniform(0.05, 0.1), 0.02, rng.uniform(0.04, 0.08), (0.7, 0.72, 0.76, 1))
    m = meshes.rock('chip', rng, (0.55, 0.4, 0.3, 1), sub=1)
    m.setScale(rng.uniform(0.04, 0.08))
    return m
