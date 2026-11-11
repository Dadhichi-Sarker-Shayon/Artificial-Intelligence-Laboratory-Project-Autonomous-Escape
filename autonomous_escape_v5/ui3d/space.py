"""Version 5: the same game in open space above Mars.

The grid becomes a flight plane (z = 0). Blocked tiles are clusters of
tumbling asteroids; open space is filled with drifting debris and dust well
above and below the plane, with a faint navigation lattice marking the lanes.
The start is a launch dock off Phobos, the goal is the docking port of an
orbital station.
"""
import math

import numpy as np
from panda3d.core import (AmbientLight, DirectionalLight, Geom, GeomNode, GeomPoints, GeomVertexData,
                          GeomVertexFormat, GeomVertexWriter, NodePath, Vec3, Vec4)
from scipy import ndimage

from . import meshes
from .models import bar, glow_card
from . import spacelook as look

TILE_PX = 36
SUN_DIR = Vec3(0.62, -0.7, -0.38).normalized()


class FlatGround:
    """Space has no ground: everything flies on the z = 0 plane."""

    def __init__(self, n):
        self.n = n

    def z(self, x, y):
        return 0.0


def _merge(parts):
    P, N, C, I, base = [], [], [], [], 0
    for p, n, c, i in parts:
        P.append(p)
        N.append(n)
        C.append(c)
        I.append(i + base)
        base += len(p)
    return np.concatenate(P), np.concatenate(N), np.concatenate(I), np.concatenate(C)


def _rock_part(rng, templates, centre, scale, colour, rot):
    p, n0, f = templates[rng.integers(len(templates))]
    sx, sy, sz = scale
    q = p * np.array([sx, sy, sz], np.float32)
    ca, sa = math.cos(rot), math.sin(rot)
    q = np.stack([q[:, 0] * ca - q[:, 1] * sa, q[:, 0] * sa + q[:, 1] * ca, q[:, 2]], -1) + np.array(centre, np.float32)
    inv = n0 / np.array([sx, sy, sz], np.float32)
    inv = np.stack([inv[:, 0] * ca - inv[:, 1] * sa, inv[:, 0] * sa + inv[:, 1] * ca, inv[:, 2]], -1)
    nrm = inv / (np.linalg.norm(inv, axis=1, keepdims=True) + 1e-9)
    face = rng.uniform(0.75, 1.1, len(q) // 3).repeat(3)
    col = np.stack([colour[0] * face, colour[1] * face, colour[2] * face, np.ones_like(face)], -1).astype(np.float32)
    return q, nrm, col, f


def build_world(parent, gmap, glow_tex):
    n = gmap.size
    rng = np.random.default_rng(gmap.seed * 7 + 11)
    root = parent.attachNewNode('space')
    spinners = []

    def template(sub, rough):
        p, f = meshes.rock_arrays(rng, sub, rough=rough)
        return meshes.flat_shade(p, f)
    big = [template(2, 0.5) for _ in range(10)]
    small = [template(1, 0.55) for _ in range(8)]

    # asteroid clusters on blocked tiles: one tumbling main body each, plus companions
    tint = [(0.5, 0.47, 0.44), (0.42, 0.4, 0.4), (0.55, 0.5, 0.45), (0.5, 0.52, 0.56)]
    trng = np.random.default_rng(77)
    templates = [look.asteroid_template(trng, np.array(tint[k % len(tint)], np.float32)) for k in range(12)]
    companions = []
    for r in range(n):
        for c in range(n):
            if not gmap.walls[r][c]:
                continue
            col = tint[rng.integers(len(tint))]
            node = NodePath('asteroid')
            s = rng.uniform(0.36, 0.55)
            holder = node.attachNewNode('shape')
            templates[rng.integers(len(templates))].instanceTo(holder)
            holder.setScale(s, s * rng.uniform(0.85, 1.1), s * rng.uniform(0.85, 1.05))
            node.setPos(c + rng.uniform(-0.1, 0.1), -r + rng.uniform(-0.1, 0.1), rng.uniform(-0.25, 0.25))
            node.setHpr(*rng.uniform(0, 360, 3))
            node.reparentTo(root)
            spinners.append((node, Vec3(*rng.uniform(-12, 12, 3))))
            for k in range(rng.integers(1, 4)):
                a = rng.uniform(0, 2 * math.pi)
                d = rng.uniform(0.25, 0.5)
                cs = rng.uniform(0.06, 0.16)
                companions.append(_rock_part(rng, small, (c + d * math.cos(a), -r + d * math.sin(a),
                                                          rng.uniform(-0.6, 0.6)), (cs, cs, cs), col,
                                             rng.uniform(0, 6.3)))
    comp = meshes.mesh('companions', *_merge(companions)[:3], _merge(companions)[3])
    comp.reparentTo(root)

    # drifting debris above and below the flight plane (never on it)
    cloud = root.attachNewNode('cloud')
    bits = []
    for k in range(int(900 * (n / 29) ** 2)):
        x, y = rng.uniform(-6, n + 6), -rng.uniform(-6, n + 6)
        z = rng.choice([-1, 1]) * rng.uniform(0.9, 6)
        bits.append(look.debris_part(rng, (x, y, z)))
    P, N, I, C = _merge(bits)
    meshes.mesh('debris', P, N, I, C).reparentTo(cloud)
    cloud.setPos(n / 2, -n / 2, 0)
    for ch in cloud.getChildren():
        ch.setPos(-n / 2, n / 2, 0)

    # big distant asteroids
    far = root.attachNewNode('far')
    for k in range(26):
        a = rng.uniform(0, 2 * math.pi)
        d = rng.uniform(35, 75)
        s = rng.uniform(1.5, 6)
        node = far.attachNewNode('farrock')
        holder = node.attachNewNode('shape')
        templates[rng.integers(len(templates))].instanceTo(holder)
        holder.setScale(s, s, s * 0.8)
        node.setPos(n / 2 + d * math.cos(a), -n / 2 + d * math.sin(a), rng.uniform(-14, 10))
        spinners.append((node, Vec3(*rng.uniform(-3, 3, 3))))
    far.setShaderOff(1)

    # space dust: faint points everywhere, for parallax when flying
    vd = GeomVertexData('dust', GeomVertexFormat.getV3c4(), Geom.UHStatic)
    vw, cw = GeomVertexWriter(vd, 'vertex'), GeomVertexWriter(vd, 'color')
    pts = GeomPoints(Geom.UHStatic)
    for i in range(4000):
        vw.addData3(rng.uniform(-10, n + 10), -rng.uniform(-10, n + 10), rng.uniform(-8, 8))
        b = rng.uniform(0.25, 0.7)
        cw.addData4(b, b, b * 1.1, 1)
        pts.addVertex(i)
    g = Geom(vd)
    g.addPrimitive(pts)
    gn = GeomNode('dust')
    gn.addGeom(g)
    dust = root.attachNewNode(gn)
    dust.setRenderModeThickness(1.5)
    dust.setLightOff(1)
    dust.setShaderOff(1)

    # navigation lattice: faint dots on every open lane tile
    vd = GeomVertexData('lattice', GeomVertexFormat.getV3c4(), Geom.UHStatic)
    vw, cw = GeomVertexWriter(vd, 'vertex'), GeomVertexWriter(vd, 'color')
    pts = GeomPoints(Geom.UHStatic)
    for i, (r, c) in enumerate(gmap.open_tiles):
        vw.addData3(c, -r, -0.02)
        cw.addData4(0.3, 0.8, 0.9, 0.5)
        pts.addVertex(i)
    g = Geom(vd)
    g.addPrimitive(pts)
    gn = GeomNode('lattice')
    gn.addGeom(g)
    lat = root.attachNewNode(gn)
    lat.setRenderModeThickness(2)
    lat.setAlphaScale(0.55)
    meshes.additive(lat)

    root.setPythonTag('spinners', spinners)
    root.setPythonTag('cloud', cloud)
    return root, FlatGround(n), {'image': minimap_image(gmap)}


def minimap_image(gmap):
    """Top-down map for the HUD: deep space with the asteroid clusters."""
    n = gmap.size
    S = n * TILE_PX
    walls = np.kron(np.array(gmap.walls, np.float32), np.ones((TILE_PX, TILE_PX), np.float32))
    rng = np.random.default_rng(gmap.seed)
    blob = ndimage.gaussian_filter(walls, 6) * (0.7 + 0.6 * ndimage.gaussian_filter(rng.random((S, S)), 3))
    img = np.zeros((S, S, 3), np.float32)
    img[...] = (10, 12, 28)
    stars = rng.random((S, S)) > 0.998
    img[stars] = (160, 160, 190)
    rock = np.clip(blob * 1.6, 0, 1)[..., None]
    img = img * (1 - rock) + np.array([120, 112, 104], np.float32) * rock
    return np.clip(img, 0, 255).astype(np.uint8)


def lights(base, render, n):
    amb = AmbientLight('ambient')
    amb.setColor(Vec4(0.2, 0.2, 0.26, 1))
    render.setLight(render.attachNewNode(amb))
    sun = DirectionalLight('sun')
    sun.setColor(Vec4(1.35, 1.28, 1.18, 1))
    sun_np = render.attachNewNode(sun)
    sun_np.lookAt(SUN_DIR)
    sun_np.setPos(n / 2 - SUN_DIR.x * 30, -n / 2 - SUN_DIR.y * 30, -SUN_DIR.z * 30)
    lens = sun.getLens()
    lens.setFilmSize(n + 8, n + 8)
    lens.setNearFar(5, 80)
    render.setLight(sun_np)
    glow = DirectionalLight('marsglow')               # warm light reflected up from Mars below
    glow.setColor(Vec4(0.3, 0.13, 0.07, 1))
    g_np = render.attachNewNode(glow)
    g_np.lookAt(Vec3(0.1, 0.2, 1))
    render.setLight(g_np)
    return sun_np


def dress_sky(sky, glow_tex):
    """Space backdrop: Mars filling the view below, nebula glow, Phobos."""
    look.skydome(sky)
    stars = sky.find('**/stars')
    if not stars.isEmpty():
        stars.setRenderModeThickness(1.6)
    mars = sky.find('**/mars')
    if not mars.isEmpty():
        mdir = Vec3(0.15, 0.25, -1).normalized()
        mars.setPos(mdir * 380)
        mars.setScale(260)
        mars.setTexture(meshes.texture(look.mars_texture(), 'mars_hd'), 1)
        halo = glow_card(sky, glow_tex, 640, (1.0, 0.55, 0.35, 0.22))
        halo.setPos(mdir * 400)
    phobos = meshes.sphere('phobos', (0.45, 0.42, 0.4, 1), sub=3)
    phobos.setScale(16, 13, 12)
    phobos.setPos(Vec3(-1, 1, -0.25).normalized() * 380)
    phobos.reparentTo(sky)


# ------------------------------------------------------------------ craft
class ShuttleModel:
    """The escape shuttle (PATHFINDER-7): faceted white hull, delta wings,
    twin tail fins, three engines whose glow follows the thrust, and RCS jets.
    Same interface as the rover: `root`, `head` (cockpit) and roll_wheels()."""

    def __init__(self, glow_tex):
        self.root = NodePath('shuttle')
        self.bank = self.root.attachNewNode('bank')
        b = self.bank
        b.setZ(0.18)
        white, grey, dark = (0.92, 0.93, 0.95, 1), (0.55, 0.57, 0.62, 1), (0.12, 0.13, 0.16, 1)
        hull = meshes.sphere('hull', white, sub=3)
        hull.setScale(0.13, 0.36, 0.1)
        hull.reparentTo(b)
        belly = meshes.sphere('belly', dark, sub=3)
        belly.setScale(0.12, 0.33, 0.06)
        belly.setZ(-0.04)
        belly.reparentTo(b)
        glass = meshes.sphere('glass', (0.1, 0.18, 0.3, 1), sub=3)
        glass.setScale(0.08, 0.1, 0.05)
        glass.setPos(0, 0.2, 0.06)
        glass.reparentTo(b)
        stripe = meshes.box('stripe', 0.262, 0.05, 0.01, (0.95, 0.5, 0.15, 1), (0, 0.05, 0.085))
        stripe.reparentTo(b)
        wing = np.array([(-0.09, 0.12, 0), (-0.42, -0.22, 0), (-0.38, -0.3, 0), (-0.09, -0.26, 0)], np.float32)
        for side in (-1, 1):
            w = wing.copy()
            w[:, 0] *= side
            top = w + np.array([0, 0, 0.012], np.float32)
            bot = w - np.array([0, 0, 0.012], np.float32)
            pos = np.concatenate([top, bot])
            idx = np.array([(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6)])
            if side < 0:
                idx = idx[:, ::-1]
            p, nn, ii = meshes.flat_shade(pos, idx)
            meshes.mesh('wing', p, nn, ii, white).reparentTo(b)
            meshes.box('tip', 0.02, 0.1, 0.012, (0.95, 0.5, 0.15, 1), (side * 0.4, -0.26, 0)).reparentTo(b)
            fin = meshes.box('fin', 0.012, 0.12, 0.14, white, (side * 0.06, -0.26, 0.1))
            fin.setR(side * 15)
            fin.reparentTo(b)
        meshes.box('engines', 0.2, 0.06, 0.08, grey, (0, -0.33, 0)).reparentTo(b)
        self.flames = []
        for dx in (-0.06, 0, 0.06):
            noz = meshes.cylinder('nozzle', 0.028, 0.05, dark, seg=12)
            noz.setH(90)
            noz.setPos(dx, -0.37, 0)
            noz.reparentTo(b)
            core = glow_card(b, glow_tex, 0.14, (0.75, 0.9, 1, 1))
            core.setPos(dx, -0.41, 0)
            plume = glow_card(b, glow_tex, 0.35, (0.35, 0.6, 1, 0.7))
            plume.setPos(dx, -0.48, 0)
            self.flames.append((core, plume))
        self.rcs = []
        for sx in (-1, 1):
            g = glow_card(b, glow_tex, 0.12, (0.9, 0.95, 1, 1))
            g.setPos(sx * 0.13, 0.26, 0)
            g.hide()
            self.rcs.append(g)
        for sx, col in ((-1, (1, 0.2, 0.2, 1)), (1, (0.2, 1, 0.3, 1))):    # navigation lights
            glow_card(b, glow_tex, 0.08, col).setPos(sx * 0.41, -0.25, 0.01)
        self.head = b.attachNewNode('head')                 # cockpit: the pilot camera sits here
        self.head.setPos(0, 0.22, 0.09)
        self.thrust = 0.0
        self.bank_angle = 0.0
        from panda3d.core import Material
        mat = Material('hull')
        mat.setSpecular((0.7, 0.7, 0.75, 1))
        mat.setShininess(48)
        self.root.setMaterial(mat)

    def roll_wheels(self, distance, steer):
        pass                                             # flight effects come from engines()

    def engines(self, a_fwd, speed, yaw_acc, now):
        """Main engines glow with forward thrust (and idle at cruise), retro jets
        fire when braking, RCS nose jets puff when yawing."""
        want = max(0.0, min(1.0, a_fwd * 0.35 + speed * 0.25))
        self.thrust += (want - self.thrust) * 0.2
        flicker = 1 + 0.08 * math.sin(now * 55)
        for core, plume in self.flames:
            core.setScale((0.35 + 0.9 * self.thrust) * flicker)
            plume.setScale((0.15 + 1.6 * self.thrust) * flicker)
            plume.setAlphaScale(0.25 + 0.75 * self.thrust)
        for k, g in enumerate(self.rcs):
            on = (yaw_acc > 60 and k == 0) or (yaw_acc < -60 and k == 1) or a_fwd < -1.2
            g.show() if on else g.hide()


def station_model(glow_tex):
    """Orbital station: hub, rotating habitat ring on spokes, solar arrays and a
    docking arm whose port sits at the root's origin, facing +y/-x."""
    root = NodePath('station')
    body = root.attachNewNode('body')
    body.setPos(2.6, -2.6, 1.6)
    white, grey, gold = (0.88, 0.89, 0.92, 1), (0.5, 0.52, 0.58, 1), (0.86, 0.66, 0.26, 1)
    hub = meshes.sphere('hub', white, sub=3)
    hub.setScale(0.75)
    hub.reparentTo(body)
    core = meshes.cylinder('core', 0.35, 3.2, grey, seg=20)
    core.setR(90)
    core.reparentTo(body)
    ring = body.attachNewNode('ring')
    for k in range(24):
        seg = meshes.box('hab', 0.62, 0.4, 0.3, white if k % 3 else gold, (0, 2.3, 0))
        seg.setH(360 * k / 24)
        seg.reparentTo(ring)
    for k in range(4):
        bar(ring, (0, 0, 0), (2.3 * math.cos(k * math.pi / 2), 2.3 * math.sin(k * math.pi / 2), 0), 0.08, grey)
    for k in range(12):
        a = 2 * math.pi * k / 12
        glow_card(ring, glow_tex, 0.18, (1, 0.9, 0.6, 1)).setPos(2.55 * math.cos(a), 2.55 * math.sin(a), 0.1)
    for sz in (-1, 1):
        mast = meshes.box('mast', 0.08, 0.08, 1.8, grey, (0, 0, sz * 1.6))
        mast.reparentTo(body)
        for sx in (-1, 1):
            panel = meshes.box('array', 2.2, 0.03, 0.8, (0.14, 0.26, 0.62, 1), (sx * 1.25, 0, sz * 2.4))
            panel.reparentTo(body)
    # docking arm from the hub down to the port at the origin
    bar(root, (2.6, -2.6, 1.6), (0.55, -0.55, 0.18), 0.22, grey)
    port = meshes.cylinder('port', 0.32, 0.3, white, seg=24, cap_color=(0.15, 0.16, 0.2, 1))
    port.setPos(0.35, -0.35, 0.18)
    port.setH(-45)
    port.reparentTo(root)
    guides = root.attachNewNode('guides')
    for k in range(8):
        a = 2 * math.pi * k / 8
        g = glow_card(guides, glow_tex, 0.12, (0.3, 1, 0.45, 1))
        g.setPos(0.2 + 0.3 * math.cos(a) * 0.7, -0.2 + 0.3 * math.cos(a) * 0.7, 0.18 + 0.3 * math.sin(a))
    beacons = root.attachNewNode('beacons')
    for pos, col in (((2.6, -2.6, 3.3), (1, 0.2, 0.2, 1)), ((2.6, -2.6, -0.1), (1, 0.2, 0.2, 1)),
                     ((4.9, -2.6, 1.6), (0.3, 1, 0.4, 1)), ((0.3, -2.6, 1.6), (1, 1, 1, 1))):
        glow_card(beacons, glow_tex, 0.3, col).setPos(*pos)
    from panda3d.core import Material
    mat = Material('station')
    mat.setSpecular((0.6, 0.6, 0.65, 1))
    mat.setShininess(30)
    root.setMaterial(mat)
    root.setPythonTag('ring', ring)
    root.setPythonTag('guides', guides)
    root.setPythonTag('beacons', beacons)
    return root


def launch_dock_model(glow_tex):
    root = NodePath('dock')
    meshes.cylinder('deck', 0.7, 0.08, (0.4, 0.42, 0.48, 1), seg=6).reparentTo(root)
    root.getChild(0).setR(90)
    root.getChild(0).setZ(-0.12)
    meshes.box('gantry', 0.08, 0.08, 1.1, (0.75, 0.76, 0.8, 1), (-0.6, 0.6, 0.4)).reparentTo(root)
    bar(root, (-0.6, 0.6, 0.9), (-0.1, 0.1, 0.5), 0.05, (0.75, 0.76, 0.8, 1))
    for k in range(6):
        a = 2 * math.pi * k / 6
        glow_card(root, glow_tex, 0.1, (1, 0.7, 0.2, 1)).setPos(0.68 * math.cos(a), 0.68 * math.sin(a), -0.06)
    return root
