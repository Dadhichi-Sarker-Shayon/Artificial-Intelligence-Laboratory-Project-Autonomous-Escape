"""The 3D Phobos scene: terrain, rocks, sky (stars, Mars, the Sun) and the
models for the rover, the alien saucer and the alien's boulders.

World axes: x = grid column, y = -grid row (north is +y), z = up.
One grid tile is one world unit.
"""
import math
import random

import numpy as np
from panda3d.core import (AmbientLight, DirectionalLight, GeomNode, GeomPoints, Geom, GeomVertexData,
                          GeomVertexFormat, GeomVertexWriter, NodePath, Vec3, Vec4)

from ui import terrain
from . import meshes

TILE_PX = 36          # resolution of the shared terrain layers (pixels per tile)
ZS = 1.6              # vertical exaggeration of the ground height
SUN_DIR = Vec3(0.62, -0.7, -0.38).normalized()    # light travels this way: low sun in the north-west
HORIZON_R = 210.0     # Phobos is tiny: the ground visibly curves away


def tile_xy(p):
    r, c = p
    return float(c), float(-r)


class Ground:
    """Height lookups on the terrain mesh, in world units."""

    def __init__(self, heights, n):
        self.h = heights
        self.n = n
        self.S = heights.shape[0]

    def z(self, x, y):
        px = (x + 0.5) * TILE_PX
        py = (-y + 0.5) * TILE_PX
        i = int(min(self.S - 1, max(0, py)))
        j = int(min(self.S - 1, max(0, px)))
        return float(self.h[i, j]) / TILE_PX * ZS


def build_world(parent, gmap, glow_tex):
    """Build the static scene under `parent`. Returns (Ground, fields)."""
    n = gmap.size
    f = terrain.fields(gmap, TILE_PX)
    root = parent.attachNewNode('world')

    # terrain mesh textured with the lit regolith image
    step_px = 4
    hs = f['ground'][::step_px, ::step_px]
    hs = np.pad(hs, ((0, 1), (0, 1)), mode='edge')
    tmesh = meshes.heightfield('terrain', hs / TILE_PX, -0.5, 0.5, step_px / TILE_PX, ZS)
    img = np.clip(f['image'].astype(np.float32) * 1.3 + 4, 0, 255).astype(np.uint8)
    tmesh.setTexture(meshes.texture(img, 'regolith'))
    detail = detail_texture()
    from panda3d.core import TextureStage
    ts = TextureStage('detail')
    ts.setMode(TextureStage.MModulate)
    tmesh.setTexture(ts, detail)
    tmesh.setTexScale(ts, n * 1.6, n * 1.6)
    tmesh.reparentTo(root)

    # the plain beyond the field, curving down to a close horizon
    edge = np.concatenate([img[:8].reshape(-1, 3), img[-8:].reshape(-1, 3)]).mean(0) / 255
    skirt = horizon_mesh(n, edge)
    skirt.setTexture(ts, detail)
    skirt.setTexScale(ts, 260, 260)
    skirt.setShaderOff(1)
    skirt.reparentTo(root)

    ground = Ground(f['ground'], n)
    _rocks(root, f['rocks'], ground)
    _horizon_rocks(root, n, ground)
    return root, ground, f


def drop(d):
    """How far the ground has curved away at distance d outside the field."""
    return -(d * d) / (2 * HORIZON_R)


def outside(x, y, n):
    dx = max(-0.5 - x, 0, x - (n - 0.5))
    dy = max(y - 0.5, 0, -(n - 0.5) - y)
    return math.hypot(dx, dy)


def horizon_mesh(n, edge, R=150, res=110):
    """A wide regolith plain around the field that curves away like a small
    moon's surface. Under the field itself it drops out of sight."""
    cx, cy = n / 2 - 0.5, -(n / 2 - 0.5)
    lin = np.linspace(-R, R, res)
    X, Y = np.meshgrid(cx + lin, cy - lin)
    dx = np.maximum(np.maximum(-0.5 - X, 0), X - (n - 0.5))
    dy = np.maximum(np.maximum(Y - 0.5, 0), -(n - 0.5) - Y)
    D = np.hypot(dx, dy)
    Z = -(D * D) / (2 * HORIZON_R) - 0.02
    Z[D == 0] = -1.5
    rng = np.random.default_rng(3)
    shade = 0.85 + 0.25 * rng.random(X.shape)
    from scipy import ndimage
    shade = ndimage.gaussian_filter(shade, 2)
    rows, cols = X.shape
    pos = np.stack([X, Y, Z], -1).reshape(-1, 3)
    gy, gx = np.gradient(Z, lin[1] - lin[0])
    nrm = np.stack([-gx, gy, np.ones_like(Z)], -1).reshape(-1, 3)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    col = np.concatenate([np.clip(edge[None, :] * 1.15 * shade.reshape(-1, 1), 0, 1),
                          np.ones((rows * cols, 1))], 1)
    uv = np.stack([(X - cx) / (2 * R) + 0.5, (Y - cy) / (2 * R) + 0.5], -1).reshape(-1, 2)
    ii, jj = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing='ij')
    a = (ii * cols + jj).reshape(-1)
    b, c, d = a + 1, a + cols, a + cols + 1
    idx = np.concatenate([np.stack([a, c, d], -1), np.stack([a, d, b], -1)])
    return meshes.mesh('horizon', pos, nrm, idx, col.astype(np.float32), uv)


_DETAIL = None


def detail_texture(size=256):
    """Tileable fine regolith grain, multiplied over the terrain up close."""
    global _DETAIL
    if _DETAIL is None:
        from scipy import ndimage
        rng = np.random.default_rng(9)
        g = ndimage.gaussian_filter(rng.random((size, size)), 0.8, mode='wrap')
        g2 = ndimage.gaussian_filter(rng.random((size, size)), 4, mode='wrap')
        v = (g - g.mean()) / (g.std() + 1e-6) * 0.07 + (g2 - g2.mean()) / (g2.std() + 1e-6) * 0.05
        v = np.clip(0.86 + v, 0.6, 1.0)
        img = np.repeat((v * 255).astype(np.uint8)[..., None], 3, -1)
        tex = meshes.texture(img, 'detail')
        from panda3d.core import Texture
        tex.setWrapU(Texture.WM_repeat)
        tex.setWrapV(Texture.WM_repeat)
        _DETAIL = tex
    return _DETAIL


def _rocks(root, rocks, ground):
    """All terrain rocks merged into one mesh, placed where the map texture has them."""
    rng = np.random.default_rng(7)
    def template(sub):
        p, f = meshes.rock_arrays(rng, sub, rough=0.4)
        return meshes.flat_shade(p, f)
    templates = {2: [template(2) for _ in range(8)], 1: [template(1) for _ in range(8)]}
    P, N, C, I = [], [], [], []
    base = 0
    for k, (x, y, r, h) in enumerate(rocks):
        sub = 2 if r > 8 else 1
        p, nrm0, f = templates[sub][k % len(templates[sub])]
        wx, wy = x / TILE_PX - 0.5, -(y / TILE_PX - 0.5)
        s = r / TILE_PX * 0.98
        sz = max(h, r) / TILE_PX * 0.95
        a = (k * 2.399) % (2 * math.pi)
        ca, sa = math.cos(a), math.sin(a)
        q = p * np.array([s, s, sz], np.float32)
        q = np.stack([q[:, 0] * ca - q[:, 1] * sa, q[:, 0] * sa + q[:, 1] * ca, q[:, 2]], -1)
        q += np.array([wx, wy, ground.z(wx, wy) + sz * 0.2], np.float32)
        inv = nrm0 / np.array([s, s, sz], np.float32)
        inv = np.stack([inv[:, 0] * ca - inv[:, 1] * sa, inv[:, 0] * sa + inv[:, 1] * ca, inv[:, 2]], -1)
        nrm = inv / (np.linalg.norm(inv, axis=1, keepdims=True) + 1e-9)
        g = 0.42 + 0.1 * ((k * 37) % 10) / 10
        face = rng.uniform(0.8, 1.1, len(q) // 3).repeat(3)
        col = np.stack([g * 1.03 * face, g * face, g * 0.96 * face, np.ones_like(face)], -1).astype(np.float32)
        P.append(q)
        N.append(nrm)
        C.append(col)
        I.append(f + base)
        base += len(q)
    m = meshes.mesh('rocks', np.concatenate(P), np.concatenate(N), np.concatenate(I), np.concatenate(C))
    m.reparentTo(root)


def _horizon_rocks(root, n, ground):
    """Scattered boulders and low hills outside the play area, so the field
    does not end at a cliff when seen from the pilot seat."""
    rng = np.random.default_rng(11)
    P, N, C, I = [], [], [], []
    base = 0
    for k in range(420):
        while True:
            x, y = rng.uniform(-45, n + 45), rng.uniform(-n - 45, 45)
            if not (-1.5 < x < n + 0.5 and -n - 0.5 < y < 1.5):
                break
        far = min(abs(x - n / 2), abs(y + n / 2))
        s = rng.uniform(0.15, 0.6) * (1 + far / 12)
        p, f = meshes.rock_arrays(rng, 1, rough=0.5)
        p, _, f = meshes.flat_shade(p, f)
        q = p * np.array([s, s, s * rng.uniform(0.5, 0.9)], np.float32) +             np.array([x, y, drop(outside(x, y, n)) - 0.05], np.float32)
        P.append(q)
        N.append(meshes.flat_shade(q, f)[1])
        C.append(np.tile(np.array([0.42, 0.4, 0.38, 1], np.float32), (len(q), 1)))
        I.append(f + base)
        base += len(q)
    m = meshes.mesh('far_rocks', np.concatenate(P), np.concatenate(N), np.concatenate(I), np.concatenate(C))
    m.setShaderOff(1)         # outside the shadow map: plain lighting
    m.reparentTo(root)


# ------------------------------------------------------------------ sky
def build_sky(base, glow_tex):
    """Stars, Mars and the Sun on a dome that follows the camera."""
    sky = base.camera.attachNewNode('sky')
    sky.setCompass()
    sky.setBin('background', 0)
    sky.setDepthWrite(False)
    sky.setDepthTest(False)
    sky.setLightOff(1)
    sky.setShaderOff(1)

    # stars
    vd = GeomVertexData('stars', GeomVertexFormat.getV3c4(), Geom.UHStatic)
    vw, cw = GeomVertexWriter(vd, 'vertex'), GeomVertexWriter(vd, 'color')
    rng = random.Random(3)
    pts = GeomPoints(Geom.UHStatic)
    for i in range(2200):
        v = Vec3(rng.gauss(0, 1), rng.gauss(0, 1), abs(rng.gauss(0, 1)) * 0.9 + 0.02).normalized() * 500
        vw.addData3(v)
        b = rng.random() ** 3 * 0.8 + 0.2
        tint = rng.choice(((1, 1, 1), (0.8, 0.85, 1), (1, 0.9, 0.8)))
        cw.addData4(b * tint[0], b * tint[1], b * tint[2], 1)
        pts.addVertex(i)
    g = Geom(vd)
    g.addPrimitive(pts)
    node = GeomNode('stars')
    node.addGeom(g)
    stars = sky.attachNewNode(node)
    stars.setRenderModeThickness(2)

    # Mars: from Phobos it fills a huge part of the sky
    mars = meshes.sphere('mars', (1, 1, 1, 1), sub=4)
    mars.setTexture(meshes.texture(_mars_texture(), 'mars'))
    _mars_uv_and_light(mars)
    mars.reparentTo(sky)
    mars.setPos(Vec3(1, 0.35, 0.2).normalized() * 380)
    mars.setScale(140)
    deimos = meshes.sphere('deimos', (0.7, 0.66, 0.62, 1), sub=2)
    deimos.reparentTo(sky)
    deimos.setPos(Vec3(-0.6, -0.9, 0.55).normalized() * 380)
    deimos.setScale(2.2)
    mars.setHpr(30, 10, -20)

    # the Sun: small and blinding, opposite its light
    sun = sky.attachNewNode('sun')
    sun.setPos(-SUN_DIR * 450)
    for size, alpha in ((60, 0.9), (180, 0.35), (420, 0.12)):
        card = _card(sun, size, glow_tex, (1, 0.97, 0.88, alpha))
        card.setBillboardPointEye()
    return sky


def _card(parent, size, tex, color):
    from panda3d.core import CardMaker
    cm = CardMaker('card')
    cm.setFrame(-size / 2, size / 2, -size / 2, size / 2)
    np_ = parent.attachNewNode(cm.generate())
    np_.setTexture(tex)
    np_.setColor(*color)
    meshes.additive(np_)
    return np_


def _mars_texture(w=512, h=256):
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    tex = np.zeros((h, w), np.float32)
    for k in range(5):
        g = rng.random((4 * 2 ** k, 8 * 2 ** k))
        from scipy import ndimage
        z = ndimage.zoom(g, (h / g.shape[0], w / g.shape[1]), order=3, mode='grid-wrap')
        tex += z[:h, :w] / 2 ** k
    tex = (tex - tex.min()) / np.ptp(tex)
    base = np.array([186, 92, 56], np.float32)
    dark = np.array([110, 52, 36], np.float32)
    img = base * (1 - tex[..., None] * 0.8) + dark * tex[..., None] * 0.8
    lat = np.abs(yy - h / 2) / (h / 2)
    cap = np.clip((lat - 0.86) * 10, 0, 1)[..., None]
    img = img * (1 - cap) + np.array([235, 228, 220]) * cap
    return np.clip(img, 0, 255).astype(np.uint8)


def _mars_uv_and_light(mars):
    """Spherical UVs plus a day/night terminator baked into vertex colours."""
    from panda3d.core import GeomVertexRewriter
    gnode = mars.node()
    geom = gnode.modifyGeom(0)
    vd = geom.modifyVertexData()
    vr = GeomVertexRewriter(vd, 'vertex')
    tw = GeomVertexRewriter(vd, 'texcoord')
    cw = GeomVertexRewriter(vd, 'color')
    light = -SUN_DIR
    while not vr.isAtEnd():
        v = vr.getData3()
        u = 0.5 + math.atan2(v.y, v.x) / (2 * math.pi)
        t = 0.5 + math.asin(max(-1, min(1, v.z))) / math.pi
        tw.setData2(u, t)
        lam = max(0.0, v.normalized().dot(light))
        b = 0.12 + 0.95 * lam ** 0.7
        cw.setData4(b, b * 0.97, b * 0.95, 1)


def lights(base, render, n):
    amb = AmbientLight('ambient')
    amb.setColor(Vec4(0.24, 0.23, 0.25, 1))       # no air: shadows stay dark
    render.setLight(render.attachNewNode(amb))
    sun = DirectionalLight('sun')
    sun.setColor(Vec4(1.3, 1.24, 1.14, 1))
    sun_np = render.attachNewNode(sun)
    sun_np.lookAt(SUN_DIR)
    sun_np.setPos(n / 2 - SUN_DIR.x * 30, -n / 2 - SUN_DIR.y * 30, -SUN_DIR.z * 30)
    lens = sun.getLens()
    lens.setFilmSize(n + 8, n + 8)
    lens.setNearFar(5, 80)
    render.setLight(sun_np)
    fill = DirectionalLight('marsshine')            # reddish light reflected off Mars
    fill.setColor(Vec4(0.10, 0.05, 0.03, 1))
    fill_np = render.attachNewNode(fill)
    fill_np.lookAt(Vec3(-0.15, -1, -0.3))
    render.setLight(fill_np)
    return sun_np


# ------------------------------------------------------------------ models
def rover_model():
    """Rover facing +y: white chassis, blue solar deck, four wheels, camera mast."""
    root = NodePath('rover')
    meshes.box('chassis', 0.42, 0.5, 0.12, (0.92, 0.93, 0.95, 1), (0, 0, 0.2)).reparentTo(root)
    meshes.box('belly', 0.36, 0.44, 0.06, (0.2, 0.22, 0.26, 1), (0, 0, 0.12)).reparentTo(root)
    deck = meshes.box('panel', 0.5, 0.42, 0.025, (0.13, 0.25, 0.65, 1), (0, -0.04, 0.275))
    deck.reparentTo(root)
    for i in range(1, 4):
        meshes.box('cell', 0.008, 0.42, 0.03, (0.35, 0.5, 0.9, 1), (-0.25 + 0.125 * i, -0.04, 0.28)).reparentTo(root)
    for sx in (-1, 1):
        for sy in (-1, 1):
            w = meshes.cylinder('wheel', 0.09, 0.07, (0.12, 0.13, 0.16, 1), cap_color=(0.45, 0.85, 0.95, 1))
            w.setPos(sx * 0.27, sy * 0.19, 0.09)
            w.reparentTo(root)
            meshes.box('strut', 0.12, 0.03, 0.03, (0.8, 0.82, 0.86, 1), (sx * 0.2, sy * 0.19, 0.15)).reparentTo(root)
    meshes.box('mast', 0.04, 0.04, 0.26, (0.9, 0.9, 0.92, 1), (0, 0.2, 0.38)).reparentTo(root)
    meshes.box('head', 0.14, 0.07, 0.07, (0.95, 0.78, 0.25, 1), (0, 0.21, 0.52)).reparentTo(root)
    meshes.box('lens', 0.05, 0.02, 0.04, (0.05, 0.05, 0.08, 1), (0.03, 0.25, 0.52)).reparentTo(root)
    meshes.box('antenna', 0.015, 0.015, 0.2, (0.85, 0.85, 0.9, 1), (-0.15, -0.18, 0.38)).reparentTo(root)
    return root


def ufo_model(glow_tex):
    root = NodePath('ufo')
    hull = meshes.sphere('hull', (0.2, 0.4, 0.85, 1), sub=3)
    hull.setScale(0.75, 0.75, 0.16)
    hull.reparentTo(root)
    rim = meshes.sphere('rim', (0.45, 0.65, 1.0, 1), sub=3)
    rim.setScale(0.62, 0.62, 0.1)
    rim.setZ(0.06)
    rim.reparentTo(root)
    dome = meshes.sphere('dome', (0.85, 0.5, 0.95, 0.8), sub=3)
    dome.setScale(0.32, 0.32, 0.26)
    dome.setZ(0.1)
    meshes.translucent(dome)
    dome.reparentTo(root)
    for k in range(10):
        a = 2 * math.pi * k / 10
        light = _card(root, 0.16, glow_tex, (1, 0.85, 0.35, 1))
        light.setPos(0.7 * math.cos(a), 0.7 * math.sin(a), 0)
        light.setBillboardPointEye()
    return root


def boulder_model(kind, variant, glow_tex):
    """The alien's boulders: warm scorched rock (soft) or dark fused rock with a
    glowing core (hard)."""
    rng = np.random.default_rng(100 + variant)
    if kind == 1:
        m = meshes.rock('soft', rng, (0.66, 0.45, 0.34, 1))
    else:
        m = meshes.rock('hard', rng, (0.24, 0.17, 0.17, 1))
    m.setScale(0.42, 0.42, 0.36)
    m.setZ(0.18)
    root = NodePath('boulder')
    m.reparentTo(root)
    g = _card(root, 1.3 if kind == 2 else 0.9, glow_tex,
              (1, 0.45, 0.12, 0.9) if kind == 2 else (1, 0.55, 0.25, 0.35))
    g.setZ(0.25)
    g.setBillboardPointEye()
    return root


def shuttle_model(glow_tex):
    """Relief shuttle standing on the pad: white body, orange band, fins,
    engine bell; `flame` is the exhaust node shown during lift-off."""
    root = NodePath('shuttle')
    body = meshes.cylinder('body', 0.28, 1.5, (0.93, 0.94, 0.96, 1), seg=24)
    body.setR(90)
    body.setZ(1.0)
    body.reparentTo(root)
    band = meshes.cylinder('band', 0.29, 0.14, (0.95, 0.5, 0.15, 1), seg=24)
    band.setR(90)
    band.setZ(1.35)
    band.reparentTo(root)
    nose = meshes.sphere('nose', (0.93, 0.94, 0.96, 1), sub=3)
    nose.setScale(0.28, 0.28, 0.55)
    nose.setZ(1.75)
    nose.reparentTo(root)
    bell = meshes.cylinder('bell', 0.18, 0.25, (0.25, 0.26, 0.3, 1), seg=20)
    bell.setR(90)
    bell.setZ(0.2)
    bell.reparentTo(root)
    for k in range(3):
        a = k * 120
        fin = meshes.box('fin', 0.05, 0.42, 0.55, (0.85, 0.3, 0.12, 1), (0, 0.38, 0.45))
        fin.setH(a)
        fin.reparentTo(root)
    meshes.box('window', 0.16, 0.05, 0.12, (0.1, 0.2, 0.35, 1), (0, 0.27, 1.55)).reparentTo(root)
    flame = root.attachNewNode('flame')
    for size, col, z in ((0.9, (1, 0.95, 0.8, 1), -0.05), (1.8, (1, 0.55, 0.15, 0.9), -0.3),
                         (3.2, (1, 0.35, 0.1, 0.5), -0.6)):
        c = _card(flame, size, glow_tex, col)
        c.setZ(z)
        c.setBillboardPointEye()
    flame.hide()
    return root, flame
