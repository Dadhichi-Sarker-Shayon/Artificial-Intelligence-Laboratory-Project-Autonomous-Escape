"""Procedural geometry for the 3D view, built from numpy arrays.

Every mesh uses one vertex format (position, normal, RGBA colour, UV, all
float32) and 32-bit triangle indices, written straight into Panda3D buffers.
"""
import math

import numpy as np
from panda3d.core import (ColorBlendAttrib, Geom, GeomNode, GeomTriangles, GeomVertexArrayFormat,
                          GeomVertexData, GeomVertexFormat, InternalName, NodePath, Texture,
                          TransparencyAttrib)

_FORMAT = None


def _fmt():
    global _FORMAT
    if _FORMAT is None:
        arr = GeomVertexArrayFormat()
        arr.addColumn(InternalName.getVertex(), 3, Geom.NT_float32, Geom.C_point)
        arr.addColumn(InternalName.getNormal(), 3, Geom.NT_float32, Geom.C_normal)
        arr.addColumn(InternalName.getColor(), 4, Geom.NT_float32, Geom.C_color)
        arr.addColumn(InternalName.getTexcoord(), 2, Geom.NT_float32, Geom.C_texcoord)
        _FORMAT = GeomVertexFormat.registerFormat(GeomVertexFormat(arr))
    return _FORMAT


def mesh(name, pos, nrm, idx, col=None, uv=None):
    """NodePath from arrays: pos/nrm (N,3), idx (M,3), col (N,4) or RGBA tuple, uv (N,2)."""
    n = len(pos)
    if col is None:
        col = (1, 1, 1, 1)
    col = np.asarray(col, np.float32)
    if col.ndim == 1:
        col = np.broadcast_to(col, (n, 4))
    if uv is None:
        uv = np.zeros((n, 2), np.float32)
    data = np.hstack([np.asarray(pos, np.float32), np.asarray(nrm, np.float32), col,
                      np.asarray(uv, np.float32)]).astype(np.float32)
    vd = GeomVertexData(name, _fmt(), Geom.UHStatic)
    vd.uncleanSetNumRows(n)
    memoryview(vd.modifyArray(0)).cast('B')[:] = data.tobytes()
    prim = GeomTriangles(Geom.UHStatic)
    prim.setIndexType(Geom.NT_uint32)
    flat = np.asarray(idx, np.uint32).reshape(-1)
    verts = prim.modifyVertices()
    verts.uncleanSetNumRows(len(flat))
    memoryview(verts).cast('B')[:] = flat.tobytes()
    geom = Geom(vd)
    geom.addPrimitive(prim)
    node = GeomNode(name)
    node.addGeom(geom)
    return NodePath(node)


def smooth_normals(pos, idx):
    pos = np.asarray(pos, np.float32)
    idx = np.asarray(idx)
    a, b, c = pos[idx[:, 0]], pos[idx[:, 1]], pos[idx[:, 2]]
    fn = np.cross(b - a, c - a)
    nrm = np.zeros_like(pos)
    for k in range(3):
        np.add.at(nrm, idx[:, k], fn)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    return nrm


# ------------------------------------------------------------------ shapes
def box_arrays(sx, sy, sz, centre=(0, 0, 0)):
    """Axis-aligned box with flat faces; returns pos, nrm, idx."""
    cx, cy, cz = centre
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    faces = [((1, 0, 0), [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)]),
             ((-1, 0, 0), [(-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1)]),
             ((0, 1, 0), [(1, 1, -1), (-1, 1, -1), (-1, 1, 1), (1, 1, 1)]),
             ((0, -1, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)]),
             ((0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]),
             ((0, 0, -1), [(-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)])]
    pos, nrm, idx = [], [], []
    for n, quad in faces:
        base = len(pos)
        for x, y, z in quad:
            pos.append((cx + x * hx, cy + y * hy, cz + z * hz))
            nrm.append(n)
        idx += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    return np.array(pos, np.float32), np.array(nrm, np.float32), np.array(idx)


def box(name, sx, sy, sz, color, centre=(0, 0, 0)):
    p, n, i = box_arrays(sx, sy, sz, centre)
    return mesh(name, p, n, i, color)


def cylinder(name, r, length, color, seg=20, cap_color=None):
    """Cylinder along the x axis (a wheel), centred on the origin."""
    pos, nrm, col, idx = [], [], [], []
    cap_color = cap_color or color
    for k in range(seg):
        a0, a1 = 2 * math.pi * k / seg, 2 * math.pi * (k + 1) / seg
        base = len(pos)
        for a in (a0, a1):
            y, z = r * math.cos(a), r * math.sin(a)
            for x in (-length / 2, length / 2):
                pos.append((x, y, z))
                nrm.append((0, math.cos(a), math.sin(a)))
                col.append(color)
        idx += [(base, base + 2, base + 3), (base, base + 3, base + 1)]
    for side in (-1, 1):
        c0 = len(pos)
        pos.append((side * length / 2, 0, 0))
        nrm.append((side, 0, 0))
        col.append(cap_color)
        for k in range(seg + 1):
            a = 2 * math.pi * k / seg
            pos.append((side * length / 2, r * math.cos(a), r * math.sin(a)))
            nrm.append((side, 0, 0))
            col.append(cap_color)
        for k in range(seg):
            tri = (c0, c0 + 1 + k, c0 + 2 + k)
            idx.append(tri if side > 0 else tri[::-1])
    return mesh(name, np.array(pos), np.array(nrm), np.array(idx), np.array(col, np.float32))


def _icosphere(sub):
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
         (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11),
         (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(sub):
        cache, nf = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = v[a] + v[b]
                v.append(m / np.linalg.norm(m))
                cache[key] = len(v) - 1
            return cache[key]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(v, np.float32), np.array(f)


_SPHERES = {}


def sphere_arrays(sub=2):
    if sub not in _SPHERES:
        _SPHERES[sub] = _icosphere(sub)
    return _SPHERES[sub]


def rock_arrays(rng, sub=2, rough=0.28):
    """Lumpy rock: an icosphere pushed around by random low-frequency waves,
    with a flattened base. Unit radius."""
    v, f = sphere_arrays(sub)
    d = np.zeros(len(v), np.float32)
    for _ in range(6):
        k = rng.normal(size=3)
        k /= np.linalg.norm(k)
        d += rng.uniform(0.3, 1.0) * np.sin(v @ k * rng.uniform(1.5, 4.5) + rng.uniform(0, 6.3))
    d = 1 + rough * d / 3
    p = v * d[:, None]
    p[:, 2] = np.maximum(p[:, 2], -0.35)
    return p.astype(np.float32), f


def flat_shade(pos, idx):
    """Un-share vertices so every triangle gets its own normal (faceted rock)."""
    p = np.asarray(pos, np.float32)[np.asarray(idx).reshape(-1)]
    a, b, c = p[0::3], p[1::3], p[2::3]
    fn = np.cross(b - a, c - a)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-9
    return p, np.repeat(fn, 3, axis=0), np.arange(len(p)).reshape(-1, 3)


def rock(name, rng, color, sub=2):
    p, f = rock_arrays(rng, sub, rough=0.45)
    p, n, i = flat_shade(p, f)
    shade = rng.uniform(0.82, 1.08, len(p) // 3).repeat(3)[:, None]
    col = np.clip(np.asarray(color, np.float32)[None, :] * np.hstack([shade, shade, shade, np.ones_like(shade)]), 0, 1)
    return mesh(name, p, n, i, col)


def sphere(name, color, sub=2):
    v, f = sphere_arrays(sub)
    return mesh(name, v, v, f, color)


def ring(name, r_in, r_out, color, seg=48):
    """Flat annulus on the ground plane, facing up."""
    pos, idx = [], []
    for k in range(seg + 1):
        a = 2 * math.pi * k / seg
        pos += [(r_in * math.cos(a), r_in * math.sin(a), 0), (r_out * math.cos(a), r_out * math.sin(a), 0)]
    for k in range(seg):
        b = 2 * k
        idx += [(b, b + 1, b + 3), (b, b + 3, b + 2)]
    pos = np.array(pos, np.float32)
    return mesh(name, pos, np.tile((0, 0, 1), (len(pos), 1)), np.array(idx), color)


def disc(name, r, color, seg=48):
    pos = [(0, 0, 0)] + [(r * math.cos(2 * math.pi * k / seg), r * math.sin(2 * math.pi * k / seg), 0)
                         for k in range(seg + 1)]
    idx = [(0, k, k + 1) for k in range(1, seg + 1)]
    pos = np.array(pos, np.float32)
    return mesh(name, pos, np.tile((0, 0, 1), (len(pos), 1)), np.array(idx), color)


def cone(name, r_top, r_bottom, height, color, seg=32):
    """Open cone from z=0 (r_top) down to z=-height (r_bottom): a tractor beam."""
    pos, idx = [], []
    for k in range(seg + 1):
        a = 2 * math.pi * k / seg
        pos += [(r_top * math.cos(a), r_top * math.sin(a), 0),
                (r_bottom * math.cos(a), r_bottom * math.sin(a), -height)]
    for k in range(seg):
        b = 2 * k
        idx += [(b, b + 1, b + 3), (b, b + 3, b + 2)]
    pos = np.array(pos, np.float32)
    nrm = pos * np.array([1, 1, 0], np.float32)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-6
    return mesh(name, pos, nrm, np.array(idx), color)


def heightfield(name, heights, x0, y0, step, zscale, uv=True):
    """Grid mesh: heights[row, col] -> z; row grows towards -y, col towards +x."""
    rows, cols = heights.shape
    jj, ii = np.meshgrid(np.arange(cols), np.arange(rows))
    x = x0 + jj * step
    y = y0 - ii * step
    z = heights * zscale
    pos = np.stack([x, y, z], -1).reshape(-1, 3)
    gy, gx = np.gradient(z, step)
    nrm = np.stack([-gx, gy, np.ones_like(z)], -1).reshape(-1, 3)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    uvs = np.stack([jj / (cols - 1), 1 - ii / (rows - 1)], -1).reshape(-1, 2)
    a = (ii[:-1, :-1] * cols + jj[:-1, :-1]).reshape(-1)
    b, c, d = a + 1, a + cols, a + cols + 1
    idx = np.concatenate([np.stack([a, c, d], -1), np.stack([a, d, b], -1)])
    return mesh(name, pos, nrm, idx, (1, 1, 1, 1), uvs)


# ------------------------------------------------------------------ textures
def texture(img, name='tex', mipmap=True):
    """Texture from a uint8 array (rows x cols x 3 or 4, top row first)."""
    img = np.ascontiguousarray(img)
    h, w, ch = img.shape
    tex = Texture(name)
    tex.setup2dTexture(w, h, Texture.T_unsigned_byte, Texture.F_rgba8 if ch == 4 else Texture.F_rgb8)
    order = [2, 1, 0, 3][:ch]                       # Panda wants BGR(A), bottom row first
    tex.setRamImage(np.ascontiguousarray(img[::-1, :, order]).tobytes())
    if mipmap:
        tex.setMinfilter(Texture.FT_linear_mipmap_linear)
        tex.setAnisotropicDegree(8)
    tex.setMagfilter(Texture.FT_linear)
    tex.setWrapU(Texture.WM_clamp)
    tex.setWrapV(Texture.WM_clamp)
    return tex


def glow_texture(size=64):
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.hypot(xx - size / 2 + 0.5, yy - size / 2 + 0.5) / (size / 2)
    a = np.clip(1 - d, 0, 1) ** 2.2
    img = np.zeros((size, size, 4), np.uint8)
    img[..., :3] = 255
    img[..., 3] = (a * 255).astype(np.uint8)
    return texture(img, 'glow', mipmap=False)


def additive(np_):
    """Glowing look: add light, ignore depth writes, no lighting."""
    np_.setAttrib(ColorBlendAttrib.make(ColorBlendAttrib.MAdd, ColorBlendAttrib.OIncomingAlpha,
                                        ColorBlendAttrib.OOne))
    np_.setDepthWrite(False)
    np_.setLightOff(1)
    np_.setShaderOff(1)
    np_.setBin('fixed', 10)
    return np_


def translucent(np_):
    np_.setTransparency(TransparencyAttrib.MAlpha)
    np_.setDepthWrite(False)
    np_.setBin('fixed', 5)
    return np_
