"""Small 2D UI kit on top of Panda3D's aspect2d: text, framed panels,
buttons and filled shapes, all in the game's palette and font."""
import math
import os

from panda3d.core import (Filename, GeomNode, LineSegs, NodePath, TextNode, Texture, TransparencyAttrib)

CYAN = (0.34, 0.86, 0.91, 1)
ORANGE = (1, 0.62, 0.25, 1)
RED = (0.94, 0.33, 0.38, 1)
GREEN = (0.43, 0.86, 0.55, 1)
YELLOW = (0.98, 0.82, 0.35, 1)
PURPLE = (0.72, 0.6, 1, 1)
TEXT = (0.93, 0.93, 0.96, 1)
MUTED = (0.62, 0.6, 0.72, 1)
DIM = (0.4, 0.38, 0.5, 1)
PANEL = (0.06, 0.05, 0.14, 0.84)
EDGE = (0.25, 0.22, 0.46, 1)


class UI:
    def __init__(self, base):
        self.base = base
        self.font = self._font('segoeui.ttf')
        self.bold = self._font('segoeuib.ttf') or self.font

    def _font(self, name):
        path = os.path.join('C:/Windows/Fonts', name)
        if not os.path.exists(path):
            return None
        f = self.base.loader.loadFont(Filename.fromOsSpecific(path).getFullpath())
        f.setPixelsPerUnit(64)
        return f

    def text(self, parent, s, pos, scale=0.05, fg=TEXT, align=TextNode.ALeft, bold=False, wrap=None, shadow=True):
        from direct.gui.OnscreenText import OnscreenText
        t = OnscreenText(text=s, pos=pos, scale=scale, fg=fg, align=align, parent=parent, mayChange=True,
                         font=self.bold if bold else self.font, shadow=(0, 0, 0, 0.6) if shadow else None,
                         wordwrap=wrap)
        return t

    def panel(self, parent, l, r, b, t, color=PANEL, edge=EDGE, thickness=1.5):
        from direct.gui.DirectGui import DirectFrame
        f = DirectFrame(parent=parent, frameColor=color, frameSize=(l, r, b, t))
        if edge:
            f.attachNewNode(outline(l, r, b, t, edge, thickness))
        return f

    def button(self, parent, label, x, y, w, h, command, args=(), selected=False, enabled=True, scale=0.05,
               color=None, fg=None):
        from direct.gui.DirectGui import DirectButton, DGG
        base = color or ((0.14, 0.13, 0.3, 0.92) if selected else (0.08, 0.07, 0.18, 0.88))
        b = DirectButton(parent=parent, frameSize=(-w / 2, w / 2, -h / 2, h / 2), frameColor=base,
                         text=label, text_font=self.bold, text_scale=scale, text_pos=(0, -scale * 0.35),
                         text_fg=fg or (TEXT if enabled else DIM), relief=DGG.FLAT, pos=(x, 0, y),
                         command=command if enabled else None, extraArgs=list(args), pressEffect=1)
        b.attachNewNode(outline(-w / 2, w / 2, -h / 2, h / 2, CYAN if selected else EDGE, 2 if selected else 1.2))
        return b


def outline(l, r, b, t, color, thickness=1.5):
    ls = LineSegs()
    ls.setColor(*color)
    ls.setThickness(thickness)
    ls.moveTo(l, 0, b)
    for x, y in ((r, b), (r, t), (l, t), (l, b)):
        ls.drawTo(x, 0, y)
    return ls.create()


def star(parent, x, y, r, color):
    """Filled five-point star in aspect2d."""
    from panda3d.core import Geom, GeomTriangles, GeomVertexData, GeomVertexFormat, GeomVertexWriter
    vd = GeomVertexData('star', GeomVertexFormat.getV3c4(), Geom.UHStatic)
    vw, cw = GeomVertexWriter(vd, 'vertex'), GeomVertexWriter(vd, 'color')
    vw.addData3(0, 0, 0)
    cw.addData4(*color)
    for k in range(11):
        a = math.pi / 2 + k * math.pi / 5
        rr = r if k % 2 == 0 else r * 0.42
        vw.addData3(rr * math.cos(a), 0, rr * math.sin(a))
        cw.addData4(*color)
    tris = GeomTriangles(Geom.UHStatic)
    for k in range(10):
        tris.addVertices(0, k + 2, k + 1)
    g = Geom(vd)
    g.addPrimitive(tris)
    node = GeomNode('star')
    node.addGeom(g)
    np_ = parent.attachNewNode(node)
    np_.setPos(x, 0, y)
    np_.setTwoSided(True)
    np_.setTransparency(TransparencyAttrib.MAlpha)
    return np_


class DynamicTexture:
    """A texture whose pixels are replaced every frame from a pygame surface."""

    def __init__(self, w, h, name):
        self.tex = Texture(name)
        self.tex.setup2dTexture(w, h, Texture.T_unsigned_byte, Texture.F_rgba8)
        self.tex.setMagfilter(Texture.FT_linear)
        self.tex.setMinfilter(Texture.FT_linear)
        self.w, self.h = w, h

    def load(self, surface):
        import pygame as pg
        data = pg.image.tobytes(pg.transform.flip(surface, False, True), 'BGRA')
        self.tex.setRamImage(data)
